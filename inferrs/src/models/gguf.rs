//! GGUF loading infrastructure for inferrs.
//!
//! This module owns:
//! - [`GgufBackend`]: a lazy, dequantizing [`SimpleBackend`] backed by a GGUF file.
//! - [`GgufNaming`]: detection of whether a GGUF uses HuggingFace or llama.cpp tensor names.
//! - [`RenamingBackend`]: a transparent wrapper that maps HF tensor names → llama.cpp names
//!   on every lookup, so model code requires no changes to load Ollama blobs.
//! - [`var_builder_from_gguf`]: the entry point for building a [`VarBuilder`] from a GGUF.
//! - Per-architecture rename tables (`*_hf_to_llama`): pure functions that convert a
//!   HuggingFace tensor name to its llama.cpp equivalent for a specific model family.
//!
//! ## Adding a new architecture
//!
//! 1. Add a `pub fn <arch>_hf_to_llama(name: &str) -> String` below, mapping all tensors.
//! 2. Add a match arm in `crate::models::load_model` to select it when the arch is detected.
//! 3. Add unit tests (see the test module at the bottom of this file).

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use std::path::Path;

// ---------------------------------------------------------------------------
// GgufBackend
// ---------------------------------------------------------------------------

/// A lazy [`candle_nn::var_builder::SimpleBackend`] backed by a GGUF file.
///
/// Tensors are dequantized on demand — only when the model calls
/// `VarBuilder::get` for that specific weight.  This avoids the huge memory
/// spike and slow startup of eager loading (all 2 000+ tensors upfront).
///
/// The GGUF file is kept open for the lifetime of the backend; a `Mutex`
/// around the `BufReader` satisfies the `Sync` requirement of `SimpleBackend`.
struct GgufBackend {
    content: candle_core::quantized::gguf_file::Content,
    reader: std::sync::Mutex<std::io::BufReader<std::fs::File>>,
    device: Device,
}

impl candle_nn::var_builder::SimpleBackend for GgufBackend {
    fn get(
        &self,
        s: candle_core::Shape,
        name: &str,
        _: candle_nn::Init,
        dtype: DType,
        dev: &Device,
    ) -> candle_core::Result<Tensor> {
        let mut reader = self.reader.lock().expect("gguf reader lock poisoned");
        // Use `dev` when it requests CPU placement (e.g. for the enormous
        // embed_tokens_per_layer table) so the data never touches GPU memory.
        let load_dev = if matches!(dev, Device::Cpu) { dev } else { &self.device };
        let qt = self
            .content
            .tensor(&mut *reader, name, load_dev)
            .map_err(|e| {
                candle_core::Error::CannotFindTensor {
                    path: format!("{name}: {e}"),
                }
                .bt()
            })?;
        let tensor = qt.dequantize(dev)?.to_dtype(dtype)?;
        if tensor.shape() != &s {
            candle_core::bail!(
                "shape mismatch for {name}: expected {s:?}, got {:?}",
                tensor.shape()
            );
        }
        Ok(tensor)
    }

    fn get_unchecked(&self, name: &str, dtype: DType, dev: &Device) -> candle_core::Result<Tensor> {
        let mut reader = self.reader.lock().expect("gguf reader lock poisoned");
        let load_dev = if matches!(dev, Device::Cpu) { dev } else { &self.device };
        let qt = self
            .content
            .tensor(&mut *reader, name, load_dev)
            .map_err(|e| {
                candle_core::Error::CannotFindTensor {
                    path: format!("{name}: {e}"),
                }
                .bt()
            })?;
        qt.dequantize(dev)?.to_dtype(dtype)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        self.content.tensor_infos.contains_key(name)
    }
}

// ---------------------------------------------------------------------------
// GgufNaming
// ---------------------------------------------------------------------------

/// Tensor naming convention used in a GGUF file.
///
/// HuggingFace GGUFs mirror the safetensors layout (`model.layers.{i}.*`
/// or `model.language_model.layers.{i}.*` depending on the architecture).
/// llama.cpp GGUFs (produced by llama.cpp / stored by Ollama) use a uniform
/// `blk.{i}.*` layout regardless of architecture.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum GgufNaming {
    /// HuggingFace / safetensors tensor names (architecture-dependent prefix).
    HuggingFace,
    /// llama.cpp names (used by Ollama): `token_embd.weight`, `blk.{i}.attn_q.weight`, …
    LlamaCpp,
}

impl GgufNaming {
    fn detect(content: &candle_core::quantized::gguf_file::Content) -> Self {
        if content.tensor_infos.contains_key("token_embd.weight") {
            Self::LlamaCpp
        } else {
            Self::HuggingFace
        }
    }
}

// ---------------------------------------------------------------------------
// GemmaNormFixBackend
// ---------------------------------------------------------------------------

/// A [`SimpleBackend`] that subtracts 1.0 from every `*norm.weight` tensor.
///
/// Compensates for the Gemma3 convention difference: HuggingFace safetensors
/// store RMSNorm weights as `actual_scale - 1.0` (initialised as zeros), while
/// llama.cpp GGUFs store the actual scale.  The candle Gemma3 implementation
/// always adds `+1.0` back, so we pre-subtract here so the net is correct.
///
/// Applied only when loading a llama.cpp-named GGUF for a Gemma3 model.
/// The rename (HF → llama.cpp names) is handled separately by
/// [`VarBuilder::rename_f`] so the name seen here is already the llama.cpp
/// name — which also consistently ends with `norm.weight` for norm tensors.
struct GemmaNormFixBackend {
    inner: GgufBackend,
}

impl candle_nn::var_builder::SimpleBackend for GemmaNormFixBackend {
    fn get(
        &self,
        s: candle_core::Shape,
        name: &str,
        init: candle_nn::Init,
        dtype: DType,
        dev: &Device,
    ) -> candle_core::Result<Tensor> {
        let t = self.inner.get(s, name, init, dtype, dev)?;
        if name.ends_with("norm.weight") { t - 1.0f64 } else { Ok(t) }
    }

    fn get_unchecked(&self, name: &str, dtype: DType, dev: &Device) -> candle_core::Result<Tensor> {
        let t = self.inner.get_unchecked(name, dtype, dev)?;
        if name.ends_with("norm.weight") { t - 1.0f64 } else { Ok(t) }
    }

    fn contains_tensor(&self, name: &str) -> bool {
        self.inner.contains_tensor(name)
    }
}

// ---------------------------------------------------------------------------
// var_builder_from_gguf
// ---------------------------------------------------------------------------

/// Build a [`VarBuilder`] backed by a GGUF file.
///
/// `arch_rename` is the architecture's HF→llama.cpp rename function (e.g.
/// [`gemma4_hf_to_llama`]).  It is applied **only** when the GGUF is found to
/// use llama.cpp naming; HF-named GGUFs (e.g. from `--gguf` with a custom
/// export) pass through unchanged.
///
/// `gemma_norm_fix` — pass `true` for Gemma 3 models.  llama.cpp GGUFs store
/// RMSNorm weights as the **actual** scale, whereas
/// `candle_transformers::models::gemma3` always adds `+1.0` to every weight
/// (the HuggingFace initialisation convention).  Setting this flag subtracts
/// 1.0 from every `*norm.weight` tensor so the net value seen by the model
/// code is the correct scale.
///
/// Tensors are dequantized lazily — only on first access — so startup is fast
/// and peak memory is bounded by the model's actual weight usage.
///
/// Also returns the detected [`GgufNaming`] so callers can apply the same
/// conditional rename to supplementary builders (e.g. `QGgufVarBuilder`).
pub(crate) fn var_builder_from_gguf(
    gguf_path: &Path,
    dtype: DType,
    device: &Device,
    arch_rename: Option<fn(&str) -> String>,
    gemma_norm_fix: bool,
) -> Result<(VarBuilder<'static>, GgufNaming)> {
    use candle_core::quantized::gguf_file;

    let file = std::fs::File::open(gguf_path)
        .with_context(|| format!("Cannot open GGUF {}", gguf_path.display()))?;
    let mut reader = std::io::BufReader::new(file);

    let content = gguf_file::Content::read(&mut reader)
        .with_context(|| format!("Failed to parse GGUF header in {}", gguf_path.display()))?;

    let naming = GgufNaming::detect(&content);
    tracing::info!(
        "Opened GGUF with {} tensors ({:?} naming): {}",
        content.tensor_infos.len(),
        naming,
        gguf_path.display()
    );

    let backend = GgufBackend {
        content,
        reader: std::sync::Mutex::new(reader),
        device: device.clone(),
    };

    // Apply rename and norm fix only for llama.cpp-named GGUFs.
    // HF-named GGUFs already use the tensor names the model code expects.
    let is_llama_cpp = naming == GgufNaming::LlamaCpp;

    // Wrap in GemmaNormFixBackend before building the VarBuilder so that
    // rename_f sees the llama.cpp name (which also ends with `norm.weight`
    // for norm tensors, so the fix fires correctly after renaming).
    let base_vb = if gemma_norm_fix && is_llama_cpp {
        VarBuilder::from_backend(Box::new(GemmaNormFixBackend { inner: backend }), dtype, device.clone())
    } else {
        VarBuilder::from_backend(Box::new(backend), dtype, device.clone())
    };

    // Apply the HF→llama.cpp rename via the built-in VarBuilder::rename_f.
    let vb = match is_llama_cpp.then_some(arch_rename).flatten() {
        Some(f) => base_vb.rename_f(f),
        None => base_vb,
    };

    Ok((vb, naming))
}

// ---------------------------------------------------------------------------
// Per-architecture HuggingFace → llama.cpp tensor name mappings
// ---------------------------------------------------------------------------
//
// Each function maps a HuggingFace-style tensor name to its llama.cpp/Ollama
// GGUF equivalent for one model family.  Unknown names pass through unchanged
// so that the backend emits a CannotFindTensor error rather than silently
// loading the wrong weight.
//
// Convention for the blk.{i}.* suffix tables (right-hand side):
//   attn_norm      ← input_layernorm
//   ffn_norm       ← pre_feedforward_layernorm  (or post_attention_layernorm)
//   post_ffw_norm  ← post_feedforward_layernorm
//   post_attention_norm ← post_attention_layernorm  (Gemma4)
//   attn_q/k/v/output   ← self_attn q/k/v/o_proj
//   attn_q_norm / attn_k_norm ← self_attn q_norm / k_norm
//   ffn_gate / ffn_up / ffn_down ← mlp gate/up/down_proj

// ── Gemma 4 ─────────────────────────────────────────────────────────────────

/// Map a HuggingFace tensor name to its llama.cpp/Ollama GGUF equivalent for Gemma 4.
pub fn gemma4_hf_to_llama(name: &str) -> String {
    match name {
        "model.language_model.embed_tokens.weight" => return "token_embd.weight".into(),
        "model.language_model.embed_tokens_per_layer.weight" => {
            return "per_layer_token_embd.weight".into()
        }
        "model.language_model.per_layer_model_projection.weight" => {
            return "per_layer_model_proj.weight".into()
        }
        "model.language_model.per_layer_projection_norm.weight" => {
            return "per_layer_proj_norm.weight".into()
        }
        "model.language_model.norm.weight" => return "output_norm.weight".into(),
        _ => {}
    }

    const PREFIX: &str = "model.language_model.layers.";
    if let Some(after_prefix) = name.strip_prefix(PREFIX) {
        if let Some(dot_pos) = after_prefix.find('.') {
            let layer_idx = &after_prefix[..dot_pos];
            let rest = &after_prefix[dot_pos + 1..];
            let suffix = match rest {
                "input_layernorm.weight" => "attn_norm.weight",
                "pre_feedforward_layernorm.weight" => "ffn_norm.weight",
                "post_feedforward_layernorm.weight" => "post_ffw_norm.weight",
                "post_attention_layernorm.weight" => "post_attention_norm.weight",
                "self_attn.q_proj.weight" => "attn_q.weight",
                "self_attn.k_proj.weight" => "attn_k.weight",
                "self_attn.v_proj.weight" => "attn_v.weight",
                "self_attn.o_proj.weight" => "attn_output.weight",
                "self_attn.q_norm.weight" => "attn_q_norm.weight",
                "self_attn.k_norm.weight" => "attn_k_norm.weight",
                "mlp.gate_proj.weight" => "ffn_gate.weight",
                "mlp.up_proj.weight" => "ffn_up.weight",
                "mlp.down_proj.weight" => "ffn_down.weight",
                "per_layer_input_gate.weight" => "inp_gate.weight",
                "per_layer_projection.weight" => "proj.weight",
                "post_per_layer_input_norm.weight" => "post_norm.weight",
                // layer_scalar has no ".weight" suffix in HF naming
                "layer_scalar" => "layer_output_scale.weight",
                _ => return name.to_string(),
            };
            return format!("blk.{layer_idx}.{suffix}");
        }
    }

    name.to_string()
}

// ── Gemma 3 ─────────────────────────────────────────────────────────────────

/// Map a HuggingFace tensor name to its llama.cpp/Ollama GGUF equivalent for Gemma 3.
///
/// Gemma 3 uses the same 4-norm-per-layer pattern as Gemma 4 but without the
/// per-layer-input tensors (no `embed_tokens_per_layer`, no `layer_scalar`).
/// The lm_head is tied to embed_tokens — no separate tensor.
pub fn gemma3_hf_to_llama(name: &str) -> String {
    match name {
        "model.embed_tokens.weight" => return "token_embd.weight".into(),
        "model.norm.weight"         => return "output_norm.weight".into(),
        _ => {}
    }

    const PREFIX: &str = "model.layers.";
    if let Some(after_prefix) = name.strip_prefix(PREFIX) {
        if let Some(dot_pos) = after_prefix.find('.') {
            let layer_idx = &after_prefix[..dot_pos];
            let rest = &after_prefix[dot_pos + 1..];
            let suffix = match rest {
                "input_layernorm.weight"            => "attn_norm.weight",
                "pre_feedforward_layernorm.weight"  => "ffn_norm.weight",
                "post_feedforward_layernorm.weight" => "post_ffw_norm.weight",
                "post_attention_layernorm.weight"   => "post_attention_norm.weight",
                "self_attn.q_proj.weight"           => "attn_q.weight",
                "self_attn.k_proj.weight"           => "attn_k.weight",
                "self_attn.v_proj.weight"           => "attn_v.weight",
                "self_attn.o_proj.weight"           => "attn_output.weight",
                "self_attn.q_norm.weight"           => "attn_q_norm.weight",
                "self_attn.k_norm.weight"           => "attn_k_norm.weight",
                "mlp.gate_proj.weight"              => "ffn_gate.weight",
                "mlp.up_proj.weight"                => "ffn_up.weight",
                "mlp.down_proj.weight"              => "ffn_down.weight",
                _ => return name.to_string(),
            };
            return format!("blk.{layer_idx}.{suffix}");
        }
    }

    name.to_string()
}

// ── Qwen 3 ──────────────────────────────────────────────────────────────────

/// Map a HuggingFace tensor name to its llama.cpp/Ollama GGUF equivalent for Qwen 3.
pub fn qwen3_hf_to_llama(name: &str) -> String {
    match name {
        "model.embed_tokens.weight" => return "token_embd.weight".into(),
        "model.norm.weight" => return "output_norm.weight".into(),
        "lm_head.weight" => return "output.weight".into(),
        _ => {}
    }

    const PREFIX: &str = "model.layers.";
    if let Some(after_prefix) = name.strip_prefix(PREFIX) {
        if let Some(dot_pos) = after_prefix.find('.') {
            let layer_idx = &after_prefix[..dot_pos];
            let rest = &after_prefix[dot_pos + 1..];
            let suffix = match rest {
                "input_layernorm.weight" => "attn_norm.weight",
                "post_attention_layernorm.weight" => "ffn_norm.weight",
                "self_attn.q_proj.weight" => "attn_q.weight",
                "self_attn.k_proj.weight" => "attn_k.weight",
                "self_attn.v_proj.weight" => "attn_v.weight",
                "self_attn.o_proj.weight" => "attn_output.weight",
                "self_attn.q_norm.weight" => "attn_q_norm.weight",
                "self_attn.k_norm.weight" => "attn_k_norm.weight",
                "mlp.gate_proj.weight" => "ffn_gate.weight",
                "mlp.up_proj.weight" => "ffn_up.weight",
                "mlp.down_proj.weight" => "ffn_down.weight",
                _ => return name.to_string(),
            };
            return format!("blk.{layer_idx}.{suffix}");
        }
    }

    name.to_string()
}

// ── Qwen 3.5 ────────────────────────────────────────────────────────────────

/// Map a HuggingFace tensor name to its llama.cpp/Ollama GGUF equivalent for Qwen 3.5.
///
/// Qwen 3.5 uses a hybrid architecture (full-attention + linear-attention layers).
/// All weights live under `model.language_model.*` (same prefix as Gemma 4).
///
/// Full-attention and MLP tensors follow the standard pattern.  The
/// linear-attention (Gated Delta Rule SSM) tensors use `ssm_*` names in
/// llama.cpp — mapped here from the HuggingFace `linear_attn.*` sub-tree.
///
/// The lm_head is tied to embed_tokens — no separate tensor.
pub fn qwen35_hf_to_llama(name: &str) -> String {
    match name {
        "model.language_model.embed_tokens.weight" => return "token_embd.weight".into(),
        "model.language_model.norm.weight"         => return "output_norm.weight".into(),
        _ => {}
    }

    const PREFIX: &str = "model.language_model.layers.";
    if let Some(after_prefix) = name.strip_prefix(PREFIX) {
        if let Some(dot_pos) = after_prefix.find('.') {
            let layer_idx = &after_prefix[..dot_pos];
            let rest = &after_prefix[dot_pos + 1..];
            let suffix = match rest {
                // Layer norms (both full-attention and linear-attention layers)
                "input_layernorm.weight"           => "attn_norm.weight",
                "post_attention_layernorm.weight"  => "post_attention_norm.weight",
                // Full-attention (GQA + QK-norm + RoPE)
                // Note: q_proj stores [num_heads*head_dim*2, hidden] — the second half
                // is the per-head output gate that attn_output_gate extracts at runtime.
                "self_attn.q_proj.weight"          => "attn_q.weight",
                "self_attn.k_proj.weight"          => "attn_k.weight",
                "self_attn.v_proj.weight"          => "attn_v.weight",
                "self_attn.o_proj.weight"          => "attn_output.weight",
                "self_attn.q_norm.weight"          => "attn_q_norm.weight",
                "self_attn.k_norm.weight"          => "attn_k_norm.weight",
                // MLP (SwiGLU, shared by both layer types)
                "mlp.gate_proj.weight"             => "ffn_gate.weight",
                "mlp.up_proj.weight"               => "ffn_up.weight",
                "mlp.down_proj.weight"             => "ffn_down.weight",
                // Linear-attention (Gated Delta Rule SSM) layers
                "linear_attn.in_proj_qkv.weight"   => "attn_qkv.weight",
                "linear_attn.in_proj_z.weight"     => "attn_gate.weight",
                "linear_attn.in_proj_a.weight"     => "ssm_alpha.weight",
                "linear_attn.in_proj_b.weight"     => "ssm_beta.weight",
                "linear_attn.conv1d.weight"        => "ssm_conv1d.weight",
                "linear_attn.A_log"                => "ssm_a",
                "linear_attn.dt_bias"              => "ssm_dt",
                "linear_attn.norm.weight"          => "ssm_norm.weight",
                "linear_attn.out_proj.weight"      => "ssm_out.weight",
                _ => return name.to_string(),
            };
            return format!("blk.{layer_idx}.{suffix}");
        }
    }

    name.to_string()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Check every `(input, expected)` pair against a rename function.
    /// Reports the failing input on assertion failure.
    fn check(f: fn(&str) -> String, cases: &[(&str, &str)]) {
        for (input, expected) in cases {
            assert_eq!(f(input), *expected, "rename({input:?})");
        }
    }

    #[test]
    fn gemma4_rename_table() {
        check(gemma4_hf_to_llama, &[
            // globals
            ("model.language_model.embed_tokens.weight",              "token_embd.weight"),
            ("model.language_model.embed_tokens_per_layer.weight",    "per_layer_token_embd.weight"),
            ("model.language_model.per_layer_model_projection.weight","per_layer_model_proj.weight"),
            ("model.language_model.per_layer_projection_norm.weight", "per_layer_proj_norm.weight"),
            ("model.language_model.norm.weight",                      "output_norm.weight"),
            // attention (spot-check a few layer indices)
            ("model.language_model.layers.0.self_attn.q_proj.weight",  "blk.0.attn_q.weight"),
            ("model.language_model.layers.5.self_attn.k_proj.weight",  "blk.5.attn_k.weight"),
            ("model.language_model.layers.12.self_attn.v_proj.weight", "blk.12.attn_v.weight"),
            ("model.language_model.layers.3.self_attn.o_proj.weight",  "blk.3.attn_output.weight"),
            ("model.language_model.layers.7.self_attn.q_norm.weight",  "blk.7.attn_q_norm.weight"),
            ("model.language_model.layers.7.self_attn.k_norm.weight",  "blk.7.attn_k_norm.weight"),
            // norms
            ("model.language_model.layers.2.input_layernorm.weight",            "blk.2.attn_norm.weight"),
            ("model.language_model.layers.2.pre_feedforward_layernorm.weight",  "blk.2.ffn_norm.weight"),
            ("model.language_model.layers.2.post_feedforward_layernorm.weight", "blk.2.post_ffw_norm.weight"),
            ("model.language_model.layers.2.post_attention_layernorm.weight",   "blk.2.post_attention_norm.weight"),
            // mlp
            ("model.language_model.layers.10.mlp.gate_proj.weight", "blk.10.ffn_gate.weight"),
            ("model.language_model.layers.10.mlp.up_proj.weight",   "blk.10.ffn_up.weight"),
            ("model.language_model.layers.10.mlp.down_proj.weight", "blk.10.ffn_down.weight"),
            // per-layer inputs
            ("model.language_model.layers.1.per_layer_input_gate.weight",    "blk.1.inp_gate.weight"),
            ("model.language_model.layers.1.per_layer_projection.weight",    "blk.1.proj.weight"),
            ("model.language_model.layers.1.post_per_layer_input_norm.weight","blk.1.post_norm.weight"),
            ("model.language_model.layers.1.layer_scalar",                   "blk.1.layer_output_scale.weight"),
            // unknown names pass through unchanged
            ("model.some_future_tensor.weight", "model.some_future_tensor.weight"),
            ("lm_head.weight",                  "lm_head.weight"),
        ]);
    }

    #[test]
    fn gemma3_rename_table() {
        check(gemma3_hf_to_llama, &[
            // globals
            ("model.embed_tokens.weight", "token_embd.weight"),
            ("model.norm.weight",         "output_norm.weight"),
            // attention
            ("model.layers.0.self_attn.q_proj.weight",  "blk.0.attn_q.weight"),
            ("model.layers.4.self_attn.k_proj.weight",  "blk.4.attn_k.weight"),
            ("model.layers.4.self_attn.v_proj.weight",  "blk.4.attn_v.weight"),
            ("model.layers.4.self_attn.o_proj.weight",  "blk.4.attn_output.weight"),
            ("model.layers.6.self_attn.q_norm.weight",  "blk.6.attn_q_norm.weight"),
            ("model.layers.6.self_attn.k_norm.weight",  "blk.6.attn_k_norm.weight"),
            // norms (all 4 per layer)
            ("model.layers.2.input_layernorm.weight",            "blk.2.attn_norm.weight"),
            ("model.layers.2.pre_feedforward_layernorm.weight",  "blk.2.ffn_norm.weight"),
            ("model.layers.2.post_feedforward_layernorm.weight", "blk.2.post_ffw_norm.weight"),
            ("model.layers.2.post_attention_layernorm.weight",   "blk.2.post_attention_norm.weight"),
            // mlp
            ("model.layers.9.mlp.gate_proj.weight", "blk.9.ffn_gate.weight"),
            ("model.layers.9.mlp.up_proj.weight",   "blk.9.ffn_up.weight"),
            ("model.layers.9.mlp.down_proj.weight", "blk.9.ffn_down.weight"),
            // unknown passthrough (no separate lm_head — tied)
            ("lm_head.weight",                  "lm_head.weight"),
            ("model.some_future_tensor.weight", "model.some_future_tensor.weight"),
        ]);
    }

    #[test]
    fn qwen35_rename_table() {
        check(qwen35_hf_to_llama, &[
            // globals
            ("model.language_model.embed_tokens.weight", "token_embd.weight"),
            ("model.language_model.norm.weight",         "output_norm.weight"),
            // full-attention layer
            ("model.language_model.layers.0.self_attn.q_proj.weight", "blk.0.attn_q.weight"),
            ("model.language_model.layers.0.self_attn.k_proj.weight", "blk.0.attn_k.weight"),
            ("model.language_model.layers.0.self_attn.v_proj.weight", "blk.0.attn_v.weight"),
            ("model.language_model.layers.0.self_attn.o_proj.weight", "blk.0.attn_output.weight"),
            ("model.language_model.layers.3.self_attn.q_norm.weight", "blk.3.attn_q_norm.weight"),
            ("model.language_model.layers.3.self_attn.k_norm.weight", "blk.3.attn_k_norm.weight"),
            // norms (both layer types share the same norm names)
            ("model.language_model.layers.1.input_layernorm.weight",          "blk.1.attn_norm.weight"),
            ("model.language_model.layers.1.post_attention_layernorm.weight", "blk.1.post_attention_norm.weight"),
            // mlp
            ("model.language_model.layers.5.mlp.gate_proj.weight", "blk.5.ffn_gate.weight"),
            ("model.language_model.layers.5.mlp.up_proj.weight",   "blk.5.ffn_up.weight"),
            ("model.language_model.layers.5.mlp.down_proj.weight", "blk.5.ffn_down.weight"),
            // linear-attention (SSM) tensors — verified against actual Ollama GGUF blk.0/blk.2
            ("model.language_model.layers.2.linear_attn.in_proj_qkv.weight", "blk.2.attn_qkv.weight"),
            ("model.language_model.layers.2.linear_attn.in_proj_z.weight",   "blk.2.attn_gate.weight"),
            ("model.language_model.layers.2.linear_attn.in_proj_a.weight",   "blk.2.ssm_alpha.weight"),
            ("model.language_model.layers.2.linear_attn.in_proj_b.weight",   "blk.2.ssm_beta.weight"),
            ("model.language_model.layers.2.linear_attn.conv1d.weight",      "blk.2.ssm_conv1d.weight"),
            ("model.language_model.layers.2.linear_attn.A_log",              "blk.2.ssm_a"),
            ("model.language_model.layers.2.linear_attn.dt_bias",            "blk.2.ssm_dt"),
            ("model.language_model.layers.2.linear_attn.norm.weight",        "blk.2.ssm_norm.weight"),
            ("model.language_model.layers.2.linear_attn.out_proj.weight",    "blk.2.ssm_out.weight"),
            // unknown passthrough
            ("model.some_future_tensor.weight", "model.some_future_tensor.weight"),
        ]);
    }

    #[test]
    fn qwen3_rename_table() {
        check(qwen3_hf_to_llama, &[
            // globals
            ("model.embed_tokens.weight", "token_embd.weight"),
            ("model.norm.weight",         "output_norm.weight"),
            ("lm_head.weight",            "output.weight"),
            // attention
            ("model.layers.0.self_attn.q_proj.weight",  "blk.0.attn_q.weight"),
            ("model.layers.3.self_attn.k_proj.weight",  "blk.3.attn_k.weight"),
            ("model.layers.3.self_attn.v_proj.weight",  "blk.3.attn_v.weight"),
            ("model.layers.3.self_attn.o_proj.weight",  "blk.3.attn_output.weight"),
            ("model.layers.5.self_attn.q_norm.weight",  "blk.5.attn_q_norm.weight"),
            ("model.layers.5.self_attn.k_norm.weight",  "blk.5.attn_k_norm.weight"),
            // norms
            ("model.layers.2.input_layernorm.weight",         "blk.2.attn_norm.weight"),
            ("model.layers.2.post_attention_layernorm.weight","blk.2.ffn_norm.weight"),
            // mlp
            ("model.layers.8.mlp.gate_proj.weight", "blk.8.ffn_gate.weight"),
            ("model.layers.8.mlp.up_proj.weight",   "blk.8.ffn_up.weight"),
            ("model.layers.8.mlp.down_proj.weight", "blk.8.ffn_down.weight"),
            // unknown passthrough
            ("model.some_future_tensor.weight", "model.some_future_tensor.weight"),
        ]);
    }

}
