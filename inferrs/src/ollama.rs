//! Ollama model discovery and HuggingFace ID resolution.
//!
//! Ollama stores models as content-addressed GGUF blobs under
//! `~/.ollama/models/`, with a Docker-style manifest alongside.
//! This module locates those blobs and maps them to HuggingFace
//! model IDs so inferrs can fetch the `config.json` and
//! `tokenizer.json` it needs for architecture detection and tokenization.

use anyhow::{Context, Result};
use serde::Deserialize;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// A locally available Ollama model and its GGUF blob path.
#[derive(Debug)]
pub struct OllamaModel {
    /// Display name, e.g. `"gemma4:e4b"`.
    pub display_name: String,
    /// Architecture family from the Ollama config blob, e.g. `"gemma4"`.
    pub family: String,
    /// Friendly size string from the Ollama config blob, e.g. `"8.0B"`.
    pub model_type: String,
    /// Quantisation label, e.g. `"Q4_K_M"`.
    pub quantization: String,
    /// Absolute path to the GGUF blob.
    pub gguf_path: PathBuf,
    /// Blob size in bytes.
    pub size_bytes: u64,
}

// ---------------------------------------------------------------------------
// Detection
// ---------------------------------------------------------------------------

/// Returns `true` if `s` looks like an Ollama model identifier.
///
/// Ollama uses `name:tag` (colon, no slash). HuggingFace IDs always have a
/// slash; local paths start with `/` or `.`.  Only the `name:tag` form is
/// treated as Ollama — bare single-word names are ambiguous and not detected.
pub fn is_ollama_model_id(s: &str) -> bool {
    !s.starts_with('/')
        && !s.starts_with('.')
        && s.contains(':')
        && !s.contains('/')
}

// ---------------------------------------------------------------------------
// Discovery
// ---------------------------------------------------------------------------

/// Root of Ollama's local model storage.
/// Respects `OLLAMA_MODELS` if set.
fn ollama_models_dir() -> PathBuf {
    if let Ok(p) = std::env::var("OLLAMA_MODELS") {
        return PathBuf::from(p);
    }
    crate::util::home_dir().join(".ollama").join("models")
}

fn blobs_dir() -> PathBuf {
    ollama_models_dir().join("blobs")
}

fn manifests_root() -> PathBuf {
    ollama_models_dir().join("manifests")
}

/// `"sha256:abc"` → `"sha256-abc"` (blob filename on disk).
fn digest_to_filename(digest: &str) -> String {
    digest.replacen(':', "-", 1)
}

/// `"name:tag"` → `("name", "tag")`.  Defaults to `"latest"` when no tag.
pub fn split_name_tag(s: &str) -> (&str, &str) {
    match s.rfind(':') {
        Some(pos) => (&s[..pos], &s[pos + 1..]),
        None => (s, "latest"),
    }
}

// ---------------------------------------------------------------------------
// Manifest / blob types (private)
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct OllamaManifest {
    config: ManifestLayer,
    layers: Vec<ManifestLayer>,
}

#[derive(Deserialize)]
struct ManifestLayer {
    #[serde(rename = "mediaType")]
    media_type: String,
    digest: String,
    size: u64,
}

#[derive(Deserialize, Default)]
struct OllamaConfigBlob {
    model_family: Option<String>,
    model_families: Option<Vec<String>>,
    model_type: Option<String>,
    file_type: Option<String>,
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Locate a specific Ollama model by its `"name:tag"` identifier.
pub fn find_ollama_model(name_tag: &str) -> Result<OllamaModel> {
    let (name, tag) = split_name_tag(name_tag);
    let root = manifests_root();
    anyhow::ensure!(
        root.exists(),
        "Ollama model directory not found at {} — is Ollama installed?",
        root.display()
    );
    // Walk: <root>/<registry>/<namespace>/<name>/<tag>
    for registry in read_dirs(&root) {
        for namespace in read_dirs(&registry) {
            for name_dir in read_dirs(&namespace) {
                if !name_dir.file_name()
                    .map(|n| n.eq_ignore_ascii_case(name))
                    .unwrap_or(false)
                {
                    continue;
                }
                for tag_path in read_dirs(&name_dir) {
                    if tag_path.file_name()
                        .map(|t| t.eq_ignore_ascii_case(tag))
                        .unwrap_or(false)
                    {
                        return parse_manifest(&tag_path, &format!("{name}:{tag}"))
                            .with_context(|| format!("Failed to parse manifest for {name_tag}"));
                    }
                }
            }
        }
    }
    anyhow::bail!(
        "Ollama model '{name_tag}' not found locally.\n\
         Run `ollama pull {name_tag}` to download it first."
    )
}

/// List all Ollama models available locally.
pub fn list_ollama_models() -> Result<Vec<OllamaModel>> {
    let root = manifests_root();
    if !root.exists() {
        return Ok(vec![]);
    }
    let mut models = Vec::new();
    for registry in read_dirs(&root) {
        for namespace in read_dirs(&registry) {
            for name_dir in read_dirs(&namespace) {
                let name = name_dir.file_name().unwrap_or_default().to_string_lossy();
                for tag_path in read_dirs(&name_dir) {
                    let tag = tag_path.file_name().unwrap_or_default().to_string_lossy();
                    let display = format!("{name}:{tag}");
                    if let Ok(m) = parse_manifest(&tag_path, &display) {
                        models.push(m);
                    }
                }
            }
        }
    }
    models.sort_by(|a, b| a.display_name.cmp(&b.display_name));
    Ok(models)
}

fn parse_manifest(manifest_path: &Path, display_name: &str) -> Result<OllamaModel> {
    let text = std::fs::read_to_string(manifest_path)
        .with_context(|| format!("Cannot read {}", manifest_path.display()))?;
    let manifest: OllamaManifest =
        serde_json::from_str(&text).context("Invalid Ollama manifest JSON")?;

    let model_layer = manifest
        .layers
        .iter()
        .find(|l| l.media_type == "application/vnd.ollama.image.model")
        .context("No model layer in Ollama manifest")?;

    let gguf_path = blobs_dir().join(digest_to_filename(&model_layer.digest));
    anyhow::ensure!(
        gguf_path.exists(),
        "GGUF blob missing: {} — try `ollama pull {display_name}`",
        gguf_path.display()
    );

    let cfg: OllamaConfigBlob = {
        let p = blobs_dir().join(digest_to_filename(&manifest.config.digest));
        std::fs::read_to_string(&p)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default()
    };

    let family = cfg
        .model_family
        .or_else(|| cfg.model_families.and_then(|v| v.into_iter().next()))
        .unwrap_or_else(|| "unknown".to_string());

    Ok(OllamaModel {
        display_name: display_name.to_string(),
        family,
        model_type: cfg.model_type.unwrap_or_else(|| "?".to_string()),
        quantization: cfg.file_type.unwrap_or_else(|| "?".to_string()),
        gguf_path,
        size_bytes: model_layer.size,
    })
}

fn read_dirs(path: &Path) -> Vec<PathBuf> {
    std::fs::read_dir(path)
        .map(|rd| {
            rd.filter_map(|e| e.ok())
                .map(|e| e.path())
                .collect()
        })
        .unwrap_or_default()
}

// ---------------------------------------------------------------------------
// HuggingFace ID resolution
// ---------------------------------------------------------------------------

/// Normalise the Ollama `model_type` field (e.g. `"8.0B"`, `"8B"`, `"3.8B"`)
/// to a one-decimal string key (e.g. `"8.0"`, `"8.0"`, `"3.8"`).
fn normalise_size(model_type: &str) -> String {
    let trimmed = model_type.trim_end_matches('B').trim();
    // Parse as f64 so "8" and "8.0" both become 8.0, then format consistently.
    match trimmed.parse::<f64>() {
        Ok(v) => {
            // Use one decimal for fractional values, still one decimal for integers.
            format!("{v:.1}")
        }
        Err(_) => trimmed.to_string(),
    }
}

fn map_key(model: &OllamaModel) -> String {
    format!("{}:{}", model.family, normalise_size(&model.model_type))
}

/// Resolve an Ollama model to its HuggingFace model ID.
///
/// Checks the user override map at `~/.config/inferrs/ollama-map.json` first,
/// then falls back to the bundled `ollama_map.json`.
///
/// Returns a descriptive error if the model is not in either map, including
/// exactly what to add and where.
pub fn resolve_hf_model_id(model: &OllamaModel) -> Result<String> {
    let key = map_key(model);

    // User override takes precedence
    if let Some(id) = user_map_lookup(&key) {
        return Ok(id);
    }

    // Bundled map
    let bundled: HashMap<String, serde_json::Value> =
        serde_json::from_str(include_str!("ollama_map.json"))
            .context("bundled ollama_map.json is invalid — please file a bug")?;

    if let Some(id) = bundled.get(&key).and_then(|v| v.as_str()) {
        return Ok(id.to_string());
    }

    let map_path = user_map_path();
    Err(anyhow::anyhow!(
        "No HuggingFace model mapping for Ollama model '{name}' \
         (family={family}, size={mtype}, key={key}).\n\
         \n\
         Add it to your override map at:\n  {map_path}\n\
         \n\
         Entry to add:\n  \"{key}\": \"<org/model-name-on-huggingface>\"\n\
         \n\
         Find the HuggingFace model ID at:\n  https://huggingface.co/models\n\
         Search for: {family} {mtype}\n\
         \n\
         Or bypass the map entirely with --gguf:\n  \
         inferrs serve <hf-model-id> --gguf {gguf}",
        name = model.display_name,
        family = model.family,
        mtype = model.model_type,
        key = key,
        map_path = map_path.display(),
        gguf = model.gguf_path.display(),
    ))
}

fn user_map_path() -> PathBuf {
    crate::util::home_dir()
        .join(".config")
        .join("inferrs")
        .join("ollama-map.json")
}

fn user_map_lookup(key: &str) -> Option<String> {
    let path = user_map_path();
    let text = std::fs::read_to_string(&path).ok()?;
    let map: HashMap<String, serde_json::Value> = serde_json::from_str(&text).ok()?;
    map.get(key)?.as_str().map(str::to_string)
}

// ---------------------------------------------------------------------------
// Engine integration
// ---------------------------------------------------------------------------

/// Resolve an Ollama `"name:tag"` model to a [`crate::hub::ModelFiles`] ready
/// for [`crate::engine::load_engine`].
///
/// Locates the GGUF blob in the local Ollama store, maps it to a HuggingFace
/// model ID via [`resolve_hf_model_id`], downloads only the metadata files
/// (`config.json`, `tokenizer.json`) from HuggingFace (cached in
/// `~/.cache/inferrs/ollama/<name>/<tag>/`), and returns a `ModelFiles` with
/// the GGUF path pre-populated and `weight_paths` empty.
pub fn load_model_files(name_tag: &str, revision: &str) -> Result<crate::hub::ModelFiles> {
    let ollama = find_ollama_model(name_tag)?;
    tracing::info!(
        "Found Ollama model '{}' — {} {} at {}",
        ollama.display_name,
        ollama.family,
        ollama.model_type,
        ollama.gguf_path.display()
    );
    let hf_id = resolve_hf_model_id(&ollama)?;
    tracing::info!("Mapped to HuggingFace model: {hf_id}");

    let (model_name, model_tag) = split_name_tag(name_tag);
    let cache_dir = crate::util::inferrs_cache_root()
        .join("ollama")
        .join(model_name)
        .join(model_tag);
    std::fs::create_dir_all(&cache_dir)
        .with_context(|| format!("Cannot create metadata cache {}", cache_dir.display()))?;

    let (config_path, tokenizer_path, tokenizer_config_path) =
        crate::hub_ollama::download_metadata_only(&hf_id, revision, &cache_dir)?;

    Ok(crate::hub::ModelFiles {
        config_path,
        tokenizer_path,
        tokenizer_config_path,
        weight_paths: vec![],
        gguf_path: Some(ollama.gguf_path),
    })
}
