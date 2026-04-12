//! HuggingFace Hub helpers for GGUF-backed model loading.
//!
//! These functions fetch only the metadata files (`config.json`,
//! `tokenizer.json`, `tokenizer_config.json`) that [`crate::engine`] needs
//! when the weight file is already available locally — either as an Ollama
//! GGUF blob or via the `--gguf` flag.

use anyhow::{Context, Result};
use hf_hub::api::sync::{Api, ApiBuilder};
use std::path::PathBuf;

/// Download `config.json`, `tokenizer.json`, and optionally
/// `tokenizer_config.json` from HuggingFace into `dest_dir`, using a custom
/// HF hub cache rooted there.
///
/// Files already present are reused without hitting the network (the HF hub
/// caching layer handles this).  The returned paths point into `dest_dir`.
///
/// Called for Ollama models, where the GGUF weights already exist locally but
/// the HF metadata must be fetched separately.
pub fn download_metadata_only(
    hf_id: &str,
    revision: &str,
    dest_dir: &std::path::Path,
) -> Result<(PathBuf, PathBuf, Option<PathBuf>)> {
    tracing::info!("Downloading metadata for {hf_id} into {}", dest_dir.display());

    let api = ApiBuilder::new()
        .with_cache_dir(dest_dir.to_path_buf())
        .build()
        .context("Failed to create HuggingFace API client")?;

    let repo = api.repo(hf_hub::Repo::with_revision(
        hf_id.to_string(),
        hf_hub::RepoType::Model,
        revision.to_string(),
    ));

    let config_path = repo
        .get("config.json")
        .map_err(|e| hf_auth_hint(anyhow::anyhow!("{e}"), hf_id))?;
    let tokenizer_path = repo
        .get("tokenizer.json")
        .map_err(|e| hf_auth_hint(anyhow::anyhow!("{e}"), hf_id))?;
    let tokenizer_config_path = repo.get("tokenizer_config.json").ok();

    Ok((config_path, tokenizer_path, tokenizer_config_path))
}

/// Fetch only `config.json` and `tokenizer.json` — no weights.
///
/// Used by [`crate::engine::load_engine`] when the caller supplies their own
/// GGUF via `--gguf`: we still need the HuggingFace metadata to detect the
/// architecture and build the tokenizer, but there is no point downloading
/// multi-gigabyte safetensors shards that will never be read.
///
/// Accepts a HuggingFace model ID, an absolute path, or a relative path
/// starting with `./`/`../`.
pub fn load_config_and_tokenizer(
    model_id: &str,
    revision: &str,
) -> Result<crate::hub::ModelFiles> {
    let as_path = std::path::Path::new(model_id);
    let (config_path, tokenizer_path, tokenizer_config_path) = if as_path.is_absolute()
        || model_id.starts_with("./")
        || model_id.starts_with("../")
        || as_path.exists()
    {
        tracing::info!(
            "Loading config and tokenizer from local path: {}",
            as_path.display()
        );
        let config_path = as_path.join("config.json");
        anyhow::ensure!(
            config_path.exists(),
            "config.json not found in {}",
            as_path.display()
        );
        let tokenizer_path = as_path.join("tokenizer.json");
        anyhow::ensure!(
            tokenizer_path.exists(),
            "tokenizer.json not found in {}",
            as_path.display()
        );
        let tokenizer_config_path = {
            let p = as_path.join("tokenizer_config.json");
            p.exists().then_some(p)
        };
        (config_path, tokenizer_path, tokenizer_config_path)
    } else {
        tracing::info!(
            "Downloading config and tokenizer for {} (revision: {})",
            model_id,
            revision
        );
        let api = Api::new().context("Failed to create HuggingFace API client")?;
        let repo = api.repo(hf_hub::Repo::with_revision(
            model_id.to_string(),
            hf_hub::RepoType::Model,
            revision.to_string(),
        ));
        let config_path = repo
            .get("config.json")
            .context("Failed to download config.json")?;
        let tokenizer_path = repo
            .get("tokenizer.json")
            .context("Failed to download tokenizer.json")?;
        let tokenizer_config_path = repo.get("tokenizer_config.json").ok();
        (config_path, tokenizer_path, tokenizer_config_path)
    };

    Ok(crate::hub::ModelFiles {
        config_path,
        tokenizer_path,
        tokenizer_config_path,
        weight_paths: vec![],
        gguf_path: None, // caller sets this
    })
}

/// Enrich a HuggingFace download error with authentication guidance when the
/// status code is 401 (Unauthorized) or 403 (Forbidden).
///
/// Many models (Gemma, Llama, etc.) are gated: you must accept the model's
/// license terms on HuggingFace before downloading, then authenticate with an
/// access token.
fn hf_auth_hint(err: anyhow::Error, model_id: &str) -> anyhow::Error {
    let msg = err.to_string();
    if msg.contains("status code 401")
        || msg.contains("status code 403")
        || msg.contains("status: 401")
        || msg.contains("status: 403")
    {
        anyhow::anyhow!(
            "{err}\n\
             \n\
             '{model_id}' is a gated model that requires a HuggingFace access token.\n\
             \n\
             1. Accept the model license at:\n   \
                https://huggingface.co/{model_id}\n\
             2. Create a token (read access is enough) at:\n   \
                https://huggingface.co/settings/tokens\n\
             3. Pass it to inferrs via the environment:\n   \
                HF_TOKEN=hf_... inferrs serve {model_id}\n\
             \n\
             Or log in once with the HuggingFace CLI:\n   \
                hf auth login"
        )
    } else {
        err.context(format!("Failed to download metadata for {model_id}"))
    }
}
