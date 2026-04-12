//! `inferrs list` — show locally cached HuggingFace models and Ollama models.

use anyhow::Result;
use clap::Parser;

use crate::util::{cache_root, dir_size, format_bytes};

#[derive(Parser, Clone)]
pub struct ListArgs {}

pub fn run(_args: ListArgs) -> Result<()> {
    let hf_entries = hf_models();
    let ollama_entries = ollama_models();

    if hf_entries.is_empty() && ollama_entries.is_empty() {
        println!("No models found.");
        return Ok(());
    }

    let name_width = hf_entries
        .iter()
        .chain(ollama_entries.iter())
        .map(|(id, _, _)| id.len())
        .max()
        .unwrap_or(0);

    if !hf_entries.is_empty() {
        println!("HuggingFace");
        for (id, size, note) in &hf_entries {
            print!("  {id:<width$}  {}", format_bytes(*size), width = name_width);
            if let Some(n) = note {
                print!("  {n}");
            }
            println!();
        }
    }

    if !ollama_entries.is_empty() {
        if !hf_entries.is_empty() {
            println!();
        }
        println!("Ollama");
        for (id, size, note) in &ollama_entries {
            print!("  {id:<width$}  {}", format_bytes(*size), width = name_width);
            if let Some(n) = note {
                print!("  {n}");
            }
            println!();
        }
    }

    Ok(())
}

/// Convert `"models--Org--Name"` back to `"Org/Name"`.
fn folder_to_model_id(folder: &str) -> String {
    folder
        .strip_prefix("models--")
        .unwrap_or(folder)
        .replace("--", "/")
}

/// Returns `(model_id, size_bytes, optional_note)` for each cached HF model.
fn hf_models() -> Vec<(String, u64, Option<String>)> {
    let cache_dir = cache_root();
    if !cache_dir.exists() {
        return vec![];
    }
    let mut entries: Vec<(String, u64, Option<String>)> = std::fs::read_dir(&cache_dir)
        .into_iter()
        .flatten()
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.file_type().map(|t| t.is_dir()).unwrap_or(false)
                && e.file_name().to_string_lossy().starts_with("models--")
        })
        .map(|e| {
            let folder = e.file_name().to_string_lossy().into_owned();
            let model_id = folder_to_model_id(&folder);
            let size = dir_size(&e.path()).unwrap_or(0);
            (model_id, size, None)
        })
        .collect();
    entries.sort_by(|a, b| a.0.cmp(&b.0));
    entries
}

/// Returns `(display_name, size_bytes, note)` for each locally available Ollama model.
fn ollama_models() -> Vec<(String, u64, Option<String>)> {
    crate::ollama::list_ollama_models()
        .unwrap_or_default()
        .into_iter()
        .map(|m| {
            let note = format!("{} {} ({})", m.family, m.model_type, m.quantization);
            (m.display_name, m.size_bytes, Some(note))
        })
        .collect()
}
