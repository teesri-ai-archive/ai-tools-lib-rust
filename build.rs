//! Prompt templates are embedded with `include_dir!` in `src/prompt_manager/prompt_template.rs`.
//! Cargo does not otherwise track files outside this crate, so edits under `flixie-prompts/`
//! would not rebuild this library and Lambdas would keep a stale embedded tree.
//! Declaring `rerun-if-changed` on the prompts directory forces a rebuild when any template changes.

use std::path::Path;

fn main() {
    let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let prompts = manifest_dir.join("../flixie-prompts/prompts");
    println!("cargo:rerun-if-changed={}", prompts.display());
    if !prompts.is_dir() {
        panic!(
            "ai-tools-lib-rust: missing prompt directory at {} (expected flixie-prompts repo as a sibling of ai-tools-lib-rust).",
            prompts.display()
        );
    }
}
