//! Build script: regenerates `include/turboquant.h` from the Rust source
//! using cbindgen so the shipped C header always matches the compiled ABI.

use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=cbindgen.toml");

    let crate_dir = match env::var("CARGO_MANIFEST_DIR") {
        Ok(value) => PathBuf::from(value),
        Err(err) => panic!("CARGO_MANIFEST_DIR is unavailable: {err}"),
    };
    let header = crate_dir.join("include").join("turboquant.h");
    let config = match cbindgen::Config::from_file(crate_dir.join("cbindgen.toml")) {
        Ok(config) => config,
        Err(err) => panic!("failed to read cbindgen.toml: {err}"),
    };

    match cbindgen::Builder::new()
        .with_crate(&crate_dir)
        .with_config(config)
        .generate()
    {
        Ok(bindings) => {
            bindings.write_to_file(&header);
        }
        Err(err) => {
            // Keep the committed header if regeneration fails (e.g. offline
            // tooling problems); fail hard only when no header exists at all.
            if header.exists() {
                println!(
                    "cargo:warning=cbindgen failed ({err}); keeping existing include/turboquant.h"
                );
            } else {
                panic!("cbindgen failed and include/turboquant.h does not exist: {err}");
            }
        }
    }
}
