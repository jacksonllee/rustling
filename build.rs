use std::path::PathBuf;
use std::process::Command;

fn main() {
    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());

    // (schema path, OUT_DIR subdirectory, pre-generated fallback file)
    let schemas: &[(&str, &str, &str)] = &[
        ("src/hmm/model.fbs", "hmm", "src/hmm/model_generated.rs"),
        (
            "src/perceptron_pos_tagger/model.fbs",
            "perceptron_pos_tagger",
            "src/perceptron_pos_tagger/model_generated.rs",
        ),
        (
            "src/wordseg/longest_string_matching/model.fbs",
            "lsm",
            "src/wordseg/longest_string_matching/model_generated.rs",
        ),
        ("src/lm/model.fbs", "lm", "src/lm/model_generated.rs"),
    ];

    let has_flatc = Command::new("flatc")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);

    for (schema_rel, out_subdir, fallback_rel) in schemas {
        let schema_path = manifest_dir.join(schema_rel);
        println!("cargo:rerun-if-changed={}", schema_path.display());

        let out_subpath = out_dir.join(out_subdir);
        std::fs::create_dir_all(&out_subpath).unwrap();

        if has_flatc {
            let status = Command::new("flatc")
                .args(["--rust", "-o"])
                .arg(&out_subpath)
                .arg(&schema_path)
                .status()
                .expect("flatc invocation failed");
            assert!(
                status.success(),
                "flatc failed for {}",
                schema_path.display()
            );
        } else {
            // Fall back to pre-generated file committed in the repository.
            let fallback_path = manifest_dir.join(fallback_rel);
            let dest = out_subpath.join("model_generated.rs");
            std::fs::copy(&fallback_path, &dest).unwrap_or_else(|e| {
                panic!(
                    "flatc not found and fallback file {} is missing: {}",
                    fallback_path.display(),
                    e
                )
            });
        }
    }
}
