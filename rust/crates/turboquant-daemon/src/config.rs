//! Daemon configuration.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Configuration for the `TurboQuant` daemon.
///
/// Loaded from a JSON file (see [`DaemonConfig::load`]); every field is
/// optional and falls back to its default.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(default, deny_unknown_fields)]
pub struct DaemonConfig {
    /// Directories to watch for new/changed `.gguf` models.
    pub watch_dirs: Vec<String>,
    /// Directory where compressed copies are written.
    pub output_dir: String,
    /// HTTP listen address for the health endpoint.
    pub listen_addr: String,
    /// Seconds to ignore repeat events for the same file (debounce).
    pub interval_secs: u64,
    /// Quantization block size (power of two >= 8).
    pub block_size: usize,
    /// Maximum number of filesystem events buffered before backpressure drops duplicates.
    pub event_queue_capacity: usize,
    /// Maximum number of compression jobs that may run concurrently.
    pub max_concurrent_jobs: usize,
    /// Maximum admitted GGUF input size in bytes.
    pub max_file_bytes: u64,
    /// Maximum number of recent paths retained by the debounce cache.
    pub debounce_cache_capacity: usize,
}

impl Default for DaemonConfig {
    fn default() -> Self {
        Self {
            watch_dirs: vec!["~/.ollama/models".to_string()],
            output_dir: "~/.turboquant/compressed".to_string(),
            listen_addr: "127.0.0.1:7460".to_string(),
            interval_secs: 30,
            block_size: 64,
            event_queue_capacity: 256,
            max_concurrent_jobs: 2,
            max_file_bytes: 64 * 1024 * 1024 * 1024,
            debounce_cache_capacity: 4096,
        }
    }
}

impl DaemonConfig {
    /// Load configuration from a JSON file. Missing fields use defaults.
    ///
    /// # Errors
    ///
    /// Returns an error on I/O failure, invalid JSON, unknown fields, or
    /// invalid field values.
    pub fn load(path: &Path) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let text = std::fs::read_to_string(path)?;
        let config: Self = serde_json::from_str(&text)?;
        config.validate()?;
        Ok(config)
    }

    /// Validate field values.
    ///
    /// # Errors
    ///
    /// Returns an error for an invalid block size or empty watch list.
    pub fn validate(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if !self.block_size.is_power_of_two() || self.block_size < 8 {
            return Err(format!(
                "block_size must be a power of two >= 8, got {}",
                self.block_size
            )
            .into());
        }
        if self.watch_dirs.is_empty() {
            return Err("watch_dirs must not be empty".into());
        }
        if self.event_queue_capacity == 0 {
            return Err("event_queue_capacity must be greater than zero".into());
        }
        if self.max_concurrent_jobs == 0 {
            return Err("max_concurrent_jobs must be greater than zero".into());
        }
        if self.max_file_bytes == 0 {
            return Err("max_file_bytes must be greater than zero".into());
        }
        if self.debounce_cache_capacity == 0 {
            return Err("debounce_cache_capacity must be greater than zero".into());
        }
        Ok(())
    }

    /// Watch directories with `~` expanded.
    #[must_use]
    pub fn expanded_watch_dirs(&self) -> Vec<PathBuf> {
        self.watch_dirs.iter().map(|d| expand_tilde(d)).collect()
    }

    /// Output directory with `~` expanded.
    #[must_use]
    pub fn expanded_output_dir(&self) -> PathBuf {
        expand_tilde(&self.output_dir)
    }
}

/// Expand a leading `~/` (or bare `~`) to `$HOME`.
#[must_use]
pub fn expand_tilde(path: &str) -> PathBuf {
    if path == "~" || path.starts_with("~/") {
        if let Ok(home) = std::env::var("HOME") {
            if path == "~" {
                return PathBuf::from(home);
            }
            return Path::new(&home).join(&path[2..]);
        }
    }
    PathBuf::from(path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_is_valid() {
        DaemonConfig::default().validate().unwrap();
    }

    #[test]
    fn parse_full_config() {
        let json = r#"{
            "watch_dirs": ["/models/a", "/models/b"],
            "output_dir": "/out",
            "listen_addr": "0.0.0.0:8080",
            "interval_secs": 5,
            "block_size": 128,
            "event_queue_capacity": 32,
            "max_concurrent_jobs": 3,
            "max_file_bytes": 1048576,
            "debounce_cache_capacity": 128
        }"#;
        let config: DaemonConfig = serde_json::from_str(json).unwrap();
        config.validate().unwrap();
        assert_eq!(config.watch_dirs, vec!["/models/a", "/models/b"]);
        assert_eq!(config.output_dir, "/out");
        assert_eq!(config.listen_addr, "0.0.0.0:8080");
        assert_eq!(config.interval_secs, 5);
        assert_eq!(config.block_size, 128);
        assert_eq!(config.event_queue_capacity, 32);
        assert_eq!(config.max_concurrent_jobs, 3);
        assert_eq!(config.max_file_bytes, 1_048_576);
        assert_eq!(config.debounce_cache_capacity, 128);
    }

    #[test]
    fn parse_partial_config_uses_defaults() {
        let config: DaemonConfig = serde_json::from_str(r#"{"output_dir": "/tmp/out"}"#).unwrap();
        assert_eq!(config.output_dir, "/tmp/out");
        assert_eq!(config.listen_addr, DaemonConfig::default().listen_addr);
        assert_eq!(config.block_size, 64);
    }

    #[test]
    fn parse_rejects_unknown_fields() {
        assert!(serde_json::from_str::<DaemonConfig>(r#"{"no_such_key": 1}"#).is_err());
    }

    #[test]
    fn validate_rejects_bad_block_size() {
        let config = DaemonConfig {
            block_size: 24,
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn validate_rejects_zero_resource_limits() {
        for config in [
            DaemonConfig {
                event_queue_capacity: 0,
                ..Default::default()
            },
            DaemonConfig {
                max_concurrent_jobs: 0,
                ..Default::default()
            },
            DaemonConfig {
                max_file_bytes: 0,
                ..Default::default()
            },
            DaemonConfig {
                debounce_cache_capacity: 0,
                ..Default::default()
            },
        ] {
            assert!(config.validate().is_err());
        }
    }

    #[test]
    fn load_from_file_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.json");
        std::fs::write(&path, r#"{"interval_secs": 2}"#).unwrap();
        let config = DaemonConfig::load(&path).unwrap();
        assert_eq!(config.interval_secs, 2);

        std::fs::write(&path, "not json").unwrap();
        assert!(DaemonConfig::load(&path).is_err());
    }

    #[test]
    fn tilde_expansion() {
        std::env::set_var("HOME", "/home/tester");
        assert_eq!(expand_tilde("~/x"), PathBuf::from("/home/tester/x"));
        assert_eq!(expand_tilde("/abs/x"), PathBuf::from("/abs/x"));
    }
}
