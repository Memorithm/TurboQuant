//! Daemon runtime: filesystem watcher, compression, health endpoint,
//! systemd watchdog keepalives, and graceful shutdown.

use crate::config::DaemonConfig;
use axum::extract::State;
use axum::routing::get;
use axum::{Json, Router};
use notify::{Event, EventKind, RecursiveMode, Watcher};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::sync::mpsc::error::TrySendError;
use tokio::sync::Semaphore;
use tokio::task::JoinSet;
use tracing::{debug, info, warn};
use turboquant_gguf::turbo::{self, TurboOptions};
use turboquant_gguf::{GgufLimits, GgufParser};

type BoxError = Box<dyn std::error::Error + Send + Sync>;

/// Floor for the watchdog keepalive interval, guarding against an
/// absurdly small `WATCHDOG_USEC` producing a busy ping loop.
const MIN_WATCHDOG_PING: Duration = Duration::from_secs(1);

/// FNV-1a 128-bit offset basis. This digest is a stable identity and
/// provenance key, not an authentication primitive.
const FNV1A_128_OFFSET: u128 = 0x6c62_272e_07bb_0142_62b8_2175_6295_c58d;
/// FNV-1a 128-bit prime.
const FNV1A_128_PRIME: u128 = 0x0000_0000_0100_0000_0000_0000_0000_013b;
/// Separates the canonical path from the file bytes in the identity digest.
const IDENTITY_SEPARATOR: &[u8] = b"\0turboquant-source-v1\0";
/// Per-process suffix for exclusive temporary publication paths.
static NEXT_TEMP_ID: AtomicU64 = AtomicU64::new(0);

/// Derive the keepalive ping interval from systemd's `WATCHDOG_USEC`
/// value: half the watchdog timeout (so a missed tick still leaves one
/// more chance before systemd fires), clamped to [`MIN_WATCHDOG_PING`].
fn watchdog_ping_interval(watchdog_usec: u64) -> Duration {
    Duration::from_micros(watchdog_usec / 2).max(MIN_WATCHDOG_PING)
}

/// Parse systemd watchdog configuration as passed via the environment
/// (`sd_watchdog_enabled(3)` semantics). Returns the keepalive interval
/// if a watchdog is armed for this process:
///
/// - `usec` (`WATCHDOG_USEC`) must be a positive integer.
/// - `pid` (`WATCHDOG_PID`), when set, must equal `own_pid`; otherwise
///   the watchdog is meant for another process and we must not ping.
fn watchdog_interval(usec: Option<&str>, pid: Option<&str>, own_pid: u32) -> Option<Duration> {
    let usec: u64 = usec?.parse().ok()?;
    if usec == 0 {
        return None;
    }
    if let Some(pid) = pid {
        if pid.parse::<u32>().ok()? != own_pid {
            return None;
        }
    }
    Some(watchdog_ping_interval(usec))
}

/// Read the watchdog configuration from the process environment.
fn watchdog_interval_from_env() -> Option<Duration> {
    watchdog_interval(
        std::env::var("WATCHDOG_USEC").ok().as_deref(),
        std::env::var("WATCHDOG_PID").ok().as_deref(),
        std::process::id(),
    )
}

/// Shared daemon state, exposed via the `/healthz` endpoint.
#[derive(Debug, Default)]
pub struct DaemonState {
    /// Files compressed since startup.
    pub files_compressed: AtomicU64,
    /// Failed compression attempts since startup.
    pub failures: AtomicU64,
    /// Filesystem events dropped when the bounded queue is saturated.
    pub events_dropped: AtomicU64,
    /// Duplicate filesystem events coalesced while a path is already queued.
    pub events_coalesced: AtomicU64,
}

/// Outcome of a single compression attempt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompressOutcome {
    /// The file was compressed; contains the output path.
    Compressed(PathBuf),
    /// Skipped: the input is already TurboQuant-compressed.
    AlreadyCompressed,
    /// Skipped: an up-to-date compressed copy already exists.
    UpToDate(PathBuf),
}

#[derive(Debug, Serialize, Deserialize)]
struct ProvenanceManifest {
    schema: String,
    source_path: String,
    source_identity: String,
    source_bytes: u64,
    digest_algorithm: String,
    output_file: String,
    block_size: usize,
}

struct DestinationLock {
    file: Option<File>,
    #[cfg(not(unix))]
    path: PathBuf,
}

impl DestinationLock {
    #[cfg(unix)]
    fn acquire(output: &Path) -> Result<Self, BoxError> {
        let path = sibling_with_suffix(output, ".lock")?;
        let file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path)?;
        use std::os::fd::AsRawFd;
        // SAFETY: `file` owns a valid descriptor for the lifetime of this
        // guard. `flock` neither takes ownership nor retains the pointer.
        if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) } != 0 {
            return Err(std::io::Error::last_os_error().into());
        }
        Ok(Self { file: Some(file) })
    }

    #[cfg(not(unix))]
    fn acquire(output: &Path) -> Result<Self, BoxError> {
        let path = sibling_with_suffix(output, ".lock")?;
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)?;
        Ok(Self {
            file: Some(file),
            path,
        })
    }
}

impl Drop for DestinationLock {
    fn drop(&mut self) {
        #[cfg(unix)]
        {
            use std::os::fd::AsRawFd;
            if let Some(file) = self.file.as_ref() {
                // SAFETY: the descriptor remains valid until this guard is dropped.
                let result = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_UN) };
                if result != 0 {
                    warn!(
                        "failed to release destination lock: {}",
                        std::io::Error::last_os_error()
                    );
                }
            }
        }
        #[cfg(not(unix))]
        {
            drop(self.file.take());
            if let Err(error) = std::fs::remove_file(&self.path) {
                if error.kind() != std::io::ErrorKind::NotFound {
                    warn!(
                        "failed to remove destination lock {}: {error}",
                        self.path.display()
                    );
                }
            }
        }
    }
}

fn sibling_with_suffix(path: &Path, suffix: &str) -> Result<PathBuf, BoxError> {
    let name = path
        .file_name()
        .ok_or("destination has no file name")?
        .to_string_lossy();
    Ok(path.with_file_name(format!("{name}{suffix}")))
}

fn fnv1a_update(mut digest: u128, bytes: &[u8]) -> u128 {
    for byte in bytes {
        digest ^= u128::from(*byte);
        digest = digest.wrapping_mul(FNV1A_128_PRIME);
    }
    digest
}

fn source_identity(input: &Path, max_file_bytes: u64) -> Result<(PathBuf, String, u64), BoxError> {
    let canonical = input.canonicalize()?;
    let mut file = File::open(&canonical)?;
    let metadata = file.metadata()?;
    if !metadata.is_file() {
        return Err("GGUF input is not a regular file".into());
    }
    if metadata.len() > max_file_bytes {
        return Err(format!(
            "GGUF input is {} bytes, exceeding max_file_bytes={max_file_bytes}",
            metadata.len()
        )
        .into());
    }

    let canonical_text = canonical.to_string_lossy();
    let mut digest = fnv1a_update(FNV1A_128_OFFSET, canonical_text.as_bytes());
    digest = fnv1a_update(digest, IDENTITY_SEPARATOR);
    let mut total = 0u64;
    let mut chunk = vec![0u8; 64 * 1024];
    loop {
        let remaining = max_file_bytes.saturating_sub(total);
        let read_limit = usize::try_from(remaining.saturating_add(1).min(chunk.len() as u64))
            .map_err(|_| "GGUF read budget does not fit usize")?;
        let count = file.read(&mut chunk[..read_limit])?;
        if count == 0 {
            break;
        }
        if count as u64 > remaining {
            return Err("GGUF input grew beyond max_file_bytes while hashing".into());
        }
        total += count as u64;
        digest = fnv1a_update(digest, &chunk[..count]);
    }
    Ok((canonical, format!("{digest:032x}"), total))
}

fn create_exclusive(path: &Path, bytes: &[u8]) -> Result<(), BoxError> {
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}

fn manifest_matches(path: &Path, source_identity: &str, block_size: usize) -> bool {
    std::fs::read(path)
        .ok()
        .and_then(|bytes| serde_json::from_slice::<ProvenanceManifest>(&bytes).ok())
        .is_some_and(|manifest| {
            manifest.schema == "turboquant.provenance.v1"
                && manifest.source_identity == source_identity
                && manifest.block_size == block_size
        })
}

/// Compress a single GGUF file into `output_dir`, using the same library
/// path as the CLI and the parser's default 64-GiB admission budget.
///
/// # Errors
///
/// Returns an error on I/O failure or malformed GGUF input.
pub fn compress_once(
    input: &Path,
    output_dir: &Path,
    block_size: usize,
) -> Result<CompressOutcome, BoxError> {
    compress_once_with_limit(
        input,
        output_dir,
        block_size,
        GgufLimits::default().max_file_bytes,
    )
}

/// Compress a single GGUF file with an explicit raw-input byte budget.
///
/// Output names contain a stable digest of the canonical source identity
/// and bytes. Publication uses exclusive temporary files, a per-destination
/// lock, atomic renames, and a sidecar provenance manifest.
///
/// # Errors
///
/// Returns an error on admission, parsing, compression, locking, or I/O failure.
pub fn compress_once_with_limit(
    input: &Path,
    output_dir: &Path,
    block_size: usize,
    max_file_bytes: u64,
) -> Result<CompressOutcome, BoxError> {
    let before = input.metadata()?;
    let (canonical, identity, source_bytes) = source_identity(input, max_file_bytes)?;
    let parsed = GgufParser::parse_file_with_limits(
        &canonical,
        GgufLimits {
            max_file_bytes,
            ..GgufLimits::default()
        },
    )?;
    if turbo::is_turbo_compressed(&parsed) {
        return Ok(CompressOutcome::AlreadyCompressed);
    }

    let after = canonical.metadata()?;
    if before.len() != after.len() || before.modified().ok() != after.modified().ok() {
        return Err("GGUF input changed while it was being admitted".into());
    }

    let stem = canonical
        .file_stem()
        .map_or_else(|| "model".to_string(), |s| s.to_string_lossy().into_owned());
    let output = output_dir.join(format!("{stem}-{identity}-b{block_size}-turbo3.gguf"));
    let manifest_path = sibling_with_suffix(&output, ".provenance.json")?;

    if output.exists() || manifest_path.exists() {
        if output.is_file() && manifest_matches(&manifest_path, &identity, block_size) {
            return Ok(CompressOutcome::UpToDate(output));
        }
        return Err("destination exists without matching provenance; refusing to replace it".into());
    }

    std::fs::create_dir_all(output_dir)?;
    let _lock = DestinationLock::acquire(&output)?;
    if output.exists() || manifest_path.exists() {
        if output.is_file() && manifest_matches(&manifest_path, &identity, block_size) {
            return Ok(CompressOutcome::UpToDate(output));
        }
        return Err("destination appeared without matching provenance; refusing to replace it".into());
    }
    let opts = TurboOptions {
        block_size,
        ..Default::default()
    };
    let (writer, stats) = turbo::compress(&parsed, &opts)?;
    let output_bytes = writer.to_bytes()?;
    let manifest = ProvenanceManifest {
        schema: "turboquant.provenance.v1".to_string(),
        source_path: canonical.to_string_lossy().into_owned(),
        source_identity: identity,
        source_bytes,
        digest_algorithm: "fnv1a-128(canonical-path || separator || source-bytes)".to_string(),
        output_file: output
            .file_name()
            .ok_or("output has no file name")?
            .to_string_lossy()
            .into_owned(),
        block_size,
    };
    let manifest_bytes = serde_json::to_vec_pretty(&manifest)?;
    let nonce = format!(
        "{}.{}",
        std::process::id(),
        NEXT_TEMP_ID.fetch_add(1, Ordering::Relaxed)
    );
    let output_tmp = sibling_with_suffix(&output, &format!(".tmp.{nonce}"))?;
    let manifest_tmp = sibling_with_suffix(&manifest_path, &format!(".tmp.{nonce}"))?;

    if let Err(error) = create_exclusive(&output_tmp, &output_bytes)
        .and_then(|()| create_exclusive(&manifest_tmp, &manifest_bytes))
        .and_then(|()| {
            std::fs::rename(&output_tmp, &output)?;
            if let Err(error) = std::fs::rename(&manifest_tmp, &manifest_path) {
                let _ = std::fs::remove_file(&output);
                return Err(error.into());
            }
            Ok(())
        })
    {
        let _ = std::fs::remove_file(&output_tmp);
        let _ = std::fs::remove_file(&manifest_tmp);
        return Err(error);
    }
    info!(
        "compressed {} -> {} ({} tensor(s) quantized, {} passed through, {:.2}x data ratio)",
        input.display(),
        output.display(),
        stats.tensors_compressed,
        stats.tensors_passthrough,
        stats.data_ratio()
    );
    Ok(CompressOutcome::Compressed(output))
}

fn should_handle(
    last_handled: &mut HashMap<PathBuf, Instant>,
    path: PathBuf,
    now: Instant,
    debounce: Duration,
    capacity: usize,
) -> bool {
    last_handled.retain(|_, seen| now.saturating_duration_since(*seen) < debounce);
    if last_handled
        .get(&path)
        .is_some_and(|seen| now.saturating_duration_since(*seen) < debounce)
    {
        return false;
    }
    if last_handled.len() >= capacity {
        if let Some(oldest) = last_handled
            .iter()
            .min_by_key(|(_, seen)| **seen)
            .map(|(cached_path, _)| cached_path.clone())
        {
            last_handled.remove(&oldest);
        }
    }
    last_handled.insert(path, now);
    true
}

fn record_compression_result(
    path: &Path,
    result: Result<CompressOutcome, BoxError>,
    state: &DaemonState,
) {
    match result {
        Ok(CompressOutcome::Compressed(_)) => {
            state.files_compressed.fetch_add(1, Ordering::Relaxed);
        }
        Ok(CompressOutcome::AlreadyCompressed) => {
            debug!("{} already compressed, skipping", path.display());
        }
        Ok(CompressOutcome::UpToDate(out)) => {
            debug!("{} up to date at {}", path.display(), out.display());
        }
        Err(error) => {
            state.failures.fetch_add(1, Ordering::Relaxed);
            warn!("failed to compress {}: {error}", path.display());
        }
    }
}

/// Run the daemon with the given configuration: serve `/healthz`, watch
/// the configured directories for `.gguf` files, and compress each into
/// the output directory. Shuts down gracefully on SIGTERM or ctrl-c.
///
/// # Errors
///
/// Returns an error if the config is invalid, the listen address cannot
/// be bound, or the watcher cannot be created.
pub async fn run(config: DaemonConfig) -> Result<(), BoxError> {
    config.validate()?;
    info!("Starting TurboQuant daemon");

    let state = Arc::new(DaemonState::default());
    let output_dir = config.expanded_output_dir();
    std::fs::create_dir_all(&output_dir)?;

    // Health endpoint.
    let app = Router::new()
        .route("/healthz", get(healthz))
        .with_state(state.clone());
    let listener = tokio::net::TcpListener::bind(&config.listen_addr).await?;
    info!("Health endpoint on http://{}/healthz", config.listen_addr);

    // Filesystem watcher. notify runs its own thread; bridge events into
    // the tokio runtime through a bounded, non-blocking channel.
    let (tx, mut rx) = tokio::sync::mpsc::channel::<PathBuf>(config.event_queue_capacity);
    let watcher_state = state.clone();
    let pending_paths = Arc::new(Mutex::new(HashSet::<PathBuf>::new()));
    let watcher_pending_paths = pending_paths.clone();
    let mut watcher = notify::recommended_watcher(move |res: notify::Result<Event>| match res {
        Ok(event) => {
            if matches!(event.kind, EventKind::Create(_) | EventKind::Modify(_)) {
                for path in event.paths {
                    if path.extension().is_some_and(|e| e == "gguf") {
                        let should_send = match watcher_pending_paths.lock() {
                            Ok(mut pending) => {
                                if pending.insert(path.clone()) {
                                    true
                                } else {
                                    watcher_state
                                        .events_coalesced
                                        .fetch_add(1, Ordering::Relaxed);
                                    false
                                }
                            }
                            Err(error) => {
                                watcher_state.failures.fetch_add(1, Ordering::Relaxed);
                                warn!("pending-path set is poisoned: {error}");
                                true
                            }
                        };
                        if !should_send {
                            continue;
                        }
                        match tx.try_send(path) {
                            Ok(()) => {}
                            Err(TrySendError::Full(path)) => {
                                watcher_state.events_dropped.fetch_add(1, Ordering::Relaxed);
                                if let Ok(mut pending) = watcher_pending_paths.lock() {
                                    pending.remove(&path);
                                }
                                warn!("event queue full; dropping {}", path.display());
                            }
                            Err(TrySendError::Closed(path)) => {
                                if let Ok(mut pending) = watcher_pending_paths.lock() {
                                    pending.remove(&path);
                                }
                                return;
                            }
                        }
                    }
                }
            }
        }
        Err(e) => warn!("watch error: {e}"),
    })?;
    let mut n_watched = 0usize;
    for dir in config.expanded_watch_dirs() {
        if dir.is_dir() {
            watcher.watch(&dir, RecursiveMode::Recursive)?;
            info!("Watching {}", dir.display());
            n_watched += 1;
        } else {
            warn!("watch directory {} does not exist, skipping", dir.display());
        }
    }
    if n_watched == 0 {
        warn!("no watch directories exist; only the health endpoint is active");
    }

    // Tell systemd we are ready (no-op outside systemd). Keep
    // NOTIFY_SOCKET set (unset_env = false): the watchdog task below
    // sends keepalives over the same socket.
    if let Err(e) = sd_notify::notify(false, &[sd_notify::NotifyState::Ready]) {
        debug!("sd_notify failed (not running under systemd?): {e}");
    }

    // systemd watchdog keepalives. Only spawned when systemd armed a
    // watchdog for this process; normal runs pay zero overhead.
    let watchdog = watchdog_interval_from_env().map(|interval| {
        debug!("systemd watchdog enabled, pinging every {interval:?}");
        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(interval);
            loop {
                ticker.tick().await;
                if let Err(e) = sd_notify::notify(false, &[sd_notify::NotifyState::Watchdog]) {
                    warn!("watchdog keepalive failed: {e}");
                }
            }
        })
    });

    // HTTP server with graceful shutdown.
    let (http_shutdown_tx, http_shutdown_rx) = tokio::sync::oneshot::channel::<()>();
    let server = tokio::spawn(async move {
        axum::serve(listener, app)
            .with_graceful_shutdown(async {
                let _ = http_shutdown_rx.await;
            })
            .await
    });

    let debounce = Duration::from_secs(config.interval_secs.max(1));
    let mut last_handled: HashMap<PathBuf, Instant> = HashMap::new();
    let block_size = config.block_size;
    let max_file_bytes = config.max_file_bytes;
    let debounce_cache_capacity = config.debounce_cache_capacity;
    let max_concurrent_jobs = config.max_concurrent_jobs;
    let semaphore = Arc::new(Semaphore::new(max_concurrent_jobs));
    let mut jobs = JoinSet::new();

    let shutdown = shutdown_signal();
    tokio::pin!(shutdown);

    loop {
        if jobs.len() >= max_concurrent_jobs {
            tokio::select! {
                () = &mut shutdown => {
                    info!("shutdown signal received");
                    break;
                }
                joined = jobs.join_next() => {
                    if let Some(Err(error)) = joined {
                        state.failures.fetch_add(1, Ordering::Relaxed);
                        warn!("compression task failed to join: {error}");
                    }
                }
            }
            continue;
        }

        tokio::select! {
            () = &mut shutdown => {
                info!("shutdown signal received");
                break;
            }
            joined = jobs.join_next(), if !jobs.is_empty() => {
                if let Some(Err(error)) = joined {
                    state.failures.fetch_add(1, Ordering::Relaxed);
                    warn!("compression task failed to join: {error}");
                }
            }
            maybe_path = rx.recv() => {
                let Some(path) = maybe_path else { break };
                if let Ok(mut pending) = pending_paths.lock() {
                    pending.remove(&path);
                }
                if path.starts_with(&output_dir) {
                    continue;
                }
                if !should_handle(
                    &mut last_handled,
                    path.clone(),
                    Instant::now(),
                    debounce,
                    debounce_cache_capacity,
                ) {
                    continue;
                }

                let out_dir = output_dir.clone();
                let task_state = state.clone();
                let permit = match semaphore.clone().try_acquire_owned() {
                    Ok(permit) => permit,
                    Err(error) => {
                        state.failures.fetch_add(1, Ordering::Relaxed);
                        warn!("compression semaphore unexpectedly unavailable: {error}");
                        continue;
                    }
                };
                jobs.spawn(async move {
                    let _permit = permit;
                    tokio::time::sleep(Duration::from_millis(200)).await;
                    let display_path = path.clone();
                    let result = tokio::task::spawn_blocking(move || {
                        compress_once_with_limit(&path, &out_dir, block_size, max_file_bytes)
                    })
                    .await;
                    match result {
                        Ok(result) => record_compression_result(&display_path, result, &task_state),
                        Err(error) => {
                            task_state.failures.fetch_add(1, Ordering::Relaxed);
                            warn!(
                                "compression worker failed for {}: {error}",
                                display_path.display()
                            );
                        }
                    }
                });
            }
        }
    }

    // Stop accepting events, then drain all tracked compression tasks.
    rx.close();
    drop(watcher);
    while let Some(joined) = jobs.join_next().await {
        if let Err(error) = joined {
            state.failures.fetch_add(1, Ordering::Relaxed);
            warn!("compression task failed during shutdown drain: {error}");
        }
    }

    // Stop the keepalives and drain the HTTP server.
    if let Some(task) = watchdog {
        task.abort();
    }
    let _ = http_shutdown_tx.send(());
    match server.await {
        Ok(result) => result?,
        Err(e) => warn!("HTTP server task failed: {e}"),
    }
    info!("daemon stopped");
    Ok(())
}

/// `GET /healthz` — JSON liveness/status report.
async fn healthz(State(state): State<Arc<DaemonState>>) -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "status": "ok",
        "files_compressed": state.files_compressed.load(Ordering::Relaxed),
        "failures": state.failures.load(Ordering::Relaxed),
        "events_dropped": state.events_dropped.load(Ordering::Relaxed),
        "events_coalesced": state.events_coalesced.load(Ordering::Relaxed),
    }))
}

/// Resolves when SIGTERM (unix) or ctrl-c is received.
async fn shutdown_signal() {
    let ctrl_c = tokio::signal::ctrl_c();
    #[cfg(unix)]
    {
        use tokio::signal::unix::{signal, SignalKind};
        match signal(SignalKind::terminate()) {
            Ok(mut term) => {
                tokio::select! {
                    _ = ctrl_c => {}
                    _ = term.recv() => {}
                }
            }
            Err(e) => {
                warn!("cannot install SIGTERM handler: {e}");
                let _ = ctrl_c.await;
            }
        }
    }
    #[cfg(not(unix))]
    {
        let _ = ctrl_c.await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use turboquant_gguf::{GgmlType, GgufValue, GgufWriter};

    fn write_test_model_named(path: &Path, name: &str) {
        let mut w = GgufWriter::new();
        w.add_metadata("general.name", GgufValue::String(name.into()));
        let values: Vec<f32> = (0..256).map(|i| (i as f32 * 0.1).sin()).collect();
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        w.add_tensor("w", vec![256], GgmlType::F32, bytes).unwrap();
        w.write_to_file(path).unwrap();
    }

    fn write_test_model(path: &Path) {
        write_test_model_named(path, "daemon-test");
    }

    #[test]
    fn compress_once_compresses_then_skips() {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("model.gguf");
        let out_dir = dir.path().join("out");
        std::fs::create_dir_all(&out_dir).unwrap();
        write_test_model(&input);

        // First run compresses.
        let outcome = compress_once(&input, &out_dir, 64).unwrap();
        let CompressOutcome::Compressed(output) = outcome else {
            panic!("expected Compressed, got {outcome:?}");
        };
        assert!(output.exists());
        let parsed = GgufParser::parse_file(&output).unwrap();
        assert!(turbo::is_turbo_compressed(&parsed));
        assert_eq!(turbo::decompress_tensor(&parsed, "w").unwrap().len(), 256);

        // Second run is a no-op (output newer than input).
        assert_eq!(
            compress_once(&input, &out_dir, 64).unwrap(),
            CompressOutcome::UpToDate(output.clone())
        );

        // Compressing an already-compressed file is refused.
        assert_eq!(
            compress_once(&output, &out_dir, 64).unwrap(),
            CompressOutcome::AlreadyCompressed
        );
    }

    #[test]
    fn watchdog_ping_interval_halves_and_clamps() {
        // WatchdogSec=60 -> ping every 30s.
        assert_eq!(watchdog_ping_interval(60_000_000), Duration::from_secs(30));
        // 4s -> 2s (the smoke-test configuration).
        assert_eq!(watchdog_ping_interval(4_000_000), Duration::from_secs(2));
        // Tiny or zero values clamp to the 1s floor instead of busy-looping.
        assert_eq!(watchdog_ping_interval(100_000), MIN_WATCHDOG_PING);
        assert_eq!(watchdog_ping_interval(0), MIN_WATCHDOG_PING);
        // Exactly 2s -> exactly the floor.
        assert_eq!(watchdog_ping_interval(2_000_000), MIN_WATCHDOG_PING);
    }

    #[test]
    fn watchdog_interval_parses_environment_values() {
        let pid = 4242;
        // Enabled: usec set, no pid restriction.
        assert_eq!(
            watchdog_interval(Some("60000000"), None, pid),
            Some(Duration::from_secs(30))
        );
        // Enabled: pid restriction matches us.
        assert_eq!(
            watchdog_interval(Some("4000000"), Some("4242"), pid),
            Some(Duration::from_secs(2))
        );
        // Disabled: watchdog armed for a different process.
        assert_eq!(watchdog_interval(Some("4000000"), Some("1"), pid), None);
        // Disabled: unset, zero, or garbage usec; garbage pid.
        assert_eq!(watchdog_interval(None, None, pid), None);
        assert_eq!(watchdog_interval(Some("0"), None, pid), None);
        assert_eq!(watchdog_interval(Some("soon"), None, pid), None);
        assert_eq!(watchdog_interval(Some("-1"), None, pid), None);
        assert_eq!(watchdog_interval(Some("4000000"), Some("pid"), pid), None);
    }

    #[test]
    fn compress_once_rejects_non_gguf() {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("junk.gguf");
        std::fs::write(&input, b"not a gguf file").unwrap();
        assert!(compress_once(&input, dir.path(), 64).is_err());
    }

    #[test]
    fn compression_rejects_input_over_explicit_byte_budget() {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("model.gguf");
        write_test_model(&input);
        let too_small = input.metadata().unwrap().len() - 1;

        assert!(compress_once_with_limit(&input, dir.path(), 64, too_small).is_err());
        assert_eq!(
            std::fs::read_dir(dir.path())
                .unwrap()
                .filter_map(Result::ok)
                .filter(|entry| entry.file_name().to_string_lossy().contains("turbo3"))
                .count(),
            0
        );
    }

    #[test]
    fn same_stem_in_distinct_directories_has_distinct_outputs_and_provenance() {
        let dir = tempfile::tempdir().unwrap();
        let source_a = dir.path().join("a");
        let source_b = dir.path().join("b");
        let output_dir = dir.path().join("out");
        std::fs::create_dir_all(&source_a).unwrap();
        std::fs::create_dir_all(&source_b).unwrap();
        let input_a = source_a.join("model.gguf");
        let input_b = source_b.join("model.gguf");
        write_test_model_named(&input_a, "model-a");
        write_test_model_named(&input_b, "model-b");

        let CompressOutcome::Compressed(output_a) =
            compress_once(&input_a, &output_dir, 64).unwrap()
        else {
            panic!("first source was not compressed");
        };
        let CompressOutcome::Compressed(output_b) =
            compress_once(&input_b, &output_dir, 64).unwrap()
        else {
            panic!("second source was not compressed");
        };

        assert_ne!(output_a, output_b);
        for (output, source) in [(&output_a, &input_a), (&output_b, &input_b)] {
            assert!(output.is_file());
            let manifest_path = sibling_with_suffix(output, ".provenance.json").unwrap();
            let manifest: ProvenanceManifest =
                serde_json::from_slice(&std::fs::read(manifest_path).unwrap()).unwrap();
            assert_eq!(
                manifest.source_path,
                source.canonicalize().unwrap().to_string_lossy()
            );
            assert_eq!(
                manifest.output_file,
                output.file_name().unwrap().to_string_lossy()
            );
        }
    }

    #[test]
    fn destination_lock_can_be_reacquired_after_guard_drop() {
        let dir = tempfile::tempdir().unwrap();
        let output = dir.path().join("model-turbo3.gguf");
        drop(DestinationLock::acquire(&output).unwrap());
        drop(DestinationLock::acquire(&output).unwrap());
    }

    #[test]
    fn failed_new_policy_publication_preserves_previous_output() {
        let dir = tempfile::tempdir().unwrap();
        let input = dir.path().join("model.gguf");
        let output_dir = dir.path().join("out");
        write_test_model(&input);
        let CompressOutcome::Compressed(previous_output) =
            compress_once(&input, &output_dir, 64).unwrap()
        else {
            panic!("initial source was not compressed");
        };
        let previous_bytes = std::fs::read(&previous_output).unwrap();

        let (_, identity, _) = source_identity(&input, u64::MAX).unwrap();
        let next_output = output_dir.join(format!("model-{identity}-b128-turbo3.gguf"));
        let blocked_sidecar = sibling_with_suffix(&next_output, ".provenance.json").unwrap();
        std::fs::create_dir(&blocked_sidecar).unwrap();

        assert!(compress_once(&input, &output_dir, 128).is_err());
        assert_eq!(std::fs::read(&previous_output).unwrap(), previous_bytes);
        assert!(!next_output.exists());
    }

    #[test]
    fn bounded_queue_and_debounce_cache_survive_sustained_load() {
        let (tx, _rx) = tokio::sync::mpsc::channel(2);
        assert!(tx.try_send(PathBuf::from("a.gguf")).is_ok());
        assert!(tx.try_send(PathBuf::from("b.gguf")).is_ok());
        assert!(matches!(
            tx.try_send(PathBuf::from("c.gguf")),
            Err(TrySendError::Full(_))
        ));

        let mut cache = HashMap::new();
        let now = Instant::now();
        for index in 0..10_000 {
            assert!(should_handle(
                &mut cache,
                PathBuf::from(format!("model-{index}.gguf")),
                now,
                Duration::from_secs(30),
                128,
            ));
            assert!(cache.len() <= 128);
        }
        assert!(!should_handle(
            &mut cache,
            PathBuf::from("model-9999.gguf"),
            now,
            Duration::from_secs(30),
            128,
        ));
        assert!(should_handle(
            &mut cache,
            PathBuf::from("after-expiry.gguf"),
            now + Duration::from_secs(31),
            Duration::from_secs(30),
            128,
        ));
        assert_eq!(cache.len(), 1);
    }
}
