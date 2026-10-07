# Daemon

`turboquant daemon` (or the `turboquant-daemon` binary) watches
configured directories for new `.gguf` files and compresses each into an
output directory. It integrates with systemd (`Type=notify` readiness)
and shuts down gracefully on SIGTERM/ctrl-c.

## systemd watchdog

When the unit sets `WatchdogSec=` (the shipped unit uses 60 seconds),
systemd exports `WATCHDOG_USEC` (and `WATCHDOG_PID`) to the daemon,
which then sends `WATCHDOG=1` keepalives at half that interval (clamped
to at least 1 second). If the daemon's event loop hangs, the keepalives
stop and systemd restarts the service. `WATCHDOG_PID` is honored: if it
names another process, no keepalives are sent. Outside systemd (no
watchdog environment), no keepalive task runs at all.

## Configuration

JSON config file (`turboquant daemon --config config.json`); defaults:

- `watch_dirs`: directories scanned for `.gguf` files
- `output_dir`: `~/.turboquant/compressed`
- `listen_addr`: `127.0.0.1:7460`
- `block_size`, `interval_secs` (debounce)
- `event_queue_capacity`: bounded filesystem-event queue (default: 256)
- `max_concurrent_jobs`: maximum tracked compression workers (default: 2)
- `max_file_bytes`: GGUF admission/read budget (default: 64 GiB)
- `debounce_cache_capacity`: bounded recent-path cache (default: 4096)

Duplicate filesystem activity is coalesced while a path is queued. When the
bounded queue is full, additional unique paths are dropped rather than forming
an unbounded backlog. Both counters are exposed by `/healthz`. Active
compression jobs are tracked and drained during shutdown.

Compressed files are named
`<stem>-<source-identity>-turbo3.gguf`. The identity covers the canonical
source path and source bytes, so equal stems in different directories cannot
overwrite each other. Each output has a `.provenance.json` sidecar. Output and
sidecar are staged in exclusive temporary files and published under a
per-destination lock.

## HTTP API

A single health endpoint:

```text
GET /healthz   ->  {"status":"ok","files_compressed":N,"failures":M,"events_dropped":D,"events_coalesced":C}
```
