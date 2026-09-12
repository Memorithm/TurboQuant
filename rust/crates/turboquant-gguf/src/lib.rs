#![deny(missing_docs)]
#![cfg_attr(not(test), warn(clippy::unwrap_used, clippy::expect_used))]
// Clippy allows for practical binary-format code (same set as turboquant-core).
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::missing_errors_doc)]

//! GGUF I/O for `TurboQuant`.
//!
//! Full GGUF v3 container support (read v2/v3, write v3): metadata
//! key/value pairs of all GGUF value types (including strings and
//! arrays), tensor infos, alignment handling, and tensor-data access
//! with F32/F16 decoding. The [`turbo`] module implements the
//! `TurboQuant` "turbo3" compressed-model format on top of the container.

/// Parser for GGUF format files.
pub mod parser;
/// TurboQuant compression of GGUF files.
pub mod turbo;
/// GGUF types and constants.
pub mod types;
/// Writer for GGUF format files.
pub mod writer;

pub use parser::{GgufFile, GgufParser};
pub use types::{GgmlType, GgufHeader, GgufTensorInfo, GgufValue, GgufValueType};
pub use writer::GgufWriter;
