#![deny(missing_docs)]
#![doc = include_str!("../README.md")]
#![cfg_attr(not(test), warn(clippy::unwrap_used, clippy::expect_used))]
// Clippy allows for practical numerical code
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::double_must_use)]
#![allow(clippy::items_after_statements)]
#![allow(clippy::option_if_let_else)]
#![allow(clippy::return_self_not_must_use)]

//! `TurboQuant` Core: fundamental algorithms for 3-bit KV cache compression.
//!
//! This crate provides:
//! - Polar rotation (QR, Householder, Hadamard)
//! - QJL 3-bit quantization with 1-bit residual correction
//! - 3-bit bit packing/unpacking
//! - Compressed KV block storage

pub mod bitpack;
pub mod kv_block;
pub mod polar;
pub mod qjl;
pub mod quantize;
pub mod rotation;

/// Common error type for TurboQuant operations.
pub mod error;

/// Result type alias for `TurboQuant` operations.
pub type Result<T> = std::result::Result<T, error::TurboQuantError>;
