#![deny(missing_docs)]
// Pedantic allows justified by the FFI nature of this crate:
// - borrow_as_ptr: `&mut x` at extern "C" call sites (tests) is the
//   idiomatic way to pass out-params to the ABI under test;
// - too_many_lines: tq_quantize/tq_dequantize are long because every
//   argument is validated up front (the panic-free contract);
// - cast_precision_loss / float_cmp: test data generation and exact
//   reference comparisons in the round-trip tests.
#![allow(clippy::borrow_as_ptr)]
#![allow(clippy::too_many_lines)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::float_cmp)]

//! C FFI bindings for `TurboQuant`.
//!
//! Exposes a stable, panic-free C ABI over the QJL 3-bit quantizer from
//! `turboquant-core`. Every function that can fail returns an `int` status
//! code (`TQ_OK` == 0 on success). All pointer arguments are validated
//! before use; no panics cross the FFI boundary (inputs are validated up
//! front and remaining core calls are additionally wrapped in
//! `catch_unwind` as defense in depth for unwinding builds).
//!
//! The C header `include/turboquant.h` is regenerated from this file by
//! cbindgen at build time (see `build.rs`), so it always matches the ABI.
//!
//! There is no global state: the only stateful object is the opaque
//! [`tq_quantizer`] handle created by [`tq_quantizer_create`] and released
//! by [`tq_quantizer_destroy`].

use std::os::raw::{c_char, c_int};
use std::panic::{catch_unwind, AssertUnwindSafe};

use half::f16;
use turboquant_core::qjl::{CompressedBlock, CorrectionMode, QjlConfig, QjlQuantizer, ScaleMode};

// ---------------------------------------------------------------------------
// Status codes
// ---------------------------------------------------------------------------

/// Operation completed successfully.
pub const TQ_OK: c_int = 0;
/// A required pointer argument was NULL.
pub const TQ_ERR_NULL_POINTER: c_int = 1;
/// An argument value was invalid (bad enum value, out-of-range parameter,
/// zero-length input, non-finite float, ...).
pub const TQ_ERR_INVALID_ARGUMENT: c_int = 2;
/// An output buffer capacity was too small. Use `tq_packed_size` /
/// `tq_corr_size` to size buffers.
pub const TQ_ERR_BUFFER_TOO_SMALL: c_int = 3;
/// An internal error occurred (should not happen for valid inputs).
pub const TQ_ERR_INTERNAL: c_int = 4;

// ---------------------------------------------------------------------------
// Scale modes
// ---------------------------------------------------------------------------

/// Scale = max(|x|) of the block (`scale_param` is ignored).
pub const TQ_SCALE_ABSMAX: c_int = 0;
/// Scale = the `scale_param`-th percentile of |x|; `scale_param` must be in
/// [0.0, 1.0].
pub const TQ_SCALE_PERCENTILE: c_int = 1;
/// Scale = standard deviation of the block (`scale_param` is ignored).
pub const TQ_SCALE_ADAPTIVE: c_int = 2;
/// Scale = `scale_param` (must be finite and > 0).
pub const TQ_SCALE_FIXED: c_int = 3;

// ---------------------------------------------------------------------------
// Opaque handle
// ---------------------------------------------------------------------------

/// Opaque quantizer handle. Create with `tq_quantizer_create`, release with
/// `tq_quantizer_destroy`. The handle is immutable after creation and safe
/// to share across threads for concurrent `tq_quantize` / `tq_dequantize`
/// calls.
#[allow(non_camel_case_types)]
pub struct tq_quantizer {
    inner: QjlQuantizer,
}

// ---------------------------------------------------------------------------
// Version
// ---------------------------------------------------------------------------

/// Returns the `TurboQuant` version as a static NUL-terminated UTF-8 string.
/// The returned pointer is valid for the lifetime of the program and must
/// not be freed.
#[no_mangle]
pub extern "C" fn tq_version() -> *const c_char {
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr().cast()
}

// ---------------------------------------------------------------------------
// Buffer sizing helpers
// ---------------------------------------------------------------------------

fn packed_size(n: usize) -> Option<usize> {
    if n == 0 {
        return None;
    }
    // n padded up to a multiple of 8 values, 3 bits per value.
    n.checked_next_multiple_of(8).map(|p| p / 8 * 3)
}

const fn corr_size(n: usize) -> usize {
    n.div_ceil(8)
}

/// Returns the number of bytes of packed output produced when quantizing
/// `n` float values (`n` is padded up to a multiple of 8 values; 3 bits per
/// value). Returns 0 if `n` is 0 or unrepresentably large.
#[no_mangle]
pub extern "C" fn tq_packed_size(n: usize) -> usize {
    packed_size(n).unwrap_or(0)
}

/// Returns the number of bytes of correction output produced when quantizing
/// `n` float values with correction enabled (1 bit per value). Returns 0 if
/// `n` is 0.
#[no_mangle]
pub extern "C" fn tq_corr_size(n: usize) -> usize {
    corr_size(n)
}

// ---------------------------------------------------------------------------
// Create / destroy
// ---------------------------------------------------------------------------

/// Creates a quantizer and stores the handle in `*out_quantizer`.
///
/// - `bits`: bits per value; only 3 is currently supported.
/// - `block_size`: nominal block size stored in the configuration. Each
///   `tq_quantize` call quantizes its entire input as one block with one
///   scale, so callers chunk their data and typically pass `block_size`
///   values per call.
/// - `scale_mode`: one of `TQ_SCALE_ABSMAX`, `TQ_SCALE_PERCENTILE`,
///   `TQ_SCALE_ADAPTIVE`, `TQ_SCALE_FIXED`.
/// - `scale_param`: percentile in `[0, 1]` for `TQ_SCALE_PERCENTILE`, fixed
///   scale (> 0) for `TQ_SCALE_FIXED`, ignored otherwise.
/// - `correction_enabled`: non-zero enables 1-bit residual correction.
/// - `correction_scale`: correction magnitude relative to the block scale
///   (a quarter step, 0.25/3.5 ≈ 0.0714, is the MSE-optimal fixed value —
///   see `turboquant_core::qjl::DEFAULT_CORRECTION_SCALE`); must be finite
///   and >= 0 when correction is enabled,
///   ignored otherwise.
///
/// Returns `TQ_OK` and a non-NULL `*out_quantizer` on success. On failure,
/// `*out_quantizer` is left untouched.
///
/// # Safety
///
/// `out_quantizer` must be NULL or a valid pointer to writable storage for
/// one `tq_quantizer*`.
#[no_mangle]
pub unsafe extern "C" fn tq_quantizer_create(
    bits: u8,
    block_size: usize,
    scale_mode: c_int,
    scale_param: f32,
    correction_enabled: c_int,
    correction_scale: f32,
    out_quantizer: *mut *mut tq_quantizer,
) -> c_int {
    if out_quantizer.is_null() {
        return TQ_ERR_NULL_POINTER;
    }
    if bits != 3 || block_size == 0 {
        return TQ_ERR_INVALID_ARGUMENT;
    }
    let scale_mode = match scale_mode {
        TQ_SCALE_ABSMAX => ScaleMode::PerBlockAbsMax,
        TQ_SCALE_PERCENTILE => {
            if !scale_param.is_finite() || !(0.0..=1.0).contains(&scale_param) {
                return TQ_ERR_INVALID_ARGUMENT;
            }
            ScaleMode::PerBlockPercentile(scale_param)
        }
        TQ_SCALE_ADAPTIVE => ScaleMode::Adaptive,
        TQ_SCALE_FIXED => {
            if !scale_param.is_finite() || scale_param <= 0.0 {
                return TQ_ERR_INVALID_ARGUMENT;
            }
            ScaleMode::Fixed(scale_param)
        }
        _ => return TQ_ERR_INVALID_ARGUMENT,
    };
    let correction = if correction_enabled != 0 {
        if !correction_scale.is_finite() || correction_scale < 0.0 {
            return TQ_ERR_INVALID_ARGUMENT;
        }
        CorrectionMode::OneBitResidual {
            learned_scale: correction_scale,
        }
    } else {
        CorrectionMode::None
    };

    let config = QjlConfig {
        bits,
        block_size,
        scale_mode,
        correction,
    };
    let handle = Box::new(tq_quantizer {
        inner: QjlQuantizer::new(config),
    });
    // SAFETY: out_quantizer was checked to be non-NULL; the caller
    // guarantees it points to writable storage.
    unsafe {
        *out_quantizer = Box::into_raw(handle);
    }
    TQ_OK
}

/// Destroys a quantizer created with `tq_quantizer_create`. Passing NULL is
/// a safe no-op. The handle must not be used after this call.
///
/// # Safety
///
/// `quantizer` must be NULL or a pointer previously returned by
/// `tq_quantizer_create` that has not already been destroyed.
#[no_mangle]
pub unsafe extern "C" fn tq_quantizer_destroy(quantizer: *mut tq_quantizer) {
    if !quantizer.is_null() {
        // SAFETY: the caller guarantees this pointer came from
        // Box::into_raw in tq_quantizer_create and is not used again.
        drop(unsafe { Box::from_raw(quantizer) });
    }
}

// ---------------------------------------------------------------------------
// Quantize / dequantize
// ---------------------------------------------------------------------------

/// Quantizes `n` float values as a single block.
///
/// - `input`: `n` finite float values (`n` > 0).
/// - `packed_out` / `packed_cap`: output buffer for the packed 3-bit data;
///   `packed_cap` must be >= `tq_packed_size(n)`.
/// - `packed_written`: receives the number of packed bytes written.
/// - `scale_out`: receives the block scale (round-trips through f16).
/// - `corr_out` / `corr_cap`: output buffer for the 1-bit correction data;
///   required (>= `tq_corr_size(n)`) when the quantizer was created with
///   correction enabled, ignored (may be NULL) otherwise.
/// - `corr_written`: optional (may be NULL); receives the number of
///   correction bytes written (0 when correction is disabled).
///
/// Returns `TQ_OK` on success.
///
/// # Safety
///
/// All non-NULL pointers must be valid for the accesses implied by the
/// documented sizes: `input` readable for `n` floats, `packed_out` writable
/// for `packed_cap` bytes, `corr_out` writable for `corr_cap` bytes,
/// `packed_written`, `scale_out` and `corr_written` writable.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn tq_quantize(
    quantizer: *const tq_quantizer,
    input: *const f32,
    n: usize,
    packed_out: *mut u8,
    packed_cap: usize,
    packed_written: *mut usize,
    scale_out: *mut f32,
    corr_out: *mut u8,
    corr_cap: usize,
    corr_written: *mut usize,
) -> c_int {
    if quantizer.is_null() || input.is_null() || packed_out.is_null() {
        return TQ_ERR_NULL_POINTER;
    }
    if packed_written.is_null() || scale_out.is_null() {
        return TQ_ERR_NULL_POINTER;
    }
    let Some(needed_packed) = packed_size(n) else {
        return TQ_ERR_INVALID_ARGUMENT;
    };
    if n > (isize::MAX as usize) / std::mem::size_of::<f32>() {
        return TQ_ERR_INVALID_ARGUMENT;
    }
    if packed_cap < needed_packed {
        return TQ_ERR_BUFFER_TOO_SMALL;
    }

    // SAFETY: quantizer is non-NULL and (per the caller contract) points to
    // a live handle.
    let handle = unsafe { &*quantizer };
    let correction_enabled = matches!(
        handle.inner.config().correction,
        CorrectionMode::OneBitResidual { .. }
    );
    let needed_corr = corr_size(n);
    if correction_enabled {
        if corr_out.is_null() {
            return TQ_ERR_NULL_POINTER;
        }
        if corr_cap < needed_corr {
            return TQ_ERR_BUFFER_TOO_SMALL;
        }
    }

    // SAFETY: input is non-NULL and readable for n floats per the caller
    // contract; n * 4 bytes was checked to fit in isize.
    let data = unsafe { std::slice::from_raw_parts(input, n) };

    let Ok(block) = catch_unwind(AssertUnwindSafe(|| handle.inner.quantize_block(data))) else {
        return TQ_ERR_INTERNAL;
    };
    if block.packed.len() != needed_packed {
        return TQ_ERR_INTERNAL;
    }

    // SAFETY: packed_out is writable for packed_cap >= needed_packed bytes.
    unsafe {
        std::ptr::copy_nonoverlapping(block.packed.as_ptr(), packed_out, block.packed.len());
        *packed_written = block.packed.len();
        *scale_out = block.scale.to_f32();
    }

    let mut corr_len = 0usize;
    if let Some(bits) = &block.correction_bits {
        if !correction_enabled || bits.len() != needed_corr {
            return TQ_ERR_INTERNAL;
        }
        // SAFETY: corr_out is non-NULL and writable for corr_cap >=
        // needed_corr bytes (checked above when correction is enabled).
        unsafe {
            std::ptr::copy_nonoverlapping(bits.as_ptr(), corr_out, bits.len());
        }
        corr_len = bits.len();
    }
    if !corr_written.is_null() {
        // SAFETY: corr_written is non-NULL and writable per the contract.
        unsafe {
            *corr_written = corr_len;
        }
    }
    TQ_OK
}

/// Dequantizes a block previously produced by `tq_quantize` back to `n`
/// float values.
///
/// - `packed` / `packed_len`: the packed 3-bit data;
///   `packed_len` must be >= `tq_packed_size(n)`.
/// - `n`: the original number of values (> 0).
/// - `scale`: the block scale returned by `tq_quantize`; it must remain finite
///   and non-zero after conversion to the core's `f16` scale representation.
/// - `corr` / `corr_len`: optional correction data; when `corr` is non-NULL,
///   `corr_len` must be >= `tq_corr_size(n)`. Correction is applied only if
///   the quantizer was created with correction enabled; passing NULL skips
///   correction.
/// - `output` / `output_cap`: output buffer; `output_cap` must be >= `n`
///   floats.
///
/// Returns `TQ_OK` on success.
///
/// # Safety
///
/// All non-NULL pointers must be valid for the accesses implied by the
/// documented sizes: `packed` readable for `packed_len` bytes, `corr`
/// readable for `corr_len` bytes, `output` writable for `output_cap` floats.
#[no_mangle]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn tq_dequantize(
    quantizer: *const tq_quantizer,
    packed: *const u8,
    packed_len: usize,
    n: usize,
    scale: f32,
    corr: *const u8,
    corr_len: usize,
    output: *mut f32,
    output_cap: usize,
) -> c_int {
    if quantizer.is_null() || packed.is_null() || output.is_null() {
        return TQ_ERR_NULL_POINTER;
    }
    let Some(needed_packed) = packed_size(n) else {
        return TQ_ERR_INVALID_ARGUMENT;
    };
    if !scale.is_finite() || scale <= 0.0 {
        return TQ_ERR_INVALID_ARGUMENT;
    }
    let scale = f16::from_f32(scale);
    if !scale.is_finite() || scale <= f16::ZERO {
        return TQ_ERR_INVALID_ARGUMENT;
    }
    if packed_len < needed_packed {
        return TQ_ERR_BUFFER_TOO_SMALL;
    }
    if output_cap < n {
        return TQ_ERR_BUFFER_TOO_SMALL;
    }
    if !corr.is_null() && corr_len < corr_size(n) {
        return TQ_ERR_BUFFER_TOO_SMALL;
    }

    // SAFETY: quantizer is non-NULL and points to a live handle; packed is
    // readable for packed_len >= needed_packed bytes; corr (when non-NULL)
    // is readable for corr_len >= corr_size(n) bytes.
    let handle = unsafe { &*quantizer };
    let packed_vec = unsafe { std::slice::from_raw_parts(packed, needed_packed) }.to_vec();
    let corr_vec = if corr.is_null() {
        None
    } else {
        Some(unsafe { std::slice::from_raw_parts(corr, corr_size(n)) }.to_vec())
    };

    let block = CompressedBlock {
        packed: packed_vec,
        scale,
        correction_bits: corr_vec,
    };

    let Ok(values) = catch_unwind(AssertUnwindSafe(|| {
        handle.inner.dequantize_block(&block, n)
    })) else {
        return TQ_ERR_INTERNAL;
    };
    if values.len() != n {
        return TQ_ERR_INTERNAL;
    }

    // SAFETY: output is writable for output_cap >= n floats.
    unsafe {
        std::ptr::copy_nonoverlapping(values.as_ptr(), output, n);
    }
    TQ_OK
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CStr;
    use std::ptr;

    fn make_quantizer(correction: bool) -> *mut tq_quantizer {
        let mut q: *mut tq_quantizer = ptr::null_mut();
        let rc = unsafe {
            tq_quantizer_create(
                3,
                64,
                TQ_SCALE_ABSMAX,
                0.0,
                c_int::from(correction),
                0.01,
                &mut q,
            )
        };
        assert_eq!(rc, TQ_OK);
        assert!(!q.is_null());
        q
    }

    #[test]
    fn version_matches_cargo() {
        let ptr = tq_version();
        assert!(!ptr.is_null());
        let s = unsafe { CStr::from_ptr(ptr) }.to_str().unwrap();
        assert_eq!(s, env!("CARGO_PKG_VERSION"));
    }

    #[test]
    fn sizing_helpers() {
        assert_eq!(tq_packed_size(0), 0);
        assert_eq!(tq_packed_size(1), 3);
        assert_eq!(tq_packed_size(8), 3);
        assert_eq!(tq_packed_size(64), 24);
        assert_eq!(tq_packed_size(usize::MAX), 0); // overflow
        assert_eq!(tq_corr_size(0), 0);
        assert_eq!(tq_corr_size(1), 1);
        assert_eq!(tq_corr_size(8), 1);
        assert_eq!(tq_corr_size(64), 8);
    }

    #[test]
    fn create_rejects_invalid_config() {
        let mut q: *mut tq_quantizer = ptr::null_mut();
        unsafe {
            // NULL out pointer.
            assert_eq!(
                tq_quantizer_create(3, 64, TQ_SCALE_ABSMAX, 0.0, 0, 0.0, ptr::null_mut()),
                TQ_ERR_NULL_POINTER
            );
            // Unsupported bit width.
            assert_eq!(
                tq_quantizer_create(4, 64, TQ_SCALE_ABSMAX, 0.0, 0, 0.0, &mut q),
                TQ_ERR_INVALID_ARGUMENT
            );
            // Zero block size.
            assert_eq!(
                tq_quantizer_create(3, 0, TQ_SCALE_ABSMAX, 0.0, 0, 0.0, &mut q),
                TQ_ERR_INVALID_ARGUMENT
            );
            // Unknown scale mode.
            assert_eq!(
                tq_quantizer_create(3, 64, 99, 0.0, 0, 0.0, &mut q),
                TQ_ERR_INVALID_ARGUMENT
            );
            // Percentile out of range.
            assert_eq!(
                tq_quantizer_create(3, 64, TQ_SCALE_PERCENTILE, 1.5, 0, 0.0, &mut q),
                TQ_ERR_INVALID_ARGUMENT
            );
            // Non-positive fixed scale.
            assert_eq!(
                tq_quantizer_create(3, 64, TQ_SCALE_FIXED, -1.0, 0, 0.0, &mut q),
                TQ_ERR_INVALID_ARGUMENT
            );
            // Non-finite correction scale.
            assert_eq!(
                tq_quantizer_create(3, 64, TQ_SCALE_ABSMAX, 0.0, 1, f32::NAN, &mut q),
                TQ_ERR_INVALID_ARGUMENT
            );
        }
        assert!(q.is_null(), "failed create must not write the handle");
    }

    #[test]
    fn destroy_null_is_noop() {
        unsafe { tq_quantizer_destroy(ptr::null_mut()) };
    }

    #[test]
    fn all_scale_modes_create() {
        for (mode, param) in [
            (TQ_SCALE_ABSMAX, 0.0),
            (TQ_SCALE_PERCENTILE, 0.99),
            (TQ_SCALE_ADAPTIVE, 0.0),
            (TQ_SCALE_FIXED, 2.5),
        ] {
            let mut q: *mut tq_quantizer = ptr::null_mut();
            let rc = unsafe { tq_quantizer_create(3, 64, mode, param, 1, 0.01, &mut q) };
            assert_eq!(rc, TQ_OK, "mode {mode}");
            unsafe { tq_quantizer_destroy(q) };
        }
    }

    fn ffi_roundtrip(q: *mut tq_quantizer, data: &[f32], with_corr: bool) -> (Vec<f32>, f32) {
        let n = data.len();
        let mut packed = vec![0u8; tq_packed_size(n)];
        let mut corr = vec![0u8; tq_corr_size(n)];
        let mut packed_written = 0usize;
        let mut corr_written = 0usize;
        let mut scale = 0f32;
        let rc = unsafe {
            tq_quantize(
                q,
                data.as_ptr(),
                n,
                packed.as_mut_ptr(),
                packed.len(),
                &mut packed_written,
                &mut scale,
                if with_corr {
                    corr.as_mut_ptr()
                } else {
                    ptr::null_mut()
                },
                if with_corr { corr.len() } else { 0 },
                &mut corr_written,
            )
        };
        assert_eq!(rc, TQ_OK);
        assert_eq!(packed_written, tq_packed_size(n));
        if with_corr {
            assert_eq!(corr_written, tq_corr_size(n));
        } else {
            assert_eq!(corr_written, 0);
        }

        let mut out = vec![0f32; n];
        let rc = unsafe {
            tq_dequantize(
                q,
                packed.as_ptr(),
                packed_written,
                n,
                scale,
                if with_corr {
                    corr.as_ptr()
                } else {
                    ptr::null()
                },
                corr_written,
                out.as_mut_ptr(),
                out.len(),
            )
        };
        assert_eq!(rc, TQ_OK);
        (out, scale)
    }

    #[test]
    fn roundtrip_matches_core_with_correction() {
        let data: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) / 10.0).collect();
        let q = make_quantizer(true);
        let (ffi_out, ffi_scale) = ffi_roundtrip(q, &data, true);
        unsafe { tq_quantizer_destroy(q) };

        // Reference result straight through turboquant-core.
        let core_q = QjlQuantizer::new(QjlConfig {
            bits: 3,
            block_size: 64,
            scale_mode: ScaleMode::PerBlockAbsMax,
            correction: CorrectionMode::OneBitResidual {
                learned_scale: 0.01,
            },
        });
        let block = core_q.quantize_block(&data);
        let core_out = core_q.dequantize_block(&block, data.len());

        assert_eq!(ffi_scale, block.scale.to_f32());
        assert_eq!(ffi_out, core_out);
    }

    #[test]
    fn roundtrip_no_correction_unaligned_len() {
        // 50 is deliberately not a multiple of 8.
        let data: Vec<f32> = (0..50).map(|i| ((i * 7) % 13) as f32 - 6.0).collect();
        let mut q: *mut tq_quantizer = ptr::null_mut();
        let rc = unsafe { tq_quantizer_create(3, 64, TQ_SCALE_PERCENTILE, 0.95, 0, 0.0, &mut q) };
        assert_eq!(rc, TQ_OK);
        let (out, scale) = ffi_roundtrip(q, &data, false);
        unsafe { tq_quantizer_destroy(q) };

        assert!(scale > 0.0);
        assert_eq!(out.len(), data.len());
        let mse: f32 = data
            .iter()
            .zip(&out)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            / data.len() as f32;
        assert!(mse < 2.0, "mse too high: {mse}");
    }

    #[test]
    fn quantize_error_codes() {
        let data: Vec<f32> = (0..64).map(|i| i as f32).collect();
        let n = data.len();
        let mut packed = vec![0u8; tq_packed_size(n)];
        let mut corr = vec![0u8; tq_corr_size(n)];
        let mut written = 0usize;
        let mut scale = 0f32;
        let q = make_quantizer(true);
        unsafe {
            // NULL quantizer / input / outputs.
            assert_eq!(
                tq_quantize(
                    ptr::null(),
                    data.as_ptr(),
                    n,
                    packed.as_mut_ptr(),
                    packed.len(),
                    &mut written,
                    &mut scale,
                    corr.as_mut_ptr(),
                    corr.len(),
                    ptr::null_mut(),
                ),
                TQ_ERR_NULL_POINTER
            );
            assert_eq!(
                tq_quantize(
                    q,
                    ptr::null(),
                    n,
                    packed.as_mut_ptr(),
                    packed.len(),
                    &mut written,
                    &mut scale,
                    corr.as_mut_ptr(),
                    corr.len(),
                    ptr::null_mut(),
                ),
                TQ_ERR_NULL_POINTER
            );
            assert_eq!(
                tq_quantize(
                    q,
                    data.as_ptr(),
                    n,
                    ptr::null_mut(),
                    packed.len(),
                    &mut written,
                    &mut scale,
                    corr.as_mut_ptr(),
                    corr.len(),
                    ptr::null_mut(),
                ),
                TQ_ERR_NULL_POINTER
            );
            assert_eq!(
                tq_quantize(
                    q,
                    data.as_ptr(),
                    n,
                    packed.as_mut_ptr(),
                    packed.len(),
                    ptr::null_mut(),
                    &mut scale,
                    corr.as_mut_ptr(),
                    corr.len(),
                    ptr::null_mut(),
                ),
                TQ_ERR_NULL_POINTER
            );
            // Correction enabled but corr_out NULL.
            assert_eq!(
                tq_quantize(
                    q,
                    data.as_ptr(),
                    n,
                    packed.as_mut_ptr(),
                    packed.len(),
                    &mut written,
                    &mut scale,
                    ptr::null_mut(),
                    0,
                    ptr::null_mut(),
                ),
                TQ_ERR_NULL_POINTER
            );
            // Zero-length input.
            assert_eq!(
                tq_quantize(
                    q,
                    data.as_ptr(),
                    0,
                    packed.as_mut_ptr(),
                    packed.len(),
                    &mut written,
                    &mut scale,
                    corr.as_mut_ptr(),
                    corr.len(),
                    ptr::null_mut(),
                ),
                TQ_ERR_INVALID_ARGUMENT
            );
            // Packed buffer one byte short.
            assert_eq!(
                tq_quantize(
                    q,
                    data.as_ptr(),
                    n,
                    packed.as_mut_ptr(),
                    packed.len() - 1,
                    &mut written,
                    &mut scale,
                    corr.as_mut_ptr(),
                    corr.len(),
                    ptr::null_mut(),
                ),
                TQ_ERR_BUFFER_TOO_SMALL
            );
            // Correction buffer one byte short.
            assert_eq!(
                tq_quantize(
                    q,
                    data.as_ptr(),
                    n,
                    packed.as_mut_ptr(),
                    packed.len(),
                    &mut written,
                    &mut scale,
                    corr.as_mut_ptr(),
                    corr.len() - 1,
                    ptr::null_mut(),
                ),
                TQ_ERR_BUFFER_TOO_SMALL
            );
            tq_quantizer_destroy(q);
        }
    }

    #[test]
    fn dequantize_error_codes() {
        let n = 64usize;
        let packed = vec![0u8; tq_packed_size(n)];
        let corr = vec![0u8; tq_corr_size(n)];
        let mut out = vec![0f32; n];
        let q = make_quantizer(true);
        unsafe {
            assert_eq!(
                tq_dequantize(
                    ptr::null(),
                    packed.as_ptr(),
                    packed.len(),
                    n,
                    1.0,
                    corr.as_ptr(),
                    corr.len(),
                    out.as_mut_ptr(),
                    out.len(),
                ),
                TQ_ERR_NULL_POINTER
            );
            assert_eq!(
                tq_dequantize(
                    q,
                    ptr::null(),
                    packed.len(),
                    n,
                    1.0,
                    corr.as_ptr(),
                    corr.len(),
                    out.as_mut_ptr(),
                    out.len(),
                ),
                TQ_ERR_NULL_POINTER
            );
            assert_eq!(
                tq_dequantize(
                    q,
                    packed.as_ptr(),
                    packed.len(),
                    n,
                    1.0,
                    corr.as_ptr(),
                    corr.len(),
                    ptr::null_mut(),
                    out.len(),
                ),
                TQ_ERR_NULL_POINTER
            );
            // Zero n.
            assert_eq!(
                tq_dequantize(
                    q,
                    packed.as_ptr(),
                    packed.len(),
                    0,
                    1.0,
                    corr.as_ptr(),
                    corr.len(),
                    out.as_mut_ptr(),
                    out.len(),
                ),
                TQ_ERR_INVALID_ARGUMENT
            );
            // Bad scales.
            for bad in [
                0.0f32,
                -1.0,
                f32::NAN,
                f32::INFINITY,
                f32::NEG_INFINITY,
                70_000.0,
                1.0e-20,
            ] {
                assert_eq!(
                    tq_dequantize(
                        q,
                        packed.as_ptr(),
                        packed.len(),
                        n,
                        bad,
                        corr.as_ptr(),
                        corr.len(),
                        out.as_mut_ptr(),
                        out.len(),
                    ),
                    TQ_ERR_INVALID_ARGUMENT,
                    "scale {bad}"
                );
            }
            // Values at both non-zero finite binary16 boundaries remain valid.
            for boundary in [f16::from_bits(1).to_f32(), f16::MAX.to_f32()] {
                assert_eq!(
                    tq_dequantize(
                        q,
                        packed.as_ptr(),
                        packed.len(),
                        n,
                        boundary,
                        corr.as_ptr(),
                        corr.len(),
                        out.as_mut_ptr(),
                        out.len(),
                    ),
                    TQ_OK,
                    "scale {boundary}"
                );
            }
            // Short packed buffer.
            assert_eq!(
                tq_dequantize(
                    q,
                    packed.as_ptr(),
                    packed.len() - 1,
                    n,
                    1.0,
                    corr.as_ptr(),
                    corr.len(),
                    out.as_mut_ptr(),
                    out.len(),
                ),
                TQ_ERR_BUFFER_TOO_SMALL
            );
            // Short output buffer.
            assert_eq!(
                tq_dequantize(
                    q,
                    packed.as_ptr(),
                    packed.len(),
                    n,
                    1.0,
                    corr.as_ptr(),
                    corr.len(),
                    out.as_mut_ptr(),
                    out.len() - 1,
                ),
                TQ_ERR_BUFFER_TOO_SMALL
            );
            // Short correction buffer (only checked when corr is non-NULL).
            assert_eq!(
                tq_dequantize(
                    q,
                    packed.as_ptr(),
                    packed.len(),
                    n,
                    1.0,
                    corr.as_ptr(),
                    corr.len() - 1,
                    out.as_mut_ptr(),
                    out.len(),
                ),
                TQ_ERR_BUFFER_TOO_SMALL
            );
            // NULL corr with correction-enabled quantizer is allowed:
            // correction is simply skipped.
            assert_eq!(
                tq_dequantize(
                    q,
                    packed.as_ptr(),
                    packed.len(),
                    n,
                    1.0,
                    ptr::null(),
                    0,
                    out.as_mut_ptr(),
                    out.len(),
                ),
                TQ_OK
            );
            tq_quantizer_destroy(q);
        }
    }
}
