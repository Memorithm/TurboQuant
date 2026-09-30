//! GGUF file parser (versions 2 and 3).

use crate::types::{
    GgmlType, GgufHeader, GgufTensorInfo, GgufValue, GgufValueType, DEFAULT_ALIGNMENT, GGUF_MAGIC,
};
use std::io::Read;
use std::path::Path;
use turboquant_core::error::TurboQuantError;

/// Maximum nesting depth for metadata arrays (defensive limit).
const MAX_ARRAY_DEPTH: u32 = 8;

/// Admission limits for owned GGUF bytes and decoded metadata allocations.
///
/// The allocation budget counts requested vector storage and string bytes;
/// it is not an RSS limit and excludes the caller-owned raw input buffer.
#[derive(Debug, Clone, Copy)]
pub struct GgufLimits {
    /// Maximum raw input size (default: 64 GiB).
    pub max_file_bytes: u64,
    /// Maximum metadata key/value count (default: 1,000,000).
    pub max_metadata_entries: usize,
    /// Maximum tensor count (default: 100,000).
    pub max_tensors: usize,
    /// Maximum elements in one metadata array (default: 1,000,000).
    pub max_array_elements: usize,
    /// Maximum bytes in one decoded string (default: 16 MiB).
    pub max_string_bytes: usize,
    /// Cumulative requested decoded allocation bytes (default: 256 MiB).
    pub max_decoded_bytes: usize,
}

impl Default for GgufLimits {
    fn default() -> Self {
        Self {
            max_file_bytes: 64 * 1024 * 1024 * 1024,
            max_metadata_entries: 1_000_000,
            max_tensors: 100_000,
            max_array_elements: 1_000_000,
            max_string_bytes: 16 * 1024 * 1024,
            max_decoded_bytes: 256 * 1024 * 1024,
        }
    }
}

fn bounded_count(count: u64, limit: usize, label: &str) -> Result<usize, TurboQuantError> {
    let count = usize::try_from(count).map_err(|_| err(format!("{label} count overflow")))?;
    if count > limit {
        return Err(err(format!("{label} count {count} exceeds limit {limit}")));
    }
    Ok(count)
}

fn err(msg: impl Into<String>) -> TurboQuantError {
    TurboQuantError::InvalidGguf(msg.into())
}

/// Parse a GGUF header (magic + version + tensor count + metadata KV
/// count) from the first 24 bytes of a file.
///
/// # Errors
///
/// Returns an error if the buffer is too short, the magic is wrong, or
/// the version is unsupported (only v2 and v3 are supported).
pub fn parse_header(data: &[u8]) -> Result<GgufHeader, TurboQuantError> {
    if data.len() < 24 {
        return Err(err("file too short for GGUF header (need 24 bytes)"));
    }
    if data[0..4] != GGUF_MAGIC {
        return Err(err("invalid GGUF magic number"));
    }
    let version = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
    if !(2..=3).contains(&version) {
        return Err(err(format!(
            "unsupported GGUF version {version} (supported: 2, 3)"
        )));
    }
    let tensor_count = u64::from_le_bytes([
        data[8], data[9], data[10], data[11], data[12], data[13], data[14], data[15],
    ]);
    let metadata_kv_count = u64::from_le_bytes([
        data[16], data[17], data[18], data[19], data[20], data[21], data[22], data[23],
    ]);
    Ok(GgufHeader {
        version,
        tensor_count,
        metadata_kv_count,
    })
}

/// A fully parsed GGUF file: header, metadata, tensor infos, and the
/// raw file bytes for tensor-data access.
#[derive(Debug, Clone)]
pub struct GgufFile {
    /// Parsed header.
    pub header: GgufHeader,
    /// Metadata key/value pairs, in file order.
    pub metadata: Vec<(String, GgufValue)>,
    /// Tensor infos, in file order.
    pub tensors: Vec<GgufTensorInfo>,
    /// Alignment of the data section (`general.alignment`, default 32).
    pub alignment: u64,
    /// Absolute file offset where the tensor-data section starts.
    pub data_start: usize,
    /// The complete file contents.
    data: Vec<u8>,
}

impl GgufFile {
    /// Look up a metadata value by key.
    #[must_use]
    pub fn metadata_value(&self, key: &str) -> Option<&GgufValue> {
        self.metadata.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    /// Look up a tensor info by name.
    #[must_use]
    pub fn tensor(&self, name: &str) -> Option<&GgufTensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// Raw bytes of a tensor's data.
    ///
    /// For element types with a known size the exact byte range is
    /// returned. For block-quantized/unknown types, the span up to the
    /// next tensor offset (or end of file) is returned, which may include
    /// trailing alignment padding.
    ///
    /// # Errors
    ///
    /// Returns an error if the tensor's byte range lies outside the file.
    pub fn tensor_data(&self, info: &GgufTensorInfo) -> Result<&[u8], TurboQuantError> {
        let start = self
            .data_start
            .checked_add(usize::try_from(info.offset).map_err(|_| err("tensor offset overflow"))?)
            .ok_or_else(|| err("tensor offset overflow"))?;
        let size = if let Some(element_size) = info.ggml_type.element_size() {
            let elements = info
                .dims
                .iter()
                .try_fold(1u64, |n, &dim| n.checked_mul(dim))
                .ok_or_else(|| err("tensor element count overflow"))?;
            let s = elements
                .checked_mul(element_size as u64)
                .ok_or_else(|| err("tensor size overflow"))?;
            usize::try_from(s).map_err(|_| err("tensor size overflow"))?
        } else {
            let next = self
                .tensors
                .iter()
                .map(|t| t.offset)
                .filter(|&o| o > info.offset)
                .min();
            if let Some(n) = next {
                usize::try_from(n - info.offset).map_err(|_| err("tensor size overflow"))?
            } else {
                self.data.len().saturating_sub(start)
            }
        };
        let end = start
            .checked_add(size)
            .ok_or_else(|| err("tensor extent overflow"))?;
        if end > self.data.len() {
            return Err(err(format!(
                "tensor '{}' data range {start}..{end} exceeds file size {}",
                info.name,
                self.data.len()
            )));
        }
        Ok(&self.data[start..end])
    }

    /// Decode a tensor's data to `f32`. Supported element types: F32, F16.
    ///
    /// # Errors
    ///
    /// Returns an error for other element types or out-of-range data.
    pub fn tensor_f32(&self, info: &GgufTensorInfo) -> Result<Vec<f32>, TurboQuantError> {
        let bytes = self.tensor_data(info)?;
        match info.ggml_type {
            GgmlType::F32 => Ok(bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()),
            GgmlType::F16 => Ok(bytes
                .chunks_exact(2)
                .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
                .collect()),
            other => Err(err(format!(
                "tensor '{}' has type {other:?}; only F32/F16 can be decoded to f32",
                info.name
            ))),
        }
    }

    /// The complete raw file contents.
    #[must_use]
    pub fn raw(&self) -> &[u8] {
        &self.data
    }
}

/// Parser for GGUF format files.
pub struct GgufParser;

impl GgufParser {
    /// Parse a GGUF file from an owned byte buffer.
    ///
    /// # Errors
    ///
    /// Returns an error on malformed or truncated input.
    pub fn parse(data: Vec<u8>) -> Result<GgufFile, TurboQuantError> {
        Self::parse_with_limits(data, GgufLimits::default())
    }

    /// Parse owned bytes under explicit admission limits before any decoded reserve.
    /// Returns `InvalidGguf` for exhausted budgets, impossible counts or allocations.
    pub fn parse_with_limits(
        data: Vec<u8>,
        limits: GgufLimits,
    ) -> Result<GgufFile, TurboQuantError> {
        if data.len() as u64 > limits.max_file_bytes {
            return Err(err("GGUF input exceeds file byte limit"));
        }
        let header = parse_header(&data)?;
        let metadata_count = bounded_count(
            header.metadata_kv_count,
            limits.max_metadata_entries,
            "metadata",
        )?;
        let tensor_count = bounded_count(header.tensor_count, limits.max_tensors, "tensor")?;
        // Even an empty key/name needs its u64 length. The smallest metadata
        // value needs a type tag and one byte; a scalar tensor needs 24 bytes.
        let minimum_body = metadata_count
            .checked_mul(13)
            .and_then(|n| tensor_count.checked_mul(24).and_then(|t| n.checked_add(t)))
            .ok_or_else(|| err("GGUF entry byte count overflow"))?;
        if minimum_body > data.len() - 24 {
            return Err(err("GGUF entry counts exceed remaining input"));
        }
        let mut r = Reader {
            buf: &data,
            pos: 24,
            limits,
            remaining_allocation: limits.max_decoded_bytes,
        };

        let mut metadata = r.reserve_vec(metadata_count)?;
        for _ in 0..metadata_count {
            let key = r.string()?;
            let ty = GgufValueType::from_u32(r.u32()?)?;
            let value = r.value(ty, 0)?;
            metadata.push((key, value));
        }

        let mut tensors = r.reserve_vec(tensor_count)?;
        for _ in 0..tensor_count {
            let name = r.string()?;
            let n_dims = r.u32()?;
            if n_dims > 8 {
                return Err(err(format!("tensor '{name}' has {n_dims} dims (max 8)")));
            }
            let mut dims = r.reserve_vec(n_dims as usize)?;
            for _ in 0..n_dims {
                dims.push(r.u64()?);
            }
            let ggml_type = GgmlType::from_u32(r.u32()?);
            let offset = r.u64()?;
            tensors.push(GgufTensorInfo {
                name,
                dims,
                ggml_type,
                offset,
            });
        }

        let alignment = match metadata
            .iter()
            .find(|(k, _)| k == "general.alignment")
            .and_then(|(_, v)| v.as_u64())
        {
            Some(a) if a.is_power_of_two() && a >= 8 => a,
            Some(a) => {
                return Err(err(format!(
                    "invalid general.alignment {a} (must be a power of two >= 8)"
                )))
            }
            None => DEFAULT_ALIGNMENT,
        };

        let alignment_usize =
            usize::try_from(alignment).map_err(|_| err("alignment overflow"))?;
        let data_start = r
            .pos
            .checked_add(alignment_usize - 1)
            .map(|n| n & !(alignment_usize - 1))
            .ok_or_else(|| err("data section offset overflow"))?;
        if data_start > data.len() {
            return Err(err("file truncated before tensor-data section"));
        }

        let file = GgufFile {
            header,
            metadata,
            tensors,
            alignment,
            data_start,
            data,
        };

        for info in &file.tensors {
            file.tensor_data(info)?;
        }
        Ok(file)
    }

    /// Read and parse a GGUF file from disk.
    ///
    /// # Errors
    ///
    /// Returns an error on I/O failure or malformed input.
    pub fn parse_file(path: impl AsRef<Path>) -> Result<GgufFile, TurboQuantError> {
        Self::parse_file_with_limits(path, GgufLimits::default())
    }

    /// Read a regular GGUF file under an explicit byte budget, including file growth.
    /// This does not provide path confinement or an I/O wall-clock deadline.
    pub fn parse_file_with_limits(
        path: impl AsRef<Path>,
        limits: GgufLimits,
    ) -> Result<GgufFile, TurboQuantError> {
        let path = path.as_ref();
        // Reject obvious non-regular inputs before open, then check the actual
        // descriptor as well. This is not a no-follow security boundary.
        if !std::fs::metadata(path)?.is_file() {
            return Err(err("GGUF input is not a regular file"));
        }
        let mut file = std::fs::File::open(path)?;
        let metadata = file.metadata()?;
        if !metadata.is_file() || metadata.len() > limits.max_file_bytes {
            return Err(err("GGUF file exceeds admission limits"));
        }
        let mut data = Vec::new();
        let mut chunk = [0u8; 8192];
        loop {
            // Read only the remaining budget plus one byte to detect growth.
            let remaining = limits.max_file_bytes.saturating_sub(data.len() as u64);
            let read_limit = remaining.saturating_add(1).min(chunk.len() as u64) as usize;
            let count = file.read(&mut chunk[..read_limit])?;
            if count == 0 {
                break;
            }
            if count as u64 > remaining {
                return Err(err("GGUF file exceeds file byte limit during read"));
            }
            data.try_reserve(count)
                .map_err(|_| err("GGUF raw input allocation failed"))?;
            data.extend_from_slice(&chunk[..count]);
        }
        Self::parse_with_limits(data, limits)
    }
}

struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
    limits: GgufLimits,
    remaining_allocation: usize,
}

impl Reader<'_> {
    fn reserve_vec<T>(&mut self, count: usize) -> Result<Vec<T>, TurboQuantError> {
        let bytes = count
            .checked_mul(std::mem::size_of::<T>())
            .ok_or_else(|| err("decoded allocation size overflow"))?;
        self.remaining_allocation = self
            .remaining_allocation
            .checked_sub(bytes)
            .ok_or_else(|| err("GGUF decoded allocation budget exceeded"))?;
        let mut values = Vec::new();
        values
            .try_reserve_exact(count)
            .map_err(|_| err("GGUF decoded allocation failed"))?;
        Ok(values)
    }

    fn take(&mut self, n: usize) -> Result<&[u8], TurboQuantError> {
        let end = self
            .pos
            .checked_add(n)
            .filter(|&e| e <= self.buf.len())
            .ok_or_else(|| err(format!("unexpected end of file at offset {}", self.pos)))?;
        let out = &self.buf[self.pos..end];
        self.pos = end;
        Ok(out)
    }

    fn array<const N: usize>(&mut self) -> Result<[u8; N], TurboQuantError> {
        self.take(N)?
            .try_into()
            .map_err(|_| err("internal reader width mismatch"))
    }

    fn u8(&mut self) -> Result<u8, TurboQuantError> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16, TurboQuantError> {
        Ok(u16::from_le_bytes(self.array()?))
    }

    fn u32(&mut self) -> Result<u32, TurboQuantError> {
        Ok(u32::from_le_bytes(self.array()?))
    }

    fn u64(&mut self) -> Result<u64, TurboQuantError> {
        Ok(u64::from_le_bytes(self.array()?))
    }

    fn f32(&mut self) -> Result<f32, TurboQuantError> {
        Ok(f32::from_le_bytes(self.array()?))
    }

    fn f64(&mut self) -> Result<f64, TurboQuantError> {
        Ok(f64::from_le_bytes(self.array()?))
    }

    fn string(&mut self) -> Result<String, TurboQuantError> {
        let len = self.u64()?;
        let len = usize::try_from(len).map_err(|_| err("string length overflow"))?;
        if len > self.limits.max_string_bytes || len > self.buf.len() - self.pos {
            return Err(err("GGUF string exceeds string limit or remaining input"));
        }
        let mut owned = self.reserve_vec(len)?;
        let bytes = self.take(len)?;
        owned.extend_from_slice(bytes);
        String::from_utf8(owned).map_err(|_| err("string is not valid UTF-8"))
    }

    #[allow(clippy::cast_possible_wrap)]
    fn value(&mut self, ty: GgufValueType, depth: u32) -> Result<GgufValue, TurboQuantError> {
        Ok(match ty {
            GgufValueType::U8 => GgufValue::U8(self.u8()?),
            GgufValueType::I8 => GgufValue::I8(self.u8()? as i8),
            GgufValueType::U16 => GgufValue::U16(self.u16()?),
            GgufValueType::I16 => GgufValue::I16(self.u16()? as i16),
            GgufValueType::U32 => GgufValue::U32(self.u32()?),
            GgufValueType::I32 => GgufValue::I32(self.u32()? as i32),
            GgufValueType::F32 => GgufValue::F32(self.f32()?),
            GgufValueType::Bool => GgufValue::Bool(self.u8()? != 0),
            GgufValueType::String => GgufValue::String(self.string()?),
            GgufValueType::U64 => GgufValue::U64(self.u64()?),
            GgufValueType::I64 => GgufValue::I64(self.u64()? as i64),
            GgufValueType::F64 => GgufValue::F64(self.f64()?),
            GgufValueType::Array => {
                if depth >= MAX_ARRAY_DEPTH {
                    return Err(err("metadata array nesting too deep"));
                }
                let elem_ty = GgufValueType::from_u32(self.u32()?)?;
                let raw_count = self.u64()?;
                let count = bounded_count(raw_count, self.limits.max_array_elements, "array")?;
                let minimum_width = match elem_ty {
                    GgufValueType::U8 | GgufValueType::I8 | GgufValueType::Bool => 1,
                    GgufValueType::U16 | GgufValueType::I16 => 2,
                    GgufValueType::U32 | GgufValueType::I32 | GgufValueType::F32 => 4,
                    GgufValueType::Array => 12,
                    _ => 8,
                };
                let minimum_bytes = count
                    .checked_mul(minimum_width)
                    .ok_or_else(|| err("array byte count overflow"))?;
                if minimum_bytes > self.buf.len() - self.pos {
                    return Err(err(format!("array count {count} exceeds remaining file")));
                }
                let mut values = self.reserve_vec(count)?;
                for _ in 0..count {
                    values.push(self.value(elem_ty, depth + 1)?);
                }
                GgufValue::Array(elem_ty, values)
            }
        })
    }
}
