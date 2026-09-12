//! GGUF file parser (versions 2 and 3).

use crate::types::{
    GgmlType, GgufHeader, GgufTensorInfo, GgufValue, GgufValueType, DEFAULT_ALIGNMENT, GGUF_MAGIC,
};
use std::path::Path;
use turboquant_core::error::TurboQuantError;

/// Maximum nesting depth for metadata arrays (defensive limit).
const MAX_ARRAY_DEPTH: u32 = 8;

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
        let size = if let Some(s) = info.data_size() {
            usize::try_from(s).map_err(|_| err("tensor size overflow"))?
        } else {
            // Span until the next tensor's offset, or end of file.
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
        let header = parse_header(&data)?;
        let mut r = Reader {
            buf: &data,
            pos: 24,
        };

        let mut metadata =
            Vec::with_capacity(usize::try_from(header.metadata_kv_count).unwrap_or(0));
        for _ in 0..header.metadata_kv_count {
            let key = r.string()?;
            let ty = GgufValueType::from_u32(r.u32()?)?;
            let value = r.value(ty, 0)?;
            metadata.push((key, value));
        }

        let mut tensors = Vec::with_capacity(usize::try_from(header.tensor_count).unwrap_or(0));
        for _ in 0..header.tensor_count {
            let name = r.string()?;
            let n_dims = r.u32()?;
            if n_dims > 8 {
                return Err(err(format!("tensor '{name}' has {n_dims} dims (max 8)")));
            }
            let mut dims = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                dims.push(r.u64()?);
            }
            let ggml_type = GgmlType::from_u32(r.u32()?)?;
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

        let data_start = usize::try_from((r.pos as u64).next_multiple_of(alignment))
            .map_err(|_| err("data section offset overflow"))?;
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

        // Validate that every tensor's data lies within the file.
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
        let data = std::fs::read(path)?;
        Self::parse(data)
    }
}

/// Little-endian cursor over a byte slice.
struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl Reader<'_> {
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
        let bytes = self.take(len)?;
        String::from_utf8(bytes.to_vec()).map_err(|_| err("string is not valid UTF-8"))
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
                let count = self.u64()?;
                let count = usize::try_from(count).map_err(|_| err("array count overflow"))?;
                // Every element occupies at least one byte on the wire.
                if count > self.buf.len() - self.pos {
                    return Err(err(format!("array count {count} exceeds remaining file")));
                }
                let mut values = Vec::with_capacity(count);
                for _ in 0..count {
                    values.push(self.value(elem_ty, depth + 1)?);
                }
                GgufValue::Array(elem_ty, values)
            }
        })
    }
}
