//! Bounded malformed-input regressions for audit RT-TQ-01.

use turboquant_gguf::{GgmlType, GgufLimits, GgufParser, GgufValue, GgufValueType, GgufWriter};

fn header(tensors: u64, metadata: u64) -> Vec<u8> {
    let mut bytes = b"GGUF".to_vec();
    bytes.extend_from_slice(&3u32.to_le_bytes());
    bytes.extend_from_slice(&tensors.to_le_bytes());
    bytes.extend_from_slice(&metadata.to_le_bytes());
    bytes
}

fn metadata_file() -> Vec<u8> {
    let mut writer = GgufWriter::new();
    writer.add_metadata("one", GgufValue::U8(1));
    writer.add_metadata("two", GgufValue::String("value".into()));
    writer.to_bytes().unwrap()
}

#[test]
fn maximum_header_counts_are_rejected_without_panicking() {
    for (tensors, metadata) in [(u64::MAX, 0), (0, u64::MAX), (u64::MAX, u64::MAX)] {
        let result = std::panic::catch_unwind(|| GgufParser::parse(header(tensors, metadata)));
        assert!(result.expect("malformed counts must not panic").is_err());
    }
}

#[test]
fn structural_counts_are_checked_even_with_permissive_policy() {
    let limits = GgufLimits {
        max_tensors: usize::MAX,
        max_metadata_entries: usize::MAX,
        ..GgufLimits::default()
    };
    for (tensors, metadata) in [(1, 0), (0, 1), (u64::MAX, u64::MAX)] {
        assert!(GgufParser::parse_with_limits(header(tensors, metadata), limits).is_err());
    }
}

#[test]
fn combined_structural_minimum_is_checked_before_reserving() {
    // Each count alone could fit in 24 remaining bytes; together they cannot.
    let mut bytes = header(1, 1);
    bytes.resize(48, 0);
    assert!(GgufParser::parse(bytes).is_err());
}

#[test]
fn metadata_count_policy_is_enforced_on_valid_input() {
    assert!(GgufParser::parse_with_limits(
        metadata_file(),
        GgufLimits {
            max_metadata_entries: 1,
            ..GgufLimits::default()
        }
    )
    .is_err());
}

#[test]
fn tensor_count_policy_is_enforced_on_valid_input() {
    let mut writer = GgufWriter::new();
    writer
        .add_tensor("tensor", vec![1], GgmlType::F32, 1f32.to_le_bytes().to_vec())
        .unwrap();
    assert!(GgufParser::parse_with_limits(
        writer.to_bytes().unwrap(),
        GgufLimits {
            max_tensors: 0,
            ..GgufLimits::default()
        }
    )
    .is_err());
}

#[test]
fn decoded_allocation_budget_is_cumulative() {
    let bytes = metadata_file();
    let vector_bytes = 2 * std::mem::size_of::<(String, GgufValue)>();
    // The vector fits, then string storage exhausts the remaining budget.
    assert!(GgufParser::parse_with_limits(
        bytes,
        GgufLimits {
            max_decoded_bytes: vector_bytes,
            ..GgufLimits::default()
        }
    )
    .is_err());
}

#[test]
fn string_budget_rejects_before_copying() {
    assert!(GgufParser::parse_with_limits(
        metadata_file(),
        GgufLimits {
            max_string_bytes: 4,
            ..GgufLimits::default()
        }
    )
    .is_err());
}

#[test]
fn array_count_policy_is_enforced() {
    let mut writer = GgufWriter::new();
    writer.add_metadata(
        "a",
        GgufValue::Array(GgufValueType::U8, vec![GgufValue::U8(1), GgufValue::U8(2)]),
    );
    assert!(GgufParser::parse_with_limits(
        writer.to_bytes().unwrap(),
        GgufLimits {
            max_array_elements: 1,
            ..GgufLimits::default()
        }
    )
    .is_err());
}

#[test]
fn array_minimum_wire_width_is_checked() {
    let mut bytes = header(0, 1);
    bytes.extend_from_slice(&0u64.to_le_bytes()); // Empty key.
    bytes.extend_from_slice(&9u32.to_le_bytes()); // Array.
    bytes.extend_from_slice(&11u32.to_le_bytes()); // u64 elements.
    bytes.extend_from_slice(&8u64.to_le_bytes()); // Eight elements need 64 bytes.
    bytes.extend_from_slice(&[0; 8]); // Count <= bytes, but wire width cannot fit.
    assert!(GgufParser::parse(bytes).is_err());
}

#[test]
fn maximum_array_count_is_rejected_without_panicking() {
    let mut bytes = header(0, 1);
    bytes.extend_from_slice(&0u64.to_le_bytes());
    bytes.extend_from_slice(&9u32.to_le_bytes());
    bytes.extend_from_slice(&0u32.to_le_bytes());
    bytes.extend_from_slice(&u64::MAX.to_le_bytes());
    assert!(std::panic::catch_unwind(|| GgufParser::parse(bytes))
        .expect("array count must not panic")
        .is_err());
}

#[test]
fn dimension_product_overflow_is_an_error() {
    let mut bytes = header(1, 0);
    bytes.extend_from_slice(&0u64.to_le_bytes());
    bytes.extend_from_slice(&2u32.to_le_bytes());
    bytes.extend_from_slice(&u64::MAX.to_le_bytes());
    bytes.extend_from_slice(&2u64.to_le_bytes());
    bytes.extend_from_slice(&0u32.to_le_bytes()); // F32.
    bytes.extend_from_slice(&0u64.to_le_bytes());
    assert!(std::panic::catch_unwind(|| GgufParser::parse(bytes))
        .expect("dimensions must not panic")
        .is_err());
}

#[test]
fn raw_byte_budget_has_an_exact_boundary() {
    let bytes = metadata_file();
    let limit = bytes.len() as u64;
    let limits = GgufLimits {
        max_file_bytes: limit,
        ..GgufLimits::default()
    };
    assert!(GgufParser::parse_with_limits(bytes.clone(), limits).is_ok());
    assert!(GgufParser::parse_with_limits(
        bytes,
        GgufLimits {
            max_file_bytes: limit - 1,
            ..limits
        }
    )
    .is_err());
}

#[test]
fn file_reader_enforces_byte_budget_and_rejects_directories() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("tiny.gguf");
    let bytes = metadata_file();
    std::fs::write(&path, &bytes).unwrap();
    assert!(GgufParser::parse_file_with_limits(
        &path,
        GgufLimits {
            max_file_bytes: bytes.len() as u64 - 1,
            ..GgufLimits::default()
        }
    )
    .is_err());
    assert!(GgufParser::parse_file(&path).is_ok());
    assert!(GgufParser::parse_file(dir.path()).is_err());
}
