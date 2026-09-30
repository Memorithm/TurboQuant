# GGUF admission limits

Audit RT-TQ-01: header and array counts must fit both the remaining encoded
bytes and the configured policy before a vector reserve. All decoded vector
and string storage is charged to one cumulative requested-allocation budget;
reserves are fallible. Tensor extents and alignment use checked arithmetic.

`GgufParser::parse` and `parse_file` use `GgufLimits::default()`: 64 GiB raw
input, 1,000,000 metadata entries, 100,000 tensors, 1,000,000 elements per array,
16 MiB per string and 256 MiB cumulative decoded allocations. Applications can
choose stricter or larger limits through `parse_with_limits` and
`parse_file_with_limits`. This is an admission policy, not a change to the
GGUF wire format or a claim about real-model quality.

The file reader checks the regular-file size before reading, uses a byte quota
while reading to cover growth, and uses fallible buffer growth. It does not
provide no-follow path confinement, an I/O deadline or a process RSS limit.
The raw buffer is separate from the decoded budget, and allocator overhead is
not counted. Applications handling hostile paths still need their own bounded
and confined file-opening policy.

`tests/parser_limits.rs` contains tiny malformed-input cases and exact-budget
boundaries. Existing round-trip tests remain the compatibility oracle. These
fixtures can inform the import-budget consumers identified by audit RT-X-02;
no downstream consumer is claimed to have adopted this policy yet.
