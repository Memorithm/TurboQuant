#![deny(missing_docs)]
#![cfg_attr(not(test), warn(clippy::unwrap_used, clippy::expect_used))]
// Clippy allows for practical numerical code
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::items_after_statements)]
#![allow(clippy::new_without_default)]

//! CPU backend for `TurboQuant`.
//!
//! Provides a CPU implementation using rayon for parallelism.
//! (No explicit SIMD path yet; scalar code is autovectorized by LLVM.)

use ndarray::{Array2, ArrayView2};
use rayon::prelude::*;
use turboquant_core::kv_block::KvBlock;
use turboquant_core::polar::PolarQuant;
use turboquant_core::qjl::{QjlConfig, QjlQuantizer};
use turboquant_core::rotation::QrRotation;

/// CPU backend supporting all rotation and quantization operations.
#[allow(dead_code)]
pub struct CpuBackend;

impl Default for CpuBackend {
    fn default() -> Self {
        Self
    }
}

impl CpuBackend {
    /// Create a new CPU backend.
    ///
    /// # Examples
    ///
    /// ```
    /// use turboquant_cpu::CpuBackend;
    ///
    /// let backend = CpuBackend::new();
    /// ```
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Benchmark compression throughput on CPU.
    ///
    /// Returns throughput in GB/s.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn bench_compression(
        &self,
        head_dim: usize,
        num_heads: usize,
        num_iterations: usize,
    ) -> f64 {
        let rot = QrRotation::new(head_dim, Some(42));
        let polar = PolarQuant::new(rot);

        let config = QjlConfig::default();
        let quantizer = QjlQuantizer::new(config);

        let tensor = Array2::from_shape_fn((num_heads, head_dim), |_| {
            (rand::random::<f32>() - 0.5) * 2.0
        });

        let start = std::time::Instant::now();
        for _ in 0..num_iterations {
            let _ = turboquant_core::quantize::compress_tensor(&polar, &quantizer, &tensor.view());
        }
        let elapsed = start.elapsed();

        let total_bytes = num_heads * head_dim * 4 * num_iterations;
        let gb = total_bytes as f64 / 1e9;
        gb / elapsed.as_secs_f64()
    }

    /// Parallel compress multiple KV blocks.
    ///
    /// # Errors
    ///
    /// Returns an error if any compression fails.
    pub fn parallel_compress_kv(
        &self,
        k_tensors: &[ArrayView2<f32>],
        v_tensors: &[ArrayView2<f32>],
        rot: &QrRotation,
        quantizer: &QjlQuantizer,
    ) -> turboquant_core::Result<(Vec<KvBlock>, Vec<KvBlock>)> {
        let polar = PolarQuant::new(rot.clone());

        let k_blocks: Vec<KvBlock> = k_tensors
            .par_iter()
            .map(|k| turboquant_core::quantize::compress_tensor(&polar, quantizer, k))
            .collect::<turboquant_core::Result<Vec<_>>>()?;

        let v_blocks: Vec<KvBlock> = v_tensors
            .par_iter()
            .map(|v| turboquant_core::quantize::compress_tensor(&polar, quantizer, v))
            .collect::<turboquant_core::Result<Vec<_>>>()?;

        Ok((k_blocks, v_blocks))
    }

    /// Run a full compression-benchmark pipeline and return stats.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn full_benchmark(
        &self,
        head_dim: usize,
        seq_len: usize,
        num_heads: usize,
    ) -> BenchmarkResults {
        let rot = QrRotation::new(head_dim, Some(42));

        let config = QjlConfig::default();
        let quantizer = QjlQuantizer::new(config);

        let start = std::time::Instant::now();

        let total_values = num_heads * head_dim;
        let mut k_blocks = Vec::with_capacity(num_heads);
        let mut v_blocks = Vec::with_capacity(num_heads);

        for _ in 0..num_heads {
            let k_tensor =
                Array2::from_shape_fn((1, head_dim), |_| (rand::random::<f32>() - 0.5) * 2.0);
            let v_tensor =
                Array2::from_shape_fn((1, head_dim), |_| (rand::random::<f32>() - 0.5) * 2.0);
            let polar = PolarQuant::new(rot.clone());

            if let Ok(block) =
                turboquant_core::quantize::compress_tensor(&polar, &quantizer, &k_tensor.view())
            {
                k_blocks.push(block);
            }
            if let Ok(block) =
                turboquant_core::quantize::compress_tensor(&polar, &quantizer, &v_tensor.view())
            {
                v_blocks.push(block);
            }
        }

        let elapsed = start.elapsed();
        let throughput_gbps = (total_values * 4) as f64 / 1e9 / elapsed.as_secs_f64();

        let k_mem: usize = k_blocks
            .iter()
            .map(turboquant_core::kv_block::KvBlock::memory_usage_bytes)
            .sum();
        let v_mem: usize = v_blocks
            .iter()
            .map(turboquant_core::kv_block::KvBlock::memory_usage_bytes)
            .sum();
        let fp16_mem = total_values * 2 * 2;

        BenchmarkResults {
            compression_throughput_gbps: throughput_gbps,
            compressed_k_mb: k_mem as f64 / 1e6,
            compressed_v_mb: v_mem as f64 / 1e6,
            fp16_total_mb: fp16_mem as f64 / 1e6,
            compression_ratio: fp16_mem as f64 / (k_mem + v_mem) as f64,
            head_dim,
            seq_len,
            num_heads,
            elapsed_ms: elapsed.as_millis() as f64,
        }
    }
}

/// Results from a full benchmark run.
#[derive(Debug, Clone)]
pub struct BenchmarkResults {
    /// Compression throughput in GB/s.
    pub compression_throughput_gbps: f64,
    /// Compressed K cache size in MB.
    pub compressed_k_mb: f64,
    /// Compressed V cache size in MB.
    pub compressed_v_mb: f64,
    /// FP16 total size in MB.
    pub fp16_total_mb: f64,
    /// Actual compression ratio (including overhead).
    pub compression_ratio: f64,
    /// Head dimension.
    pub head_dim: usize,
    /// Sequence length.
    pub seq_len: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Elapsed time in ms.
    pub elapsed_ms: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_creation() {
        let backend = CpuBackend::new();
        let results = backend.full_benchmark(64, 1, 4);
        assert!(results.compression_ratio > 3.0);
        assert!(results.elapsed_ms > 0.0);
    }

    #[test]
    fn test_compression_benchmark() {
        let backend = CpuBackend::new();
        let throughput = backend.bench_compression(64, 8, 100);
        assert!(throughput > 0.0);
    }

    #[test]
    fn test_parallel_compress() {
        let rot = QrRotation::new(64, Some(42));
        let config = QjlConfig::default();
        let quantizer = QjlQuantizer::new(config);

        let k_tensors: Vec<Array2<f32>> = (0..4)
            .map(|_| Array2::from_shape_fn((1, 64), |_| (rand::random::<f32>() - 0.5) * 2.0))
            .collect();

        let vs: Vec<Array2<f32>> = k_tensors.clone();
        let k_views: Vec<ArrayView2<f32>> =
            k_tensors.iter().map(ndarray::ArrayBase::view).collect();
        let v_views: Vec<ArrayView2<f32>> = vs.iter().map(ndarray::ArrayBase::view).collect();

        let backend = CpuBackend::new();
        let (k_blocks, v_blocks) = backend
            .parallel_compress_kv(&k_views, &v_views, &rot, &quantizer)
            .unwrap();

        assert_eq!(k_blocks.len(), 4);
        assert_eq!(v_blocks.len(), 4);
    }
}
