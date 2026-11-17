//! The safetensors container format.
//!
//! ```text
//! ┌────────────────┬─────────────────────────┬──────────────────────────┐
//! │ 8 bytes, u64 LE│ JSON header, N bytes     │ raw tensor data          │
//! │ header length N│ name → {dtype, shape,    │ concatenated, offsets    │
//! │                │        data_offsets}     │ relative to this point   │
//! └────────────────┴─────────────────────────┴──────────────────────────┘
//! ```
//!
//! That is the entire format. It is deliberately boring — which is the point of it,
//! versus a pickle, which is arbitrary code execution wearing a data format's hat.
//!
//! The file is memory-mapped rather than read. For a ~1 GB checkpoint that means the
//! header parse touches a few KB instead of faulting in the whole file, and the OS
//! page cache decides what stays resident. Individual tensors are materialised as
//! `f32` on demand.

use memmap2::Mmap;
use std::collections::BTreeMap;
use std::fs::File;
use std::path::Path;
use tensor::Tensor;

#[derive(Debug)]
pub enum LoadError {
    Io(std::io::Error),
    Format(String),
    MissingTensor(String),
}

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoadError::Io(e) => write!(f, "io error: {e}"),
            LoadError::Format(m) => write!(f, "malformed safetensors file: {m}"),
            LoadError::MissingTensor(n) => write!(f, "no tensor named {n:?} in the file"),
        }
    }
}

impl std::error::Error for LoadError {}

impl From<std::io::Error> for LoadError {
    fn from(e: std::io::Error) -> Self {
        LoadError::Io(e)
    }
}

/// The element types we can read. The checkpoint for Qwen2.5-0.5B-Instruct is
/// `BF16`; `F32` and `F16` are here because they cost four lines each and their
/// absence would be discovered at the least convenient moment.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Dtype {
    F32,
    F16,
    BF16,
}

impl Dtype {
    fn parse(s: &str) -> Option<Dtype> {
        match s {
            "F32" => Some(Dtype::F32),
            "F16" => Some(Dtype::F16),
            "BF16" => Some(Dtype::BF16),
            _ => None,
        }
    }

    pub fn size_bytes(self) -> usize {
        match self {
            Dtype::F32 => 4,
            Dtype::F16 | Dtype::BF16 => 2,
        }
    }
}

#[derive(Clone, Debug)]
pub struct TensorInfo {
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    /// Byte range within the data blob, i.e. *after* the header.
    pub start: usize,
    pub end: usize,
}

impl TensorInfo {
    pub fn elements(&self) -> usize {
        self.shape.iter().product()
    }
}

pub struct SafeTensors {
    mmap: Mmap,
    data_start: usize,
    tensors: BTreeMap<String, TensorInfo>,
}

impl SafeTensors {
    pub fn open(path: impl AsRef<Path>) -> Result<SafeTensors, LoadError> {
        let file = File::open(path.as_ref())?;
        // SAFETY: we map the file read-only and never hand out a reference that
        // outlives `self`. The invariant we are trusting is that no other process
        // truncates the checkpoint while we hold it open — if that happens the
        // process gets SIGBUS. For a local read-only weight file this is the
        // standard tradeoff every mmap-based loader makes.
        let mmap = unsafe { Mmap::map(&file)? };

        if mmap.len() < 8 {
            return Err(LoadError::Format(
                "file shorter than the 8-byte length prefix".into(),
            ));
        }
        let header_len = u64::from_le_bytes(mmap[0..8].try_into().unwrap()) as usize;

        let header_end = 8usize
            .checked_add(header_len)
            .ok_or_else(|| LoadError::Format("header length overflows".into()))?;
        if header_end > mmap.len() {
            return Err(LoadError::Format(format!(
                "header claims {header_len} bytes but the file is only {} bytes",
                mmap.len()
            )));
        }

        let header: serde_json::Value = serde_json::from_slice(&mmap[8..header_end])
            .map_err(|e| LoadError::Format(format!("header is not valid JSON: {e}")))?;
        let obj = header
            .as_object()
            .ok_or_else(|| LoadError::Format("header is not a JSON object".into()))?;

        let blob_len = mmap.len() - header_end;
        let mut tensors = BTreeMap::new();

        for (name, spec) in obj {
            // `__metadata__` holds free-form strings (framework, format version).
            // It is not a tensor and has no data_offsets.
            if name == "__metadata__" {
                continue;
            }
            let spec = spec
                .as_object()
                .ok_or_else(|| LoadError::Format(format!("{name}: entry is not an object")))?;

            let dtype_str = spec
                .get("dtype")
                .and_then(|v| v.as_str())
                .ok_or_else(|| LoadError::Format(format!("{name}: missing dtype")))?;
            let dtype = Dtype::parse(dtype_str).ok_or_else(|| {
                LoadError::Format(format!("{name}: unsupported dtype {dtype_str}"))
            })?;

            let shape: Vec<usize> = spec
                .get("shape")
                .and_then(|v| v.as_array())
                .ok_or_else(|| LoadError::Format(format!("{name}: missing shape")))?
                .iter()
                .map(|v| v.as_u64().map(|n| n as usize))
                .collect::<Option<Vec<_>>>()
                .ok_or_else(|| LoadError::Format(format!("{name}: shape is not integral")))?;

            let offsets = spec
                .get("data_offsets")
                .and_then(|v| v.as_array())
                .ok_or_else(|| LoadError::Format(format!("{name}: missing data_offsets")))?;
            if offsets.len() != 2 {
                return Err(LoadError::Format(format!(
                    "{name}: data_offsets is not a pair"
                )));
            }
            let start = offsets[0].as_u64().unwrap_or(u64::MAX) as usize;
            let end = offsets[1].as_u64().unwrap_or(0) as usize;

            // Validate rather than trust. A corrupt or truncated download shows up
            // here as a clear message instead of an out-of-bounds slice later.
            if end < start || end > blob_len {
                return Err(LoadError::Format(format!(
                    "{name}: data_offsets [{start}, {end}] outside the {blob_len}-byte blob"
                )));
            }
            let expected: usize = shape.iter().product::<usize>() * dtype.size_bytes();
            if end - start != expected {
                return Err(LoadError::Format(format!(
                    "{name}: shape {shape:?} of {dtype:?} needs {expected} bytes, \
                     but data_offsets span {}",
                    end - start
                )));
            }

            tensors.insert(
                name.clone(),
                TensorInfo {
                    dtype,
                    shape,
                    start,
                    end,
                },
            );
        }

        Ok(SafeTensors {
            mmap,
            data_start: header_end,
            tensors,
        })
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.tensors.keys().map(|s| s.as_str())
    }

    pub fn info(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.get(name)
    }

    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// Total elements across every tensor — i.e. the parameter count.
    pub fn total_parameters(&self) -> usize {
        self.tensors.values().map(|t| t.elements()).sum()
    }

    /// Materialise a tensor as `f32`, converting from whatever it was stored as.
    ///
    /// Per docs/ARCHITECTURE.md everything is `f32` in v1, so the conversion happens
    /// once at load rather than in the kernels. For this checkpoint that turns ~1 GB
    /// of `bf16` on disk into ~2 GB of `f32` in memory, which is the price of not
    /// having a dtype-generic kernel for every operation.
    pub fn tensor(&self, name: &str) -> Result<Tensor, LoadError> {
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| LoadError::MissingTensor(name.to_string()))?;

        let bytes = &self.mmap[self.data_start + info.start..self.data_start + info.end];
        let n = info.elements();
        let mut out = Vec::with_capacity(n);

        match info.dtype {
            Dtype::F32 => {
                for c in bytes.chunks_exact(4) {
                    out.push(f32::from_le_bytes(c.try_into().unwrap()));
                }
            }
            // bf16 is simply the top 16 bits of an f32 — same 8 exponent bits, 7
            // mantissa bits instead of 23. So widening is a shift, with no
            // exponent rebiasing and no special cases. That is the entire reason
            // bf16 exists: it trades mantissa precision for a free conversion and
            // the same dynamic range as f32.
            Dtype::BF16 => {
                for c in bytes.chunks_exact(2) {
                    let bits = u16::from_le_bytes(c.try_into().unwrap());
                    out.push(f32::from_bits((bits as u32) << 16));
                }
            }
            // f16 is a different layout entirely — 5 exponent bits, 10 mantissa —
            // so it needs real rebiasing rather than a shift.
            Dtype::F16 => {
                for c in bytes.chunks_exact(2) {
                    let bits = u16::from_le_bytes(c.try_into().unwrap());
                    out.push(f16_to_f32(bits));
                }
            }
        }

        Ok(Tensor::new(out, info.shape.clone()))
    }
}

/// IEEE 754 binary16 → binary32.
fn f16_to_f32(bits: u16) -> f32 {
    let sign = (bits as u32 & 0x8000) << 16;
    let exp = (bits >> 10) & 0x1f;
    let frac = bits as u32 & 0x3ff;

    let out = match exp {
        // Zero or subnormal.
        0 => {
            if frac == 0 {
                sign
            } else {
                // Renormalise: shift the fraction left until the implicit leading
                // bit reaches position 10, then rebias.
                //
                // A subnormal f16 is `frac × 2^-24`. After `k` shifts the value is
                // `1.xxx × 2^(-14-k)`, so the f32 exponent field is
                // `(-14 - k) + 127 = 113 - k`. Starting `e` at 1 and decrementing
                // once per shift makes `127 - 15 + e` come out at exactly that.
                let mut e = 1i32;
                let mut f = frac;
                while f & 0x400 == 0 {
                    f <<= 1;
                    e -= 1;
                }
                let exp32 = (127 - 15 + e) as u32;
                sign | (exp32 << 23) | ((f & 0x3ff) << 13)
            }
        }
        // Inf or NaN.
        0x1f => sign | 0x7f80_0000 | (frac << 13),
        // Normal: rebias the exponent from 15 to 127.
        _ => sign | ((exp as u32 + 127 - 15) << 23) | (frac << 13),
    };
    f32::from_bits(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// Build a safetensors file in memory. Note this fixture is written from the *format
    /// spec*, not from our parser, so it is an independent oracle rather than a
    /// circular one.
    fn write_fixture(path: &std::path::Path, entries: &[(&str, Dtype, Vec<usize>, Vec<u8>)]) {
        let mut header = serde_json::Map::new();
        let mut blob: Vec<u8> = Vec::new();
        for (name, dtype, shape, data) in entries {
            let start = blob.len();
            blob.extend_from_slice(data);
            let end = blob.len();
            let dtype_str = match dtype {
                Dtype::F32 => "F32",
                Dtype::F16 => "F16",
                Dtype::BF16 => "BF16",
            };
            header.insert(
                name.to_string(),
                serde_json::json!({
                    "dtype": dtype_str,
                    "shape": shape,
                    "data_offsets": [start, end],
                }),
            );
        }
        let header_bytes = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
        let mut f = std::fs::File::create(path).unwrap();
        f.write_all(&(header_bytes.len() as u64).to_le_bytes())
            .unwrap();
        f.write_all(&header_bytes).unwrap();
        f.write_all(&blob).unwrap();
    }

    fn tmp(name: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("foundry_st_test_{name}.safetensors"));
        p
    }

    #[test]
    fn reads_f32_tensors_and_shapes() {
        let path = tmp("f32");
        let values: Vec<f32> = vec![1.0, -2.5, 3.25, 4.0, 0.0, -1.0];
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        write_fixture(&path, &[("w", Dtype::F32, vec![2, 3], bytes)]);

        let st = SafeTensors::open(&path).unwrap();
        assert_eq!(st.len(), 1);
        assert_eq!(st.info("w").unwrap().shape, vec![2, 3]);
        assert_eq!(st.total_parameters(), 6);

        let t = st.tensor("w").unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.data(), values.as_slice());
        std::fs::remove_file(&path).ok();
    }

    /// bf16 widening must be exact for any value that bf16 can represent, because
    /// every bf16 is exactly representable in f32.
    #[test]
    fn bf16_widening_is_exact() {
        let path = tmp("bf16");
        // Values chosen to be exactly representable in bf16 (7 mantissa bits).
        let values: Vec<f32> = vec![1.0, -2.0, 0.5, 3.5, 0.0, -0.015625];
        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
            .collect();
        write_fixture(&path, &[("w", Dtype::BF16, vec![6], bytes)]);

        let st = SafeTensors::open(&path).unwrap();
        let t = st.tensor("w").unwrap();
        assert_eq!(t.data(), values.as_slice());
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn f16_conversion_handles_normals_zero_and_subnormals() {
        // 0x3C00 = 1.0, 0xC000 = -2.0, 0x0000 = 0.0, 0x0001 = smallest subnormal
        assert_eq!(f16_to_f32(0x3C00), 1.0);
        assert_eq!(f16_to_f32(0xC000), -2.0);
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert_eq!(f16_to_f32(0x8000), -0.0);
        // Smallest positive subnormal is exactly 2^-24, and the largest is
        // 1023 × 2^-24. Both must be exact — f32 can represent every f16.
        assert_eq!(f16_to_f32(0x0001), 2f32.powi(-24));
        assert_eq!(f16_to_f32(0x03ff), 1023.0 * 2f32.powi(-24));
        // Largest subnormal and smallest normal must be adjacent.
        assert_eq!(f16_to_f32(0x0400), 1024.0 * 2f32.powi(-24));
        assert!(f16_to_f32(0x7C00).is_infinite());
        assert!(f16_to_f32(0x7E00).is_nan());
    }

    #[test]
    fn multiple_tensors_get_distinct_slices() {
        let path = tmp("multi");
        let a: Vec<u8> = [1.0f32, 2.0].iter().flat_map(|v| v.to_le_bytes()).collect();
        let b: Vec<u8> = [9.0f32, 8.0, 7.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        write_fixture(
            &path,
            &[
                ("first", Dtype::F32, vec![2], a),
                ("second", Dtype::F32, vec![3], b),
            ],
        );
        let st = SafeTensors::open(&path).unwrap();
        assert_eq!(st.tensor("first").unwrap().data(), &[1.0, 2.0]);
        assert_eq!(st.tensor("second").unwrap().data(), &[9.0, 8.0, 7.0]);
        assert!(matches!(
            st.tensor("absent"),
            Err(LoadError::MissingTensor(_))
        ));
        std::fs::remove_file(&path).ok();
    }

    /// A shape that disagrees with the byte span is corruption, and must be caught
    /// at open time rather than producing a silently wrong tensor.
    #[test]
    fn rejects_offsets_inconsistent_with_the_shape() {
        let path = tmp("bad");
        let bytes: Vec<u8> = vec![0u8; 8]; // 2 f32s
        write_fixture(&path, &[("w", Dtype::F32, vec![99], bytes)]);
        assert!(matches!(
            SafeTensors::open(&path),
            Err(LoadError::Format(_))
        ));
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn rejects_a_truncated_file() {
        let path = tmp("trunc");
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(&(9999u64).to_le_bytes()).unwrap();
        f.write_all(b"{}").unwrap();
        drop(f);
        assert!(matches!(
            SafeTensors::open(&path),
            Err(LoadError::Format(_))
        ));
        std::fs::remove_file(&path).ok();
    }
}
