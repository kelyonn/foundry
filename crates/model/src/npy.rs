//! A minimal `.npy` reader, so the parity harness can consume the reference dumps.
//!
//! Format: a 6-byte magic `\x93NUMPY`, a version pair, a header length, then a
//! Python dict literal describing dtype/order/shape, then the raw data. We only
//! ever read arrays this project's own script wrote, so the parser handles
//! little-endian `f4`/`f8`/`i8` in C order and rejects everything else rather than
//! pretending to be general.

use tensor::Tensor;

#[derive(Debug)]
pub struct NpyError(pub String);

impl std::fmt::Display for NpyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "npy error: {}", self.0)
    }
}

impl std::error::Error for NpyError {}

fn err<T>(msg: impl Into<String>) -> Result<T, NpyError> {
    Err(NpyError(msg.into()))
}

/// Read a `.npy` file as `f32`, widening `f8` (float64) if needed.
pub fn read_f32(path: impl AsRef<std::path::Path>) -> Result<Tensor, NpyError> {
    let raw = std::fs::read(path.as_ref())
        .map_err(|e| NpyError(format!("cannot read {}: {e}", path.as_ref().display())))?;

    if raw.len() < 10 || &raw[0..6] != b"\x93NUMPY" {
        return err("not a .npy file");
    }
    let major = raw[6];
    // v1 uses a 2-byte header length, v2+ uses 4.
    let (header_len, header_start) = if major == 1 {
        (u16::from_le_bytes([raw[8], raw[9]]) as usize, 10)
    } else {
        (
            u32::from_le_bytes([raw[8], raw[9], raw[10], raw[11]]) as usize,
            12,
        )
    };
    let header_end = header_start + header_len;
    if header_end > raw.len() {
        return err("header runs past end of file");
    }
    let header = std::str::from_utf8(&raw[header_start..header_end])
        .map_err(|_| NpyError("header is not UTF-8".into()))?;

    if header.contains("'fortran_order': True") {
        return err("Fortran-order arrays are not supported");
    }

    let descr = extract(header, "'descr': '").ok_or_else(|| NpyError("no descr".into()))?;
    let shape_str = extract(header, "'shape': (").ok_or_else(|| NpyError("no shape".into()))?;

    let shape: Vec<usize> = shape_str
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| {
            s.parse::<usize>()
                .map_err(|_| NpyError(format!("bad dim {s:?}")))
        })
        .collect::<Result<_, _>>()?;
    // A 0-d array has an empty shape tuple; treat it as a single element.
    let shape = if shape.is_empty() { vec![1] } else { shape };
    let n: usize = shape.iter().product();

    let body = &raw[header_end..];
    let data = match descr {
        "<f4" | "=f4" => {
            if body.len() < n * 4 {
                return err("file too short for its shape");
            }
            body[..n * 4]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect()
        }
        "<f8" | "=f8" => {
            if body.len() < n * 8 {
                return err("file too short for its shape");
            }
            body[..n * 8]
                .chunks_exact(8)
                .map(|c| f64::from_le_bytes(c.try_into().unwrap()) as f32)
                .collect()
        }
        other => return err(format!("unsupported dtype {other:?}")),
    };

    Ok(Tensor::new(data, shape))
}

/// Read a `.npy` of int64 token ids.
pub fn read_u32(path: impl AsRef<std::path::Path>) -> Result<Vec<u32>, NpyError> {
    let raw = std::fs::read(path.as_ref())
        .map_err(|e| NpyError(format!("cannot read {}: {e}", path.as_ref().display())))?;
    if raw.len() < 10 || &raw[0..6] != b"\x93NUMPY" {
        return err("not a .npy file");
    }
    let major = raw[6];
    let (header_len, header_start) = if major == 1 {
        (u16::from_le_bytes([raw[8], raw[9]]) as usize, 10)
    } else {
        (
            u32::from_le_bytes([raw[8], raw[9], raw[10], raw[11]]) as usize,
            12,
        )
    };
    let header_end = header_start + header_len;
    let header = std::str::from_utf8(&raw[header_start..header_end])
        .map_err(|_| NpyError("header is not UTF-8".into()))?;
    let descr = extract(header, "'descr': '").ok_or_else(|| NpyError("no descr".into()))?;
    let body = &raw[header_end..];

    match descr {
        "<i8" | "=i8" => Ok(body
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes(c.try_into().unwrap()) as u32)
            .collect()),
        "<i4" | "=i4" => Ok(body
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes(c.try_into().unwrap()) as u32)
            .collect()),
        other => err(format!("unsupported int dtype {other:?}")),
    }
}

fn extract<'a>(header: &'a str, key: &str) -> Option<&'a str> {
    let start = header.find(key)? + key.len();
    let rest = &header[start..];
    let end = if key.ends_with('(') {
        rest.find(')')?
    } else {
        rest.find('\'')?
    };
    Some(&rest[..end])
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_npy(path: &std::path::Path, header: &str, body: &[u8]) {
        let mut f = std::fs::File::create(path).unwrap();
        f.write_all(b"\x93NUMPY\x01\x00").unwrap();
        // numpy pads the header to a 64-byte boundary; the length field is what
        // actually matters to us.
        f.write_all(&(header.len() as u16).to_le_bytes()).unwrap();
        f.write_all(header.as_bytes()).unwrap();
        f.write_all(body).unwrap();
    }

    #[test]
    fn reads_a_2d_f32_array() {
        let mut p = std::env::temp_dir();
        p.push("foundry_npy_test.npy");
        let values: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let body: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        write_npy(
            &p,
            "{'descr': '<f4', 'fortran_order': False, 'shape': (2, 3), }",
            &body,
        );
        let t = read_f32(&p).unwrap();
        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.data(), values.as_slice());
        std::fs::remove_file(&p).ok();
    }

    #[test]
    fn rejects_fortran_order() {
        let mut p = std::env::temp_dir();
        p.push("foundry_npy_fortran.npy");
        write_npy(
            &p,
            "{'descr': '<f4', 'fortran_order': True, 'shape': (2, 2), }",
            &[0u8; 16],
        );
        assert!(read_f32(&p).is_err());
        std::fs::remove_file(&p).ok();
    }

    #[test]
    fn handles_a_1d_shape_with_trailing_comma() {
        let mut p = std::env::temp_dir();
        p.push("foundry_npy_1d.npy");
        let body: Vec<u8> = [1.0f32, 2.0].iter().flat_map(|v| v.to_le_bytes()).collect();
        write_npy(
            &p,
            "{'descr': '<f4', 'fortran_order': False, 'shape': (2,), }",
            &body,
        );
        let t = read_f32(&p).unwrap();
        assert_eq!(t.shape(), &[2]);
        std::fs::remove_file(&p).ok();
    }
}
