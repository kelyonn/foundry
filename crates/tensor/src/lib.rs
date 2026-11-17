//! Hand-written tensor storage and kernels. No dependencies.
//!
//! Per docs/ARCHITECTURE.md: **row-major, contiguous, `f32`, no strides, no views,
//! no broadcasting engine.** A `Tensor` is a `Vec<f32>` plus a shape, and that is
//! the whole abstraction. The cost is extra copies on transpose; the benefit is
//! that stride bugs — the classic time sink in a hand-rolled tensor library — are
//! not representable.
//!
//! Correctness before speed (hard rule 2). Everything here is scalar. A matmul
//! that takes 90 seconds per token is a pass if the numbers are right.

pub mod ops;
pub mod pool;
pub mod quant;

/// A row-major, contiguous `f32` tensor.
#[derive(Clone, PartialEq)]
pub struct Tensor {
    data: Vec<f32>,
    shape: Vec<usize>,
}

impl Tensor {
    /// Wrap existing data. Panics if the length disagrees with the shape —
    /// per docs/RULES.md, panics on shape mismatch are acceptable in week 1–2.
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Tensor {
        let expected: usize = shape.iter().product();
        assert_eq!(
            data.len(),
            expected,
            "data length {} does not match shape {:?} (expected {})",
            data.len(),
            shape,
            expected
        );
        Tensor { data, shape }
    }

    pub fn zeros(shape: Vec<usize>) -> Tensor {
        let n = shape.iter().product();
        Tensor {
            data: vec![0.0; n],
            shape,
        }
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn data(&self) -> &[f32] {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }

    pub fn into_data(self) -> Vec<f32> {
        self.data
    }

    /// Reinterpret the shape without touching the data. Contiguity makes this free.
    pub fn reshape(mut self, shape: Vec<usize>) -> Tensor {
        let expected: usize = shape.iter().product();
        assert_eq!(self.data.len(), expected, "reshape changes element count");
        self.shape = shape;
        self
    }

    /// Row `i` of a 2-D tensor.
    pub fn row(&self, i: usize) -> &[f32] {
        assert_eq!(self.shape.len(), 2, "row() needs a 2-D tensor");
        let cols = self.shape[1];
        &self.data[i * cols..(i + 1) * cols]
    }

    /// Largest absolute difference against another tensor of the same shape.
    ///
    /// This is the parity metric used everywhere in this project — docs/TESTING.md
    /// specifies max absolute error, not mean, because a mean hides a single
    /// catastrophically wrong element among thousands of correct ones.
    pub fn max_abs_diff(&self, other: &Tensor) -> f32 {
        assert_eq!(self.shape, other.shape, "shape mismatch in comparison");
        self.data
            .iter()
            .zip(&other.data)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max)
    }
}

impl std::fmt::Debug for Tensor {
    /// Prints the shape and a few elements — never the whole tensor, which is
    /// how you accidentally print 151,936 floats into a terminal.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Tensor{:?} [", self.shape)?;
        for (i, v) in self.data.iter().take(6).enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{v:.5}")?;
        }
        if self.data.len() > 6 {
            write!(f, ", … {} more", self.data.len() - 6)?;
        }
        write!(f, "]")
    }
}
