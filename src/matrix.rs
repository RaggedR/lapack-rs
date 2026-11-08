//! Safe matrix operations, each annotated with its grading ρ.
//!
//! These are the rungs above the scalar routines. Two of them — [`dot`] and
//! [`matmul`] — are *bilinear*, so they are **multigraded**: each argument
//! block carries its own scaling action, and the result transports by the sum
//! of the block exponents. The other two — [`gram_schmidt`] and
//! [`jacobi_eigen`] — are not single-scale homogeneous; they are built from the
//! safe primitives, and the grading appears in their *outputs* (an orthonormal
//! Q is degree 0, eigenvalues are degree 1, eigenvectors are degree 0).
//!
//! The safety claim throughout is about overflow/underflow, not cancellation
//! (scaling cannot cure an ill-conditioned subtraction — that is a separate
//! concern, as it is in LAPACK).

use crate::equivariant::{ilogb2, ldexp, nrm2};

fn maxabs(xs: impl Iterator<Item = f64>) -> f64 {
    xs.fold(0.0_f64, |a, v| a.max(v.abs()))
}

/// The exponent that reduces a block into the fundamental domain (0 if the
/// block is all-zero or non-finite).
fn block_exp(m: f64) -> i32 {
    if m > 0.0 && m.is_finite() {
        ilogb2(m)
    } else {
        0
    }
}

/// Inner product ⟨x, y⟩ = Σ xᵢyᵢ, bilinear ⇒ **ρ = ⟨1; 1⟩**.
///
/// x and y are scaled into range *independently* (that is the multigrading),
/// every product is formed O(1), and the result is transported by `2^(eₓ+e_y)`.
/// This survives intermediate overflow from large-magnitude products that
/// cancel — where the naive loop yields `inf − inf = NaN`.
pub fn dot(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len(), "dot: length mismatch");
    let ex = block_exp(maxabs(x.iter().copied()));
    let ey = block_exp(maxabs(y.iter().copied()));
    let mut s = 0.0;
    for i in 0..x.len() {
        s += ldexp(x[i], -ex) * ldexp(y[i], -ey);
    }
    ldexp(s, ex + ey)
}

/// The naive inner product, kept as a foil (overflows/NaNs under big products).
pub fn dot_naive(x: &[f64], y: &[f64]) -> f64 {
    x.iter().zip(y).map(|(a, b)| a * b).sum()
}

/// A dense row-major matrix.
#[derive(Clone, Debug)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

impl Matrix {
    pub fn zeros(rows: usize, cols: usize) -> Matrix {
        Matrix {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    pub fn from_rows(rows: &[Vec<f64>]) -> Matrix {
        let r = rows.len();
        let c = rows.first().map_or(0, |row| row.len());
        let mut m = Matrix::zeros(r, c);
        for (i, row) in rows.iter().enumerate() {
            assert_eq!(row.len(), c, "from_rows: ragged input");
            for (j, &v) in row.iter().enumerate() {
                m.set(i, j, v);
            }
        }
        m
    }

    pub fn identity(n: usize) -> Matrix {
        let mut m = Matrix::zeros(n, n);
        for i in 0..n {
            m.set(i, i, 1.0);
        }
        m
    }

    #[inline]
    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.data[i * self.cols + j]
    }

    #[inline]
    pub fn set(&mut self, i: usize, j: usize, v: f64) {
        self.data[i * self.cols + j] = v;
    }

    pub fn transpose(&self) -> Matrix {
        let mut t = Matrix::zeros(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                t.set(j, i, self.get(i, j));
            }
        }
        t
    }

    pub fn maxabs(&self) -> f64 {
        maxabs(self.data.iter().copied())
    }

    pub fn col(&self, j: usize) -> Vec<f64> {
        (0..self.rows).map(|i| self.get(i, j)).collect()
    }

    /// Row `i` as a contiguous slice (row-major storage).
    pub fn row(&self, i: usize) -> &[f64] {
        &self.data[i * self.cols..(i + 1) * self.cols]
    }
}

/// Matrix product A·B, bilinear ⇒ **ρ = ⟨1; 1⟩**. A and B are each reduced into
/// range independently; every scalar product is O(1); entries transport by
/// `2^(e_A+e_B)`.
pub fn matmul(a: &Matrix, b: &Matrix) -> Matrix {
    assert_eq!(a.cols, b.rows, "matmul: shape mismatch");
    let ea = block_exp(a.maxabs());
    let eb = block_exp(b.maxabs());
    let mut c = Matrix::zeros(a.rows, b.cols);
    for i in 0..a.rows {
        for j in 0..b.cols {
            let mut s = 0.0;
            for k in 0..a.cols {
                s += ldexp(a.get(i, k), -ea) * ldexp(b.get(k, j), -eb);
            }
            c.set(i, j, ldexp(s, ea + eb));
        }
    }
    c
}

/// Result of a (thin) QR via Gram–Schmidt: `q` columns are orthonormal, `r` is
/// upper-triangular, and `q[j]·r = v[j]`.
pub struct QR {
    pub q: Vec<Vec<f64>>,
    pub r: Matrix,
}

/// Modified Gram–Schmidt on column vectors `v`.
///
/// Built from the safe [`dot`] and [`nrm2`], so it never overflows even when
/// the inputs are astronomically scaled. Grading of the *outputs*: Q is
/// **degree 0** — orthonormal vectors are scale-invariant (scaling every input
/// by 2ᵏ leaves Q unchanged) — while R is **degree 1**. Modified (not classical)
/// GS: projections are removed sequentially, which is the more stable order.
pub fn gram_schmidt(v: &[Vec<f64>]) -> QR {
    let n = v.len();
    let mut q: Vec<Vec<f64>> = Vec::with_capacity(n);
    let mut r = Matrix::zeros(n, n);
    for j in 0..n {
        let mut w = v[j].clone();
        for i in 0..j {
            let rij = dot(&q[i], &w); // safe inner product
            r.set(i, j, rij);
            for t in 0..w.len() {
                w[t] -= rij * q[i][t];
            }
        }
        let rjj = nrm2(&w); // safe norm
        r.set(j, j, rjj);
        if rjj > 0.0 {
            for t in 0..w.len() {
                w[t] /= rjj;
            }
        }
        q.push(w);
    }
    QR { q, r }
}

/// Symmetric eigendecomposition by the cyclic-pivot Jacobi rotation method.
///
/// Returns `(eigenvalues, eigenvectors)` with the eigenvectors as the columns
/// of the returned matrix, so that `A = V · diag(λ) · Vᵀ`. The input is first
/// reduced into range by one power-of-two scale; eigenvalues are therefore
/// **degree 1** (A ↦ 2ᵏA scales every λ by 2ᵏ) and eigenvectors **degree 0**
/// (unchanged by scaling). Requires a symmetric `a`.
pub fn jacobi_eigen(a: &Matrix) -> (Vec<f64>, Matrix) {
    assert_eq!(a.rows, a.cols, "jacobi_eigen: not square");
    let n = a.rows;

    // Reduce into range: work on 2^(-e)·A, then scale eigenvalues back by 2^e.
    let e = block_exp(a.maxabs());
    let mut m = a.clone();
    for x in m.data.iter_mut() {
        *x = ldexp(*x, -e);
    }
    let mut v = Matrix::identity(n);

    let off = |m: &Matrix| -> f64 {
        let mut s = 0.0;
        for i in 0..n {
            for j in (i + 1)..n {
                s += m.get(i, j) * m.get(i, j);
            }
        }
        s.sqrt()
    };

    let tol = 1e-15 * (n as f64);
    for _sweep in 0..100 {
        if off(&m) <= tol {
            break;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = m.get(p, q);
                if apq == 0.0 {
                    continue;
                }
                // rotation that zeros (p,q)
                let theta = (m.get(q, q) - m.get(p, p)) / (2.0 * apq);
                let t = if theta == 0.0 {
                    1.0
                } else {
                    theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt())
                };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;

                // A ← Jᵀ A J : rotate columns p,q then rows p,q
                for i in 0..n {
                    let aip = m.get(i, p);
                    let aiq = m.get(i, q);
                    m.set(i, p, c * aip - s * aiq);
                    m.set(i, q, s * aip + c * aiq);
                }
                for i in 0..n {
                    let api = m.get(p, i);
                    let aqi = m.get(q, i);
                    m.set(p, i, c * api - s * aqi);
                    m.set(q, i, s * api + c * aqi);
                }
                // V ← V J
                for i in 0..n {
                    let vip = v.get(i, p);
                    let viq = v.get(i, q);
                    v.set(i, p, c * vip - s * viq);
                    v.set(i, q, s * vip + c * viq);
                }
            }
        }
    }

    let eig = (0..n).map(|i| ldexp(m.get(i, i), e)).collect();
    (eig, v)
}
