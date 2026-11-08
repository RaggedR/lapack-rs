//! Carrier-generic linear algebra: **one** implementation per operation, run
//! over any [`Scalar`] carrier. Instantiate `::<f64>` for the fast,
//! magnitude-safe path, or `::<CReal>` for the slow, fully-exact
//! (cancellation-proof) path. The source is identical for both — only the type
//! parameter changes at the call site.
//!
//! The safety per carrier lives in `T::dot` / `T::nrm2` (f64 overrides them with
//! safe scaling; CReal uses the exact naive defaults, since exact reals never
//! overflow). The algorithm bodies below therefore read like the textbook
//! definitions. The only visible cost of genericity is `.clone()`: a general
//! `Scalar` is not `Copy` (CReal holds an `Rc`). Structurally nothing else
//! changes between the f64 and the exact version.

use crate::Scalar;

/// Inner product — dispatches to the carrier's safe `dot`.
pub fn dot<T: Scalar>(x: &[T], y: &[T]) -> T {
    T::dot(x, y)
}

/// Matrix product A·B (row-major `Vec<Vec<T>>`): each entry is a `dot`.
pub fn matmul<T: Scalar>(a: &[Vec<T>], b: &[Vec<T>]) -> Vec<Vec<T>> {
    let (m, k, n) = (a.len(), b.len(), b[0].len());
    let cols: Vec<Vec<T>> = (0..n)
        .map(|j| (0..k).map(|i| b[i][j].clone()).collect())
        .collect();
    (0..m)
        .map(|i| (0..n).map(|j| T::dot(&a[i], &cols[j])).collect())
        .collect()
}

/// Modified Gram–Schmidt. Returns `(Q, R)` with orthonormal `Q` columns.
/// Assumes the inputs are linearly independent (every norm nonzero) — exact
/// zero is undecidable over `CReal`, so we neither test nor branch on it.
pub fn gram_schmidt<T: Scalar>(v: &[Vec<T>]) -> (Vec<Vec<T>>, Vec<Vec<T>>) {
    let n = v.len();
    let mut q: Vec<Vec<T>> = Vec::with_capacity(n);
    let mut r = vec![vec![T::zero(); n]; n];
    for j in 0..n {
        let mut w = v[j].clone();
        for i in 0..j {
            let rij = T::dot(&q[i], &w); // safe inner product
            for t in 0..w.len() {
                w[t] = w[t].clone() - rij.clone() * q[i][t].clone(); // remove projection
            }
            r[i][j] = rij;
        }
        let rjj = T::nrm2(&w); // safe norm
        for t in 0..w.len() {
            w[t] = w[t].clone() / rjj.clone(); // normalise
        }
        r[j][j] = rjj;
        q.push(w);
    }
    (q, r)
}

fn identity<T: Scalar>(n: usize) -> Vec<Vec<T>> {
    (0..n)
        .map(|i| (0..n).map(|j| if i == j { T::from_f64(1.0) } else { T::zero() }).collect())
        .collect()
}

/// Symmetric eigendecomposition by cyclic Jacobi rotations. Returns
/// `(eigenvalues, eigenvectors-as-columns)` with `A = V·diag(λ)·Vᵀ`.
///
/// Carrier notes: over `f64` this omits the power-of-two pre-scale of
/// `matrix::jacobi_eigen` (which is f64-specific), so use it for normal-range
/// inputs; the `matrix` version remains the magnitude-safe f64 path. Over
/// `CReal` it is exact, but each rotation deepens the lazy expression trees, so
/// it is practical only for small matrices (a proper exact solver would force
/// each entry to a bounded precision between sweeps).
pub fn jacobi_eigen<T: Scalar>(a: &[Vec<T>]) -> (Vec<T>, Vec<Vec<T>>) {
    let n = a.len();
    let mut m: Vec<Vec<T>> = a.to_vec();
    let mut v = identity::<T>(n);
    let one = T::from_f64(1.0);
    let two = T::from_f64(2.0);
    let tol = T::from_f64(1e-15);

    for _ in 0..100 {
        let mut off = T::zero();
        for i in 0..n {
            for j in (i + 1)..n {
                off = off + m[i][j].clone() * m[i][j].clone();
            }
        }
        if off.sqrt() <= tol {
            break;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = m[p][q].clone();
                if apq.abs() <= tol {
                    continue;
                }
                // rotation that zeros (p,q)
                let theta = (m[q][q].clone() - m[p][p].clone()) / (two.clone() * apq.clone());
                let denom = theta.abs() + (theta.clone() * theta.clone() + one.clone()).sqrt();
                let t = if theta > T::zero() {
                    one.clone() / denom
                } else if theta < T::zero() {
                    -(one.clone() / denom)
                } else {
                    one.clone()
                };
                let c = one.clone() / (t.clone() * t.clone() + one.clone()).sqrt();
                let s = t.clone() * c.clone();

                for i in 0..n {
                    let (aip, aiq) = (m[i][p].clone(), m[i][q].clone());
                    m[i][p] = c.clone() * aip.clone() - s.clone() * aiq.clone();
                    m[i][q] = s.clone() * aip + c.clone() * aiq;
                }
                for i in 0..n {
                    let (api, aqi) = (m[p][i].clone(), m[q][i].clone());
                    m[p][i] = c.clone() * api.clone() - s.clone() * aqi.clone();
                    m[q][i] = s.clone() * api + c.clone() * aqi;
                }
                for i in 0..n {
                    let (vip, viq) = (v[i][p].clone(), v[i][q].clone());
                    v[i][p] = c.clone() * vip.clone() - s.clone() * viq.clone();
                    v[i][q] = s.clone() * vip + c.clone() * viq;
                }
            }
        }
    }
    let eig = (0..n).map(|i| m[i][i].clone()).collect();
    (eig, v)
}
