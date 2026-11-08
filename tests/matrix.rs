//! The four matrix operations: correctness, grading, and overflow survival.

use lapack_rs::matrix::{dot, dot_naive, gram_schmidt, jacobi_eigen, matmul, Matrix};

fn rel(got: f64, want: f64) -> f64 {
    if want == 0.0 {
        got.abs()
    } else {
        (got - want).abs() / want.abs()
    }
}

// --- inner product (ρ = ⟨1;1⟩) ---------------------------------------------

#[test]
fn dot_is_correct_and_bilinear() {
    assert_eq!(dot(&[3.0, 4.0], &[1.0, 2.0]), 11.0);
    // bilinear: ⟨2x, 3y⟩ = 6⟨x,y⟩
    let x = [1.5, -2.0, 0.5];
    let y = [4.0, 1.0, -3.0];
    let base = dot(&x, &y);
    let sx: Vec<f64> = x.iter().map(|v| 2.0 * v).collect();
    let sy: Vec<f64> = y.iter().map(|v| 3.0 * v).collect();
    assert!(rel(dot(&sx, &sy), 6.0 * base) < 1e-14);
}

#[test]
fn dot_survives_intermediate_overflow_under_cancellation() {
    // products are ±1e309 (overflow), but they cancel to 0.
    let x = [1e200, 1e200];
    let y = [1e109, -1e109];
    assert!(dot_naive(&x, &y).is_nan(), "naive should be inf-inf = NaN");
    assert_eq!(dot(&x, &y), 0.0); // bigraded scaling keeps each product O(1)
}

// --- matrix multiply (ρ = ⟨1;1⟩) -------------------------------------------

#[test]
fn matmul_identity_and_values() {
    let a = Matrix::from_rows(&[vec![1.0, 2.0], vec![3.0, 4.0]]);
    let id = Matrix::identity(2);
    let ai = matmul(&a, &id);
    for i in 0..2 {
        for j in 0..2 {
            assert_eq!(ai.get(i, j), a.get(i, j));
        }
    }
    // [[1,2],[3,4]] · [[5,6],[7,8]] = [[19,22],[43,50]]
    let b = Matrix::from_rows(&[vec![5.0, 6.0], vec![7.0, 8.0]]);
    let c = matmul(&a, &b);
    assert_eq!((c.get(0, 0), c.get(0, 1), c.get(1, 0), c.get(1, 1)), (19.0, 22.0, 43.0, 50.0));
}

#[test]
fn matmul_scales_bilinearly() {
    // (2A)(4B) = 8 AB, even with A huge and B tiny (independent block scales).
    let a = Matrix::from_rows(&[vec![1e200, 2e200], vec![3e200, 4e200]]);
    let b = Matrix::from_rows(&[vec![1e-200, 0.0], vec![0.0, 1e-200]]);
    let c = matmul(&a, &b); // ≈ [[1,2],[3,4]]
    assert!(rel(c.get(0, 0), 1.0) < 1e-12);
    assert!(rel(c.get(1, 1), 4.0) < 1e-12);
    assert!(c.data.iter().all(|v| v.is_finite()));
}

// --- Gram–Schmidt (Q degree 0, R degree 1) ----------------------------------

fn check_orthonormal(q: &[Vec<f64>]) {
    for i in 0..q.len() {
        for j in 0..q.len() {
            let want = if i == j { 1.0 } else { 0.0 };
            assert!((dot(&q[i], &q[j]) - want).abs() < 1e-12, "q{i}·q{j} ≠ {want}");
        }
    }
}

#[test]
fn gram_schmidt_produces_orthonormal_basis() {
    let v = vec![vec![1.0, 1.0, 0.0], vec![1.0, 0.0, 1.0], vec![0.0, 1.0, 1.0]];
    let qr = gram_schmidt(&v);
    check_orthonormal(&qr.q);
    // R is upper triangular
    for i in 0..3 {
        for j in 0..i {
            assert_eq!(qr.r.get(i, j), 0.0);
        }
    }
}

#[test]
fn gram_schmidt_q_is_degree_zero_and_overflow_safe() {
    let v = vec![vec![1.0, 1.0, 0.0], vec![1.0, 0.0, 1.0], vec![0.0, 1.0, 1.0]];
    let base = gram_schmidt(&v);

    // Scaling every input by 2^k leaves Q bit-identical (degree 0) and scales
    // the diagonal of R by exactly 2^k (degree 1).
    for k in [-600_i32, 40, 600] {
        let scale = 2f64.powi(k.clamp(-1000, 1000));
        let vs: Vec<Vec<f64>> = v.iter().map(|col| col.iter().map(|x| x * scale).collect()).collect();
        let qr = gram_schmidt(&vs);
        check_orthonormal(&qr.q); // still orthonormal at extreme scale (overflow-safe)
        for j in 0..3 {
            for t in 0..3 {
                assert_eq!(qr.q[j][t], base.q[j][t], "Q changed under scaling (k={k})");
            }
            assert!(rel(qr.r.get(j, j), base.r.get(j, j) * scale) < 1e-12);
        }
    }
}

// --- diagonalization (eigenvalues degree 1, eigenvectors degree 0) ----------

fn reconstruct(eig: &[f64], v: &Matrix) -> Matrix {
    // V · diag(eig) · Vᵀ
    let n = eig.len();
    let mut d = Matrix::zeros(n, n);
    for i in 0..n {
        d.set(i, i, eig[i]);
    }
    matmul(&matmul(v, &d), &v.transpose())
}

fn frob_diff(a: &Matrix, b: &Matrix) -> f64 {
    a.data.iter().zip(&b.data).map(|(x, y)| (x - y) * (x - y)).sum::<f64>().sqrt()
}

/// Max-absolute relative difference — safe at any scale (no squaring, so no
/// overflow even when entries are ~1e181).
fn max_rel_diff(a: &Matrix, b: &Matrix) -> f64 {
    let m = a.data.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));
    let d = a.data.iter().zip(&b.data).fold(0.0_f64, |acc, (x, y)| acc.max((x - y).abs()));
    if m == 0.0 { d } else { d / m }
}

#[test]
fn jacobi_diagonalizes_a_symmetric_matrix() {
    let a = Matrix::from_rows(&[
        vec![4.0, 1.0, 2.0],
        vec![1.0, 3.0, 0.0],
        vec![2.0, 0.0, 5.0],
    ]);
    let (eig, v) = jacobi_eigen(&a);
    // A ≈ V Λ Vᵀ
    assert!(frob_diff(&reconstruct(&eig, &v), &a) < 1e-10);
    // V is orthogonal: VᵀV ≈ I
    let vtv = matmul(&v.transpose(), &v);
    assert!(frob_diff(&vtv, &Matrix::identity(3)) < 1e-10);
    // eigenvalues sum to the trace (4+3+5 = 12)
    assert!(rel(eig.iter().sum::<f64>(), 12.0) < 1e-12);
}

#[test]
fn jacobi_eigenvalues_are_degree_one_and_overflow_safe() {
    let a = Matrix::from_rows(&[
        vec![4.0, 1.0, 2.0],
        vec![1.0, 3.0, 0.0],
        vec![2.0, 0.0, 5.0],
    ]);
    let (base_eig, _) = jacobi_eigen(&a);
    let mut base_sorted = base_eig.clone();
    base_sorted.sort_by(|x, y| x.partial_cmp(y).unwrap());

    // Scale A by 2^600 — naive diagonalization would overflow forming A².
    let scale = 2f64.powi(600);
    let mut a_big = a.clone();
    for x in a_big.data.iter_mut() {
        *x *= scale;
    }
    let (big_eig, big_v) = jacobi_eigen(&a_big);
    assert!(big_eig.iter().all(|v| v.is_finite()));

    let mut big_sorted = big_eig.clone();
    big_sorted.sort_by(|x, y| x.partial_cmp(y).unwrap());
    for (b, s) in big_sorted.iter().zip(&base_sorted) {
        assert!(rel(*b, s * scale) < 1e-12, "eigenvalue not degree 1");
    }
    // reconstruction still holds at the huge scale (measured with a safe norm)
    assert!(max_rel_diff(&reconstruct(&big_eig, &big_v), &a_big) < 1e-10);
}
