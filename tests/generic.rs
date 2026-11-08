//! The carrier-generic operations, run over BOTH f64 and CReal from one body.

use lapack_rs::generic;
use lapack_rs::matrix::{self, Matrix};
use lapack_rs::real::CReal;

fn rel(got: f64, want: f64) -> f64 {
    if want == 0.0 {
        got.abs()
    } else {
        (got - want).abs() / want.abs()
    }
}

// --- conversions for the CReal instantiations -------------------------------

fn to_r(v: &[f64]) -> Vec<CReal> {
    v.iter().map(|&x| CReal::from_f64(x)).collect()
}

// --- f64 instantiation agrees with the hand-written matrix oracle -----------

#[test]
fn generic_f64_matches_matrix_oracle() {
    let x = [1.5, -2.0, 0.5, 7.0];
    let y = [4.0, 1.0, -3.0, 2.0];
    assert!(rel(generic::dot(&x, &y), matrix::dot(&x, &y)) <= 1e-14);

    // matmul
    let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
    let b = vec![vec![5.0, 6.0], vec![7.0, 8.0]];
    let c = generic::matmul(&a, &b);
    let oracle = matrix::matmul(&Matrix::from_rows(&a), &Matrix::from_rows(&b));
    for i in 0..2 {
        for j in 0..2 {
            assert!(rel(c[i][j], oracle.get(i, j)) <= 1e-14);
        }
    }

    // gram-schmidt: orthonormal Q
    let v = vec![vec![1.0, 1.0, 0.0], vec![1.0, 0.0, 1.0], vec![0.0, 1.0, 1.0]];
    let (q, _r) = generic::gram_schmidt(&v);
    for i in 0..3 {
        for j in 0..3 {
            let want = if i == j { 1.0 } else { 0.0 };
            assert!((generic::dot(&q[i], &q[j]) - want).abs() < 1e-12);
        }
    }

    // jacobi (f64, normal magnitude) matches the matrix oracle's eigenvalues
    let sym = vec![vec![4.0, 1.0, 2.0], vec![1.0, 3.0, 0.0], vec![2.0, 0.0, 5.0]];
    let (mut e1, _) = generic::jacobi_eigen(&sym);
    let (mut e2, _) = matrix::jacobi_eigen(&Matrix::from_rows(&sym));
    e1.sort_by(|a, b| a.partial_cmp(b).unwrap());
    e2.sort_by(|a, b| a.partial_cmp(b).unwrap());
    for (a, b) in e1.iter().zip(&e2) {
        assert!(rel(*a, *b) < 1e-10);
    }
}

// --- the SAME code over CReal: exact, and cancellation-proof ----------------

#[test]
fn generic_creal_dot_is_exact_under_cancellation() {
    // ⟨[1,1,1], [1e16, 1, -1e16]⟩ = 1e16 + 1 − 1e16 = 1 exactly; every value is
    // representable. But the `1` sits below the ulp of 1e16, so the f64 sum
    // loses it — the classic big + small + (−big) cancellation.
    let x = [1.0, 1.0, 1.0];
    let y = [1e16, 1.0, -1e16];
    let f = generic::dot(&x, &y); // f64 (scaled, but scaling is by powers of 2 — still cancels)
    let r = generic::dot(&to_r(&x), &to_r(&y)).extract(); // CReal, exact
    assert!(rel(r, 1.0) < 1e-12, "CReal should give 1, got {r}");
    assert!((f - 1.0).abs() > 0.5, "f64 should lose the 1, got {f}");
}

// NOTE: `generic::gram_schmidt` and `generic::jacobi_eigen` are carrier-generic
// and run correctly over CReal, but impractically slowly (minutes for a 3×3),
// because exact-real division re-probes at escalating precision through every
// nested reciprocal. The exact carrier is practical for *shallow* expressions
// (the dot cancellation test above, and the scalar √(x+1)−√(x) demo in
// real.rs), not for division-heavy compositional algorithms. See generic.rs.
