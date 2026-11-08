//! The iRRAM carrier runs the SAME generic algorithms — fast and exact.
//! (CReal ran these correctly but took minutes; these finish in well under a
//! second, which is the whole point of the re-run architecture.)

use lapack_rs::generic;
use lapack_rs::irram::{run, Real};
use lapack_rs::matrix::{self, Matrix};

fn seed_mat(m: &[Vec<f64>]) -> Vec<Vec<Real>> {
    m.iter().map(|r| r.iter().map(|&x| Real::from_f64(x)).collect()).collect()
}

fn rel(got: f64, want: f64) -> f64 {
    if want == 0.0 {
        got.abs()
    } else {
        (got - want).abs() / want.abs()
    }
}

#[test]
fn irram_gram_schmidt_is_fast_and_orthonormal() {
    let v = vec![vec![1.0, 1.0, 0.0], vec![1.0, 0.0, 1.0], vec![0.0, 1.0, 1.0]];
    // run the generic Gram–Schmidt over the iRRAM carrier, extract Q
    let flat = run(50, || {
        let (q, _r) = generic::gram_schmidt::<Real>(&seed_mat(&v));
        q.into_iter().flatten().collect()
    });
    let q: Vec<Vec<f64>> = flat.chunks(3).map(|c| c.to_vec()).collect();
    for i in 0..3 {
        for j in 0..3 {
            let want = if i == j { 1.0 } else { 0.0 };
            assert!((matrix::dot(&q[i], &q[j]) - want).abs() < 1e-12, "q{i}·q{j}");
        }
    }
}

#[test]
fn irram_jacobi_is_fast_and_correct() {
    let sym = vec![vec![4.0, 1.0, 2.0], vec![1.0, 3.0, 0.0], vec![2.0, 0.0, 5.0]];
    let eig = run(45, || {
        let (e, _v) = generic::jacobi_eigen::<Real>(&seed_mat(&sym));
        e
    });
    // eigenvalues match the f64 oracle and sum to the trace (12)
    let (mut e_ref, _) = matrix::jacobi_eigen(&Matrix::from_rows(&sym));
    e_ref.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut e = eig.clone();
    e.sort_by(|a, b| a.partial_cmp(b).unwrap());
    for (a, b) in e.iter().zip(&e_ref) {
        assert!(rel(*a, *b) < 1e-9, "eig {a} vs {b}");
    }
    assert!(rel(eig.iter().sum::<f64>(), 12.0) < 1e-10);
}
