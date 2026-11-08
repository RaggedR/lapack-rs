//! The multigraded DSL: `dot` and `matmul` written as naive formulas, made
//! safe by the bidegree, verified against the hand-written `matrix` oracles.

use lapack_rs::matrix::{self, Matrix};
use lapack_rs::tensor::{self, sum, xvar, yvar};
use proptest::prelude::*;

fn rel(got: f64, want: f64) -> f64 {
    if want == 0.0 {
        got.abs()
    } else {
        (got - want).abs() / want.abs()
    }
}

#[test]
fn dsl_dot_matches_handwritten_and_is_correct() {
    assert_eq!(tensor::dot(&[3.0, 4.0], &[1.0, 2.0]), 11.0);
    let x = [1.5, -2.0, 0.5, 7.0];
    let y = [4.0, 1.0, -3.0, 2.0];
    assert!(rel(tensor::dot(&x, &y), matrix::dot(&x, &y)) <= 1e-14);
}

#[test]
fn dsl_dot_survives_intermediate_overflow() {
    // ±1e309 products that cancel — naive is NaN, the DSL expression is 0.
    let x = [1e200, 1e200];
    let y = [1e109, -1e109];
    let expr = sum(xvar() * yvar());
    assert!(expr.eval_naive(&x, &y).is_nan());
    assert_eq!(tensor::dot(&x, &y), 0.0);
}

#[test]
fn dsl_dot_is_bilinear() {
    let x = [1.0, 2.0, 3.0];
    let y = [4.0, 5.0, 6.0];
    let base = tensor::dot(&x, &y);
    let sx: Vec<f64> = x.iter().map(|v| 8.0 * v).collect();
    let sy: Vec<f64> = y.iter().map(|v| 0.25 * v).collect();
    // ρ = ⟨1;1⟩ ⇒ scaling x by 8 and y by 1/4 scales the result by 2.
    assert!(rel(tensor::dot(&sx, &sy), 2.0 * base) < 1e-14);
}

#[test]
fn dsl_matmul_matches_handwritten() {
    let a = Matrix::from_rows(&[vec![1.0, 2.0], vec![3.0, 4.0]]);
    let b = Matrix::from_rows(&[vec![5.0, 6.0], vec![7.0, 8.0]]);
    let c = tensor::matmul(&a, &b);
    let oracle = matrix::matmul(&a, &b);
    for k in 0..c.data.len() {
        assert!(rel(c.data[k], oracle.data[k]) <= 1e-14);
    }
    assert_eq!((c.get(0, 0), c.get(1, 1)), (19.0, 50.0));
}

#[test]
fn dsl_matmul_is_block_bilinear_and_overflow_safe() {
    // A huge, B tiny — each block scales independently; result finite & correct.
    let a = Matrix::from_rows(&[vec![1e200, 2e200], vec![3e200, 4e200]]);
    let b = Matrix::from_rows(&[vec![1e-200, 0.0], vec![0.0, 1e-200]]);
    let c = tensor::matmul(&a, &b); // ≈ [[1,2],[3,4]]
    assert!(c.data.iter().all(|v| v.is_finite()));
    assert!(rel(c.get(0, 0), 1.0) < 1e-12);
    assert!(rel(c.get(1, 1), 4.0) < 1e-12);
}

#[test]
fn degree_is_inferred_as_one_one() {
    assert_eq!(sum(xvar() * yvar()).degree(), (1, 1));
    // NOTE: `xvar() + yvar()` is intentionally a COMPILE error — adding a
    // ⟨1,0⟩ vector to a ⟨0,1⟩ vector mixes blocks. See tests/compile_fail_notes.md.
}

fn wide() -> impl Strategy<Value = f64> {
    (any::<bool>(), 1.0f64..2.0, -200i32..=200i32).prop_map(|(neg, m, e)| {
        let v = m * 2f64.powi(e);
        if neg { -v } else { v }
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(2048))]

    /// The DSL dot and the hand-written dot agree across wide-dynamic-range input.
    #[test]
    fn dsl_dot_matches_oracle(x in prop::collection::vec(wide(), 1..32),
                              y in prop::collection::vec(wide(), 1..32)) {
        let n = x.len().min(y.len());
        let (x, y) = (&x[..n], &y[..n]);
        let got = tensor::dot(x, y);
        let want = matrix::dot(x, y);
        prop_assert!(got.is_finite() == want.is_finite());
        if got.is_finite() {
            prop_assert!(rel(got, want) <= 1e-12, "got={got}, want={want}");
        }
    }
}
