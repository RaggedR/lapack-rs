//! The graded DSL delivers safety from the type alone.

use lapack_rs::equivariant as eq;
use lapack_rs::graded::{konst, var};
use lapack_rs::real::CReal;
use proptest::prelude::*;

#[test]
fn naive_formula_is_made_safe_by_the_grading() {
    let norm = (var(0) * var(0) + var(1) * var(1)).sqrt();
    assert_eq!(norm.degree(), 1);

    // naive overflows / underflows exactly as the bare formula would...
    assert!(norm.eval_naive(&[1e200, 1e200]).is_infinite());
    assert_eq!(norm.eval_naive(&[1e-200, 1e-200]), 0.0);

    // ...but eval_safe, driven only by the type's degree, is correct.
    let big = norm.eval_safe(&[1e200, 1e200]);
    let tiny = norm.eval_safe(&[1e-200, 1e-200]);
    assert!((big - 1e200 * 2f64.sqrt()).abs() / (1e200 * 2f64.sqrt()) < 1e-15);
    assert!((tiny - 1e-200 * 2f64.sqrt()).abs() / (1e-200 * 2f64.sqrt()) < 1e-15);

    // exact small case
    assert_eq!(norm.eval_safe(&[3.0, 4.0]), 5.0);
}

#[test]
fn givens_cosine_has_inferred_degree_zero_and_is_scale_invariant() {
    let r = (var(0) * var(0) + var(1) * var(1)).sqrt();
    let c = var(0) / r;
    assert_eq!(c.degree(), 0);

    // degree 0 ⇒ the safe value is invariant under input scaling.
    let base = c.eval_safe(&[3.0, 4.0]);
    assert!((base - 0.6).abs() < 1e-15);
    // naive would divide by an overflowed r and give 0; safe stays correct.
    assert_eq!(c.eval_naive(&[1e200, 1e200]), 0.0); // 1e200 / inf
    assert!((c.eval_safe(&[1e200, 1e200]) - 1.0 / 2f64.sqrt()).abs() < 1e-15);
}

#[test]
fn a_degree_mixing_constant_still_grades_cleanly_when_multiplied() {
    // konst is degree 0; multiplying it in is fine (2 * x*x : degree 2).
    let scaled_sq = konst(2.0) * (var(0) * var(0));
    assert_eq!(scaled_sq.degree(), 2);
    assert_eq!(scaled_sq.eval_safe(&[3.0]), 18.0);
    // NOTE: `var(0)*var(0) + konst(5.0)` is intentionally a *compile error*
    // (degree 2 + degree 0). See tests/compile_fail_notes.md.
}

#[test]
fn same_expression_over_float_and_exact_carriers() {
    // ONE expression, two carriers. Over f64 the naive path overflows and the
    // safe path rescues it; over CReal there is no overflow to rescue — the
    // float only appears at `extract`.
    let norm = (var(0).square() + var(1).square()).sqrt();
    let inputs = [1e200_f64, 1e200];
    let expected = 1e200 * 2f64.sqrt();

    // f64 carrier
    assert!(norm.eval_naive(&inputs).is_infinite()); // leaks
    assert!((norm.eval_safe(&inputs) - expected).abs() / expected < 1e-15); // repaired

    // ℝ-native carrier: (1e200)² is just a 400-digit integer; no repair needed.
    let exact = norm.eval_exact(&[CReal::from_f64(1e200), CReal::from_f64(1e200)]);
    assert!((exact.extract() - expected).abs() / expected < 1e-14);
}

fn wide_f64() -> impl Strategy<Value = f64> {
    (any::<bool>(), 1.0f64..2.0, -290i32..=290i32).prop_map(|(neg, m, e)| {
        let v = m * 2f64.powi(e);
        if neg { -v } else { v }
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(4096))]

    /// The DSL's safe norm agrees with the standalone equivariant routine.
    #[test]
    fn dsl_norm_matches_equivariant_nrm2(a in wide_f64(), b in wide_f64()) {
        let norm = (var(0) * var(0) + var(1) * var(1)).sqrt();
        let got = norm.eval_safe(&[a, b]);
        let want = eq::nrm2(&[a, b]);
        let rel = if want == 0.0 { got.abs() } else { (got - want).abs() / want.abs() };
        prop_assert!(got.is_finite());
        prop_assert!(rel <= 64.0 * f64::EPSILON, "got={got}, want={want}, rel={rel:e}");
    }
}
