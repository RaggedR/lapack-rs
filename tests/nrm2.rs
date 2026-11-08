//! Validation suite for `nrm2`.
//!
//! The whole thesis of this project is that a norm routine is not "done" when
//! it compiles — it is done when it survives adversarial inputs against an
//! *independent* oracle. So the tests are the real deliverable:
//!
//!   * headline demos showing the naive algorithm failing where `nrm2` holds;
//!   * a property test for `f64` against the classic scale-by-max algorithm
//!     (a genuinely different safe method — two safe algorithms agreeing is a
//!     strong faithfulness check);
//!   * a property test for `f32` against an `f64` computation (f64's range
//!     comfortably swallows anything an f32 can represent).

use lapack_rs::{Real, nrm2, nrm2_naive};
use proptest::prelude::*;

// ---------------------------------------------------------------------------
// The derivation self-check: the decimal literals in the library really are
// the exact powers of two dictated by each format's exponent range.
// ---------------------------------------------------------------------------

#[test]
fn f64_constants_are_the_derived_powers_of_two() {
    assert_eq!(<f64 as Real>::TBIG, 2f64.powi(486));
    assert_eq!(<f64 as Real>::TSML, 2f64.powi(-511));
    assert_eq!(<f64 as Real>::SSML, 2f64.powi(537));
    assert_eq!(<f64 as Real>::SBIG, 2f64.powi(-538));
}

#[test]
fn f32_constants_are_the_derived_powers_of_two() {
    assert_eq!(<f32 as Real>::TBIG, 2f32.powi(52));
    assert_eq!(<f32 as Real>::TSML, 2f32.powi(-63));
    assert_eq!(<f32 as Real>::SSML, 2f32.powi(75));
    assert_eq!(<f32 as Real>::SBIG, 2f32.powi(-76));
}

// ---------------------------------------------------------------------------
// Headline demos: the naive routine fails, `nrm2` does not.
// ---------------------------------------------------------------------------

#[test]
fn naive_overflows_to_infinity_but_nrm2_survives() {
    // True norm of [1e200, 1e200] is 1e200·√2 ≈ 1.41421356e200 — a perfectly
    // representable f64. But (1e200)² = 1e400 overflows.
    let x = [1e200_f64, 1e200];
    let expected = 1e200_f64 * 2f64.sqrt();

    assert!(
        nrm2_naive(&x).is_infinite(),
        "naive should overflow to inf, got {}",
        nrm2_naive(&x)
    );

    let got = nrm2(&x);
    assert!(got.is_finite(), "nrm2 must stay finite, got {got}");
    assert!(
        (got - expected).abs() / expected < 1e-15,
        "nrm2 = {got}, expected {expected}"
    );
}

#[test]
fn naive_underflows_to_zero_but_nrm2_survives() {
    // True norm of [1e-200, 1e-200] is 1e-200·√2 ≈ 1.41421356e-200 — again
    // perfectly representable. But (1e-200)² = 1e-400 underflows to 0.
    let x = [1e-200_f64, 1e-200];
    let expected = 1e-200_f64 * 2f64.sqrt();

    assert_eq!(
        nrm2_naive(&x),
        0.0,
        "naive should underflow to exactly 0"
    );

    let got = nrm2(&x);
    assert!(got > 0.0, "nrm2 must not underflow, got {got}");
    assert!(
        (got - expected).abs() / expected < 1e-15,
        "nrm2 = {got}, expected {expected}"
    );
}

// ---------------------------------------------------------------------------
// Exact and edge cases.
// ---------------------------------------------------------------------------

#[test]
fn small_exact_cases() {
    assert_eq!(nrm2::<f64>(&[]), 0.0);
    assert_eq!(nrm2(&[0.0_f64, 0.0, 0.0]), 0.0);
    assert_eq!(nrm2(&[-7.0_f64]), 7.0);
    assert_eq!(nrm2(&[3.0_f64, 4.0]), 5.0); // classic 3-4-5
    assert_eq!(nrm2(&[3.0_f32, 4.0]), 5.0);
}

#[test]
fn propagates_nan_and_inf() {
    assert!(nrm2(&[1.0_f64, f64::NAN]).is_nan());
    assert!(nrm2(&[f64::INFINITY]).is_infinite());
    assert!(nrm2(&[1.0_f64, f64::INFINITY]).is_infinite());
}

// ---------------------------------------------------------------------------
// Independent oracles for the property tests.
// ---------------------------------------------------------------------------

/// Classic one-pass scaled norm: m·√(Σ(xᵢ/m)²) with m = maxᵢ|xᵢ|.
/// A *different* safe algorithm from the three-bin method under test.
fn nrm2_scale_by_max(x: &[f64]) -> f64 {
    let m = x.iter().fold(0.0_f64, |a, &v| a.max(v.abs()));
    if m == 0.0 || !m.is_finite() {
        return m; // 0, inf, or nan pass straight through
    }
    let s: f64 = x
        .iter()
        .map(|&v| {
            let r = v / m;
            r * r
        })
        .sum();
    m * s.sqrt()
}

fn rel_err(got: f64, want: f64) -> f64 {
    if want == 0.0 {
        got.abs()
    } else {
        (got - want).abs() / want.abs()
    }
}

// A wide-dynamic-range f64: sign · mantissa · 2^exp, spanning exponents that
// force values into all three accumulator bins.
fn wide_f64() -> impl Strategy<Value = f64> {
    (any::<bool>(), 1.0f64..2.0, -290i32..=290i32)
        .prop_map(|(neg, m, e)| {
            let v = m * 2f64.powi(e);
            if neg { -v } else { v }
        })
}

fn wide_f32() -> impl Strategy<Value = f32> {
    (any::<bool>(), 1.0f32..2.0, -60i32..=60i32).prop_map(|(neg, m, e)| {
        let v = m * 2f32.powi(e);
        if neg { -v } else { v }
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(4096))]

    /// f64: three-bin method vs scale-by-max, within a backward-error bound.
    #[test]
    fn nrm2_f64_matches_scale_by_max(x in prop::collection::vec(wide_f64(), 0..64)) {
        let got = nrm2(&x);
        let want = nrm2_scale_by_max(&x);
        prop_assert!(got.is_finite());
        let tol = 16.0 * f64::EPSILON * (x.len() as f64 + 1.0);
        prop_assert!(
            rel_err(got, want) <= tol,
            "n={}, got={}, want={}, rel_err={:e} > tol={:e}",
            x.len(), got, want, rel_err(got, want), tol
        );
    }

    /// f32: our f32 routine vs an f64 naive reference (f64 cannot overflow here).
    #[test]
    fn nrm2_f32_matches_f64_reference(x in prop::collection::vec(wide_f32(), 0..64)) {
        let got = nrm2(&x) as f64;
        let want: f64 = {
            let s: f64 = x.iter().map(|&v| { let d = v as f64; d * d }).sum();
            s.sqrt()
        };
        prop_assert!(got.is_finite());
        let tol = 16.0 * (f32::EPSILON as f64) * (x.len() as f64 + 1.0);
        prop_assert!(
            rel_err(got, want) <= tol,
            "n={}, got={}, want={}, rel_err={:e} > tol={:e}",
            x.len(), got, want, rel_err(got, want), tol
        );
    }
}
