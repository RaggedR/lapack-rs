//! Does the equivariant-extension operator actually deliver safe routines?
//!
//! Three things to establish:
//!   1. `equivariant::nrm2` reaches the same correctness as the hand-written
//!      three-bin [`lapack_rs::nrm2`] and the independent scale-by-max oracle —
//!      i.e. equivariance alone buys the whole safety story.
//!   2. The *representation* is real, not decorative: for Givens, c and s are
//!      degree 0 (invariant under input scaling) and r is degree 1 — checkable
//!      to the bit.
//!   3. The honest limit: where, if anywhere, the single-scale operator parts
//!      company with the three-bin routine.

use lapack_rs::equivariant as eq;
use lapack_rs::nrm2 as nrm2_threebin;
use proptest::prelude::*;

// --- independent oracle (a *different* safe algorithm) ----------------------

fn nrm2_scale_by_max(x: &[f64]) -> f64 {
    let m = x.iter().fold(0.0_f64, |a, &v| a.max(v.abs()));
    if m == 0.0 || !m.is_finite() {
        return m;
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

// --- 1. the operator-derived nrm2 is safe -----------------------------------

#[test]
fn derived_nrm2_survives_overflow_and_underflow() {
    let big = [1e200_f64, 1e200];
    let tiny = [1e-200_f64, 1e-200];
    let expect_big = 1e200 * 2f64.sqrt();
    let expect_tiny = 1e-200 * 2f64.sqrt();

    assert!(eq::nrm2(&big).is_finite());
    assert!(rel_err(eq::nrm2(&big), expect_big) < 1e-15);
    assert!(eq::nrm2(&tiny) > 0.0);
    assert!(rel_err(eq::nrm2(&tiny), expect_tiny) < 1e-15);
}

#[test]
fn derived_nrm2_small_exact_cases() {
    assert_eq!(eq::nrm2(&[]), 0.0);
    assert_eq!(eq::nrm2(&[0.0, 0.0]), 0.0);
    assert_eq!(eq::nrm2(&[3.0, 4.0]), 5.0);
    assert_eq!(eq::hypot(3.0, 4.0), 5.0);
}

// --- 2. the representation is real: Givens degrees ⟨0,0,1⟩ ------------------

#[test]
fn givens_zeroes_the_second_component() {
    for &(a, b) in &[(3.0_f64, 4.0), (-5.0, 12.0), (1.0, 0.0), (0.0, 7.0)] {
        let (c, s, r) = eq::rotg(a, b);
        // [[c, s], [-s, c]] · (a, b)ᵀ = (r, 0)
        assert!((c * a + s * b - r).abs() <= 1e-13 * r.abs().max(1.0));
        assert!((-s * a + c * b).abs() <= 1e-13 * r.abs().max(1.0));
        // orthogonality
        assert!((c * c + s * s - 1.0).abs() <= 1e-13);
    }
}

#[test]
fn givens_c_s_are_degree_zero_and_r_is_degree_one() {
    // ρ = ⟨0, 0, 1⟩ made empirical. Scaling the input by 2ᵏ must leave c, s
    // *bit-identical* (trivial subrepresentation) and scale r by exactly 2ᵏ.
    let (c0, s0, r0) = eq::rotg(3.0, 4.0);
    for k in [-700_i32, -53, -1, 10, 200, 700] {
        let a = eq_ldexp(3.0, k);
        let b = eq_ldexp(4.0, k);
        let (c, s, r) = eq::rotg(a, b);
        assert_eq!(c, c0, "cosine must be scale-invariant (degree 0), k={k}");
        assert_eq!(s, s0, "sine must be scale-invariant (degree 0), k={k}");
        assert_eq!(r, eq_ldexp(r0, k), "r must scale by exactly 2^{k} (degree 1)");
    }
}

#[test]
fn givens_survives_overflow_where_naive_fails() {
    // Naive Givens forms a²+b² directly ⇒ overflow ⇒ r=inf, c=s=0.
    let (a, b) = (1e200_f64, 1e200);
    let naive_r = (a * a + b * b).sqrt();
    assert!(naive_r.is_infinite(), "naive r should overflow");

    let (c, s, r) = eq::rotg(a, b);
    assert!(r.is_finite());
    assert!(rel_err(r, 1e200 * 2f64.sqrt()) < 1e-15);
    assert!((c - 1.0 / 2f64.sqrt()).abs() < 1e-15);
    assert!((s - 1.0 / 2f64.sqrt()).abs() < 1e-15);
}

// A local copy of the exact power-of-two scale, so the test exercises the same
// lossless shift the library uses rather than a rounding `* 2f64.powi(k)`.
fn eq_ldexp(x: f64, mut k: i32) -> f64 {
    if x == 0.0 || !x.is_finite() {
        return x;
    }
    let up = f64::from_bits((1023u64 + 512) << 52);
    let dn = f64::from_bits((1023u64 - 512) << 52);
    let mut r = x;
    while k > 512 {
        r *= up;
        k -= 512;
    }
    while k < -512 {
        r *= dn;
        k += 512;
    }
    r * f64::from_bits(((1023 + k) as u64) << 52)
}

// --- 3. property test: operator vs oracle, and vs the three-bin routine ------

fn wide_f64() -> impl Strategy<Value = f64> {
    (any::<bool>(), 1.0f64..2.0, -290i32..=290i32).prop_map(|(neg, m, e)| {
        let v = m * 2f64.powi(e);
        if neg { -v } else { v }
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(4096))]

    /// The operator-derived nrm2 matches the independent oracle to a backward
    /// error bound — equivariance alone reaches the correctness tier.
    #[test]
    fn derived_nrm2_matches_oracle(x in prop::collection::vec(wide_f64(), 0..64)) {
        let got = eq::nrm2(&x);
        let want = nrm2_scale_by_max(&x);
        prop_assert!(got.is_finite());
        let tol = 16.0 * f64::EPSILON * (x.len() as f64 + 1.0);
        prop_assert!(
            rel_err(got, want) <= tol,
            "n={}, got={}, want={}, rel_err={:e} > tol={:e}",
            x.len(), got, want, rel_err(got, want), tol
        );
    }

    /// And it agrees with the hand-written three-bin routine: the "for free"
    /// single-scale extension lands at the same place as the bespoke code.
    #[test]
    fn derived_nrm2_agrees_with_threebin(x in prop::collection::vec(wide_f64(), 0..64)) {
        let a = eq::nrm2(&x);
        let b = nrm2_threebin(&x);
        let tol = 16.0 * f64::EPSILON * (x.len() as f64 + 1.0);
        prop_assert!(
            rel_err(a, b) <= tol,
            "operator={a}, threebin={b}, rel_err={:e} > tol={:e}",
            rel_err(a, b), tol
        );
    }
}
