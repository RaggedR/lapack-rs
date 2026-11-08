//! KAM / Hénon-map tests. The anchors are *exact rationals* — determinant 1,
//! residues ±1/5 — so the `CReal` carrier is checked bit-for-bit, and the `f64`
//! carrier is shown to drift exactly where cancellation predicts it should.

use lapack_rs::comonad::{iterate, Stream};
use lapack_rs::kam::*;
use lapack_rs::real::CReal;

fn pt_f64(x: f64, y: f64) -> Pt<f64> {
    Pt { x, y }
}
fn pt_c(x: f64, y: f64) -> Pt<CReal> {
    Pt { x: CReal::from_f64(x), y: CReal::from_f64(y) }
}
fn tiny() -> CReal {
    CReal::from_f64(1e-30)
}

// ---------------------------------------------------------------------------
// The comonad itself
// ---------------------------------------------------------------------------

#[test]
fn iterate_unfolds_the_orbit() {
    let s = iterate(|n: &i64| n + 1, 0);
    assert_eq!(s.take(5), vec![0, 1, 2, 3, 4]);
}

#[test]
fn counit_cancels_comultiplication() {
    // extract ∘ duplicate = id   (the left counit law, on a finite prefix)
    let s = iterate(|n: &i64| n + 1, 10);
    let d: Stream<Stream<i64>> = s.duplicate();
    assert_eq!(d.extract().take(4), s.take(4));
    // and the second suffix really is the tail
    assert_eq!(d.tail().extract().take(3), vec![11, 12, 13]);
}

#[test]
fn extend_with_extract_is_identity() {
    // extend ε = id
    let s = iterate(|n: &i64| n + 1, 0);
    let t = s.extend(|w: &Stream<i64>| w.extract());
    assert_eq!(t.take(6), s.take(6));
}

// ---------------------------------------------------------------------------
// Symplecticity: det J = 1, exactly — and f64 cancellation at large x
// ---------------------------------------------------------------------------

#[test]
fn determinant_is_exactly_one_over_creal() {
    // c² + s² = 9/25 + 16/25 = 1, for every x, with no rounding.
    for &(x, y) in &[(0.0, 0.0), (0.37, -1.2), (1.0, 0.5), (1e6, -3e5)] {
        let d = det2(&jacobian(&pt_c(x, y)));
        assert!((d - CReal::from_i64(1)).abs() < tiny(), "det not exactly 1 at x={x}");
    }
}

#[test]
fn f64_determinant_drifts_where_creal_stays_exact() {
    // At large x the determinant is a difference of ~2csx-sized terms that must
    // cancel to 1; the roundoff grows like x·ε. Scan magnitudes and take the
    // worst — by 10¹² the loss is unmissable. f64 drifts; CReal does not.
    let worst = [1e6, 1e8, 1e10, 1e12]
        .into_iter()
        .map(|x| (det2(&jacobian(&pt_f64(x, 0.0))) - 1.0).abs())
        .fold(0.0_f64, f64::max);
    assert!(worst > 1e-9, "expected visible f64 cancellation, worst drift was {worst:e}");

    // CReal stays exact at every one of those magnitudes.
    for &x in &[1e6, 1e8, 1e10, 1e12] {
        let dc = det2(&jacobian(&pt_c(x, 0.0)));
        assert!((dc - CReal::from_i64(1)).abs() < tiny(), "CReal det not exact at x={x}");
    }
}

// ---------------------------------------------------------------------------
// Greene residues of the two fixed points: exactly +1/5 and −1/5
// ---------------------------------------------------------------------------

#[test]
fn fixed_points_are_fixed() {
    // origin and (1, 1/2) are the two fixed points — verified over CReal.
    let o = henon_step(&pt_c(0.0, 0.0));
    assert!(o.x.abs() < tiny() && o.y.abs() < tiny());

    let h = henon_step(&pt_c(1.0, 0.5));
    assert!((h.x - CReal::from_i64(1)).abs() < tiny());
    assert!((h.y - CReal::from_f64(0.5)).abs() < tiny());
}

#[test]
fn residues_are_exact_fifths() {
    let fifth = CReal::from_i64(1) / CReal::from_i64(5);

    // origin: tr J = 2c = 6/5, R = (2 − 6/5)/4 = 1/5  (elliptic → on a torus)
    let r0 = fixed_point_residue(&pt_c(0.0, 0.0));
    assert!((r0 - fifth.clone()).abs() < tiny(), "origin residue should be exactly 1/5");

    // (1, 1/2): tr J = 14/5, R = (2 − 14/5)/4 = −1/5  (hyperbolic → chaotic)
    let r1 = fixed_point_residue(&pt_c(1.0, 0.5));
    assert!((r1 + fifth).abs() < tiny(), "hyperbolic residue should be exactly −1/5");

    // sign of the residue is the elliptic/hyperbolic classifier
    assert!(r0_elliptic(), "origin must classify elliptic");
}

fn r0_elliptic() -> bool {
    fixed_point_residue(&pt_c(0.0, 0.0)) > CReal::from_i64(0)
}

// ---------------------------------------------------------------------------
// Detecting the torus: a regular orbit stays bounded and has λ ≈ 0
// ---------------------------------------------------------------------------

#[test]
fn small_orbit_lies_on_a_torus() {
    // A seed just off the elliptic centre lies on a KAM torus: bounded forever,
    // zero Lyapunov exponent.
    let seed = pt_f64(0.12, 0.0);
    let pts = orbit(seed.clone()).take(4000);
    for p in &pts {
        assert!(p.x.abs() < 1.0 && p.y.abs() < 1.0, "orbit escaped: a torus should be bounded");
    }
    let lam = lyapunov_renorm(seed, 4000);
    assert!(lam.abs() < 0.02, "a torus should have λ ≈ 0, got {lam}");
}

#[test]
fn creal_and_f64_orbits_agree_for_a_while() {
    // Same seed, two carriers: they track the same torus to high precision.
    // The exact orbit is kept short — a quadratic map doubles the rational
    // height each step, so exact iteration is only feasible for ~15 steps.
    let n = 12;
    let f = orbit(pt_f64(0.12, 0.0)).take(n);
    let c = orbit(pt_c(0.12, 0.0)).take(n);
    for (a, b) in f.iter().zip(c.iter()) {
        assert!((a.x - b.x.extract()).abs() < 1e-9);
        assert!((a.y - b.y.extract()).abs() < 1e-9);
    }
}
