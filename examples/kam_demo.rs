//! KAM tori of Hénon's area-preserving map, over both carriers.
//!
//! Run with:  cargo run --example kam_demo
//!
//! The thesis of the repo, applied to a Hamiltonian map: the *same* generic
//! code runs over `f64` (fast, leaks) and `CReal` (exact). Symplecticity
//! (det J = 1) and the Greene residues (±1/5) are exact rationals — so `CReal`
//! nails them and `f64` drifts exactly where cancellation says it must.

use lapack_rs::comonad::{iterate, Stream};
use lapack_rs::kam::*;
use lapack_rs::real::CReal;
use lapack_rs::Scalar;

fn main() {
    println!("\n=== Hénon area-preserving twist map, rotation (c,s) = (3/5, 4/5) ===\n");

    // --- 1. The co-Kleisli iteration: the orbit as a comonadic stream -------
    println!("1. The orbit is iterate(henon_step, seed) — the cofree stream.");
    let seed = Pt { x: 0.12_f64, y: 0.0 };
    let pts = orbit(seed).take(6);
    for (k, p) in pts.iter().enumerate() {
        println!("     step {k}:  x = {:+.6}   y = {:+.6}", p.x, p.y);
    }
    println!("   (bounded, winding around the elliptic centre — a KAM torus)\n");

    // --- 2. Symplecticity, and where f64 cancels ----------------------------
    println!("2. det J = c² + s² = 1  for every x — but it is computed through");
    println!("   terms 2csx that cancel. At large x, f64 loses the cancellation:\n");
    for &x in &[0.5_f64, 1e3, 1e6, 1e9, 1e12] {
        let df = det2(&jacobian(&Pt { x, y: 0.0 }));
        let dc = det2(&jacobian(&Pt { x: CReal::from_f64(x), y: CReal::zero() }));
        println!(
            "     x = {x:>8.0e}:   f64 det = {df:.15}      CReal det = {}",
            dc.extract()
        );
    }
    println!("   f64's error is bounded by ~x·ε but erratic — exact at some");
    println!("   magnitudes, off by ~1e-7 at others; CReal is exactly 1 at all.\n");

    // --- 3. The two fixed points and their exact Greene residues ------------
    println!("3. Greene residue R = (2 − tr J)/4 at the two fixed points");
    println!("   (sign of R is the elliptic/hyperbolic classifier):\n");
    report_residue("origin   (0, 0)  ", Pt { x: 0.0, y: 0.0 });
    report_residue("hyperbolic (1,1/2)", Pt { x: 1.0, y: 0.5 });
    println!("   CReal gives EXACTLY ±1/5; these are hand-checkable rationals.\n");

    // --- 4. Torus detection via extend (a co-Kleisli observable) ------------
    println!("4. Lyapunov exponent along the orbit (extend of a local rule).");
    println!("   This map is unbounded outside its island, so the three outcomes");
    println!("   are: torus (λ≈0), bounded chaos (λ>0), or escape to ∞:\n");
    for &a in &[0.12_f64, 0.5, 0.74, 0.9] {
        let seed = Pt { x: a, y: 0.0 };
        // Boundedness decides the honest label — an escaping orbit is not chaos.
        let escaped = orbit(seed.clone()).take(2_000).iter().any(|p| p.x.abs() > 10.0);
        let verdict = if escaped {
            "escapes to ∞".to_string()
        } else {
            let lam = lyapunov_renorm(seed, 20_000);
            if lam.abs() < 0.01 {
                format!("λ = {lam:+.4}   torus")
            } else {
                format!("λ = {lam:+.4}   bounded chaos")
            }
        };
        println!("     seed x={a:<4}:  {verdict}");
    }

    // --- 5. The area defect as an extend over the stream --------------------
    println!("\n5. The area defect det J(zₙ) − 1, run as a co-Kleisli arrow");
    println!("   extend'ed along the exact orbit — identically zero over CReal:");
    let orbit_c: Stream<Pt<CReal>> =
        iterate(henon_step::<CReal>, Pt { x: CReal::from_f64(0.12), y: CReal::zero() });
    let defect: Stream<CReal> =
        orbit_c.extend(|s: &Stream<Pt<CReal>>| det2(&jacobian(&s.extract())) - CReal::from_i64(1));
    let ds: Vec<f64> = defect.take(5).iter().map(|d| d.extract()).collect();
    println!("     defects: {ds:?}");
    println!();
}

fn report_residue(label: &str, p_f64: Pt<f64>) {
    let rf = fixed_point_residue(&p_f64);
    let rc = fixed_point_residue(&Pt {
        x: CReal::from_f64(p_f64.x),
        y: CReal::from_f64(p_f64.y),
    });
    let kind = if rf > 0.0 && rf < 1.0 { "elliptic " } else { "hyperbolic" };
    println!("     {label}:  f64 R = {rf:+.15}   CReal R = {:+.6}   [{kind}]", rc.extract());
}
