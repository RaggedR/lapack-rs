//! The whole argument in one screen of output.
//!
//! Run with:  cargo run --example overflow_demo

use lapack_rs::{nrm2, nrm2_naive};

fn show(label: &str, x: &[f64]) {
    let truth = x[0].abs() * 2f64.sqrt(); // both entries equal in magnitude here
    println!("{label}");
    println!("    true  ‖x‖₂ ≈ {truth:e}");
    println!("    naive      = {:e}   <- single accumulator", nrm2_naive(x));
    println!("    nrm2       = {:e}   <- three-bin safe scaling", nrm2(x));
    println!();
}

fn main() {
    println!("\nEuclidean norm of [a, a] — the answer is always a·√2.\n");
    show("a = 1e200  (huge):", &[1e200, 1e200]);
    show("a = 1e-200 (tiny):", &[1e-200, 1e-200]);
    println!("The naive column is wrong twice (inf, then 0); the true norm was");
    println!("representable both times. That gap is the entire reason LAPACK's");
    println!("dnrm2 looks the way it does.\n");
}
