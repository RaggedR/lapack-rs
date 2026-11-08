//! "Write the naive formula; the grading makes it safe."
//!
//!   cargo run --example graded_demo

use lapack_rs::graded::var;
use lapack_rs::real::CReal;

fn main() {
    // The user writes ONLY the textbook formula. No scaling code anywhere.
    let norm = (var(0) * var(0) + var(1) * var(1)).sqrt();
    println!("norm = sqrt(x*x + y*y)   inferred degree = {}\n", norm.degree());

    for &a in &[3.0_f64, 1e200, 1e-200] {
        let inputs = [a, a];
        let exact = norm
            .eval_exact(&[CReal::from_f64(a), CReal::from_f64(a)])
            .extract();
        println!("x = y = {a:e}");
        println!("    eval_naive = {:e}   (f64, formula as written)", norm.eval_naive(&inputs));
        println!("    eval_safe  = {:e}   (f64, grading inserted scaling)", norm.eval_safe(&inputs));
        println!("    eval_exact = {:e}   (ℝ-native, float only at extract)", exact);
        println!("    true       = {:e}\n", a * 2f64.sqrt());
    }

    // Rung 3: the safe routine the grading generates, degree baked in from the type.
    println!("--- compile_safe() : grading as codegen ---\n{}\n", norm.compile_safe());

    // Givens cosine: degree 1 - 1 = 0, inferred by the type system.
    let r = (var(0) * var(0) + var(1) * var(1)).sqrt();
    let c = var(0) / r;
    println!("c = x / sqrt(x*x + y*y)  inferred degree = {} (scale-invariant)\n", c.degree());
    for &a in &[3.0_f64, 1e200] {
        println!("    x=y={a:e}:  eval_naive c = {:e},  eval_safe c = {:e}",
            c.eval_naive(&[a, a]), c.eval_safe(&[a, a]));
    }

    println!("\nThe degree in each type IS the representation ρ. The compiler reads");
    println!("it to decide how to transport the result back after scaling — and it");
    println!("rejects any formula (like x*x + 5) where no such ρ exists.");
}
