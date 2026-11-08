//! Benchmark harness for our implementations.
//!
//!   cargo run --release --example bench
//!
//! Prints timings for the f64 generic routines at several sizes, and for the
//! exact iRRAM carrier at n=4. See BENCHMARKS.md for the LAPACK (Accelerate)
//! and Arb (FLINT) comparison harnesses and the collected numbers.

use lapack_rs::irram::{run, Real};
use lapack_rs::{equivariant, generic};
use std::time::Instant;

fn rnd(s: &mut u64) -> f64 {
    *s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
    ((*s >> 33) as f64) / ((1u64 << 31) as f64)
}
fn vecf(n: usize, s: &mut u64) -> Vec<f64> {
    (0..n).map(|_| rnd(s)).collect()
}
fn matf(n: usize, s: &mut u64) -> Vec<Vec<f64>> {
    (0..n).map(|_| vecf(n, s)).collect()
}
fn sv(f: &[f64]) -> Vec<Real> {
    f.iter().map(|&x| Real::from_f64(x)).collect()
}
fn sm(f: &[Vec<f64>]) -> Vec<Vec<Real>> {
    f.iter().map(|r| sv(r)).collect()
}
fn sink<T>(_x: T) {}
fn bench<F: FnMut()>(mut f: F, reps: usize) -> f64 {
    f();
    let t = Instant::now();
    for _ in 0..reps {
        f();
    }
    t.elapsed().as_secs_f64() / reps as f64
}

fn main() {
    let mut s = 12345u64;

    println!("=== ours, f64 (generic::<f64>, release) ===");
    for &n in &[10usize, 100, 300] {
        let a = matf(n, &mut s);
        let b = matf(n, &mut s);
        let x = vecf(n, &mut s);
        let r = if n <= 10 { 2000 } else if n <= 100 { 20 } else { 2 };
        let tn = bench(|| sink(std::hint::black_box(equivariant::nrm2(&x))), 200000);
        let tm = bench(|| sink(std::hint::black_box(generic::matmul(&a, &b))), r);
        let tq = bench(|| sink(std::hint::black_box(generic::gram_schmidt(&a))), r);
        println!("n={n:>3}:  norm={:8.3} us   matmul={:12.2} us   qr(GS)={:12.2} us", tn * 1e6, tm * 1e6, tq * 1e6);
    }

    println!("\n=== ours, exact (iRRAM Real), n=4 ===");
    let n = 4;
    let xf = vecf(n, &mut s);
    let yf = vecf(n, &mut s);
    let af = matf(n, &mut s);
    let bf = matf(n, &mut s);
    let sf: Vec<Vec<f64>> = (0..n).map(|i| (0..n).map(|j| af[i][j] + af[j][i]).collect()).collect();
    let td = bench(|| drop(run(50, || vec![generic::dot(&sv(&xf), &sv(&yf))])), 5000);
    let tm = bench(|| drop(run(50, || generic::matmul(&sm(&af), &sm(&bf)).into_iter().flatten().collect())), 2000);
    let tq = bench(|| drop(run(50, || { let (q, _) = generic::gram_schmidt(&sm(&af)); q.into_iter().flatten().collect() })), 2000);
    let te = bench(|| drop(run(45, || { let (e, _) = generic::jacobi_eigen(&sm(&sf)); e })), 500);
    println!("n=  4:  dot={:.3} us  matmul={:.3} us  qr={:.3} us  eig={:.3} us", td * 1e6, tm * 1e6, tq * 1e6, te * 1e6);
}
