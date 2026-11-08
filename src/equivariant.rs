//! Layer 3 of the categorical-floating-point sketch: **the scaling trick as an
//! equivariant extension.**
//!
//! The claim being tested here is concrete. Let
//!
//!   G = { 2ᵏ : k ∈ ℤ } ≅ (ℤ, +)
//!
//! act on the nonzero `f64`s by multiplication. This action is *exact* —
//! multiplying by a power of two shifts the exponent field and leaves the
//! mantissa untouched — so it is a genuine group of automorphisms, not a
//! rounding map. Many kernels `f : ℝⁿ → ℝᵐ` are **equivariant**: each output
//! component scales by some integer power of the input scale,
//!
//!   f(2ᵏ·x)ⱼ = 2^(dⱼ·k) · f(x)ⱼ,
//!
//! where the vector of exponents `(d₁,…,dₘ)` *is* the representation ρ of G on
//! the output (a direct sum of characters). The naive kernel is correct only
//! on a **fundamental domain** W (the "safe window" where squaring neither
//! overflows nor underflows). Because every float is a power of two times an
//! element of W, the safe routine is forced to be the unique G-equivariant
//! extension of `kernel|_W`:
//!
//!   E[kernel](x) = ρ(g)⁻¹ · kernel(g·x),   g chosen so g·x ∈ W.
//!
//! [`scale_safe`] is `E`. `nrm2`, `hypot` and Givens `rotg` below are *the same
//! operator* applied to three different (kernel, ρ) pairs. The scaling
//! machinery is written exactly once — the thing LAPACK gestured at with
//! `xLASSQ`/`xLASCL`, here as a higher-order combinator with ρ made explicit.
//!
//! Scope: this is the single-scale *correctness* tier (compare
//! [`crate::nrm2`]'s hand-written three-bin *speed* tier). It allocates and is
//! not optimised; it exists to demonstrate that the equivariance theory yields
//! a safe algorithm for free. `f64` only — generalising to `f32` is the same
//! bit-layout logic with different field widths.

/// ⌊log₂|x|⌋ read straight off the IEEE-754 exponent field. Exact, no rounding.
/// `x` must be finite and nonzero (callers guard the zero/non-finite cases).
pub(crate) fn ilogb2(x: f64) -> i32 {
    let bits = x.abs().to_bits();
    let biased = ((bits >> 52) & 0x7ff) as i32;
    if biased == 0 {
        // Subnormal: value = mant · 2^(-1074); ⌊log₂⌋ = msb(mant) − 1074.
        let mant = bits & 0x000f_ffff_ffff_ffff;
        let msb = 63 - mant.leading_zeros() as i32; // highest set bit, 0..=51
        msb - 1074
    } else {
        // Normal: value ∈ [2^(biased−1023), 2^(biased−1022)).
        biased - 1023
    }
}

/// x · 2ᵏ, computed by exponent arithmetic so the multiplier never transiently
/// overflows to ∞. Exact (a product with a power of two) up to genuine
/// over/underflow of the *result*.
pub(crate) fn ldexp(x: f64, mut k: i32) -> f64 {
    if x == 0.0 || !x.is_finite() {
        return x;
    }
    let up = f64::from_bits((1023u64 + 512) << 52); // 2^512
    let dn = f64::from_bits((1023u64 - 512) << 52); // 2^-512
    let mut r = x;
    while k > 512 {
        r *= up;
        k -= 512;
    }
    while k < -512 {
        r *= dn;
        k += 512;
    }
    // |k| ≤ 512 ⇒ 1023+k ∈ [511,1535], a valid normal exponent ⇒ 2ᵏ exact.
    r * f64::from_bits(((1023 + k) as u64) << 52)
}

/// The equivariant-extension operator **E**.
///
/// `kernel` is the naive computation (correct inside the safe window); `rho` is
/// the representation — one integer *degree* per output component, saying how
/// that output scales when the whole input is scaled. Returns the safe routine.
///
/// The input always carries the diagonal action (scale every coordinate by the
/// same `g`); `rho` dictates how the outputs come back.
pub fn scale_safe<F>(kernel: F, rho: Vec<i32>) -> impl Fn(&[f64]) -> Vec<f64>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    move |x: &[f64]| {
        let m = x.iter().fold(0.0_f64, |a, &v| a.max(v.abs()));
        if m == 0.0 || !m.is_finite() {
            // No useful scale (all zero, or an inf/NaN present) — let the
            // kernel produce the correctly-propagated value directly.
            return kernel(x);
        }
        // Reduce into the fundamental domain: pick the exact power of two that
        // lands the largest coordinate in [1, 2). Note 2^(−e) is applied by
        // ldexp, which is lossless — strictly better than dividing by `m`,
        // which would round.
        let e = ilogb2(m);
        let scaled: Vec<f64> = x.iter().map(|&v| ldexp(v, -e)).collect();
        let raw = kernel(&scaled);
        // Transport each output back by its character ρ(g)⁻¹ = 2^(e·dⱼ).
        raw.iter()
            .zip(rho.iter())
            .map(|(&y, &d)| ldexp(y, e * d))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Three routines, one operator. Each is a (kernel, ρ) pair.
// ---------------------------------------------------------------------------

/// Kernel shared by `nrm2` and `hypot`: √(Σ xᵢ²). Degree-1 scalar output.
fn sum_of_squares(x: &[f64]) -> Vec<f64> {
    let mut s = 0.0;
    for &v in x {
        s += v * v;
    }
    vec![s.sqrt()]
}

/// Euclidean norm, derived purely from equivariance (ρ = ⟨1⟩).
pub fn nrm2(x: &[f64]) -> f64 {
    scale_safe(sum_of_squares, vec![1])(x)[0]
}

/// `hypot(a, b)` — identical kernel to `nrm2`, two coordinates (ρ = ⟨1⟩).
pub fn hypot(a: f64, b: f64) -> f64 {
    scale_safe(sum_of_squares, vec![1])(&[a, b])[0]
}

/// Kernel for a Givens rotation: given (a, b) return (c, s, r) with r = √(a²+b²),
/// c = a/r, s = b/r, so that [[c, s], [−s, c]]·(a, b)ᵀ = (r, 0).
///
/// Note the mixed degrees in the *result*: r is degree 1, but **c and s are
/// degree 0** — scale-invariant. That is exactly why Givens cosines/sines are
/// numerically forgiving, and it is encoded below as ρ = ⟨0, 0, 1⟩.
fn givens_kernel(x: &[f64]) -> Vec<f64> {
    let (a, b) = (x[0], x[1]);
    let r = (a * a + b * b).sqrt();
    if r == 0.0 {
        return vec![1.0, 0.0, 0.0]; // LAPACK convention for (0,0): c=1, s=0, r=0
    }
    vec![a / r, b / r, r]
}

/// Givens rotation coefficients `(c, s, r)`, derived from the mixed
/// representation ρ = ⟨0, 0, 1⟩. (Mathematically clean convention: r ≥ 0; this
/// omits LAPACK `drotg`'s sign/`z` bookkeeping, which is orthogonal to the
/// scaling story being demonstrated.)
pub fn rotg(a: f64, b: f64) -> (f64, f64, f64) {
    let o = scale_safe(givens_kernel, vec![0, 0, 1])(&[a, b]);
    (o[0], o[1], o[2])
}

#[cfg(test)]
mod unit {
    use super::*;

    #[test]
    fn ldexp_and_ilogb_are_exact_inverses_in_range() {
        for &x in &[1.0_f64, 3.0, 4.0, 0.75, 6.022e23, 1e-300, 1e300] {
            let e = ilogb2(x);
            // reduction lands the value in [1, 2)
            let reduced = ldexp(x, -e);
            assert!((1.0..2.0).contains(&reduced), "x={x}, reduced={reduced}");
            // and the scale is exactly invertible
            assert_eq!(ldexp(reduced, e), x, "roundtrip failed for x={x}");
        }
    }
}
