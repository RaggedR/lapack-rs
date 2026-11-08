//! An iRRAM-style exact-real carrier: fast *and* adaptive *and* correct.
//!
//! The lazy [`crate::real::CReal`] is correct but re-walks a growing DAG at
//! escalating precision, which is hopeless for iterative algorithms. This module
//! uses the architecture real exact-real libraries (iRRAM) use instead:
//!
//!  1. A value is an **interval** `[lo, hi]`. Because every result is rounded to
//!     a multiple of `2^(-p)` at the working precision `p`, the endpoints are
//!     stored as **integer mantissas** `m` meaning `m·2^(-p)` — dyadic
//!     fixed-point. No rationals, so **no GCD, no denominator growth**:
//!     add/sub are integer add/sub, `√` is an *integer* square root, and
//!     comparison is a bare integer compare.
//!  2. A **comparison that cannot be resolved** at `p` (the intervals overlap)
//!     raises a thread-local *reiterate* signal instead of guessing.
//!  3. A **driver** runs the whole computation at precision `p`; if the result
//!     did not resolve tightly enough (or a comparison reiterated), it doubles
//!     `p` and runs the *entire thing again*.
//!
//! [`Real`] implements [`Scalar`], so [`crate::generic`]'s Gram–Schmidt and
//! Jacobi run over it **unchanged** — fast, exact, and adaptive.

use crate::Scalar;
use core::cell::Cell;
use core::cmp::Ordering;
use core::ops::{Add, Div, Mul, Neg, Sub};
use num_bigint::{BigInt, Sign};
use num_rational::BigRational;
use num_traits::{One, Signed, ToPrimitive, Zero};

thread_local! {
    static PREC: Cell<i64> = const { Cell::new(64) };
    static REITERATE: Cell<bool> = const { Cell::new(false) };
}

fn prec() -> i64 {
    PREC.with(|c| c.get())
}
fn signal_reiterate() {
    REITERATE.with(|c| c.set(true));
}

// --- integer helpers: floor/ceil division, and ±∞-safe power-of-two shifts ---

fn fdiv(a: &BigInt, b: &BigInt) -> BigInt {
    let q = a / b;
    let r = a % b;
    if !r.is_zero() && r.sign() != b.sign() {
        q - 1
    } else {
        q
    }
}
fn cdiv(a: &BigInt, b: &BigInt) -> BigInt {
    let q = a / b;
    let r = a % b;
    if !r.is_zero() && r.sign() == b.sign() {
        q + 1
    } else {
        q
    }
}
/// `floor(x / 2^k)` via arithmetic right shift (rounds toward −∞ for all signs).
fn shr_floor(x: &BigInt, k: i64) -> BigInt {
    x >> (k as usize)
}
/// `ceil(x / 2^k)`.
fn shr_ceil(x: &BigInt, k: i64) -> BigInt {
    -(&(-x) >> (k as usize))
}

/// A real number as an enclosing interval of dyadic fixed-point numbers:
/// the true value lies in `[lo·2^(-p), hi·2^(-p)]` at the working precision `p`.
#[derive(Clone)]
pub struct Real {
    lo: BigInt,
    hi: BigInt,
}

impl Real {
    /// Exact: an `f64` is a dyadic rational, placed as a (near-)point interval.
    pub fn from_f64(x: f64) -> Real {
        let p = prec();
        let q = BigRational::from_float(x).expect("finite f64");
        let num = q.numer() << (p as usize); // numer · 2^p
        let d = q.denom();
        Real {
            lo: fdiv(&num, d),
            hi: cdiv(&num, d),
        }
    }
    pub fn from_i64(n: i64) -> Real {
        let m = BigInt::from(n) << (prec() as usize);
        Real { lo: m.clone(), hi: m }
    }
    /// Collapse to an `f64` (the interval midpoint).
    pub fn to_f64(&self) -> f64 {
        // value = (lo+hi)/2 · 2^(-p) = (lo+hi) · 2^(-(p+1))
        mantissa_to_f64(&(&self.lo + &self.hi), prec() + 1)
    }
}

/// `m · 2^(-s)` as an `f64`, without an `inf/inf → NaN` trap.
fn mantissa_to_f64(m: &BigInt, s: i64) -> f64 {
    if m.is_zero() {
        return 0.0;
    }
    let neg = m.sign() == Sign::Minus;
    let mag = m.magnitude();
    let b = mag.bits() as i64;
    let k = (b - 64).max(0);
    let topf = (mag >> (k as usize)).to_f64().unwrap_or(f64::INFINITY);
    let mut v = topf * 2f64.powi((k - s) as i32);
    if neg {
        v = -v;
    }
    v
}

// --- interval arithmetic in dyadic fixed-point --------------------------------

fn add_iv(a: &Real, b: &Real) -> Real {
    Real {
        lo: &a.lo + &b.lo,
        hi: &a.hi + &b.hi,
    } // exact: same scale
}
fn sub_iv(a: &Real, b: &Real) -> Real {
    Real {
        lo: &a.lo - &b.hi,
        hi: &a.hi - &b.lo,
    }
}
fn neg_iv(a: &Real) -> Real {
    Real {
        lo: -&a.hi,
        hi: -&a.lo,
    }
}
fn mul_iv(a: &Real, b: &Real) -> Real {
    // products are at scale 2^(-2p); shift back to 2^(-p), rounding outward.
    let p = prec();
    let c = [&a.lo * &b.lo, &a.lo * &b.hi, &a.hi * &b.lo, &a.hi * &b.hi];
    let lo_c = c.iter().min().unwrap();
    let hi_c = c.iter().max().unwrap();
    Real {
        lo: shr_floor(lo_c, p),
        hi: shr_ceil(hi_c, p),
    }
}
fn recip_iv(b: &Real) -> Real {
    let p = prec();
    let z = BigInt::zero();
    if b.lo <= z && b.hi >= z {
        signal_reiterate(); // interval straddles 0 — reciprocal unbounded
        let one = BigInt::one() << (p as usize);
        return Real { lo: -&one, hi: one };
    }
    // value recip at scale 2^(-p): mantissa = 2^(2p) / b_endpoint
    let num = BigInt::one() << (2 * p as usize);
    Real {
        lo: fdiv(&num, &b.hi),
        hi: cdiv(&num, &b.lo),
    }
}
fn div_iv(a: &Real, b: &Real) -> Real {
    mul_iv(a, &recip_iv(b))
}
fn abs_iv(a: &Real) -> Real {
    let z = BigInt::zero();
    if a.lo >= z {
        a.clone()
    } else if a.hi <= z {
        neg_iv(a)
    } else {
        Real {
            lo: z,
            hi: (-&a.lo).max(a.hi.clone()),
        }
    }
}
fn isqrt_floor(x: &BigInt) -> BigInt {
    x.sqrt() // floor integer sqrt, x >= 0
}
fn isqrt_ceil(x: &BigInt) -> BigInt {
    let s = x.sqrt();
    if &(&s * &s) == x {
        s
    } else {
        s + 1
    }
}
fn sqrt_iv(a: &Real) -> Real {
    let p = prec();
    let z = BigInt::zero();
    let lo_in = if a.lo < z { z.clone() } else { a.lo.clone() };
    let hi_in = if a.hi < z { z.clone() } else { a.hi.clone() };
    // √(m·2^-p) at scale 2^-p has mantissa isqrt(m·2^p)
    Real {
        lo: isqrt_floor(&(lo_in << (p as usize))),
        hi: isqrt_ceil(&(hi_in << (p as usize))),
    }
}

// --- operators, ordering, Scalar impl ----------------------------------------

impl Add for Real {
    type Output = Real;
    fn add(self, o: Real) -> Real {
        add_iv(&self, &o)
    }
}
impl Sub for Real {
    type Output = Real;
    fn sub(self, o: Real) -> Real {
        sub_iv(&self, &o)
    }
}
impl Mul for Real {
    type Output = Real;
    fn mul(self, o: Real) -> Real {
        mul_iv(&self, &o)
    }
}
impl Div for Real {
    type Output = Real;
    fn div(self, o: Real) -> Real {
        div_iv(&self, &o)
    }
}
impl Neg for Real {
    type Output = Real;
    fn neg(self) -> Real {
        neg_iv(&self)
    }
}

impl PartialEq for Real {
    fn eq(&self, o: &Real) -> bool {
        matches!(self.partial_cmp(o), Some(Ordering::Equal))
    }
}
impl PartialOrd for Real {
    fn partial_cmp(&self, o: &Real) -> Option<Ordering> {
        if self.hi < o.lo {
            Some(Ordering::Less)
        } else if self.lo > o.hi {
            Some(Ordering::Greater)
        } else {
            signal_reiterate();
            None
        }
    }
}

impl Scalar for Real {
    fn from_f64(x: f64) -> Self {
        Real::from_f64(x)
    }
    fn abs(&self) -> Self {
        abs_iv(self)
    }
    fn sqrt(&self) -> Self {
        sqrt_iv(self)
    }
    fn powi(&self, n: i32) -> Self {
        let mut r = Real::from_i64(1);
        for _ in 0..n {
            r = mul_iv(&r, self);
        }
        r
    }
}

// --- driver: run the whole computation, re-run at 2× precision if loose -------

fn width_ok(iv: &Real, target: i64) -> bool {
    let p = prec();
    let w = &iv.hi - &iv.lo;
    let thresh_abs = if p >= target {
        BigInt::one() << ((p - target) as usize)
    } else {
        BigInt::zero()
    };
    if w <= thresh_abs {
        return true; // absolute width < 2^-target
    }
    // relative: w·2^target ≤ |mid|, with mid·2 = lo+hi
    let mid2 = (&iv.lo + &iv.hi).abs();
    (w << (target as usize)) * 2 <= mid2
}

/// Run `f` at increasing working precision until every output is accurate to
/// `target` bits and no comparison reiterated, then collapse to `f64`. `f` must
/// rebuild its `Real`s from their seeds each call.
pub fn run<F: Fn() -> Vec<Real>>(target: i64, f: F) -> Vec<f64> {
    let mut p = 64i64;
    loop {
        PREC.with(|c| c.set(p));
        REITERATE.with(|c| c.set(false));
        let out = f();
        let reiterated = REITERATE.with(|c| c.get());
        if !reiterated && out.iter().all(|iv| width_ok(iv, target)) {
            return out.iter().map(|iv| iv.to_f64()).collect();
        }
        if p > (1 << 15) {
            return out.iter().map(|iv| iv.to_f64()).collect(); // best effort
        }
        p *= 2;
    }
}

/// Convenience: run a scalar computation.
pub fn run_scalar<F: Fn() -> Real>(target: i64, f: F) -> f64 {
    run(target, || vec![f()])[0]
}

#[cfg(test)]
mod unit {
    use super::*;

    #[test]
    fn sqrt_two_and_cancellation() {
        let r2 = run_scalar(50, || Real::from_i64(2).sqrt());
        assert!((r2 - 2f64.sqrt()).abs() < 1e-14);

        let x = 1e14_f64;
        let got = run_scalar(50, || {
            let xr = Real::from_f64(x);
            sub_iv(&(add_iv(&xr, &Real::from_i64(1))).sqrt(), &xr.sqrt())
        });
        let want = 1.0 / ((x + 1.0).sqrt() + x.sqrt());
        assert!((got - want).abs() / want < 1e-12, "got {got}, want {want}");
    }
}
