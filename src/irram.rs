//! An iRRAM-style exact-real carrier: fast *and* adaptive *and* correct.
//!
//! The lazy [`crate::real::CReal`] is correct but re-walks a growing DAG at
//! escalating precision, which is hopeless for iterative algorithms. This module
//! uses the architecture real exact-real libraries (iRRAM) use instead:
//!
//!  1. A value is an **interval** `[lo, hi]` of multi-precision dyadic rationals.
//!     Every operation rounds its endpoints *outward* to the current working
//!     precision, so the numbers stay bounded in size — each run is plain
//!     imperative arithmetic, no lazy trees.
//!  2. A **comparison that cannot be resolved** at the current precision (the
//!     intervals overlap) raises a *reiterate* signal instead of guessing.
//!  3. A **driver** runs the whole computation at precision `p`; if the result
//!     did not resolve tightly enough (or a comparison reiterated), it doubles
//!     `p` and runs the *entire thing again*. The precision search is around the
//!     whole computation, not per operation.
//!
//! The win: [`Real`] implements [`Scalar`], so [`crate::generic`]'s
//! Gram–Schmidt and Jacobi run over it **unchanged** — and now fast, because
//! each pass is bounded-size interval arithmetic and only a handful of passes
//! are needed. Correctness is preserved: the driver never returns an
//! under-resolved answer; worst case it re-runs at higher precision.

use crate::Scalar;
use core::cell::Cell;
use core::cmp::Ordering;
use core::ops::{Add, Div, Mul, Neg, Sub};
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, Signed, ToPrimitive, Zero};

thread_local! {
    static PREC: Cell<i64> = const { Cell::new(64) };     // working precision (bits)
    static REITERATE: Cell<bool> = const { Cell::new(false) };
}

fn prec() -> i64 {
    PREC.with(|c| c.get())
}
fn signal_reiterate() {
    REITERATE.with(|c| c.set(true));
}

/// `2^(−p)` as a rational.
fn pow2(p: i64) -> BigRational {
    if p >= 0 {
        BigRational::new(BigInt::one(), BigInt::one() << (p as u64))
    } else {
        BigRational::from(BigInt::one() << ((-p) as u64))
    }
}

/// Round `q` to a multiple of `2^(−prec)`, outward (`up` ⇒ toward +∞, else −∞).
/// Keeps every endpoint a bounded-size dyadic, which is what makes this fast.
fn round_abs(q: &BigRational, prec: i64, up: bool) -> BigRational {
    let scale = BigInt::one() << (prec.max(0) as u64);
    let scaled = q * BigRational::from(scale.clone());
    let int = if up { scaled.ceil() } else { scaled.floor() }.to_integer();
    BigRational::new(int, scale)
}

/// A real number as an enclosing interval at the current working precision.
#[derive(Clone)]
pub struct Real {
    lo: BigRational,
    hi: BigRational,
}

fn mk(lo: BigRational, hi: BigRational) -> Real {
    let p = prec();
    Real {
        lo: round_abs(&lo, p, false),
        hi: round_abs(&hi, p, true),
    }
}

impl Real {
    /// Exact: an `f64` is a dyadic rational, so its interval is a point.
    pub fn from_f64(x: f64) -> Real {
        let q = BigRational::from_float(x).expect("finite f64");
        Real { lo: q.clone(), hi: q }
    }
    pub fn from_i64(n: i64) -> Real {
        let q = BigRational::from(BigInt::from(n));
        Real { lo: q.clone(), hi: q }
    }
    fn mid(&self) -> BigRational {
        (&self.lo + &self.hi) / BigRational::from(BigInt::from(2))
    }
    fn width(&self) -> BigRational {
        &self.hi - &self.lo
    }
    /// Collapse to an `f64` (the interval midpoint).
    pub fn to_f64(&self) -> f64 {
        ratio_to_f64(&self.mid())
    }
}

// --- interval arithmetic (each op rounds endpoints to the working precision) --

fn add_iv(a: &Real, b: &Real) -> Real {
    mk(&a.lo + &b.lo, &a.hi + &b.hi)
}
fn sub_iv(a: &Real, b: &Real) -> Real {
    mk(&a.lo - &b.hi, &a.hi - &b.lo)
}
fn neg_iv(a: &Real) -> Real {
    mk(-&a.hi, -&a.lo)
}
fn mul_iv(a: &Real, b: &Real) -> Real {
    let p = [
        &a.lo * &b.lo,
        &a.lo * &b.hi,
        &a.hi * &b.lo,
        &a.hi * &b.hi,
    ];
    let lo = p.iter().min().unwrap().clone();
    let hi = p.iter().max().unwrap().clone();
    mk(lo, hi)
}
fn recip_iv(b: &Real) -> Real {
    let z = BigRational::zero();
    if b.lo <= z && b.hi >= z {
        // interval straddles 0 — reciprocal is unbounded; re-run at higher precision
        signal_reiterate();
        return mk(-BigRational::one(), BigRational::one());
    }
    let r1 = BigRational::one() / &b.lo;
    let r2 = BigRational::one() / &b.hi;
    let (lo, hi) = if r1 <= r2 { (r1, r2) } else { (r2, r1) };
    mk(lo, hi)
}
fn div_iv(a: &Real, b: &Real) -> Real {
    mul_iv(a, &recip_iv(b))
}
fn abs_iv(a: &Real) -> Real {
    let z = BigRational::zero();
    if a.lo >= z {
        a.clone()
    } else if a.hi <= z {
        neg_iv(a)
    } else {
        let m = (-&a.lo).max(a.hi.clone());
        mk(z, m)
    }
}

/// √c to within `2^(−q)`, for c ≥ 0 (Heron on rationals).
fn sqrt_rat(c: &BigRational, q: i64) -> BigRational {
    if c.is_zero() {
        return BigRational::zero();
    }
    let (bn, bd) = (c.numer().bits() as i64, c.denom().bits() as i64);
    let half = (bn - bd) / 2;
    let mut x = if half >= 0 {
        BigRational::from(BigInt::one() << (half as u64))
    } else {
        BigRational::new(BigInt::one(), BigInt::one() << ((-half) as u64))
    };
    if x <= BigRational::zero() {
        x = BigRational::one();
    }
    let tol = pow2(q + 2);
    let two = BigRational::from(BigInt::from(2));
    for _ in 0..1000 {
        let next = (x.clone() + c / x.clone()) / two.clone();
        let diff = (next.clone() - x.clone()).abs();
        x = next;
        if diff < tol {
            break;
        }
    }
    x
}
fn sqrt_iv(a: &Real) -> Real {
    let p = prec();
    let z = BigRational::zero();
    let lo_in = if a.lo < z { z.clone() } else { a.lo.clone() };
    let hi_in = if a.hi < z { z.clone() } else { a.hi.clone() };
    let eps = pow2(p + 3);
    // outward bounds on √ of each endpoint
    let lo = sqrt_rat(&lo_in, p + 4) - &eps;
    let hi = sqrt_rat(&hi_in, p + 4) + &eps;
    mk(lo.max(z), hi)
}

/// Convert a (possibly huge) rational to `f64` without an `inf/inf → NaN` trap.
fn ratio_to_f64(q: &BigRational) -> f64 {
    if q.is_zero() {
        return 0.0;
    }
    let neg = q.is_negative();
    let n = q.numer().abs();
    let d = q.denom().abs();
    let (bn, bd) = (n.bits() as i64, d.bits() as i64);
    let (kn, kd) = ((bn - 64).max(0), (bd - 64).max(0));
    let ntop = (n >> kn as usize).to_f64().unwrap_or(f64::INFINITY);
    let dtop = (d >> kd as usize).to_f64().unwrap_or(f64::INFINITY);
    let mut v = (ntop / dtop) * 2f64.powi((kn - kd) as i32);
    if neg {
        v = -v;
    }
    v
}

// --- operators, ordering, and the Scalar impl -------------------------------

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
            // intervals overlap — undecidable at this precision; ask to re-run
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
    // dot / nrm2 use the naive trait defaults — correct over intervals.
}

// --- the driver: run the whole computation, re-run at 2× precision if loose --

fn width_ok(iv: &Real, target: i64) -> bool {
    let thresh = pow2(target);
    let m = iv.mid().abs();
    iv.width() < thresh * (m + BigRational::one()) // relative-or-absolute
}

/// Run `f` (a computation producing enclosing [`Real`]s) at increasing working
/// precision until every output is accurate to `target` bits and no comparison
/// reiterated, then collapse the outputs to `f64`. `f` must rebuild its `Real`s
/// from their seeds each call — that is what makes re-running at higher
/// precision meaningful.
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
        if p > (1 << 20) {
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

        // √(x+1) − √x at x = 1e14: catastrophic cancellation in f64, exact here.
        let x = 1e14_f64;
        let got = run_scalar(50, || {
            let xr = Real::from_f64(x);
            sub_iv(&(add_iv(&xr, &Real::from_i64(1))).sqrt(), &xr.sqrt())
        });
        let want = 1.0 / ((x + 1.0).sqrt() + x.sqrt());
        assert!((got - want).abs() / want < 1e-12, "got {got}, want {want}");
    }
}
