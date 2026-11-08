//! Rung 2: the **ℝ-native** layer — a real number as a lazy refinable process,
//! with `float` appearing only at `extract`.
//!
//! A [`CReal`] is represented in the classic *Cauchy-sequence-with-modulus*
//! style: a function `prec ↦ q` returning a rational `q` within `2^(−prec)` of
//! the true value. Equivalently it is the interval-domain refiner — the "stream
//! of ever-tighter approximations" from the earlier discussion, i.e. a
//! coinductive/comonadic object. [`CReal::extract`] is the comonadic counit:
//! it collapses the process to a single `f64` at a chosen precision.
//!
//! The headline consequence: arithmetic here is over *arbitrary-precision
//! rationals*, so **nothing ever overflows**. `(1e200)²` is just a 400-digit
//! integer. The entire safe-scaling apparatus of `nrm2`/`equivariant`/`graded`
//! is *unnecessary* in this layer — the floating-point haunting reappears only
//! at the `extract` boundary, and only if the final answer itself is out of
//! `f64` range. That is precisely the difference between a "float-native" and
//! an "ℝ-native" computation.
//!
//! Scope/honesty: the multiplication modulus and the Heron `sqrt` are
//! demonstration-grade (correct to the requested precision for the finite
//! expressions used here), not a certified exact-real library à la iRRAM.
//! Division/`recip` is deliberately omitted for now.

use crate::Scalar;
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, Signed, Zero};
use std::rc::Rc;

/// A computable real: ask it for a precision, it returns a rational that close.
#[derive(Clone)]
pub struct CReal {
    /// Contract: `(approx)(p)` returns `q` with `|q − x| ≤ 2^(−p)`.
    approx: Rc<dyn Fn(i64) -> BigRational>,
}

fn of(f: impl Fn(i64) -> BigRational + 'static) -> CReal {
    CReal { approx: Rc::new(f) }
}

fn two() -> BigRational {
    BigRational::from(BigInt::from(2))
}

/// `2^(−p)` as a rational (p may be negative).
fn pow2(p: i64) -> BigRational {
    if p >= 0 {
        BigRational::new(BigInt::one(), BigInt::one() << (p as u64))
    } else {
        BigRational::from(BigInt::one() << ((-p) as u64))
    }
}

impl CReal {
    /// Evaluate the refiner at precision `p`: a rational within `2^(−p)`.
    pub fn eval(&self, p: i64) -> BigRational {
        (self.approx)(p)
    }

    pub fn from_i64(n: i64) -> CReal {
        let q = BigRational::from(BigInt::from(n));
        of(move |_| q.clone())
    }

    /// Exact: an `f64` *is* a dyadic rational.
    pub fn from_f64(x: f64) -> CReal {
        let q = BigRational::from_float(x).expect("finite f64");
        of(move |_| q.clone())
    }

    pub fn add(&self, o: &CReal) -> CReal {
        let (a, b) = (self.clone(), o.clone());
        // each summand to 2^-(p+1) ⇒ sum to 2^-p
        of(move |p| a.eval(p + 1) + b.eval(p + 1))
    }

    pub fn sub(&self, o: &CReal) -> CReal {
        let (a, b) = (self.clone(), o.clone());
        of(move |p| a.eval(p + 1) - b.eval(p + 1))
    }

    pub fn neg(&self) -> CReal {
        let a = self.clone();
        of(move |p| -a.eval(p))
    }

    pub fn abs(&self) -> CReal {
        let a = self.clone();
        of(move |p| a.eval(p).abs()) // |·| is 1-Lipschitz ⇒ error preserved
    }

    pub fn mul(&self, o: &CReal) -> CReal {
        let (a, b) = (self.clone(), o.clone());
        of(move |p| {
            // |a|≤2^A, |b|≤2^B ⇒ evaluating a to 2^-(p+B+2), b to 2^-(p+A+2)
            // gives a product within 2^-p.
            let ba = mag_log2(&a);
            let bb = mag_log2(&b);
            a.eval(p + bb + 2) * b.eval(p + ba + 2)
        })
    }

    /// √x for x ≥ 0 (negative inputs from approximation error are clamped to 0).
    pub fn sqrt(&self) -> CReal {
        let a = self.clone();
        of(move |p| {
            let c = a.eval(2 * p + 4);
            if c <= BigRational::zero() {
                return BigRational::zero();
            }
            heron(&c, p)
        })
    }

    pub fn powi(&self, n: i32) -> CReal {
        assert!(n >= 0, "CReal::powi supports n ≥ 0 (no recip yet)");
        let mut r = CReal::from_i64(1);
        for _ in 0..n {
            r = r.mul(self);
        }
        r
    }

    /// Observe at precision `p` and round to `f64`.
    pub fn observe(&self, p: i64) -> f64 {
        ratio_to_f64(&self.eval(p))
    }

    /// The comonadic counit: collapse the refinement process to a single `f64`.
    ///
    /// Crucially this is *magnitude-aware*. The refiner's contract is absolute
    /// (`within 2^(−p)`), but an `f64` wants ~53 bits of **relative** accuracy,
    /// which for a value near 2^L means absolute precision ≈ 2^(L−53). So we
    /// first probe increasing precision until the value resolves as clearly
    /// nonzero — bounding its magnitude — then request enough for ~80 relative
    /// bits. (Asking for a fixed absolute precision is wrong for tiny values,
    /// and silently loses significant digits — the underflow-of-significance
    /// twin of the overflow problem.)
    pub fn extract(&self) -> f64 {
        let mut p: i64 = 24;
        while p <= 1 << 16 {
            let q = self.eval(p);
            // clearly nonzero: |q| > 2·2^(−p) ⇒ |x| > 2^(−p)
            if q.abs() > two() * pow2(p) {
                let log2 = (q.numer().bits() as i64) - (q.denom().bits() as i64);
                let want = (80 - log2).max(80);
                return ratio_to_f64(&self.eval(want));
            }
            p *= 2;
        }
        0.0 // indistinguishable from zero at 2^-65536
    }
}

/// An upper bound `B` with `|x| ≤ 2^B`, read cheaply off a coarse approximation.
fn mag_log2(r: &CReal) -> i64 {
    let q = r.eval(2).abs() + BigRational::one(); // ≥ |x|
    let bn = q.numer().bits() as i64;
    let bd = q.denom().bits() as i64;
    (bn - bd + 1).max(0)
}

/// Heron (Newton) iteration for √c on rationals, to within `2^(−p)`.
fn heron(c: &BigRational, p: i64) -> BigRational {
    // Start within a factor of ~2 of √c using the bit lengths, so the quadratic
    // convergence needs only a handful of steps even for astronomically large c.
    let bn = c.numer().bits() as i64;
    let bd = c.denom().bits() as i64;
    let half = (bn - bd) / 2;
    let mut x = if half >= 0 {
        BigRational::from(BigInt::one() << (half as u64))
    } else {
        BigRational::new(BigInt::one(), BigInt::one() << ((-half) as u64))
    };
    if x <= BigRational::zero() {
        x = BigRational::one();
    }
    let tol = pow2(p + 2);
    for _ in 0..200 {
        let next = (x.clone() + c / x.clone()) / two();
        let diff = (next.clone() - x.clone()).abs();
        x = next;
        if diff < tol {
            break;
        }
    }
    x
}

/// Convert a (possibly astronomically large) rational to `f64` without the
/// `numer.to_f64() / denom.to_f64()` → `inf/inf` → `NaN` trap: scale by bit
/// lengths and only ever call `to_f64` on a value below 2^64.
fn ratio_to_f64(q: &BigRational) -> f64 {
    use num_traits::ToPrimitive;
    if q.is_zero() {
        return 0.0;
    }
    let neg = q.is_negative();
    let n = q.numer().abs();
    let d = q.denom().abs();
    let bn = n.bits() as i64;
    let bd = d.bits() as i64;
    let kn = (bn - 64).max(0);
    let kd = (bd - 64).max(0);
    let ntop = (n >> kn as usize).to_f64().unwrap_or(f64::INFINITY);
    let dtop = (d >> kd as usize).to_f64().unwrap_or(f64::INFINITY);
    let mut val = (ntop / dtop) * 2f64.powi((kn - kd) as i32);
    if neg {
        val = -val;
    }
    val
}

impl Scalar for CReal {
    fn from_f64(x: f64) -> Self {
        CReal::from_f64(x)
    }
    fn add(&self, o: &Self) -> Self {
        CReal::add(self, o)
    }
    fn sub(&self, o: &Self) -> Self {
        CReal::sub(self, o)
    }
    fn mul(&self, o: &Self) -> Self {
        CReal::mul(self, o)
    }
    fn div(&self, _o: &Self) -> Self {
        unimplemented!("exact-real division is future work; the ℝ-native demos are division-free")
    }
    fn neg(&self) -> Self {
        CReal::neg(self)
    }
    fn abs(&self) -> Self {
        CReal::abs(self)
    }
    fn sqrt(&self) -> Self {
        CReal::sqrt(self)
    }
    fn powi(&self, n: i32) -> Self {
        CReal::powi(self, n)
    }
}

#[cfg(test)]
mod unit {
    use super::*;

    fn close(a: f64, b: f64, rel: f64) -> bool {
        if b == 0.0 {
            a.abs() < rel
        } else {
            (a - b).abs() / b.abs() < rel
        }
    }

    #[test]
    fn sqrt_two_is_correct() {
        let two = CReal::from_i64(2);
        assert!(close(two.sqrt().extract(), 2f64.sqrt(), 1e-15));
    }

    #[test]
    fn no_overflow_until_extract() {
        // x = 1e200; x*x = 1e400 — impossible in f64, trivial as a rational.
        let x = CReal::from_f64(1e200);
        let sumsq = x.mul(&x).add(&x.mul(&x)); // 2e400, exact
        let norm = sumsq.sqrt(); // ≈ 1.414e200, representable
        assert!(close(norm.extract(), 1e200 * 2f64.sqrt(), 1e-14));
    }

    #[test]
    fn tiny_norm_keeps_relative_accuracy() {
        // The underflow-of-significance case that caught the fixed-precision bug.
        for &a in &[1e-200_f64, 1e-150, 1e-300] {
            let x = CReal::from_f64(a);
            let norm = x.mul(&x).add(&x.mul(&x)).sqrt(); // a·√2
            assert!(
                close(norm.extract(), a * 2f64.sqrt(), 1e-14),
                "a={a:e}: got {}, want {}",
                norm.extract(),
                a * 2f64.sqrt()
            );
        }
    }

    #[test]
    fn to_f64_handles_huge_and_tiny() {
        assert!(close(CReal::from_f64(1e300).extract(), 1e300, 1e-12));
        assert!(close(CReal::from_f64(1e-300).extract(), 1e-300, 1e-12));
    }
}
