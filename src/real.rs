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
//! Supports `+ − × ÷`, `√`, ordering, and the operators (so it can carry the
//! generic routines in [`crate::generic`]). `eval` is memoized, so a shared
//! subexpression is computed once per precision rather than re-walked.
//!
//! Scope/honesty: the multiplication modulus and the Heron `sqrt` are
//! demonstration-grade (correct to the requested precision for the finite
//! expressions used here), not a certified exact-real library à la iRRAM. It is
//! fast for *shallow* expressions (cancellation-safe dot products, `√(x+1)−√x`)
//! but **impractically slow for deep/iterative algorithms** (Gram–Schmidt,
//! Jacobi) because division re-probes at escalating precision through every
//! nested reciprocal — a known exact-real cost that needs precision-bounded
//! "force/collapse" to tame, which is out of scope here.

use crate::Scalar;
use core::cmp::Ordering;
use core::ops::{Add, Div, DivAssign, Mul, Neg, Sub, SubAssign};
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, Signed, Zero};
use std::cell::RefCell;
use std::rc::Rc;

/// A computable real: ask it for a precision, it returns a rational that close.
#[derive(Clone)]
pub struct CReal {
    /// Contract: `(approx)(p)` returns `q` with `|q − x| ≤ 2^(−p)`.
    approx: Rc<dyn Fn(i64) -> BigRational>,
    /// Memo of the highest precision evaluated so far and its result. Shared by
    /// clones (an `Rc`), so a subexpression reused many times — the norm in
    /// every projection of Gram–Schmidt, say — is computed once per precision
    /// instead of re-walked each time. Without this, exact-real evaluation of a
    /// shared DAG is exponential; with it, linear.
    memo: Rc<RefCell<Option<(i64, BigRational)>>>,
}

fn of(f: impl Fn(i64) -> BigRational + 'static) -> CReal {
    CReal {
        approx: Rc::new(f),
        memo: Rc::new(RefCell::new(None)),
    }
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
    /// Memoized: a cached result at precision `pc ≥ p` already satisfies the
    /// `2^(−p)` contract, so it is returned directly.
    pub fn eval(&self, p: i64) -> BigRational {
        if let Some((pc, qc)) = self.memo.borrow().as_ref() {
            if *pc >= p {
                return qc.clone();
            }
        }
        let q = (self.approx)(p);
        *self.memo.borrow_mut() = Some((p, q.clone()));
        q
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
            r = CReal::mul(&r, self);
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

    /// An exponent `L` with `|x| ≥ 2^(−L)`, found by probing until the value
    /// resolves as clearly nonzero. Panics if `x` looks like zero (division and
    /// sign of an exact zero are undecidable — a fundamental limit, not a bug).
    fn lower_bound_exp(&self) -> i64 {
        let mut k = 1i64;
        while k <= 1 << 16 {
            if self.eval(k).abs() > two() * pow2(k) {
                return k; // |x| > 2^(-k)
            }
            k *= 2;
        }
        panic!("CReal: value indistinguishable from zero (recip/sign undecidable)")
    }

    /// The sign of `x`: `+1`, `-1`, or `0` if it cannot be resolved (i.e. `x`
    /// is zero, or too close to zero to tell within the probe budget).
    fn sign_probe(&self) -> i32 {
        let mut k = 1i64;
        while k <= 1 << 16 {
            let q = self.eval(k);
            if q.abs() > two() * pow2(k) {
                return if q.is_positive() { 1 } else { -1 };
            }
            k *= 2;
        }
        0
    }

    /// Reciprocal `1/x` for `x ≠ 0`. To deliver `2^(−p)` the input is evaluated
    /// to `2^(−(p+2L+2))`, where `2^(−L)` lower-bounds `|x|` — because the
    /// sensitivity of `1/x` is `1/x²`, tiny values need correspondingly more
    /// input precision. (No premature rounding ⇒ cancellation-safe.)
    pub fn recip(&self) -> CReal {
        let a = self.clone();
        let l = a.lower_bound_exp();
        of(move |p| {
            let q = (p + 2 * l + 2).max(l + 1);
            a.eval(q).recip()
        })
    }

    /// Division `x / y` = `x · (1/y)`.
    pub fn div(&self, o: &CReal) -> CReal {
        CReal::mul(self, &o.recip())
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
    fn abs(&self) -> Self {
        CReal::abs(self)
    }
    fn sqrt(&self) -> Self {
        CReal::sqrt(self)
    }
    fn powi(&self, n: i32) -> Self {
        CReal::powi(self, n)
    }
    // dot / nrm2 use the naive trait defaults — exact, because CReal never overflows.
}

// Operator overloading, so generic numeric code reads the same over CReal as
// over f64 (`a * b`, `a - b`, `w /= nrm`, ...). The Rc clone is cheap.
impl Add for CReal {
    type Output = CReal;
    fn add(self, o: CReal) -> CReal {
        CReal::add(&self, &o)
    }
}
impl Sub for CReal {
    type Output = CReal;
    fn sub(self, o: CReal) -> CReal {
        CReal::sub(&self, &o)
    }
}
impl Mul for CReal {
    type Output = CReal;
    fn mul(self, o: CReal) -> CReal {
        CReal::mul(&self, &o)
    }
}
impl Div for CReal {
    type Output = CReal;
    fn div(self, o: CReal) -> CReal {
        CReal::div(&self, &o)
    }
}
impl Neg for CReal {
    type Output = CReal;
    fn neg(self) -> CReal {
        CReal::neg(&self)
    }
}
impl SubAssign for CReal {
    fn sub_assign(&mut self, o: CReal) {
        *self = CReal::sub(self, &o);
    }
}
impl DivAssign for CReal {
    fn div_assign(&mut self, o: CReal) {
        *self = CReal::div(self, &o);
    }
}

// Order and equality by resolving the sign of the difference. Equality of exact
// reals is only semi-decidable: an exact zero difference falls through to
// `Equal` after the probe budget, which is the honest best a machine can do.
impl PartialEq for CReal {
    fn eq(&self, o: &CReal) -> bool {
        CReal::sub(self, o).sign_probe() == 0
    }
}
impl PartialOrd for CReal {
    fn partial_cmp(&self, o: &CReal) -> Option<Ordering> {
        Some(match CReal::sub(self, o).sign_probe() {
            -1 => Ordering::Less,
            1 => Ordering::Greater,
            _ => Ordering::Equal,
        })
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
        let sumsq = x.clone() * x.clone() + x.clone() * x.clone(); // 2e400, exact
        let norm = sumsq.sqrt(); // ≈ 1.414e200, representable
        assert!(close(norm.extract(), 1e200 * 2f64.sqrt(), 1e-14));
    }

    #[test]
    fn division_is_correct() {
        // 1/3 to full precision
        assert!(close((CReal::from_i64(1) / CReal::from_i64(3)).extract(), 1.0 / 3.0, 1e-15));
        // 22/7, and a tiny-denominator case that stresses the recip modulus
        assert!(close((CReal::from_i64(22) / CReal::from_i64(7)).extract(), 22.0 / 7.0, 1e-15));
        let tiny = CReal::from_f64(1e-100);
        assert!(close((CReal::from_i64(1) / tiny).extract(), 1e100, 1e-12));
        // (a/b)*b == a
        let (a, b) = (CReal::from_f64(2.5), CReal::from_f64(0.7));
        assert!(close(((a.clone() / b.clone()) * b.clone()).extract(), 2.5, 1e-14));
    }

    #[test]
    fn ordering_and_sign_resolve() {
        assert!(CReal::from_f64(0.1) < CReal::from_f64(0.2));
        assert!(CReal::from_i64(5) > CReal::from_i64(3));
        assert!(CReal::from_f64(2.0).sqrt() > CReal::from_i64(1));
        // operators compose as expected
        let x = CReal::from_f64(3.0);
        let y = CReal::from_f64(4.0);
        assert!(close((x.clone() * x + y.clone() * y).sqrt().extract(), 5.0, 1e-15));
    }

    #[test]
    fn tiny_norm_keeps_relative_accuracy() {
        // The underflow-of-significance case that caught the fixed-precision bug.
        for &a in &[1e-200_f64, 1e-150, 1e-300] {
            let x = CReal::from_f64(a);
            let norm = (x.clone() * x.clone() + x.clone() * x.clone()).sqrt(); // a·√2
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
