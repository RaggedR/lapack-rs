//! A faithful, idiomatic Rust reimplementation of LAPACK/BLAS routines,
//! grown one routine at a time and validated against independent oracles.
//!
//! The guiding rule of this project: **the correctness lives in the numerics,
//! not in the control flow.** A routine is only "reimplemented" once it matches
//! a reference computation to within a backward-error bound across adversarial
//! inputs — not merely once it compiles and returns a plausible number.
//!
//! First routine: [`nrm2`], the Euclidean norm ‖x‖₂.

use core::ops::{Add, Div, Mul};

pub mod equivariant;
pub mod graded;
pub mod matrix;
pub mod real;
pub mod tensor;

/// A numeric scalar the graded DSL can evaluate over. Implemented by `f64` (the
/// leaky, finite carrier) and by [`real::CReal`] (the exact, arbitrary-precision
/// carrier). The *same* [`graded::Expr`] runs over either — which is the whole
/// point: over `f64` it needs safe scaling, over `CReal` it needs none.
pub trait Scalar: Clone {
    fn from_f64(x: f64) -> Self;
    fn add(&self, o: &Self) -> Self;
    fn sub(&self, o: &Self) -> Self;
    fn mul(&self, o: &Self) -> Self;
    fn div(&self, o: &Self) -> Self;
    fn neg(&self) -> Self;
    fn abs(&self) -> Self;
    fn sqrt(&self) -> Self;
    fn powi(&self, n: i32) -> Self;
}

impl Scalar for f64 {
    fn from_f64(x: f64) -> Self {
        x
    }
    fn add(&self, o: &Self) -> Self {
        self + o
    }
    fn sub(&self, o: &Self) -> Self {
        self - o
    }
    fn mul(&self, o: &Self) -> Self {
        self * o
    }
    fn div(&self, o: &Self) -> Self {
        self / o
    }
    fn neg(&self) -> Self {
        -self
    }
    fn abs(&self) -> Self {
        (*self).abs()
    }
    fn sqrt(&self) -> Self {
        (*self).sqrt()
    }
    fn powi(&self, n: i32) -> Self {
        (*self).powi(n)
    }
}

/// The scalar interface the routines need: ordered arithmetic plus the
/// safe-scaling constants, which are *derived from the floating-point format's
/// exponent range* and therefore differ per type.
///
/// This is the generic that replaces LAPACK's four hand-copied precision
/// variants (`s`/`d`/`c`/`z`). One algorithm, specialised by the associated
/// constants below.
pub trait Real:
    Copy + PartialOrd + Add<Output = Self> + Mul<Output = Self> + Div<Output = Self>
{
    const ZERO: Self;
    const ONE: Self;

    /// Values `> TBIG` are scaled down by `SBIG` before squaring (anti-overflow).
    const TBIG: Self;
    /// Values `< TSML` are scaled up by `SSML` before squaring (anti-underflow).
    const TSML: Self;
    /// Scale-up factor for the small bin.
    const SSML: Self;
    /// Scale-down factor for the big bin.
    const SBIG: Self;

    fn abs(self) -> Self;
    fn sqrt(self) -> Self;
    fn is_nan(self) -> bool;
}

impl Real for f64 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    // Derived for IEEE binary64 (radix 2, minexp -1021, maxexp 1024, digits 53):
    //   TBIG = 2^floor((1024-53+1)/2) = 2^486
    //   TSML = 2^ceil((-1021-1)/2)    = 2^-511
    //   SSML = 2^-floor((-1021-53)/2) = 2^537
    //   SBIG = 2^-ceil((1024+53-1)/2) = 2^-538
    // (These decimal literals round to exactly those powers of two; see tests.)
    const TBIG: Self = 1.997_919_072_202_235_0e+146;
    const TSML: Self = 1.491_668_146_240_041_3e-154;
    const SSML: Self = 4.498_913_794_543_196_4e+161;
    const SBIG: Self = 1.111_379_374_742_538_7e-162;

    #[inline]
    fn abs(self) -> Self {
        self.abs()
    }
    #[inline]
    fn sqrt(self) -> Self {
        self.sqrt()
    }
    #[inline]
    fn is_nan(self) -> bool {
        self.is_nan()
    }
}

impl Real for f32 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    // Derived for IEEE binary32 (radix 2, minexp -125, maxexp 128, digits 24):
    //   TBIG = 2^52, TSML = 2^-63, SSML = 2^75, SBIG = 2^-76
    const TBIG: Self = 4.503_599_6e+15;
    const TSML: Self = 1.084_202_2e-19;
    const SSML: Self = 3.777_893_2e+22;
    const SBIG: Self = 1.323_489_0e-23;

    #[inline]
    fn abs(self) -> Self {
        self.abs()
    }
    #[inline]
    fn sqrt(self) -> Self {
        self.sqrt()
    }
    #[inline]
    fn is_nan(self) -> bool {
        self.is_nan()
    }
}

/// The Euclidean norm ‖x‖₂ = √(Σ xᵢ²), computed with safe scaling so it is
/// correct across the entire exponent range — no spurious overflow to `inf`
/// and no spurious underflow to `0`.
///
/// This is a direct reimplementation of Edward Anderson's three-accumulator
/// algorithm (LAPACK's `dnrm2`/`snrm2`, 2021 rewrite). Each element is binned
/// by magnitude and squared in a scale that keeps the partial sums in range;
/// the bins are then recombined.
///
/// Returns `0` for an empty slice. Propagates `NaN`/`inf` the way LAPACK does.
pub fn nrm2<T: Real>(x: &[T]) -> T {
    // Three accumulators of *sums of squares*, each in its own scale.
    let mut abig = T::ZERO; // scaled down  (for huge values)
    let mut amed = T::ZERO; // unscaled     (for mid-range values)
    let mut asml = T::ZERO; // scaled up    (for tiny values)

    // Once any genuinely huge value is seen, tiny values are negligible to the
    // norm and we stop accumulating them — matching the reference.
    let mut notbig = true;

    for &xi in x {
        let ax = xi.abs();
        if ax > T::TBIG {
            let t = ax * T::SBIG;
            abig = abig + t * t;
            notbig = false;
        } else if ax < T::TSML {
            if notbig {
                let t = ax * T::SSML;
                asml = asml + t * t;
            }
        } else {
            // Mid-range: square directly, no scaling needed.
            amed = amed + ax * ax;
        }
    }

    // Recombine the accumulators. At most two are non-empty in practice, but a
    // NaN must still be allowed to poison the result, hence the `is_nan` guards.
    let scl;
    let sumsq;
    if abig > T::ZERO {
        // Big values present: fold any mid-range contribution into the big scale.
        if amed > T::ZERO || amed.is_nan() {
            abig = abig + (amed * T::SBIG) * T::SBIG;
        }
        scl = T::ONE / T::SBIG;
        sumsq = abig;
    } else if asml > T::ZERO {
        if amed > T::ZERO || amed.is_nan() {
            // Both small and mid-range present: combine as magnitudes, carefully,
            // via ymax·√(1 + (ymin/ymax)²) to avoid re-introducing overflow.
            let amed_mag = amed.sqrt();
            let asml_mag = asml.sqrt() / T::SSML;
            let (ymin, ymax) = if asml_mag > amed_mag {
                (amed_mag, asml_mag)
            } else {
                (asml_mag, amed_mag)
            };
            scl = T::ONE;
            let r = ymin / ymax;
            sumsq = ymax * ymax * (T::ONE + r * r);
        } else {
            scl = T::ONE / T::SSML;
            sumsq = asml;
        }
    } else {
        // All mid-range (the common case) or all zero.
        scl = T::ONE;
        sumsq = amed;
    }

    scl * sumsq.sqrt()
}

/// The naive textbook norm: √(Σ xᵢ²) with a single unscaled accumulator.
///
/// Kept deliberately as a foil for [`nrm2`]: it overflows to `inf` for large
/// inputs and underflows to `0` for tiny ones, even when the true norm is a
/// perfectly representable number. The tests demonstrate exactly that.
pub fn nrm2_naive<T: Real>(x: &[T]) -> T {
    let mut s = T::ZERO;
    for &xi in x {
        s = s + xi * xi;
    }
    s.sqrt()
}
