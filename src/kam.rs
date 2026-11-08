//! KAM tori of Hénon's area-preserving quadratic twist map — the dynamics
//! layer, written once over any [`Scalar`] carrier.
//!
//! # Why this map (and not the sine standard map)
//!
//! The canonical KAM test-bed is Chirikov's standard map `p' = p + K·sin θ`.
//! But `sin` is not in the [`Scalar`] trait, and neither carrier has it: a
//! generic `sin` would be a `CReal`-only bolt-on *and* a truncated Taylor
//! series — re-introducing approximation at the very spot the exact carrier
//! exists to remove. So we use Hénon's 1969 **polynomial** area-preserving map,
//! which needs only `+ − × ` and therefore runs `::<f64>` and `::<CReal>` from
//! one body, exactly like the linear-algebra routines.
//!
//! With the Pythagorean rotation `(c, s) = (3/5, 4/5)` the map has **rational
//! coefficients**, so the orbit of a rational seed is a sequence of *exact
//! rationals* over `CReal` — the floating-point haunting appears only at
//! `extract`. And it has two fixed points at rational coordinates whose Greene
//! residues are the exact rationals `+1/5` (the elliptic centre at the origin)
//! and `−1/5` (the hyperbolic point at `(1, 1/2)`): a hand-checkable anchor.
//!
//! # What "modelling KAM tori" needs — and does not
//!
//! Everything here is the **trace** of a 2×2 symplectic matrix; no
//! (non-symmetric) eigensolver is required. Elliptic vs hyperbolic is decided
//! by `|tr J| < 2` vs `> 2`, i.e. by the residue `R = (2 − tr)/4`. Torus vs
//! chaos is read from the largest Lyapunov exponent, and in 2D area-preserving
//! flow the two exponents sum to zero, so a *single* tangent vector suffices —
//! Gram–Schmidt (needed only for the full spectrum in ≥3D) never appears.

use crate::comonad::{iterate, Stream};
use crate::Scalar;

/// A point in the `(x, y)` phase plane.
#[derive(Clone)]
pub struct Pt<T> {
    pub x: T,
    pub y: T,
}

/// The `(3/5, 4/5)` rotation, built from exact integer ratios. Because `3.0`,
/// `4.0`, `5.0` are all exactly representable in `f64`, `from_f64` is exact on
/// both carriers; the *division* is then exact over `CReal` (`3/5`, `4/5`) and
/// merely nearest-double over `f64`. This is how the map stays rational over
/// the exact carrier while remaining ordinary generic code.
fn rotation<T: Scalar>() -> (T, T) {
    let c = T::from_f64(3.0) / T::from_f64(5.0);
    let s = T::from_f64(4.0) / T::from_f64(5.0);
    (c, s)
}

/// One step of Hénon's area-preserving quadratic twist map:
///
/// ```text
/// v  = y − x²
/// x' = c·x − s·v
/// y' = s·x + c·v
/// ```
///
/// The nonlinearity is the single `x²`; the rotation supplies the twist.
pub fn henon_step<T: Scalar>(p: &Pt<T>) -> Pt<T> {
    let (c, s) = rotation::<T>();
    let v = p.y.clone() - p.x.clone() * p.x.clone();
    Pt {
        x: c.clone() * p.x.clone() - s.clone() * v.clone(),
        y: s * p.x.clone() + c * v,
    }
}

/// The Jacobian (tangent map) `∂(x',y')/∂(x,y)` at `p`, as a 2×2 row-major
/// matrix:
///
/// ```text
/// J = [ c + 2s·x    −s    ]
///     [ s − 2c·x     c    ]
/// ```
///
/// Its determinant is `c² + s² = 1` for every `x` — the map is symplectic
/// (area-preserving) *exactly*, which [`det2`] confirms bit-for-bit over
/// `CReal`.
pub fn jacobian<T: Scalar>(p: &Pt<T>) -> Vec<Vec<T>> {
    let (c, s) = rotation::<T>();
    let twox = T::from_f64(2.0) * p.x.clone();
    vec![
        vec![c.clone() + s.clone() * twox.clone(), -s.clone()],
        vec![s - c.clone() * twox, c],
    ]
}

/// Trace of a 2×2 matrix.
pub fn trace2<T: Scalar>(m: &[Vec<T>]) -> T {
    m[0][0].clone() + m[1][1].clone()
}

/// Determinant of a 2×2 matrix.
pub fn det2<T: Scalar>(m: &[Vec<T>]) -> T {
    m[0][0].clone() * m[1][1].clone() - m[0][1].clone() * m[1][0].clone()
}

/// Greene's residue `R = (2 − tr M)/4` of a 2×2 symplectic matrix. For the
/// monodromy of a periodic orbit: `0 < R < 1` elliptic (on a torus), `R < 0`
/// or `R > 1` hyperbolic (chaotic), `R ∈ {0, 1}` parabolic (bifurcation).
pub fn residue<T: Scalar>(m: &[Vec<T>]) -> T {
    (T::from_f64(2.0) - trace2(m)) / T::from_f64(4.0)
}

/// Convenience: the Greene residue of a fixed point `p` (its own Jacobian is
/// the monodromy of the period-1 orbit).
pub fn fixed_point_residue<T: Scalar>(p: &Pt<T>) -> T {
    residue(&jacobian(p))
}

/// The phase-space orbit from `seed` as a comonadic stream — the co-Kleisli
/// iteration of [`henon_step`].
pub fn orbit<T: Scalar + 'static>(seed: Pt<T>) -> Stream<Pt<T>> {
    iterate(henon_step::<T>, seed)
}

// ---------------------------------------------------------------------------
// Tangent dynamics for the Lyapunov exponent
// ---------------------------------------------------------------------------

/// A phase point carrying a tangent vector `w` — the state of the linearised
/// flow alongside the orbit.
#[derive(Clone)]
pub struct Tan<T> {
    pub p: Pt<T>,
    pub w: [T; 2],
}

/// One step of the augmented (point + tangent) flow: advance the point by the
/// map and the tangent by the Jacobian, `w ↦ J(p)·w`. **No renormalisation** —
/// over `CReal` the tangent grows in rational height but never overflows, so
/// the naive thing just works; the `f64` path needs the rescaling in
/// [`lyapunov_renorm`] to stay finite.
pub fn tangent_step<T: Scalar>(t: &Tan<T>) -> Tan<T> {
    let j = jacobian(&t.p);
    let w0 = j[0][0].clone() * t.w[0].clone() + j[0][1].clone() * t.w[1].clone();
    let w1 = j[1][0].clone() * t.w[0].clone() + j[1][1].clone() * t.w[1].clone();
    Tan { p: henon_step(&t.p), w: [w0, w1] }
}

/// The augmented trajectory from `seed` with the unit tangent `(1, 0)`.
pub fn tangent_orbit<T: Scalar + 'static>(seed: Pt<T>) -> Stream<Tan<T>> {
    iterate(tangent_step::<T>, Tan { p: seed, w: [T::from_f64(1.0), T::zero()] })
}

/// **Naive** largest Lyapunov exponent: evolve one tangent vector for `n` steps
/// *without* renormalising, then `λ ≈ ln‖wₙ‖ / n` — the single transcendental
/// at the very end, mirroring `nrm2`'s "haunting only at `extract`" story.
///
/// Returns `‖wₙ‖` as a carrier value; the caller takes the `ln` (which no
/// carrier provides) after collapsing to `f64`. Over `f64` the norm **overflows
/// to ∞** once the orbit is chaotic — the naive leak. Over `CReal` it is exact
/// for any `n` the DAG depth allows.
pub fn tangent_growth<T: Scalar + 'static>(seed: Pt<T>, n: usize) -> T {
    let mut t = Tan { p: seed, w: [T::from_f64(1.0), T::zero()] };
    for _ in 0..n {
        t = tangent_step(&t);
    }
    T::nrm2(&t.w)
}

/// **Safe** largest Lyapunov exponent over `f64`: renormalise the tangent each
/// step and accumulate `ln` of the stretching, so nothing overflows however
/// chaotic the orbit. This is the `f64`-specific counterpart to the naive
/// generic [`tangent_growth`] — the same split as `nrm2_naive` vs `nrm2`.
pub fn lyapunov_renorm(seed: Pt<f64>, n: usize) -> f64 {
    let mut t = Tan { p: seed, w: [1.0, 0.0] };
    let mut sum = 0.0;
    for _ in 0..n {
        t = tangent_step(&t);
        let norm = f64::nrm2(&t.w);
        sum += norm.ln();
        t.w = [t.w[0] / norm, t.w[1] / norm];
    }
    sum / n as f64
}
