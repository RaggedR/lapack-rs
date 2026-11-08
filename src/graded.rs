//! A graded expression DSL: **the homogeneity degree lives in the type, and the
//! compiler uses it to insert safe scaling automatically.**
//!
//! This is the speculative payoff of the whole conversation. An expression has
//! a type `Expr<D>` where `D` is a *type-level integer* (via `typenum`) — its
//! degree under the exact-scaling action G = 2^ℤ. The operators compose the
//! grading, exactly as a graded ring would:
//!
//! | construct        | degree rule                     |
//! |------------------|---------------------------------|
//! | `var(i)`         | 1   (inputs scale with the action) |
//! | `konst(c)`       | 0   (constants are fixed)       |
//! | `a + b`, `a - b` | **requires `deg a = deg b`**, result that degree |
//! | `a * b`          | `deg a + deg b`                 |
//! | `a / b`          | `deg a - deg b`                 |
//! | `a.sqrt()`       | `deg a / 2` (**only if even**)  |
//!
//! The `+` rule is the load-bearing one. `x*x + y*y` type-checks (both degree
//! 2); `x*x + konst(5.0)` does **not compile** (degree 2 vs 0). That rejection
//! is not pedantry: `x² + 5` is genuinely *not* homogeneous, so the
//! fundamental-domain trick would give a wrong answer for it. The type system
//! admits exactly the fragment on which safe scaling is a theorem.
//!
//! [`Expr::eval_safe`] then reads the degree *off the type* (`D::to_i32()`) and
//! performs one fundamental-domain reduction + transport — the `scale_safe`
//! operator from [`crate::equivariant`], but now the user writes only the naive
//! formula and the grading obliges the result to be safe.

use crate::equivariant::{ilogb2, ldexp};
use crate::real::CReal;
use crate::Scalar;
use core::marker::PhantomData;
use core::ops::{Add, Div, Mul, Neg, Sub};
use std::rc::Rc;
use typenum::operator_aliases::{Diff, PartialQuot, Prod, Sum};
use typenum::type_operators::PartialDiv;
use typenum::{Integer, P1, P2, Z0};

/// Degree-erased abstract syntax. The degree lives only in `Expr<D>`'s type.
enum Node {
    Var(usize),
    Const(f64),
    Add(Rc<Node>, Rc<Node>),
    Sub(Rc<Node>, Rc<Node>),
    Mul(Rc<Node>, Rc<Node>),
    Div(Rc<Node>, Rc<Node>),
    Sqrt(Rc<Node>),
    Neg(Rc<Node>),
    Abs(Rc<Node>),
    Powi(Rc<Node>, i32),
}

/// An expression of homogeneity degree `D` (a `typenum` integer).
pub struct Expr<D> {
    node: Rc<Node>,
    _d: PhantomData<D>,
}

// Hand-written so we don't pick up a spurious `D: Clone` bound from derive.
impl<D> Clone for Expr<D> {
    fn clone(&self) -> Self {
        Expr {
            node: self.node.clone(),
            _d: PhantomData,
        }
    }
}

impl<D> Expr<D> {
    fn wrap(n: Node) -> Self {
        Expr {
            node: Rc::new(n),
            _d: PhantomData,
        }
    }
}

/// An input slot — degree 1, because inputs carry the scaling action.
pub fn var(slot: usize) -> Expr<P1> {
    Expr::wrap(Node::Var(slot))
}

/// A literal constant — degree 0, because it does not move under scaling.
pub fn konst(c: f64) -> Expr<Z0> {
    Expr::wrap(Node::Const(c))
}

// --- the grading, enforced by the operator type signatures ------------------

impl<D> Add for Expr<D> {
    type Output = Expr<D>; // addition is homogeneous ONLY between equal degrees
    fn add(self, rhs: Expr<D>) -> Expr<D> {
        Expr::wrap(Node::Add(self.node, rhs.node))
    }
}

impl<D> Sub for Expr<D> {
    type Output = Expr<D>;
    fn sub(self, rhs: Expr<D>) -> Expr<D> {
        Expr::wrap(Node::Sub(self.node, rhs.node))
    }
}

impl<D> Neg for Expr<D> {
    type Output = Expr<D>;
    fn neg(self) -> Expr<D> {
        Expr::wrap(Node::Neg(self.node))
    }
}

impl<A, B> Mul<Expr<B>> for Expr<A>
where
    A: Add<B>,
{
    type Output = Expr<Sum<A, B>>; // degrees add
    fn mul(self, rhs: Expr<B>) -> Expr<Sum<A, B>> {
        Expr::wrap(Node::Mul(self.node, rhs.node))
    }
}

impl<A, B> Div<Expr<B>> for Expr<A>
where
    A: Sub<B>,
{
    type Output = Expr<Diff<A, B>>; // degrees subtract
    fn div(self, rhs: Expr<B>) -> Expr<Diff<A, B>> {
        Expr::wrap(Node::Div(self.node, rhs.node))
    }
}

impl<D> Expr<D>
where
    D: PartialDiv<P2>, // `PartialDiv` is implemented ONLY when 2 divides D exactly
{
    /// √ of an expression: halves the degree, and only type-checks when the
    /// degree is even (so `√(x·x)` is fine, `√x` is a compile error).
    pub fn sqrt(self) -> Expr<PartialQuot<D, P2>> {
        Expr::wrap(Node::Sqrt(self.node))
    }
}

impl<D> Expr<D> {
    /// Absolute value — degree-preserving, since |2ᵏ·x| = 2ᵏ·|x| for 2ᵏ > 0.
    pub fn abs(self) -> Expr<D> {
        Expr::wrap(Node::Abs(self.node))
    }

    /// Square — degree doubles. (`x.square()` instead of `x.clone()*x`.)
    pub fn square(self) -> Expr<Sum<D, D>>
    where
        D: Add<D>,
    {
        let n = self.node.clone();
        Expr::wrap(Node::Mul(n.clone(), n))
    }

    /// Integer power with a *type-level* exponent: degree becomes D·N.
    /// e.g. `var(0).powi::<P3>()` has degree 3.
    pub fn powi<N: Integer>(self) -> Expr<Prod<D, N>>
    where
        D: Mul<N>,
    {
        Expr::wrap(Node::Powi(self.node, N::to_i32()))
    }
}

// --- evaluation -------------------------------------------------------------

// Evaluation is generic over the carrier `S`: run it over `f64` (the leaky,
// finite carrier — needs safe scaling) or over `CReal` (exact, arbitrary
// precision — needs none). Same syntax tree, same traversal.
fn eval_node<S: Scalar>(n: &Node, inp: &[S]) -> S {
    match n {
        Node::Var(i) => inp[*i].clone(),
        Node::Const(c) => S::from_f64(*c),
        Node::Add(a, b) => eval_node::<S>(a, inp) + eval_node::<S>(b, inp),
        Node::Sub(a, b) => eval_node::<S>(a, inp) - eval_node::<S>(b, inp),
        Node::Mul(a, b) => eval_node::<S>(a, inp) * eval_node::<S>(b, inp),
        Node::Div(a, b) => eval_node::<S>(a, inp) / eval_node::<S>(b, inp),
        Node::Sqrt(a) => eval_node::<S>(a, inp).sqrt(),
        Node::Neg(a) => -eval_node::<S>(a, inp),
        Node::Abs(a) => eval_node::<S>(a, inp).abs(),
        Node::Powi(a, n) => eval_node::<S>(a, inp).powi(*n),
    }
}

impl<D: Integer> Expr<D> {
    /// The degree, recovered at runtime from the type.
    pub fn degree(&self) -> i32 {
        D::to_i32()
    }

    /// Naive evaluation over `f64` — the formula as written. Overflows/underflows.
    pub fn eval_naive(&self, inputs: &[f64]) -> f64 {
        eval_node::<f64>(&self.node, inputs)
    }

    /// Exact evaluation over the ℝ-native carrier: no scaling, no overflow.
    /// `float` appears only when you [`CReal::extract`] the result. This is the
    /// same expression as `eval_safe`, but safety is a non-issue because the
    /// arithmetic never leaves ℝ until the final observation.
    pub fn eval_exact(&self, inputs: &[CReal]) -> CReal {
        eval_node::<CReal>(&self.node, inputs)
    }

    /// Safe evaluation. Because the *type* certifies this expression is
    /// homogeneous of degree `D`, one fundamental-domain reduction of the
    /// inputs followed by a transport by `2^(e·D)` is guaranteed correct — and
    /// the degree is read straight off the type. The user wrote no scaling code.
    pub fn eval_safe(&self, inputs: &[f64]) -> f64 {
        let m = inputs.iter().fold(0.0_f64, |a, &v| a.max(v.abs()));
        if m == 0.0 || !m.is_finite() {
            return eval_node(&self.node, inputs);
        }
        let e = ilogb2(m);
        let scaled: Vec<f64> = inputs.iter().map(|&v| ldexp(v, -e)).collect();
        let raw = eval_node::<f64>(&self.node, &scaled);
        ldexp(raw, e * D::to_i32())
    }
}

// --- rung 3: the grading as a compile-time codegen pass ---------------------

fn render(n: &Node) -> String {
    match n {
        Node::Var(i) => format!("x{i}"),
        Node::Const(c) => format!("{c}"),
        Node::Add(a, b) => format!("({} + {})", render(a), render(b)),
        Node::Sub(a, b) => format!("({} - {})", render(a), render(b)),
        Node::Mul(a, b) => format!("({} * {})", render(a), render(b)),
        Node::Div(a, b) => format!("({} / {})", render(a), render(b)),
        Node::Sqrt(a) => format!("sqrt({})", render(a)),
        Node::Neg(a) => format!("(-{})", render(a)),
        Node::Abs(a) => format!("abs({})", render(a)),
        Node::Powi(a, k) => format!("{}^{}", render(a), k),
    }
}

impl<D: Integer> Expr<D> {
    /// The naive formula as source text.
    pub fn show(&self) -> String {
        render(&self.node)
    }

    /// Emit the safe-scaled routine the grading obliges — **codegen driven by
    /// the type-level degree `D`**. The user never wrote the scaling; the
    /// compiler derives it from the homogeneity the type already proved. This
    /// is `scale_safe` moved from a runtime combinator to a compile-time pass.
    pub fn compile_safe(&self) -> String {
        format!(
            "fn safe(x: &[f64]) -> f64 {{\n    \
             let e = ilogb2(max_abs(x));       // reduce into the fundamental domain\n    \
             let x = rescale(x, -e);           // exact power-of-two shift\n    \
             let y = {body};\n    \
             ldexp(y, e * {d})                 // transport back by degree {d} (= ρ)\n}}",
            body = render(&self.node),
            d = D::to_i32(),
        )
    }
}

#[cfg(test)]
mod unit {
    use super::*;

    #[test]
    fn degrees_compose_through_the_type() {
        // norm(x, y) = sqrt(x*x + y*y) : degree (1+1)/2 = 1
        let norm = (var(0) * var(0) + var(1) * var(1)).sqrt();
        assert_eq!(norm.degree(), 1);

        // a Givens cosine c = x / sqrt(x*x + y*y) : degree 1 - 1 = 0
        let r = (var(0) * var(0) + var(1) * var(1)).sqrt();
        let c = var(0) / r;
        assert_eq!(c.degree(), 0);
    }

    #[test]
    fn extended_ops_carry_the_right_degrees() {
        use typenum::P3;
        assert_eq!(var(0).square().degree(), 2);
        assert_eq!(var(0).powi::<P3>().degree(), 3);
        assert_eq!(var(0).abs().degree(), 1);

        // norm via square(): sqrt(x² + y²)
        let norm = (var(0).square() + var(1).square()).sqrt();
        assert_eq!(norm.degree(), 1);
        assert_eq!(norm.eval_safe(&[3.0, 4.0]), 5.0);
        assert!(norm.eval_safe(&[1e200, 1e200]).is_finite());

        // x³ is homogeneous of degree 3: safe-eval transports by 2^(3e).
        let cube = var(0).powi::<P3>();
        assert_eq!(cube.eval_safe(&[2.0]), 8.0);
    }

    #[test]
    fn compile_safe_bakes_in_the_type_level_degree() {
        let norm = (var(0).square() + var(1).square()).sqrt();
        let src = norm.compile_safe();
        // the generated transport uses the degree the type inferred (1)
        assert!(src.contains("e * 1"), "generated:\n{src}");
        assert!(src.contains("sqrt(((x0 * x0) + (x1 * x1)))"), "generated:\n{src}");

        // a degree-3 expression bakes in e * 3, mechanically
        use typenum::P3;
        let cube = var(0).powi::<P3>();
        assert!(cube.compile_safe().contains("e * 3"));
    }
}
