//! A **multigraded vector DSL** — the rung that finally pulls the matrix
//! operations into the grading, so they stop carrying hand-written scaling.
//!
//! The scalar DSL in [`crate::graded`] had one degree per expression. Bilinear
//! operations need **two**: an inner product ⟨x, y⟩ scales independently in its
//! x-argument and its y-argument. So here an expression carries a pair of
//! type-level degrees, `Expr<Dx, Dy>` — degree `Dx` under scaling the X-block,
//! `Dy` under scaling the Y-block. The operators compose the pair componentwise
//! (a Hadamard product adds both; `sum` preserves both), and evaluation does
//! **two** fundamental-domain reductions — one per block — reading the degrees
//! off the type and transporting by `2^(eₓ·Dx + e_y·Dy)`.
//!
//! With this, `dot` is literally `sum(xvar() * yvar())` — the naive formula,
//! with the `block_exp`/`ldexp` machinery generated, not typed by hand. And
//! `matmul` is a composition over that DSL `dot`, exactly as Gram–Schmidt is a
//! composition over `dot`/`nrm2` — so it, too, carries no scaling code.
//!
//! Design choice (kept deliberately): only the **degree** lives in the type.
//! Vector **lengths/shapes** stay runtime values (checked with `assert_eq!`),
//! which keeps this out of dependent-type territory.

use crate::equivariant::{ilogb2, ldexp};
use crate::matrix::Matrix;
use core::marker::PhantomData;
use core::ops::{Add, Mul};
use std::rc::Rc;
use typenum::operator_aliases::Sum;
use typenum::{Integer, P1, Z0};

// --- the vector expression tree (degree-erased) -----------------------------

enum VNode {
    XVar, // the X-block input vector
    YVar, // the Y-block input vector
    Hadamard(Rc<VNode>, Rc<VNode>),
    Add(Rc<VNode>, Rc<VNode>),
}

/// A vector-valued expression of bidegree ⟨Dx, Dy⟩.
pub struct VExpr<Dx, Dy> {
    node: Rc<VNode>,
    _p: PhantomData<(Dx, Dy)>,
}

/// A scalar-valued expression of bidegree ⟨Dx, Dy⟩ (a reduced vector).
pub struct SExpr<Dx, Dy> {
    node: Rc<VNode>,
    _p: PhantomData<(Dx, Dy)>,
}

impl<Dx, Dy> Clone for VExpr<Dx, Dy> {
    fn clone(&self) -> Self {
        VExpr {
            node: self.node.clone(),
            _p: PhantomData,
        }
    }
}

impl<Dx, Dy> VExpr<Dx, Dy> {
    fn wrap(n: VNode) -> Self {
        VExpr {
            node: Rc::new(n),
            _p: PhantomData,
        }
    }
}

/// The X-block input vector — bidegree ⟨1, 0⟩.
pub fn xvar() -> VExpr<P1, Z0> {
    VExpr::wrap(VNode::XVar)
}

/// The Y-block input vector — bidegree ⟨0, 1⟩.
pub fn yvar() -> VExpr<Z0, P1> {
    VExpr::wrap(VNode::YVar)
}

// Hadamard (elementwise) product: bidegrees add componentwise.
impl<Ax, Ay, Bx, By> Mul<VExpr<Bx, By>> for VExpr<Ax, Ay>
where
    Ax: Add<Bx>,
    Ay: Add<By>,
{
    type Output = VExpr<Sum<Ax, Bx>, Sum<Ay, By>>;
    fn mul(self, rhs: VExpr<Bx, By>) -> VExpr<Sum<Ax, Bx>, Sum<Ay, By>> {
        VExpr::wrap(VNode::Hadamard(self.node, rhs.node))
    }
}

// Elementwise sum: homogeneous ⇒ both bidegrees must match (rejects mixing an
// X-block vector with a Y-block vector additively).
impl<Dx, Dy> Add for VExpr<Dx, Dy> {
    type Output = VExpr<Dx, Dy>;
    fn add(self, rhs: VExpr<Dx, Dy>) -> VExpr<Dx, Dy> {
        VExpr::wrap(VNode::Add(self.node, rhs.node))
    }
}

/// Contract a vector expression to a scalar by summing its entries. Preserves
/// the bidegree (Σ of ⟨Dx,Dy⟩ terms is ⟨Dx,Dy⟩).
pub fn sum<Dx, Dy>(v: VExpr<Dx, Dy>) -> SExpr<Dx, Dy> {
    SExpr {
        node: v.node,
        _p: PhantomData,
    }
}

// --- evaluation -------------------------------------------------------------

fn eval_vec(n: &VNode, xs: &[f64], ys: &[f64]) -> Vec<f64> {
    match n {
        VNode::XVar => xs.to_vec(),
        VNode::YVar => ys.to_vec(),
        VNode::Hadamard(a, b) => {
            let va = eval_vec(a, xs, ys);
            let vb = eval_vec(b, xs, ys);
            assert_eq!(va.len(), vb.len(), "tensor: hadamard length mismatch");
            va.iter().zip(&vb).map(|(p, q)| p * q).collect()
        }
        VNode::Add(a, b) => {
            let va = eval_vec(a, xs, ys);
            let vb = eval_vec(b, xs, ys);
            assert_eq!(va.len(), vb.len(), "tensor: add length mismatch");
            va.iter().zip(&vb).map(|(p, q)| p + q).collect()
        }
    }
}

fn maxabs(xs: &[f64]) -> f64 {
    xs.iter().fold(0.0_f64, |a, v| a.max(v.abs()))
}

fn block_exp(m: f64) -> i32 {
    if m > 0.0 && m.is_finite() {
        ilogb2(m)
    } else {
        0
    }
}

impl<Dx: Integer, Dy: Integer> SExpr<Dx, Dy> {
    /// The inferred bidegree ⟨Dx, Dy⟩, read off the type.
    pub fn degree(&self) -> (i32, i32) {
        (Dx::to_i32(), Dy::to_i32())
    }

    /// Naive reduction — overflows/NaNs just like the bare formula.
    pub fn eval_naive(&self, xs: &[f64], ys: &[f64]) -> f64 {
        eval_vec(&self.node, xs, ys).iter().sum()
    }

    /// Safe reduction. The two blocks are reduced into range **independently**
    /// (the multigrading), the naive formula runs on the in-range vectors, and
    /// the result is transported by `2^(eₓ·Dx + e_y·Dy)` — with the degrees
    /// read off the type. No scaling is written by the caller.
    pub fn eval_safe(&self, xs: &[f64], ys: &[f64]) -> f64 {
        let ex = block_exp(maxabs(xs));
        let ey = block_exp(maxabs(ys));
        let xss: Vec<f64> = xs.iter().map(|v| ldexp(*v, -ex)).collect();
        let yss: Vec<f64> = ys.iter().map(|v| ldexp(*v, -ey)).collect();
        let s: f64 = eval_vec(&self.node, &xss, &yss).iter().sum();
        ldexp(s, ex * Dx::to_i32() + ey * Dy::to_i32())
    }
}

// --- the operations, now as DSL expressions ---------------------------------

/// Inner product ⟨x, y⟩ — the naive formula `sum(x ⊙ y)`, made safe by the
/// bidegree ⟨1, 1⟩. Compare `matrix::dot`, which writes the scaling by hand.
pub fn dot(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len(), "dot: length mismatch");
    let expr = sum(xvar() * yvar()); // SExpr<P1, P1>
    expr.eval_safe(x, y)
}

/// Matrix product A·B, as a composition over the DSL [`dot`] — one bidegree-⟨1,1⟩
/// expression evaluated per entry. No scaling boilerplate here: it lives in the
/// generated `eval_safe`, exactly as Gram–Schmidt delegates to its primitives.
pub fn matmul(a: &Matrix, b: &Matrix) -> Matrix {
    assert_eq!(a.cols, b.rows, "matmul: shape mismatch");
    let expr = sum(xvar() * yvar());
    let mut c = Matrix::zeros(a.rows, b.cols);
    for i in 0..a.rows {
        let row = a.row(i);
        for j in 0..b.cols {
            let col = b.col(j);
            c.set(i, j, expr.eval_safe(row, &col));
        }
    }
    c
}

#[cfg(test)]
mod unit {
    use super::*;

    #[test]
    fn dot_expression_has_bidegree_one_one() {
        let expr = sum(xvar() * yvar());
        assert_eq!(expr.degree(), (1, 1));
    }
}
