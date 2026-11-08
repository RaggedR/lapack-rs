# Compile-fail evidence for the graded DSL

The grading in `src/graded.rs` rejects non-homogeneous expressions *at compile
time*, because those are exactly the expressions on which the safe-scaling
transform would be unsound. This is the point of the whole layer, so it is worth
recording that the rejection is real rather than claimed.

## The rejected program

```rust
use lapack_rs::graded::{konst, var};
fn main() {
    // x*x has degree 2; konst(5.0) has degree 0. x² + 5 is NOT homogeneous.
    let _bad = var(0) * var(0) + konst(5.0);
}
```

## The actual compiler error (rustc 1.99.1)

```
error[E0308]: mismatched types
 --> examples/_badgrade.rs:4:34
  |
4 |     let _bad = var(0) * var(0) + konst(5.0);
  |                                  ^^^^^^^^^^ expected `Expr<PInt<UInt<UInt<UTerm, B1>, B0>>>`,
  |                                             found `Expr<Z0>`
```

`PInt<UInt<UInt<UTerm, B1>, B0>>` is typenum's binary encoding of **+2**; `Z0`
is **0**. `Add for Expr<D>` is implemented only for equal `D`, so the degree-2
left operand cannot be added to the degree-0 constant. The type system admits
precisely the fragment on which `eval_safe`'s fundamental-domain transport is a
theorem.

## Reproduce

Drop the program into `examples/_badgrade.rs` and run
`cargo build --example _badgrade` — it will fail to compile. (It is not kept in
the tree because a non-compiling file would break `cargo build`; a `trybuild`
harness could pin it as a first-class test.)

## The multigraded version (`src/tensor.rs`)

The same guard holds for the bidegree. `xvar()` is ⟨1,0⟩ and `yvar()` is ⟨0,1⟩,
so adding them mixes blocks and is rejected:

```rust
use lapack_rs::tensor::{xvar, yvar, sum};
fn main() { let _bad = sum(xvar() + yvar()); }
```

```
error[E0308]: mismatched types
  |     let _bad = sum(xvar() + yvar());
  |                            ^^^^^^ expected `VExpr<PInt<UInt<UTerm, B1>>, Z0>`,
  |                                   found `VExpr<Z0, PInt<UInt<UTerm, B1>>>`
```

`VExpr<+1, 0>` versus `VExpr<0, +1>`: `Add for VExpr` is implemented only for an
equal bidegree, so an X-block vector cannot be added to a Y-block vector.
