# lapack_rs

An idiomatic Rust reimplementation of LAPACK/BLAS routines, grown **one routine
at a time** and validated against independent oracles.

This is not a mechanical port of the Fortran (that already exists — `f2c`'s
CLAPACK — and nobody loves it). It is a from-scratch reimplementation in the
spirit of [`faer`](https://github.com/sarah-quinones/faer-rs): re-express the
*algorithms* natively, with generics instead of four copy-pasted precision
variants, and prove faithfulness by test rather than by assertion.

## Guiding rule

> The correctness lives in the numerics, not the control flow.

A routine is "reimplemented" only once it matches a reference computation to
within a backward-error bound across adversarial inputs — not merely once it
compiles and returns a plausible number.

## Three layers

The repo grew a small theory of *why* these routines look the way they do:

1. **`src/lib.rs` — the routine.** A faithful three-bin `nrm2`, generic over
   `f32`/`f64`, validated against independent oracles.
2. **`src/equivariant.rs` — the trick, named.** Safe scaling is an *equivariant
   extension*: the exact-scaling group G = 2^ℤ acts losslessly on floats, and a
   safe routine is the unique G-equivariant extension of the naive kernel from
   the "safe window" (a fundamental domain). A single operator `scale_safe`,
   parameterised by the representation ρ (a list of output *degrees*), derives
   `nrm2`, `hypot`, and Givens `rotg` from their naive kernels. The 4096-case
   test shows this single-scale extension reaches the *same* correctness as the
   hand-written three-bin code — so the three-bin machinery was only ever about
   speed, not safety.
3. **`src/graded.rs` — the trick, in the type system.** A graded expression DSL
   where the homogeneity degree lives in the type (via `typenum`). Operators
   compose the grading (`×` adds degrees, `÷` subtracts, `√` halves, `powi`
   multiplies, `+`/`abs` preserve, `+` demands equal degrees); `eval_safe` reads
   the degree *off the type* and inserts the scaling automatically.
   Un-homogeneous formulas like `x*x + 5` are a **compile error** — exactly the
   expressions where safe scaling would be unsound (see
   `tests/compile_fail_notes.md`). `compile_safe()` emits the safe routine as
   source, with the transport degree baked in from the type — the grading as a
   codegen pass.
4. **`src/real.rs` — the ℝ-native alternative.** A computable real `CReal` as a
   lazy refinable process (a rational approximation to any requested precision —
   the interval-domain refiner / comonadic stream). `extract` is the counit:
   collapse to one `f64`. Because arithmetic is arbitrary-precision, **nothing
   overflows** — `(1e200)²` is just a 400-digit integer — so the safe-scaling
   apparatus is unnecessary; the floating-point haunting reappears only at
   `extract`. A `Scalar` trait lets the *same* `graded::Expr` run over `f64`
   (needs scaling, `eval_safe`) or over `CReal` (needs none, `eval_exact`).
   This is the "float-native vs ℝ-native" distinction made executable.

Demo of layers 3–4: `cargo run --example graded_demo`.

## Routines so far

| Routine | LAPACK name | Status | Oracle |
|---------|-------------|--------|--------|
| `nrm2`  | `snrm2`/`dnrm2` | ✅ | scale-by-max (f64) · f64 naive (f32) |
| `hypot` | — (BLAS-adjacent) | ✅ | via `equivariant` |
| `rotg`  | `srotg`/`drotg` (no sign/z convention) | ✅ | Givens identities |
| `dot`   | `sdot`/`ddot` | ✅ | bilinear, ρ=⟨1;1⟩ · cancellation-overflow test |
| `matmul`| `gemm` (no α/β) | ✅ | bilinear, ρ=⟨1;1⟩ · independent block scales |
| `gram_schmidt` | `geqrf`-adjacent (MGS) | ✅ | Q orthonormal & degree-0, R degree-1 |
| `jacobi_eigen` | `syev` (symmetric) | ✅ | A=VΛVᵀ · λ degree-1, V degree-0 |

### Multigrading (`src/matrix.rs`, `src/tensor.rs`)

`dot` and `matmul` are **bilinear**, so a single scaling action is not enough —
each argument block scales independently (ρ = ⟨1; 1⟩) and the result transports
by the *sum* of the block exponents. `gram_schmidt` and `jacobi_eigen` are not
single-scale homogeneous; they are built from the safe primitives, and the
grading surfaces in their outputs — an orthonormal basis is degree 0, and a
symmetric matrix's eigenvalues are degree 1 while its eigenvectors are degree 0.

`src/matrix.rs` writes the scaling for `dot`/`matmul` **by hand**.
`src/tensor.rs` instead expresses them in a **multigraded DSL**: an expression
carries a *pair* of type-level degrees `Expr<Dx, Dy>` (one per block), operators
compose the pair componentwise, and `eval_safe` does one fundamental-domain
reduction per block, reading the degrees off the type. So `dot` becomes the
naive formula with no scaling in sight —

```rust
pub fn dot(x: &[f64], y: &[f64]) -> f64 {
    let expr = sum(xvar() * yvar());   // bidegree ⟨1,1⟩, inferred
    expr.eval_safe(x, y)               // the block_exp/ldexp is generated
}
```

— and `matmul` is a composition over that DSL `dot`. Mixing blocks additively
(`xvar() + yvar()`, ⟨1,0⟩ + ⟨0,1⟩) is a compile error
(`tests/compile_fail_notes.md`). The hand-written `matrix` versions stay as the
test oracles.

### `nrm2` — Euclidean norm ‖x‖₂

A direct reimplementation of Edward Anderson's three-accumulator algorithm
(*Algorithm 978: Safe Scaling in the Level-1 BLAS*, the 2021 LAPACK rewrite).
Each element is binned by magnitude — **big** values scaled down, **small**
values scaled up, **mid-range** left alone — so the partial sums never overflow
or underflow, then the bins are recombined.

The four scaling constants (`TBIG`, `TSML`, `SSML`, `SBIG`) are *derived from
each float type's exponent range*, so they live as associated constants on a
`Real` trait — one generic algorithm specialising to `f32` and `f64`.

See the whole argument in one screen:

```
cargo run --example overflow_demo
```

```
a = 1e200  (huge):   naive = inf   nrm2 = 1.414…e200   (true: 1.414…e200)
a = 1e-200 (tiny):   naive = 0     nrm2 = 1.414…e-200  (true: 1.414…e-200)
```

## Test

```
cargo test
```

Includes two property tests (4096 random cases each) over wide-dynamic-range
vectors, plus headline demos, exact cases (3-4-5), and NaN/inf propagation.
