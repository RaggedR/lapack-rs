# Benchmarks

All measured on one Apple-silicon machine, `--release` / `-O2`, single-threaded.
LAPACK/BLAS is Apple **Accelerate**; the exact-arithmetic peer is **Arb**
(via FLINT 3.6). Numbers are per-call averages; treat them as order-of-magnitude
(small-matrix timings include fixed call overhead).

## Reproduce

```sh
# ours
cargo run --release --example bench

# LAPACK (Accelerate)
clang -O2 -framework Accelerate benchmarks/lapack_n4.c -o /tmp/lpk && /tmp/lpk

# Arb (needs FLINT: `brew install flint`)
clang -O2 $(pkg-config --cflags flint) benchmarks/arb_n4.c \
  -o /tmp/arb $(pkg-config --libs flint) && /tmp/arb
```

## Our f64 routines vs LAPACK (varying n)

| n | | norm (BLAS-1) | matmul (BLAS-3) | QR |
|---|---|---|---|---|
| 10  | LAPACK | 0.012 µs | 0.17 µs | 0.98 µs |
|     | ours   | 0.108 µs | 3.73 µs | 2.94 µs |
|     | slower | 9× | 22× | 3× |
| 100 | LAPACK | 0.025 µs | 22.1 µs | 156 µs |
|     | ours   | 0.148 µs | 1 547 µs | 898 µs |
|     | slower | 6× | 70× | 6× |
| 300 | LAPACK | 0.065 µs | 126 µs | 652 µs |
|     | ours   | 0.274 µs | 38 801 µs | 24 161 µs |
|     | slower | 4× | **307×** | 37× |

The **norm** gap *shrinks* with n (memory-bound; both hit the bandwidth wall);
the **matmul** gap *grows* (compute-bound; LAPACK's cache-blocking + SIMD pull
further ahead). That's the BLAS-1 vs BLAS-3 distinction, measured.

## Exact arithmetic: ours vs Arb vs LAPACK (n=4, ~64-bit)

| operation | LAPACK (f64) | Arb (exact) | ours (exact) | ours ÷ Arb | Arb ÷ LAPACK |
|---|---|---|---|---|---|
| dot | 0.006 µs | 0.072 µs | 9.07 µs | **126×** | 12× |
| matmul | 0.037 µs | 0.439 µs | 25.98 µs | **59×** | 12× |
| factorization* | 0.15 µs | 0.73 µs | 21.5 µs | **~29×** | ~5× |

\* different algorithms — LAPACK QR (`dgeqrf`), Arb LU (`arb_mat_lu`), our
Gram–Schmidt — so that row is "an O(n³) factorization," rough.

**Ladder:** LAPACK (fixed 53-bit) → ~12× → Arb (certified, float + ball + GMP)
→ ~30–125× → ours (certified, dyadic fixed-point + endpoint intervals +
pure-Rust bigint).

## The fixed-point optimization (`BigRational` → dyadic `BigInt`)

Representing interval endpoints as integer mantissas at the working-precision
exponent (removing all rational/GCD overhead) at n=4:

| op | BigRational | fixed-point | speedup |
|---|---|---|---|
| dot | 61.7 µs | 9.1 µs | 6.8× |
| matmul | 826.9 µs | 26.0 µs | 32× |
| Gram–Schmidt | 4 518.9 µs | 21.5 µs | 210× |
| diagonalization | 562 125 µs | 669 µs | **840×** |

One data-structure change; no algorithm or architecture change. The GS+Jacobi
integration tests dropped from 4.15 s to 0.01 s. This closed the gap vs LAPACK
from ~10⁴–10⁶× to ~140–1500×, i.e. to within ~30–125× of Arb.

## What remains between us and Arb

1. **Fused kernels** — Arb's `arb_dot` accumulates with one rounding; ours builds
   an interval per product (worst gap: `dot`, 126×).
2. **Ball arithmetic** (midpoint + cheap radius) vs our endpoint intervals (~2×).
3. **GMP assembly** bignum vs pure-Rust `num-bigint` (~2–5×).
4. **Float-with-exponent** (`arf`) vs our fixed-point — a large magnitude inflates
   our integer size (`1e16` → ~117-bit integer); a real float carries only the
   significant bits. Highest-leverage next step.
