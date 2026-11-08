//! The **stream comonad** — the engine that turns a one-step map into a
//! trajectory, and a *local* observation into a trajectory-wide one.
//!
//! A discrete dynamical system is a map `step : S → S`. Its orbit from a seed
//! `s₀` is the infinite stream `[s₀, step s₀, step² s₀, …]`. That stream is the
//! **cofree comonad** on `step`, and this module gives it the three comonad
//! operations:
//!
//! * `extract : Stream T → T`            — the counit (the head / "now"),
//! * `duplicate : Stream T → Stream (Stream T)` — every suffix ("all futures"),
//! * `extend : (Stream T → B) → Stream T → Stream B` — run a **co-Kleisli
//!   arrow** at every position.
//!
//! A co-Kleisli arrow `Stream T → B` is a *local rule*: it reads the orbit from
//! some instant onward and returns one observation (a stretching factor, an
//! area defect, a residue). `extend` slides it along the whole orbit. So "run
//! the map, then measure it everywhere" is literally `iterate` followed by
//! `extend` — the comonadic reading of a backtest, here pointed at a
//! Hamiltonian map instead of a price series.
//!
//! Laziness is by `Rc<dyn Fn() -> Stream<T>>`: the tail is a thunk, so the
//! orbit is infinite but only forced as far as it is observed. Nothing here is
//! carrier-specific — the same `Stream<T>` carries `f64` points or exact
//! [`crate::real::CReal`] points unchanged.

use std::rc::Rc;

/// A lazy, infinite stream: a head and a thunk for the rest.
pub struct Stream<T> {
    head: T,
    tail: Rc<dyn Fn() -> Stream<T>>,
}

// `Stream` is `Clone` whenever `T` is — the tail is shared (`Rc`), the head
// copied. A manual impl is needed because `#[derive(Clone)]` would wrongly
// demand `Rc<dyn Fn>: Clone` via a `T: Clone`-independent bound.
impl<T: Clone> Clone for Stream<T> {
    fn clone(&self) -> Self {
        Stream { head: self.head.clone(), tail: Rc::clone(&self.tail) }
    }
}

impl<T: Clone + 'static> Stream<T> {
    /// Cons a head onto a lazy tail.
    pub fn cons(head: T, tail: impl Fn() -> Stream<T> + 'static) -> Stream<T> {
        Stream { head, tail: Rc::new(tail) }
    }

    /// The counit ε: read the current element.
    pub fn extract(&self) -> T {
        self.head.clone()
    }

    /// Drop the head — the orbit one step on.
    pub fn tail(&self) -> Stream<T> {
        (self.tail)()
    }

    /// Force the first `n` elements.
    pub fn take(&self, n: usize) -> Vec<T> {
        let mut out = Vec::with_capacity(n);
        let mut s = self.clone();
        for _ in 0..n {
            out.push(s.extract());
            s = s.tail();
        }
        out
    }

    /// The comultiplication δ: the stream of all suffixes. `extract ∘ duplicate
    /// = id`; `duplicate`'s head is the whole original stream.
    pub fn duplicate(&self) -> Stream<Stream<T>> {
        let me = self.clone();
        Stream { head: me.clone(), tail: Rc::new(move || me.tail().duplicate()) }
    }

    /// Coextension: apply the co-Kleisli arrow `g` at every position. This is
    /// `map g ∘ duplicate` — `g` sees the orbit from each instant onward and
    /// returns one observation, and `extend` collects them into a new stream.
    pub fn extend<B, G>(&self, g: G) -> Stream<B>
    where
        B: Clone + 'static,
        G: Fn(&Stream<T>) -> B + Clone + 'static,
    {
        let me = self.clone();
        let g_tail = g.clone();
        Stream {
            head: g(&me),
            tail: Rc::new(move || me.tail().extend(g_tail.clone())),
        }
    }
}

/// The cofree orbit of a map: `[seed, step seed, step² seed, …]`.
///
/// This *is* "the co-Kleisli iteration": unfolding `step` into the stream
/// comonad. Point it at [`crate::kam::henon_step`] and you get a phase-space
/// trajectory; point it at [`crate::kam::tangent_step`] and you get the
/// trajectory with its linearised flow carried alongside (for Lyapunov).
pub fn iterate<T, F>(step: F, seed: T) -> Stream<T>
where
    T: Clone + 'static,
    F: Fn(&T) -> T + Clone + 'static,
{
    let next = step(&seed);
    Stream { head: seed, tail: Rc::new(move || iterate(step.clone(), next.clone())) }
}
