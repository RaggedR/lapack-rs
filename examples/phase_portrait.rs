//! Emit the Poincaré phase portrait of the Hénon twist map as `points.js`.
//!
//! Run with:  cargo run --example phase_portrait > points.js
//!
//! Fans a family of seeds out from the elliptic centre along y = 0, iterates
//! each with the real [`henon_step`], and prints the bounded portion of every
//! orbit as `window.ORBITS`. The structure draws itself: nested closed curves
//! are KAM tori, chains of loops are resonant islands, a scattered haze is the
//! chaotic layer, and seeds past the last torus escape and are dropped.
//!
//! The portrait runs over `f64` — here the *structure* is the point, not
//! exactness; the exact `CReal` carrier is for the measurements (det J = 1,
//! residues ±1/5) that the portrait cannot show.

use lapack_rs::kam::{orbit, Pt};

fn main() {
    let steps = 1600; // points per orbit
    let bound = 4.0; // escape threshold
    let mut orbits: Vec<Vec<(f64, f64)>> = Vec::new();

    // A fan of seeds from just off the centre out past the last torus.
    let mut x = 0.015;
    while x < 0.95 {
        let mut pts = Vec::with_capacity(steps);
        let mut s = orbit(Pt { x, y: 0.0 });
        let mut escaped = false;
        for _ in 0..steps {
            let p = s.extract();
            if p.x.abs() > bound || p.y.abs() > bound {
                escaped = true;
                break;
            }
            pts.push((p.x, p.y));
            s = s.tail();
        }
        // Keep orbits that stayed bounded the whole way (tori + persistent
        // chaos); drop seeds that escape, which only clutter the frame.
        if !escaped && pts.len() > 40 {
            orbits.push(pts);
        }
        x += 0.012;
    }

    // Emit compact JS: window.ORBITS = [[[x,y],...],...]; rounded to 4 dp.
    print!("window.ORBITS=[");
    for (i, o) in orbits.iter().enumerate() {
        if i > 0 {
            print!(",");
        }
        print!("[");
        for (j, (px, py)) in o.iter().enumerate() {
            if j > 0 {
                print!(",");
            }
            print!("[{:.4},{:.4}]", px, py);
        }
        print!("]");
    }
    println!("];");
    // The two fixed points: elliptic centre and hyperbolic saddle.
    println!("window.FIXED=[[0,0,\"elliptic\"],[1,0.5,\"hyperbolic\"]];");
    eprintln!("emitted {} bounded orbits", orbits.len());
}
