//! Rasterise a static phase portrait to `phase_portrait.ppm` (convert to PNG
//! with `sips -s format png`). Run with:  cargo run --release --example render_png
//!
//! A self-contained f64 renderer: it iterates the same Hénon map as
//! `kam::henon_step`, generalised to an arbitrary rotation `(c, s)` so it can
//! render the dramatic ρ = 1/5 resonance (five islands + a chaotic sea) rather
//! than the repo's quiet exact-rational setting c = 3/5. The committed figure is
//! for documentation; the exact arithmetic lives in `kam.rs` and its tests.

use std::fs;

const W: usize = 1400;
const BG: [u8; 3] = [10, 12, 21];

fn step(c: f64, s: f64, x: f64, y: f64) -> (f64, f64) {
    let v = y - x * x;
    (c * x - s * v, s * x + c * v)
}

// cool core → warm rim, matching the interactive plate
fn ramp(t: f64) -> [u8; 3] {
    let stops = [
        (0.00, [91.0, 214.0, 230.0]),
        (0.34, [111.0, 140.0, 255.0]),
        (0.67, [176.0, 107.0, 255.0]),
        (1.00, [255.0, 122.0, 92.0]),
    ];
    let t = t.clamp(0.0, 1.0);
    for w in stops.windows(2) {
        let (a, b) = (w[0], w[1]);
        if t <= b.0 {
            let u = (t - a.0) / (b.0 - a.0);
            return [0, 1, 2].map(|k| (a.1[k] + u * (b.1[k] - a.1[k])) as u8);
        }
    }
    [255, 122, 92]
}

fn main() {
    // Near the ρ = 1/5 resonance (α = 76°, ρ ≈ 0.211): five islands + chaos.
    let alpha = 76.0_f64.to_radians();
    let (c, s) = (alpha.cos(), alpha.sin());

    // generate bounded orbits
    let mut orbits: Vec<Vec<(f64, f64)>> = Vec::new();
    for k in 0..92 {
        let (mut x, mut y): (f64, f64) = (0.012 + k as f64 * 0.0145, 0.0);
        let mut pts = Vec::new();
        let mut escaped = false;
        for _ in 0..1500 {
            if x.abs() > 4.0 || y.abs() > 4.0 {
                escaped = true;
                break;
            }
            pts.push((x, y));
            let n = step(c, s, x, y);
            x = n.0;
            y = n.1;
        }
        if !escaped && pts.len() > 80 {
            orbits.push(pts);
        }
    }

    // isotropic fit
    let (mut lo_x, mut hi_x, mut lo_y, mut hi_y) = (f64::MAX, f64::MIN, f64::MAX, f64::MIN);
    for o in &orbits {
        for &(px, py) in o {
            lo_x = lo_x.min(px);
            hi_x = hi_x.max(px);
            lo_y = lo_y.min(py);
            hi_y = hi_y.max(py);
        }
    }
    let cx = (lo_x + hi_x) / 2.0;
    let cy = (lo_y + hi_y) / 2.0;
    let half = (hi_x - lo_x).max(hi_y - lo_y) / 2.0 * 1.07;
    let scale = (W as f64) / (2.0 * half);
    let to_px = |x: f64, y: f64| -> (i64, i64) {
        (
            ((x - cx) * scale + W as f64 / 2.0) as i64,
            (W as f64 / 2.0 - (y - cy) * scale) as i64,
        )
    };

    // Additive luminosity: energy accumulates per pixel, so dense curves and
    // island cores saturate bright while the thin chaotic haze stays dim —
    // the glow of the interactive plate, without an image library.
    let mut acc = vec![[0.0f32; 3]; W * W];
    let add = |acc: &mut Vec<[f32; 3]>, px: i64, py: i64, col: [u8; 3], w: f32| {
        if px < 0 || py < 0 || px >= W as i64 || py >= W as i64 {
            return;
        }
        let idx = py as usize * W + px as usize;
        for k in 0..3 {
            acc[idx][k] += col[k] as f32 * w;
        }
    };

    // amplitude → colour
    let maxamp = orbits
        .iter()
        .map(|o| o.iter().map(|&(x, y)| x.hypot(y)).fold(0.0, f64::max))
        .fold(1e-9, f64::max);
    for o in &orbits {
        let amp = o.iter().map(|&(x, y)| x.hypot(y)).fold(0.0, f64::max);
        let col = ramp(amp / maxamp);
        for &(x, y) in o {
            let (px, py) = to_px(x, y);
            add(&mut acc, px, py, col, 0.55); // core pixel
            for &(dx, dy) in &[(1, 0), (-1, 0), (0, 1), (0, -1)] {
                add(&mut acc, px + dx, py + dy, col, 0.16); // soft 2px halo
            }
        }
    }
    // elliptic centre ring at the origin
    let (ox, oy) = to_px(0.0, 0.0);
    for d in -8..=8_i64 {
        let h = (64 - d * d).max(0).isqrt();
        add(&mut acc, ox + d, oy + h, [255, 255, 255], 1.0);
        add(&mut acc, ox + d, oy - h, [255, 255, 255], 1.0);
    }

    // compose additively over the plate background and clamp
    let buf: Vec<[u8; 3]> = acc
        .iter()
        .map(|e| [0, 1, 2].map(|k| (BG[k] as f32 + e[k]).min(255.0) as u8))
        .collect();

    // write binary PPM (P6)
    let mut out = format!("P6\n{} {}\n255\n", W, W).into_bytes();
    out.reserve(W * W * 3);
    for px in &buf {
        out.extend_from_slice(px);
    }
    fs::write("phase_portrait.ppm", out).expect("write ppm");
    eprintln!("wrote phase_portrait.ppm ({} orbits, ρ = 1/5)", orbits.len());
}
