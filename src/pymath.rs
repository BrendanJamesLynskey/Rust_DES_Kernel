//! The few Python numeric semantics a bit-exact port has to copy.
//!
//! * `statistics.fmean` sums with `math.fsum`, which is *correctly rounded*,
//!   not a left-to-right loop. A plain loop differs in the last bits.
//! * `a // b` on floats is CPython's `fmod`-based floor division.
//! * `round(x)` rounds half to even.

/// `math.fsum`: the correctly rounded sum (Shewchuk's algorithm, as in CPython).
pub fn fsum<I: IntoIterator<Item = f64>>(xs: I) -> f64 {
    let mut p: Vec<f64> = Vec::new();
    for mut x in xs {
        debug_assert!(x.is_finite(), "fsum is only ported for finite inputs");
        let mut i = 0;
        for j in 0..p.len() {
            let mut y = p[j];
            if x.abs() < y.abs() {
                std::mem::swap(&mut x, &mut y);
            }
            let hi = x + y;
            let lo = y - (hi - x);
            if lo != 0.0 {
                p[i] = lo;
                i += 1;
            }
            x = hi;
        }
        p.truncate(i);
        if x != 0.0 {
            p.push(x);
        }
    }
    let mut n = p.len();
    if n == 0 {
        return 0.0;
    }
    n -= 1;
    let mut hi = p[n];
    let mut lo = 0.0;
    while n > 0 {
        let x = hi;
        n -= 1;
        let y = p[n];
        hi = x + y;
        let yr = hi - x;
        lo = y - yr;
        if lo != 0.0 {
            break;
        }
    }
    if n > 0 && ((lo < 0.0 && p[n - 1] < 0.0) || (lo > 0.0 && p[n - 1] > 0.0)) {
        let y = lo * 2.0;
        let x = hi + y;
        let yr = x - hi;
        if y == yr {
            hi = x;
        }
    }
    hi
}

/// `statistics.fmean`: `fsum(xs) / len(xs)`; NaN for no data (the simulator's convention).
pub fn fmean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return f64::NAN;
    }
    fsum(xs.iter().copied()) / xs.len() as f64
}

/// Python's float floor division `a // b`.
pub fn floordiv(a: f64, b: f64) -> f64 {
    let m = a % b; // C fmod
    let mut div = (a - m) / b;
    if m != 0.0 && ((b < 0.0) != (m < 0.0)) {
        div -= 1.0;
    }
    if div != 0.0 {
        let f = div.floor();
        if div - f > 0.5 { f + 1.0 } else { f }
    } else {
        0.0f64.copysign(a / b)
    }
}

/// Python's `round(x)` for a float: nearest integer, ties to even.
pub fn round_half_even(x: f64) -> f64 {
    x.round_ties_even()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fsum_is_correctly_rounded() {
        // A left-to-right loop gives 0.9999999999999999; math.fsum gives 1.0.
        assert_eq!(fsum([0.1; 10]), 1.0);
        assert_eq!(
            fsum([1e100, 1.0, -1e100, 1e-100, 1e50, -1.0, -1e50]),
            1e-100
        );
        assert_eq!(fsum([]), 0.0);
    }

    #[test]
    fn floordiv_matches_python() {
        assert_eq!(floordiv(7.0, 2.0), 3.0);
        assert_eq!(floordiv(-7.0, 2.0), -4.0);
        assert_eq!(floordiv(0.3, 0.1), 2.0); // Python: 0.3 // 0.1 == 2.0
    }

    #[test]
    fn rounding_is_half_even() {
        assert_eq!(round_half_even(2.5), 2.0);
        assert_eq!(round_half_even(3.5), 4.0);
    }
}
