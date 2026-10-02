//! Requests and synthetic workloads: a port of `disagg_sim/workload.py`.

use crate::pymath::round_half_even;
use crate::pyrand::PyRandom;

/// One inference request, plus every timestamp the simulator stamps on it.
/// Stamps are `None` until the request reaches that point.
#[derive(Debug, Clone, PartialEq)]
pub struct Request {
    pub rid: usize,
    pub arrival: f64,
    pub prompt_len: i64,
    pub output_len: i64,

    pub prefill_start: Option<f64>,
    pub first_token: Option<f64>,
    pub kv_start: Option<f64>,
    pub kv_ready: Option<f64>,
    pub decode_start: Option<f64>,
    pub finish: Option<f64>,

    pub tokens_out: i64,
    pub last_token: f64,
    pub itls: Vec<f64>,
}

impl Request {
    pub fn new(rid: usize, arrival: f64, prompt_len: i64, output_len: i64) -> Self {
        Request {
            rid,
            arrival,
            prompt_len,
            output_len,
            prefill_start: None,
            first_token: None,
            kv_start: None,
            kv_ready: None,
            decode_start: None,
            finish: None,
            tokens_out: 0,
            last_token: 0.0,
            itls: Vec::new(),
        }
    }

    pub fn ttft(&self) -> f64 {
        self.first_token.unwrap() - self.arrival
    }

    /// Mean time per output token after the first (DistServe's definition).
    pub fn tpot(&self) -> Option<f64> {
        (self.output_len >= 2).then(|| {
            (self.finish.unwrap() - self.first_token.unwrap()) / (self.output_len - 1) as f64
        })
    }

    pub fn e2e(&self) -> f64 {
        self.finish.unwrap() - self.arrival
    }

    /// Where this request's time went, in `STAGES` order. Sums to e2e.
    pub fn stages(&self) -> [f64; 6] {
        let first = self.first_token.unwrap();
        let prefill_start = self.prefill_start.unwrap();
        let handoff = self.kv_ready.unwrap_or(first);
        let mut s = [
            prefill_start - self.arrival,
            first - prefill_start,
            0.0,
            0.0,
            0.0,
            0.0,
        ];
        if let (Some(ks), Some(kr)) = (self.kv_start, self.kv_ready) {
            s[2] = ks - first;
            s[3] = kr - ks;
        }
        if let Some(ds) = self.decode_start {
            s[4] = ds - handoff;
            s[5] = self.finish.unwrap() - ds;
        }
        s
    }
}

/// Token-length distribution: lognormal with a given mean and coefficient of
/// variation, clipped to [lo, hi]. `cv = 0` gives a fixed length.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LengthDist {
    pub mean: f64,
    pub cv: f64,
    pub lo: i64,
    pub hi: i64,
}

impl LengthDist {
    pub fn new(mean: f64, cv: f64) -> Self {
        LengthDist {
            mean,
            cv,
            lo: 1,
            hi: 32768,
        }
    }

    pub fn sample(&self, rng: &mut PyRandom) -> i64 {
        let x = if self.cv <= 0.0 {
            self.mean
        } else {
            let sigma2 = (1.0 + self.cv.powf(2.0)).ln();
            rng.lognormvariate(self.mean.ln() - sigma2 / 2.0, sigma2.sqrt())
        };
        (round_half_even(x) as i64).clamp(self.lo, self.hi)
    }
}

/// `n` requests with exponential inter-arrival times (a Poisson process),
/// identical to `poisson_workload` in Python for the same seed.
pub fn poisson_workload(
    rate: f64,
    n: usize,
    prompt: LengthDist,
    output: LengthDist,
    seed: u64,
) -> Vec<Request> {
    let mut rng = PyRandom::new(seed);
    let mut t = 0.0;
    (0..n)
        .map(|i| {
            t += rng.expovariate(rate);
            let p = prompt.sample(&mut rng);
            let o = output.sample(&mut rng);
            Request::new(i, t, p, o)
        })
        .collect()
}
