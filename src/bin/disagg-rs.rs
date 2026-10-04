//! `disagg-rs`: the Rust engine from the command line.
//!
//! ```text
//! disagg-rs                                  # 70B, 1P1D, 4 req/s, 1000 requests
//! disagg-rs --mode colocated --rate 3
//! disagg-rs --prefill 2 --link eth-25g --rate 6
//! disagg-rs --sweep 2 3 4 5 6 8              # parallel rate sweep (rayon), CSV out
//! disagg-rs --config cfg.json --json         # any config field, JSON summary out
//! disagg-rs --model llama3-8b --devices 1 --prefill-device h100 --decode-device a100
//! ```

use std::process::ExitCode;
use std::time::Instant;

use rayon::prelude::*;
use rust_des_kernel::disagg::{
    ConfigSpec, LengthDist, format_report, poisson_workload, simulate, summarise,
};

struct Args {
    spec: ConfigSpec,
    rate: f64,
    n: usize,
    seed: u64,
    prompt: (f64, f64),
    output: (f64, f64),
    sweep: Vec<f64>,
    json: bool,
}

fn parse() -> Result<Args, String> {
    let mut a = Args {
        spec: ConfigSpec::default(),
        rate: 4.0,
        n: 1000,
        seed: 0,
        prompt: (2048.0, 0.5),
        output: (256.0, 0.5),
        sweep: Vec::new(),
        json: false,
    };
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut it = argv.iter().peekable();
    let num = |v: Option<&String>, flag: &str| -> Result<f64, String> {
        v.ok_or(format!("{flag} needs a value"))?
            .parse()
            .map_err(|e| format!("{flag}: {e}"))
    };
    while let Some(flag) = it.next() {
        match flag.as_str() {
            "--config" => {
                let path = it.next().ok_or("--config needs a file")?;
                let text = std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
                a.spec = serde_json::from_str(&text).map_err(|e| format!("{path}: {e}"))?;
            }
            "--mode" => a.spec.mode = it.next().ok_or("--mode needs a value")?.clone(),
            "--model" => a.spec.model = it.next().ok_or("--model needs a value")?.clone(),
            "--device" => a.spec.device = it.next().ok_or("--device needs a value")?.clone(),
            "--devices" => a.spec.devices_per_instance = num(it.next(), flag)? as i64,
            "--prefill-device" => {
                a.spec.prefill_device =
                    Some(it.next().ok_or("--prefill-device needs a value")?.clone())
            }
            "--decode-device" => {
                a.spec.decode_device =
                    Some(it.next().ok_or("--decode-device needs a value")?.clone())
            }
            "--prefill-devices-per-instance" => {
                a.spec.prefill_devices_per_instance = Some(num(it.next(), flag)? as i64)
            }
            "--decode-devices-per-instance" => {
                a.spec.decode_devices_per_instance = Some(num(it.next(), flag)? as i64)
            }
            "--link" => a.spec.link = it.next().ok_or("--link needs a value")?.clone(),
            "--channels" => a.spec.link_channels = num(it.next(), flag)? as usize,
            "--prefill" => a.spec.n_prefill = num(it.next(), flag)? as usize,
            "--decode" => a.spec.n_decode = num(it.next(), flag)? as usize,
            "--colocated" => a.spec.n_colocated = num(it.next(), flag)? as usize,
            "--power-cap" => a.spec.power_cap_w = Some(num(it.next(), flag)?),
            "--dvfs" => a.spec.dvfs = true,
            "--rate" => a.rate = num(it.next(), flag)?,
            "--n" => a.n = num(it.next(), flag)? as usize,
            "--seed" => a.seed = num(it.next(), flag)? as u64,
            "--prompt" => a.prompt = (num(it.next(), flag)?, num(it.next(), flag)?),
            "--output" => a.output = (num(it.next(), flag)?, num(it.next(), flag)?),
            "--json" => a.json = true,
            "--sweep" => {
                while let Some(v) = it.peek().filter(|v| !v.starts_with("--")) {
                    a.sweep
                        .push(v.parse().map_err(|e| format!("--sweep: {e}"))?);
                    it.next();
                }
            }
            "-h" | "--help" => {
                println!(
                    "{}",
                    include_str!("disagg-rs.rs")
                        .lines()
                        .take(10)
                        .collect::<Vec<_>>()
                        .join("\n")
                );
                std::process::exit(0);
            }
            other => return Err(format!("unknown flag {other}")),
        }
    }
    Ok(a)
}

fn main() -> ExitCode {
    let a = match parse() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("disagg-rs: {e}");
            return ExitCode::from(2);
        }
    };
    let cfg = match a.spec.build() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("disagg-rs: {e}");
            return ExitCode::from(2);
        }
    };
    let wl = |rate: f64| {
        poisson_workload(
            rate,
            a.n,
            LengthDist::new(a.prompt.0, a.prompt.1),
            LengthDist::new(a.output.0, a.output.1),
            a.seed,
        )
    };
    if !a.sweep.is_empty() {
        // Independent runs share nothing, so rayon can farm them out safely:
        // the compiler checks that each closure only reads `cfg`.
        let rows: Vec<String> = a
            .sweep
            .par_iter()
            .map(|&rate| {
                let m = summarise(&simulate(cfg.clone(), wl(rate)).expect("valid config"));
                let g = |p: &[&str]| {
                    p.iter()
                        .fold(&m, |v, k| &v[*k])
                        .as_f64()
                        .unwrap_or(f64::NAN)
                };
                format!(
                    "{rate},{:.6},{:.6},{:.4},{:.4}",
                    g(&["latency_s", "ttft", "p99"]),
                    g(&["latency_s", "tpot", "p99"]),
                    g(&["throughput", "slo_attainment"]),
                    g(&["energy", "J_per_output_token"])
                )
            })
            .collect();
        println!("rate,ttft_p99_s,tpot_p99_s,slo_attainment,j_per_token");
        rows.iter().for_each(|r| println!("{r}"));
        return ExitCode::SUCCESS;
    }
    let t0 = Instant::now();
    let res = simulate(cfg, wl(a.rate)).expect("valid config");
    let wall = t0.elapsed().as_secs_f64();
    let m = summarise(&res);
    if a.json {
        println!("{m}");
    } else {
        println!("{}", format_report(&m));
        eprintln!(
            "{} events in {:.1} ms ({:.2} M events/s)",
            res.events,
            1e3 * wall,
            res.events as f64 / wall / 1e6
        );
    }
    ExitCode::SUCCESS
}
