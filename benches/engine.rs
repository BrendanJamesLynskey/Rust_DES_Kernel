//! Criterion benchmarks: the bare kernel, and the serving engine.
//!
//!     cargo bench --bench engine

use std::hint::black_box;

use criterion::{BatchSize, Criterion, Throughput, criterion_group, criterion_main};
use rust_des_kernel::disagg::{ConfigSpec, LengthDist, poisson_workload, simulate};
use rust_des_kernel::queueing::md1_mean_wait;

fn kernel(c: &mut Criterion) {
    let mut g = c.benchmark_group("kernel");
    let (_, events) = md1_mean_wait(0.8, 1.0, 100_000, 1);
    g.throughput(Throughput::Elements(events));
    g.bench_function("md1_100k_customers", |b| {
        b.iter(|| md1_mean_wait(black_box(0.8), 1.0, 100_000, 1))
    });
    g.finish();
}

fn engine(c: &mut Criterion) {
    let mut g = c.benchmark_group("disagg");
    g.sample_size(20);
    for (name, spec) in [
        ("1P1D_1000req", ConfigSpec::default()),
        (
            "colocated_1000req",
            ConfigSpec {
                mode: "colocated".into(),
                ..Default::default()
            },
        ),
    ] {
        let cfg = spec.build().unwrap();
        let wl = poisson_workload(
            4.0,
            1000,
            LengthDist::new(2048.0, 0.5),
            LengthDist::new(256.0, 0.5),
            0,
        );
        let events = simulate(cfg.clone(), wl.clone()).unwrap().events;
        g.throughput(Throughput::Elements(events));
        g.bench_function(name, |b| {
            b.iter_batched(
                || (cfg.clone(), wl.clone()),
                |(c, w)| simulate(c, w).unwrap(),
                BatchSize::SmallInput,
            )
        });
    }
    g.finish();
}

criterion_group!(benches, kernel, engine);
criterion_main!(benches);
