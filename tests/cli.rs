//! The `disagg-rs` binary, end to end.

use std::process::Command;

fn disagg_rs(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_disagg-rs"))
        .args(args)
        .output()
        .expect("binary runs")
}

#[test]
fn json_summary_is_valid_and_complete() {
    let out = disagg_rs(&["--n", "60", "--rate", "3", "--json"]);
    assert!(out.status.success());
    let m: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(m["requests"]["completed"], 60);
    assert!(m["latency_s"]["ttft"]["p99"].as_f64().unwrap() > 0.0);
}

#[test]
fn text_report_names_the_hot_spot() {
    let out = disagg_rs(&[
        "--n",
        "60",
        "--prefill",
        "2",
        "--link",
        "eth-25g",
        "--rate",
        "6",
    ]);
    let text = String::from_utf8(out.stdout).unwrap();
    assert!(text.contains("hot-spot     stage="), "{text}");
}

#[test]
fn sweep_prints_one_csv_row_per_rate() {
    let out = disagg_rs(&["--n", "40", "--sweep", "1", "2", "4"]);
    let text = String::from_utf8(out.stdout).unwrap();
    let lines: Vec<_> = text.lines().collect();
    assert_eq!(
        lines[0],
        "rate,ttft_p99_s,tpot_p99_s,slo_attainment,j_per_token"
    );
    assert_eq!(lines.len(), 4);
}

#[test]
fn bad_arguments_exit_with_status_2() {
    assert_eq!(disagg_rs(&["--bogus"]).status.code(), Some(2));
    assert_eq!(
        disagg_rs(&["--link", "carrier-pigeon"]).status.code(),
        Some(2)
    );
}

#[test]
fn heterogeneous_pools_from_the_command_line() {
    let out = disagg_rs(&[
        "--n",
        "40",
        "--model",
        "llama3-8b",
        "--devices",
        "1",
        "--prefill-device",
        "h100",
        "--decode-device",
        "a100",
        "--decode-devices-per-instance",
        "2",
        "--prefill-devices-per-instance",
        "1",
    ]);
    assert!(out.status.success());
    let text = String::from_utf8(out.stdout).unwrap();
    assert!(
        text.contains("pools        prefill 1x H100-SXM   decode 2x A100-SXM"),
        "{text}"
    );
    let rejected = disagg_rs(&["--prefill-device", "optical-fft"]);
    assert_eq!(rejected.status.code(), Some(2));
    let err = String::from_utf8(rejected.stderr).unwrap();
    assert!(err.contains("not in the Rust port"), "{err}");
}
