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
