// Jenkins pipeline for Rust_DES_Kernel.
//
// Stages: lint -> Rust tests (JUnit via cargo-nextest) -> Python differential
// tests (JUnit via pytest) -> coverage (cargo-llvm-cov, Cobertura) -> benchmarks
// and a performance-regression gate -> results.md -> an optional nightly sweep.
//
// Needs on the agent: rustup toolchain, cargo-nextest, cargo-llvm-cov, Python 3.10+.
// Plugins: Pipeline, Git, JUnit, Coverage.

pipeline {
    agent any

    parameters {
        booleanParam(name: 'NIGHTLY', defaultValue: false,
                     description: 'Also run the parameterised rate sweep (the nightly job sets this)')
        string(name: 'SWEEP_RATES', defaultValue: '1 2 3 4 5 6 8 10',
               description: 'Request rates (req/s) for the sweep stage')
        string(name: 'PERF_MARGIN', defaultValue: '0.25',
               description: 'Allowed slow-down against ci/perf_baseline.json before the build fails')
    }

    options {
        buildDiscarder(logRotator(numToKeepStr: '30'))
        timeout(time: 60, unit: 'MINUTES')
    }

    environment {
        PATH = "${env.HOME}/.cargo/bin:${env.PATH}"
        CARGO_TERM_COLOR = 'never'
    }

    stages {
        stage('Lint') {
            steps {
                sh 'cargo fmt --check'
                sh 'cargo clippy --all-targets -- -D warnings'
                sh 'cargo clippy --features python -- -D warnings'
            }
        }

        stage('Rust tests') {
            steps {
                sh 'cargo nextest run --release --profile ci'
            }
        }

        stage('Python differential tests') {
            steps {
                sh '''
                    python3 -m venv .venv
                    .venv/bin/pip install -q maturin
                    .venv/bin/maturin develop --release -q -E dev
                    .venv/bin/pytest pytests --junitxml=pytest-junit.xml
                '''
            }
        }

        stage('Coverage') {
            steps {
                sh 'cargo llvm-cov --release --cobertura --output-path coverage.xml'
                recordCoverage(tools: [[parser: 'COBERTURA', pattern: 'coverage.xml']],
                               sourceCodeRetention: 'LAST_BUILD')
            }
        }

        stage('Benchmarks and performance gate') {
            steps {
                sh 'cargo bench --bench engine -- --noplot --warm-up-time 1 --measurement-time 3'
                sh ".venv/bin/python ci/perf_gate.py --margin ${params.PERF_MARGIN}"
            }
            post {
                always { archiveArtifacts artifacts: 'perf_report.md', allowEmptyArchive: true }
            }
        }

        stage('Results') {
            steps {
                sh '.venv/bin/python examples/results.py > /dev/null'
                archiveArtifacts artifacts: 'examples/results.md'
            }
        }

        stage('Nightly sweep') {
            when { expression { params.NIGHTLY } }
            steps {
                sh "cargo run --release -q --bin disagg-rs -- --n 2000 --sweep ${params.SWEEP_RATES} > sweep.csv"
                sh 'cat sweep.csv'
                archiveArtifacts artifacts: 'sweep.csv'
            }
        }
    }

    post {
        always {
            junit testResults: 'target/nextest/ci/junit.xml, pytest-junit.xml', allowEmptyResults: true
        }
    }
}
