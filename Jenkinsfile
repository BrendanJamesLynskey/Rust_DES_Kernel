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
        CARGO_TERM_COLOR = 'never'
        // env.HOME is null in Groovy and a PATH set here does not reach sh steps, so each step
        // that needs cargo puts ~/.cargo/bin on its own PATH (set CARGO_BIN to override).
        // Not named CARGO: cargo and maturin read $CARGO as the path of the cargo binary.
        WITH_CARGO = 'export PATH="${CARGO_BIN:-$HOME/.cargo/bin}:$PATH"; '
    }

    stages {
        stage('Lint') {
            steps {
                sh(env.WITH_CARGO + 'cargo fmt --check')
                sh(env.WITH_CARGO + 'cargo clippy --all-targets -- -D warnings')
                sh(env.WITH_CARGO + 'cargo clippy --features python -- -D warnings')
            }
        }

        stage('Rust tests') {
            steps {
                sh(env.WITH_CARGO + 'cargo nextest run --release --profile ci')
            }
        }

        stage('Python differential tests') {
            steps {
                sh(env.WITH_CARGO + '''
                    python3 -m venv .venv
                    .venv/bin/pip install -q maturin
                    .venv/bin/maturin develop --release -q -E dev
                    .venv/bin/pytest pytests --junitxml=pytest-junit.xml
                ''')
            }
        }

        stage('Coverage') {
            steps {
                sh(env.WITH_CARGO + 'cargo llvm-cov --release --cobertura --output-path coverage.xml')
                // cargo-llvm-cov lists each generic instantiation as its own method, and the
                // Coverage plugin's Cobertura parser rejects duplicate names: skip them.
                recordCoverage(tools: [[parser: 'COBERTURA', pattern: 'coverage.xml']],
                               sourceCodeRetention: 'LAST_BUILD', ignoreParsingErrors: true)
            }
        }

        stage('Benchmarks and performance gate') {
            steps {
                sh(env.WITH_CARGO + 'cargo bench --bench engine -- --noplot --warm-up-time 1 --measurement-time 3')
                sh ".venv/bin/python ci/perf_gate.py --margin ${params.PERF_MARGIN ?: '0.25'}"
            }
            post {
                always { archiveArtifacts artifacts: 'perf_report.md', allowEmptyArchive: true }
            }
        }

        stage('Results') {
            steps {
                sh(env.WITH_CARGO + '.venv/bin/python examples/results.py > /dev/null')
                archiveArtifacts artifacts: 'examples/results.md'
            }
        }

        stage('Nightly sweep') {
            when { expression { params.NIGHTLY } }
            steps {
                sh(env.WITH_CARGO + "cargo run --release -q --bin disagg-rs -- --n 2000 --sweep ${params.SWEEP_RATES ?: '1 2 3 4 5 6 8 10'} > sweep.csv")
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
