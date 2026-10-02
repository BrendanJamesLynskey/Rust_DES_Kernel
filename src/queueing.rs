//! The smallest useful model on the kernel: an M/D/1 queue (Poisson arrivals,
//! deterministic service, one server). Its mean wait has a closed form, the
//! Pollaczek–Khinchine formula, so it checks the kernel against theory:
//!
//! ```text
//! Wq = rho * D / (2 * (1 - rho)),   rho = lambda * D
//! ```

use std::collections::VecDeque;

use crate::kernel::{Model, Scheduler, run};
use crate::pyrand::PyRandom;

/// Events are a plain enum: the compiler checks every handler covers every case.
#[derive(Debug, Clone, Copy)]
pub enum Event {
    Arrive,
    Depart,
}

pub struct Md1 {
    rng: PyRandom,
    rate: f64,
    service: f64,
    customers: u64,
    arrived: u64,
    queue: VecDeque<f64>, // arrival times of waiting customers
    busy: bool,
    pub served: u64,
    pub total_wait: f64,
}

impl Md1 {
    pub fn new(rate: f64, service: f64, customers: u64, seed: u64) -> Self {
        Md1 {
            rng: PyRandom::new(seed),
            rate,
            service,
            customers,
            arrived: 0,
            queue: VecDeque::new(),
            busy: false,
            served: 0,
            total_wait: 0.0,
        }
    }

    fn start_service(&mut self, s: &mut Scheduler<Event>, arrived_at: f64) {
        self.total_wait += s.now() - arrived_at;
        self.busy = true;
        s.schedule_in(self.service, Event::Depart);
    }
}

impl Model for Md1 {
    type Event = Event;

    fn handle(&mut self, s: &mut Scheduler<Event>, ev: Event) -> bool {
        match ev {
            Event::Arrive => {
                self.arrived += 1;
                if self.arrived < self.customers {
                    let gap = self.rng.expovariate(self.rate);
                    s.schedule_in(gap, Event::Arrive);
                }
                if self.busy {
                    self.queue.push_back(s.now());
                } else {
                    self.start_service(s, s.now());
                }
            }
            Event::Depart => {
                self.served += 1;
                self.busy = false;
                if let Some(t) = self.queue.pop_front() {
                    self.start_service(s, t);
                }
            }
        }
        self.served < self.customers
    }
}

/// Simulate `customers` customers; returns (mean wait, events processed).
pub fn md1_mean_wait(rate: f64, service: f64, customers: u64, seed: u64) -> (f64, u64) {
    let mut m = Md1::new(rate, service, customers, seed);
    let mut s = Scheduler::new();
    s.schedule_in(0.0, Event::Arrive);
    run(&mut m, &mut s);
    (m.total_wait / m.served as f64, s.processed())
}

/// The Pollaczek–Khinchine mean wait for M/D/1.
pub fn md1_theory(rate: f64, service: f64) -> f64 {
    let rho = rate * service;
    rho * service / (2.0 * (1.0 - rho))
}
