//! A small, generic discrete-event kernel.
//!
//! The kernel knows nothing about the model. It keeps a priority queue of
//! `(time, priority, sequence)`-ordered events and hands them back one at a
//! time, advancing the clock. The ordering rule is SimPy's:
//!
//! * earlier time first;
//! * at equal times, [`Priority::Urgent`] before [`Priority::Normal`];
//! * otherwise first scheduled, first served (the sequence number).
//!
//! The last rule is what makes a simulation deterministic. A model that
//! schedules the same events in the same order gets the same answer, to the
//! bit, every run.

use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;

/// Simulated time, in seconds.
pub type Time = f64;

/// Tie-break class for events at the same time (SimPy's URGENT and NORMAL).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Priority {
    Urgent = 0,
    Normal = 1,
}

/// One scheduled event: the key the heap orders by, plus the payload.
#[derive(Debug)]
struct Entry<E> {
    time: Time,
    priority: Priority,
    seq: u64,
    event: E,
}

// `f64` is not `Ord` (NaN), so order by `total_cmp`. Times are never NaN:
// `schedule` rejects them.
impl<E> Ord for Entry<E> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.time
            .total_cmp(&other.time)
            .then(self.priority.cmp(&other.priority))
            .then(self.seq.cmp(&other.seq))
    }
}

impl<E> PartialOrd for Entry<E> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<E> PartialEq for Entry<E> {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl<E> Eq for Entry<E> {}

/// The event queue and the clock.
#[derive(Debug)]
pub struct Scheduler<E> {
    now: Time,
    seq: u64,
    processed: u64,
    // `BinaryHeap` is a max-heap; `Reverse` turns it into the min-heap a DES needs.
    heap: BinaryHeap<Reverse<Entry<E>>>,
}

impl<E> Default for Scheduler<E> {
    fn default() -> Self {
        Self::new()
    }
}

impl<E> Scheduler<E> {
    pub fn new() -> Self {
        Self {
            now: 0.0,
            seq: 0,
            processed: 0,
            heap: BinaryHeap::new(),
        }
    }

    /// The current simulated time.
    pub fn now(&self) -> Time {
        self.now
    }

    /// Events handed out so far.
    pub fn processed(&self) -> u64 {
        self.processed
    }

    /// Events still waiting.
    pub fn pending(&self) -> usize {
        self.heap.len()
    }

    /// Schedule `event` at absolute time `at` (never in the past).
    pub fn schedule_at(&mut self, at: Time, priority: Priority, event: E) {
        assert!(!at.is_nan(), "event time is NaN");
        assert!(
            at >= self.now,
            "event scheduled in the past: {at} < {}",
            self.now
        );
        self.heap.push(Reverse(Entry {
            time: at,
            priority,
            seq: self.seq,
            event,
        }));
        self.seq += 1;
    }

    /// Schedule `event` after `delay` seconds, as SimPy's `env.timeout(delay)` does:
    /// the event time is `now + delay`, rounded once.
    pub fn schedule_in(&mut self, delay: Time, event: E) {
        assert!(delay >= 0.0, "negative delay {delay}");
        self.schedule_at(self.now + delay, Priority::Normal, event);
    }

    /// Schedule `event` at the current time, behind everything already due now.
    pub fn schedule_now(&mut self, event: E) {
        self.schedule_at(self.now, Priority::Normal, event);
    }

    /// Bytes per queued event (key plus payload), for sizing very large runs.
    pub fn entry_size(&self) -> usize {
        std::mem::size_of::<Reverse<Entry<E>>>()
    }

    /// Remove the next event and advance the clock to its time.
    pub fn pop(&mut self) -> Option<(Time, E)> {
        let Reverse(e) = self.heap.pop()?;
        self.now = e.time;
        self.processed += 1;
        Some((e.time, e.event))
    }
}

/// A model reacts to one event at a time and may schedule more.
pub trait Model {
    type Event;

    /// Handle one event. Return `false` to stop the run.
    fn handle(&mut self, sched: &mut Scheduler<Self::Event>, event: Self::Event) -> bool;
}

/// Run `model` until it asks to stop or the queue empties. Returns the final time.
pub fn run<M: Model>(model: &mut M, sched: &mut Scheduler<M::Event>) -> Time {
    while let Some((_, ev)) = sched.pop() {
        if !model.handle(sched, ev) {
            break;
        }
    }
    sched.now()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn earlier_first_then_urgent_then_fifo() {
        let mut s = Scheduler::new();
        s.schedule_at(2.0, Priority::Normal, "late");
        s.schedule_at(1.0, Priority::Normal, "n1");
        s.schedule_at(1.0, Priority::Normal, "n2");
        s.schedule_at(1.0, Priority::Urgent, "urgent");
        let order: Vec<_> = std::iter::from_fn(|| s.pop().map(|(_, e)| e)).collect();
        assert_eq!(order, ["urgent", "n1", "n2", "late"]);
        assert_eq!(s.now(), 2.0);
        assert_eq!(s.processed(), 4);
    }

    #[test]
    fn equality_and_pending_follow_the_ordering_key() {
        let e = |t, seq| Entry {
            time: t,
            priority: Priority::Normal,
            seq,
            event: (),
        };
        assert!(e(1.0, 0) == e(1.0, 0));
        assert!(e(1.0, 0) != e(1.0, 1));
        let mut s = Scheduler::new();
        assert_eq!(s.pending(), 0);
        s.schedule_now(());
        s.schedule_in(1.0, ());
        assert_eq!(s.pending(), 2);
    }

    #[test]
    #[should_panic(expected = "in the past")]
    fn cannot_schedule_in_the_past() {
        let mut s = Scheduler::new();
        s.schedule_at(1.0, Priority::Normal, ());
        s.pop();
        s.schedule_at(0.5, Priority::Normal, ());
    }

    #[test]
    fn schedule_in_rounds_like_simpy() {
        let mut s = Scheduler::new();
        s.schedule_at(0.1, Priority::Normal, 0);
        s.pop();
        s.schedule_in(0.2, 1);
        let (t, _) = s.pop().unwrap();
        assert_eq!(t, 0.1 + 0.2); // 0.30000000000000004, exactly as SimPy computes it
    }
}
