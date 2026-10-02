//! The kernel against queueing theory and against its own ordering contract.

use proptest::prelude::*;
use rust_des_kernel::kernel::{Priority, Scheduler};
use rust_des_kernel::queueing::{md1_mean_wait, md1_theory};

#[test]
fn md1_mean_wait_matches_pollaczek_khinchine() {
    // A service time other than 1 s, so that multiplying and dividing by it differ.
    let d = 2.5;
    for rho in [0.3, 0.6, 0.8] {
        let (sim, _) = md1_mean_wait(rho / d, d, 400_000, 11);
        let theory = md1_theory(rho / d, d);
        // (A mutant that made the theory negative once passed this check: dividing by a
        // negative number makes every relative error "small". Mutation testing found it.)
        assert!(theory > 0.0 && sim > 0.0);
        let err = (sim - theory).abs() / theory;
        assert!(
            err < 0.03,
            "rho {rho}: simulated {sim:.4}, theory {theory:.4}"
        );
    }
}

proptest! {
    /// Whatever is scheduled, events come out in (time, priority, insertion) order.
    #[test]
    fn pops_in_time_priority_fifo_order(ops in prop::collection::vec((0u8..20, any::<bool>()), 1..300)) {
        let mut s = Scheduler::new();
        for (k, &(t, urgent)) in ops.iter().enumerate() {
            let p = if urgent { Priority::Urgent } else { Priority::Normal };
            s.schedule_at(t as f64, p, (t, p, k));
        }
        let mut last: Option<(u8, Priority, usize)> = None;
        while let Some((time, e)) = s.pop() {
            prop_assert_eq!(time, e.0 as f64);
            if let Some(l) = last {
                prop_assert!(l < e, "{:?} came out before {:?}", l, e);
            }
            last = Some(e);
        }
        prop_assert_eq!(s.processed() as usize, ops.len());
    }
}
