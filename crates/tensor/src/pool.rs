//! A persistent worker pool, std only.
//!
//! **Why this exists (DECISIONS D22).** The first threaded matmul used
//! `std::thread::scope` and so spawned fresh OS threads on every call. A forward
//! pass makes ~170 matmul calls per token; at ~10-20 µs per spawn × cores that is a
//! fixed tax that prefill amortises over large matrices but decode — one token, tiny
//! work per call — pays in full. This pool starts its workers once and hands them
//! work through a mutex + condvar, so the per-call cost is a wake-up, not a spawn.
//!
//! The contract is the same as `thread::scope`: [`parallel_for`] does not return
//! until every task has finished, so the closure may borrow from the caller's stack.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{Condvar, Mutex, OnceLock};

/// Lifetime-erased pointer to the caller's closure.
///
/// SAFETY invariant: a `Job` is only reachable through `State::job`, and
/// `parallel_for` clears `State::job` and waits for `finished == total` before it
/// returns — so the pointee (a borrow of the caller's stack) outlives every use.
#[derive(Clone, Copy)]
struct Job(*const (dyn Fn(usize) + Sync));

// SAFETY: the pointee is `Sync` (it is a `dyn Fn + Sync`), and the pointer is only
// dereferenced while the owning `parallel_for` frame is blocked waiting for it.
unsafe impl Send for Job {}

struct State {
    job: Option<Job>,
    next: usize,
    total: usize,
    finished: usize,
    panicked: bool,
}

struct Pool {
    state: Mutex<State>,
    work_cv: Condvar,
    done_cv: Condvar,
    /// Serialises callers: one job in flight at a time. Concurrent callers (e.g.
    /// `cargo test` running tests on several threads) simply queue.
    submit: Mutex<()>,
    workers: usize,
}

static POOL: OnceLock<&'static Pool> = OnceLock::new();

fn pool() -> &'static Pool {
    POOL.get_or_init(|| {
        let cores = std::thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(1);
        // The calling thread works too, so spawn one fewer.
        let workers = cores.saturating_sub(1);
        // Leaked on purpose: the pool lives for the whole process.
        let p: &'static Pool = Box::leak(Box::new(Pool {
            state: Mutex::new(State {
                job: None,
                next: 0,
                total: 0,
                finished: 0,
                panicked: false,
            }),
            work_cv: Condvar::new(),
            done_cv: Condvar::new(),
            submit: Mutex::new(()),
            workers,
        }));
        for _ in 0..workers {
            std::thread::spawn(move || worker_loop(p));
        }
        p
    })
}

/// Number of threads that can make progress on a job (workers + the caller).
pub fn threads() -> usize {
    pool().workers + 1
}

/// Claim and run tasks until none are left. Used by workers and by the caller.
fn run_tasks<'a>(p: &'a Pool, mut guard: std::sync::MutexGuard<'a, State>) {
    loop {
        let job = match guard.job {
            Some(j) if guard.next < guard.total => j,
            _ => return,
        };
        let idx = guard.next;
        guard.next += 1;
        drop(guard);

        // SAFETY: see `Job`. The submitting frame is blocked until `finished == total`.
        let f: &(dyn Fn(usize) + Sync) = unsafe { &*job.0 };
        let ok = catch_unwind(AssertUnwindSafe(|| f(idx))).is_ok();

        guard = p.state.lock().unwrap();
        guard.finished += 1;
        if !ok {
            guard.panicked = true;
        }
        if guard.finished == guard.total {
            p.done_cv.notify_all();
        }
    }
}

fn worker_loop(p: &'static Pool) {
    let mut guard = p.state.lock().unwrap();
    loop {
        while !(guard.job.is_some() && guard.next < guard.total) {
            guard = p.work_cv.wait(guard).unwrap();
        }
        run_tasks(p, guard);
        guard = p.state.lock().unwrap();
    }
}

/// Run `f(0) .. f(n_tasks - 1)` across the pool and the calling thread; returns
/// when all have finished. If any task panicked, panics after all have finished.
///
/// Must not be called from inside a task (it would deadlock on `submit`).
pub fn parallel_for(n_tasks: usize, f: &(dyn Fn(usize) + Sync)) {
    if n_tasks == 0 {
        return;
    }
    let p = pool();
    if n_tasks == 1 || p.workers == 0 {
        for t in 0..n_tasks {
            f(t);
        }
        return;
    }

    // Poison-tolerant: a panicking task re-panics here while holding this guard,
    // and that must not wedge the pool for later callers.
    let _submit = p.submit.lock().unwrap_or_else(|e| e.into_inner());

    // SAFETY: erase the borrow's lifetime. We block below until every task has
    // finished and clear the job before returning, so no use outlives `f`.
    let erased: *const (dyn Fn(usize) + Sync) = unsafe {
        std::mem::transmute::<&(dyn Fn(usize) + Sync), &'static (dyn Fn(usize) + Sync)>(f)
    };

    {
        let mut st = p.state.lock().unwrap();
        st.job = Some(Job(erased));
        st.next = 0;
        st.total = n_tasks;
        st.finished = 0;
        st.panicked = false;
    }
    p.work_cv.notify_all();

    // Help out, then wait for stragglers.
    run_tasks(p, p.state.lock().unwrap());

    let mut st = p.state.lock().unwrap();
    while st.finished < st.total {
        st = p.done_cv.wait(st).unwrap();
    }
    st.job = None;
    let panicked = st.panicked;
    drop(st);
    if panicked {
        panic!("a task panicked inside tensor::pool::parallel_for");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn every_task_runs_exactly_once() {
        for n in [0usize, 1, 2, 7, 64, 1000] {
            let hits: Vec<AtomicUsize> = (0..n).map(|_| AtomicUsize::new(0)).collect();
            parallel_for(n, &|t| {
                hits[t].fetch_add(1, Ordering::SeqCst);
            });
            assert!(hits.iter().all(|h| h.load(Ordering::SeqCst) == 1), "n={n}");
        }
    }

    #[test]
    fn borrows_from_the_callers_stack() {
        let data: Vec<usize> = (0..100).collect();
        let sum = AtomicUsize::new(0);
        parallel_for(100, &|t| {
            sum.fetch_add(data[t], Ordering::SeqCst);
        });
        assert_eq!(sum.load(Ordering::SeqCst), 4950);
    }

    #[test]
    fn a_panicking_task_is_reported_and_the_pool_survives() {
        let r = std::panic::catch_unwind(|| {
            parallel_for(8, &|t| {
                if t == 3 {
                    panic!("boom");
                }
            });
        });
        assert!(r.is_err());
        // Still usable afterwards.
        let n = AtomicUsize::new(0);
        parallel_for(8, &|_| {
            n.fetch_add(1, Ordering::SeqCst);
        });
        assert_eq!(n.load(Ordering::SeqCst), 8);
    }
}
