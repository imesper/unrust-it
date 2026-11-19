//! Background worker thread for non-blocking scene compilation and motion planning
//! with monotonic generation IDs and mailbox coalescing.

use crate::scene::{run_planner_sync, BuiltWorkcell, DirtyFlags, PlaybackTrajectory, WorkcellSpec};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{mpsc, Arc};
use std::thread::{self, JoinHandle};
use std::time::Instant;

/// Command dispatched from UI/controller thread to [`PlannerWorker`].
pub enum WorkerCommand {
    CompileAndPlan {
        gen: u64,
        spec: Box<WorkcellSpec>,
        dirty: DirtyFlags,
    },
    ReplanOnly {
        gen: u64,
        workcell: Arc<BuiltWorkcell>,
        spec: Box<WorkcellSpec>,
    },
    Shutdown,
}

/// Result / progress notification sent back to the UI thread from [`PlannerWorker`].
pub enum WorkerEvent {
    Status {
        gen: u64,
        phase: &'static str,
        progress: f32,
    },
    Compiled {
        gen: u64,
        workcell: Arc<BuiltWorkcell>,
        compile_ms: f64,
    },
    PlanReady {
        gen: u64,
        trajectory: Arc<PlaybackTrajectory>,
    },
    Error {
        gen: u64,
        message: String,
    },
    Cancelled {
        gen: u64,
    },
}

/// Optional thread-safe UI wakeup callback invoked after each event is pushed.
pub type WakeupCallback = Arc<dyn Fn() + Send + Sync>;

/// Owns the dedicated `"unrust-planner-worker"` thread and generation cancellation counter.
pub struct PlannerWorker {
    cmd_tx: mpsc::Sender<WorkerCommand>,
    event_rx: mpsc::Receiver<WorkerEvent>,
    latest_gen: Arc<AtomicU64>,
    handle: Option<JoinHandle<()>>,
}

impl PlannerWorker {
    /// Spawns the background planner worker thread.
    pub fn spawn(wakeup: Option<WakeupCallback>) -> Self {
        let (cmd_tx, cmd_rx) = mpsc::channel::<WorkerCommand>();
        let (event_tx, event_rx) = mpsc::channel::<WorkerEvent>();
        let latest_gen = Arc::new(AtomicU64::new(0));
        let worker_gen = Arc::clone(&latest_gen);

        let handle = thread::Builder::new()
            .name("unrust-planner-worker".to_string())
            .spawn(move || worker_loop(cmd_rx, event_tx, worker_gen, wakeup))
            .expect("failed to spawn unrust-planner-worker thread");

        Self {
            cmd_tx,
            event_rx,
            latest_gen,
            handle: Some(handle),
        }
    }

    /// Bumps and returns the new latest generation counter (superseding older in-flight jobs).
    pub fn next_generation(&self) -> u64 {
        self.latest_gen.fetch_add(1, Ordering::AcqRel) + 1
    }

    /// Returns the most recently issued generation counter.
    #[inline]
    pub fn latest_generation(&self) -> u64 {
        self.latest_gen.load(Ordering::Acquire)
    }

    /// Increments the generation counter to mark queued work cancelled.
    pub fn cancel_pending(&self) -> u64 {
        self.next_generation()
    }

    /// Submits a command to the worker mailbox.
    pub fn submit(&self, cmd: WorkerCommand) {
        let _ = self.cmd_tx.send(cmd);
    }

    /// Non-blocking receive of the next [`WorkerEvent`].
    pub fn try_recv(&self) -> Option<WorkerEvent> {
        self.event_rx.try_recv().ok()
    }
}

impl Drop for PlannerWorker {
    fn drop(&mut self) {
        let _ = self.cmd_tx.send(WorkerCommand::Shutdown);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

fn emit(tx: &mpsc::Sender<WorkerEvent>, wakeup: &Option<WakeupCallback>, evt: WorkerEvent) {
    let _ = tx.send(evt);
    if let Some(cb) = wakeup {
        cb();
    }
}

fn worker_loop(
    cmd_rx: mpsc::Receiver<WorkerCommand>,
    event_tx: mpsc::Sender<WorkerEvent>,
    latest_gen: Arc<AtomicU64>,
    wakeup: Option<WakeupCallback>,
) {
    while let Ok(mut cmd) = cmd_rx.recv() {
        // Coalesce burst slider/control edits by keeping ONLY the newest queued command
        while let Ok(newer) = cmd_rx.try_recv() {
            let prev_gen = match &cmd {
                WorkerCommand::CompileAndPlan { gen, .. } | WorkerCommand::ReplanOnly { gen, .. } => {
                    Some(*gen)
                }
                WorkerCommand::Shutdown => None,
            };
            if matches!(cmd, WorkerCommand::Shutdown) {
                return;
            }
            if let Some(g) = prev_gen {
                emit(&event_tx, &wakeup, WorkerEvent::Cancelled { gen: g });
            }
            cmd = newer;
        }

        match cmd {
            WorkerCommand::Shutdown => return,
            WorkerCommand::CompileAndPlan { gen, spec, dirty: _ } => {
                if gen < latest_gen.load(Ordering::Acquire) {
                    emit(&event_tx, &wakeup, WorkerEvent::Cancelled { gen });
                    continue;
                }
                emit(
                    &event_tx,
                    &wakeup,
                    WorkerEvent::Status {
                        gen,
                        phase: "Compiling workcell & 3D ESDF...",
                        progress: 0.18,
                    },
                );
                let t0 = Instant::now();
                let compiled = match spec.compile(gen) {
                    Ok(wc) => Arc::new(wc),
                    Err(err) => {
                        emit(
                            &event_tx,
                            &wakeup,
                            WorkerEvent::Error {
                                gen,
                                message: err.to_string(),
                            },
                        );
                        continue;
                    }
                };
                if gen < latest_gen.load(Ordering::Acquire) {
                    emit(&event_tx, &wakeup, WorkerEvent::Cancelled { gen });
                    continue;
                }
                let compile_ms = t0.elapsed().as_secs_f64() * 1000.0;
                emit(
                    &event_tx,
                    &wakeup,
                    WorkerEvent::Compiled {
                        gen,
                        workcell: Arc::clone(&compiled),
                        compile_ms,
                    },
                );

                run_plan_and_emit(gen, &compiled, &spec, &latest_gen, &event_tx, &wakeup);
            }
            WorkerCommand::ReplanOnly {
                gen,
                workcell,
                spec,
            } => {
                if gen < latest_gen.load(Ordering::Acquire) {
                    emit(&event_tx, &wakeup, WorkerEvent::Cancelled { gen });
                    continue;
                }
                run_plan_and_emit(gen, &workcell, &spec, &latest_gen, &event_tx, &wakeup);
            }
        }
    }
}

fn run_plan_and_emit(
    gen: u64,
    workcell: &Arc<BuiltWorkcell>,
    spec: &WorkcellSpec,
    latest_gen: &Arc<AtomicU64>,
    event_tx: &mpsc::Sender<WorkerEvent>,
    wakeup: &Option<WakeupCallback>,
) {
    emit(
        event_tx,
        wakeup,
        WorkerEvent::Status {
            gen,
            phase: "Running planner & trajectory verifier...",
            progress: 0.55,
        },
    );

    match run_planner_sync(workcell, spec) {
        Ok(traj) => {
            if gen < latest_gen.load(Ordering::Acquire) {
                emit(event_tx, wakeup, WorkerEvent::Cancelled { gen });
                return;
            }
            emit(
                event_tx,
                wakeup,
                WorkerEvent::PlanReady {
                    gen,
                    trajectory: Arc::new(traj),
                },
            );
        }
        Err(err) => {
            emit(
                event_tx,
                wakeup,
                WorkerEvent::Error {
                    gen,
                    message: err,
                },
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::{preset_workcell, PresetId};
    use std::sync::atomic::AtomicUsize;
    use std::time::Duration;

    #[test]
    fn worker_compiles_coalesces_stale_gens_and_plans() {
        let wakeups = Arc::new(AtomicUsize::new(0));
        let wake_clone = Arc::clone(&wakeups);
        let worker = PlannerWorker::spawn(Some(Arc::new(move || {
            wake_clone.fetch_add(1, Ordering::Relaxed);
        })));

        let spec = Box::new(preset_workcell(PresetId::DualArmRrtToppra));
        let g1 = worker.next_generation();
        worker.submit(WorkerCommand::CompileAndPlan {
            gen: g1,
            spec: spec.clone(),
            dirty: DirtyFlags::TOPO,
        });
        let g2 = worker.next_generation();
        worker.submit(WorkerCommand::CompileAndPlan {
            gen: g2,
            spec,
            dirty: DirtyFlags::TOPO,
        });

        let deadline = Instant::now() + Duration::from_secs(10);
        let mut ready_gen = None;
        while Instant::now() < deadline {
            while let Some(evt) = worker.try_recv() {
                if let WorkerEvent::PlanReady { gen, trajectory } = evt {
                    assert!(!trajectory.frames.is_empty());
                    ready_gen = Some(gen);
                }
            }
            if ready_gen.is_some() {
                break;
            }
            thread::sleep(Duration::from_millis(10));
        }

        assert_eq!(ready_gen, Some(g2));
        assert!(wakeups.load(Ordering::Relaxed) >= 2);
    }
}
