//! Direct Rust application state, background `PlannerWorker` coordination,
//! 3D picking, playback, and ACM matrix view building for the `egui` Studio UI.

use crate::render::{
    extract_render_scene_with_highlights, pick_entity_along_ray, ImportedFrameMarker, OrbitCamera,
    OverlayFlags, PickedEntity, RenderSceneData, SceneHighlights,
};
use crate::scene::{
    arm_palette_color, build_imported_frame_markers, evaluate_frame_snapshot,
    evaluate_imported_trajectory_set, preset_workcell, resolve_trajectory_set_for_workcell,
    sample_trajectory_set_for_workcell, workcell_spec_to_world_spec, world_spec_to_workcell_spec,
    ArmSpec, ArmTargetSpec, BuiltWorkcell, DirtyFlags, FrameSnapshot, ObstacleRole, ObstacleSpec,
    PlaybackTrajectory, PoseSpec, PresetId, ShapePrimitiveSpec, ToolPreset, UrdfSource,
    WorkcellSpec,
};
use unrust_it_io::{ResolvedTrajectorySet, TrajectorySetSpec, WorldSpec};
use crate::worker::{PlannerWorker, WakeupCallback, WorkerCommand, WorkerEvent};
use crate::{generate_scenarios, render_visualizer_html, scenarios_to_json};
use nalgebra::Vector3;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};
use unrust_it::{
    BatchIkSolver, CollidingPair, CollisionEnvironment, CollisionRule, CollisionWorkspace,
    CpuSimdBatchBackend, LinkId, ObstacleId, TaskPose,
};

pub const SPEED_FACTORS: [f64; 5] = [0.1, 0.25, 0.5, 1.0, 2.0];
pub const SPEED_LABELS: [&str; 5] = ["0.1x", "0.25x", "0.5x", "1.0x", "2.0x"];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CenterViewMode {
    Viewport,
    AcmMatrix,
    Split,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SceneTab {
    Tree,
    Inspector,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlannerTab {
    PlanCbf,
    TargetsIk,
    JointsJog,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolverStatus {
    Idle,
    Running,
    Converged,
    Warning,
    Failed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AcmSliceMode {
    InterArm,
    SelfCollision,
    ToolsAndWorkpieces,
    LinksAndObstacles,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AcmRuleKind {
    Diagonal,
    Enabled,
    CustomMargin,
    DisabledAdjacent,
    DisabledNever,
    DisabledDefault,
    DisabledDynamic,
    TouchExempt,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrimitiveKind {
    Sphere,
    Cuboid,
    Cylinder,
    Capsule,
}

#[derive(Debug, Clone)]
pub struct AcmHeaderInfo {
    pub key: String,
    pub short_label: String,
    pub full_label: String,
    pub accent: [f32; 4],
}

#[derive(Debug, Clone)]
pub struct AcmCellInfo {
    pub row_key: String,
    pub col_key: String,
    pub rule: AcmRuleKind,
    pub margin_mm: f32,
    pub activation_mm: f32,
    pub label: String,
    pub live_dist_mm: f32,
    pub near_violation: bool,
    pub selected: bool,
}

#[derive(Debug, Clone)]
pub struct SelectedAcmPairInfo {
    pub valid: bool,
    pub row_key: String,
    pub col_key: String,
    pub pair_title: String,
    pub pair_kind_badge: &'static str,
    pub rule: AcmRuleKind,
    pub locked: bool,
    pub margin_mm: f32,
    pub activation_mm: f32,
    pub live_dist_mm: f32,
}

#[derive(Debug, Clone)]
pub struct AcmMatrixView {
    pub col_headers: Vec<AcmHeaderInfo>,
    pub row_headers: Vec<AcmHeaderInfo>,
    pub cells: Vec<Vec<AcmCellInfo>>,
    pub selected_pair: Option<SelectedAcmPairInfo>,
}

pub enum UiEntity {
    Link(u16),
    Payload(String),
    Obstacle(u32),
}

pub fn parse_entity(key: &str) -> Option<UiEntity> {
    if let Some(rest) = key.strip_prefix("L:") {
        rest.parse::<u16>().ok().map(UiEntity::Link)
    } else if let Some(rest) = key.strip_prefix("P:") {
        Some(UiEntity::Payload(rest.to_string()))
    } else if let Some(rest) = key.strip_prefix("O:") {
        rest.parse::<u32>().ok().map(UiEntity::Obstacle)
    } else {
        None
    }
}

/// Complete UI-thread state for the `egui` Desktop Engineering Studio.
pub struct StudioState {
    // Workcell & Planner Runtime
    pub spec: WorkcellSpec,
    pub workcell: Arc<BuiltWorkcell>,
    pub trajectory: Option<Arc<PlaybackTrajectory>>,
    pub workspace: CollisionWorkspace,
    pub env_scratch: CollisionEnvironment,
    pub last_snapshot: Option<FrameSnapshot>,
    pub current_q: Vec<f64>,
    pub current_frame_idx: usize,
    pub worker: PlannerWorker,
    pub active_preset_index: usize,
    pub gpu_adapter_label: String,

    // Worker diagnostics
    pub solver_status: SolverStatus,
    pub solver_progress: f32,
    pub solver_message: String,

    // Layout & Tab State
    pub center_mode: CenterViewMode,
    pub scene_tab: SceneTab,
    pub planner_tab: PlannerTab,
    pub left_collapsed: bool,
    pub right_collapsed: bool,
    pub plots_collapsed: bool,
    pub layers_open: bool,
    pub live_jog_mode: bool,

    // Viewport & Camera State
    pub camera: OrbitCamera,
    pub overlays: OverlayFlags,
    pub render_scene: RenderSceneData,
    pub viewport_dirty: bool,
    pub viewport_logical_size: (u32, u32),

    // Playback State
    pub playing: bool,
    pub looping: bool,
    pub playback_time: f64,
    pub speed_index: usize,
    pub last_playback_tick: Instant,

    // Selection & Tree State
    pub selected_arm_idx: usize,
    pub selected_target_arm: usize,
    pub selected_tree_key: String,
    pub selected_link_id: Option<LinkId>,
    pub selected_payload_name: Option<String>,
    pub selected_obstacle_idx: Option<usize>,
    pub selected_info_summary: String,
    pub expanded_keys: HashSet<String>,

    // ACM Matrix State
    pub acm_slice: AcmSliceMode,
    pub acm_name_filter: String,
    pub acm_only_active_or_custom: bool,
    pub acm_selected_pair: Option<(String, String)>,
    pub acm_edit_margin_mm: f32,
    pub acm_edit_act_mm: f32,

    // Imported Trajectories & Hierarchical Frame/Subframe Markers
    pub imported_trajectory_set: Option<TrajectorySetSpec>,
    pub imported_resolved: Option<ResolvedTrajectorySet>,
    pub imported_source_label: Option<String>,
    pub imported_frame_markers: Vec<ImportedFrameMarker>,
    pub imported_cbf_filter: bool,
    pub selected_imported_node: Option<String>,

    // Debounced Recompile / Replan
    pending_debounce: Option<(Instant, DirtyFlags)>,
}

impl StudioState {
    /// Constructs a new Studio session initialized to [`PresetId::ThreeArmCloseProximity`].
    pub fn new(wakeup: Option<WakeupCallback>, gpu_adapter_label: String) -> Self {
        let worker = PlannerWorker::spawn(wakeup);
        let spec = preset_workcell(PresetId::ThreeArmCloseProximity);
        let workcell = Arc::new(spec.compile(1).expect("default preset compiles"));
        let env_scratch = workcell.env.clone();
        let workspace = workcell.env.create_workspace(&workcell.model);
        let current_q = workcell.q_start.0.clone();

        let mut expanded_keys = HashSet::new();
        expanded_keys.insert("group:robots".to_string());
        expanded_keys.insert("group:obstacles".to_string());
        expanded_keys.insert("arm:0".to_string());

        let mut state = Self {
            spec,
            workcell,
            trajectory: None,
            workspace,
            env_scratch,
            last_snapshot: None,
            current_q,
            current_frame_idx: 0,
            worker,
            active_preset_index: 0,
            gpu_adapter_label,
            solver_status: SolverStatus::Idle,
            solver_progress: 0.0,
            solver_message: "Ready".to_string(),
            center_mode: CenterViewMode::Viewport,
            scene_tab: SceneTab::Tree,
            planner_tab: PlannerTab::PlanCbf,
            left_collapsed: false,
            right_collapsed: false,
            plots_collapsed: false,
            layers_open: false,
            live_jog_mode: false,
            camera: OrbitCamera::default(),
            overlays: OverlayFlags::default(),
            render_scene: RenderSceneData::default(),
            viewport_dirty: true,
            viewport_logical_size: (900, 620),
            playing: false,
            looping: true,
            playback_time: 0.0,
            speed_index: 3,
            last_playback_tick: Instant::now(),
            selected_arm_idx: 0,
            selected_target_arm: 0,
            selected_tree_key: "arm:0".to_string(),
            selected_link_id: None,
            selected_payload_name: None,
            selected_obstacle_idx: None,
            selected_info_summary: "Select an arm, link, tool, or obstacle".to_string(),
            expanded_keys,
            acm_slice: AcmSliceMode::InterArm,
            acm_name_filter: String::new(),
            acm_only_active_or_custom: false,
            acm_selected_pair: None,
            acm_edit_margin_mm: 40.0,
            acm_edit_act_mm: 160.0,
            imported_trajectory_set: None,
            imported_resolved: None,
            imported_source_label: None,
            imported_frame_markers: Vec::new(),
            imported_cbf_filter: true,
            selected_imported_node: None,
            pending_debounce: None,
        };

        state.clamp_esdf_slice_coord();
        state.refresh_pose_and_scene();
        state.dispatch_plan(DirtyFlags::PLAN);
        state
    }

    pub fn clamp_esdf_slice_coord(&mut self) {
        let axis = self.overlays.esdf_slice_axis.min(2) as usize;
        let lo = self.spec.esdf.origin[axis];
        let hi = lo + self.spec.esdf.voxel_size * (self.spec.esdf.dims[axis] as f32);
        self.overlays.esdf_slice_coord = self.overlays.esdf_slice_coord.clamp(lo, hi);
    }

    pub fn esdf_axis_bounds(&self) -> (f32, f32) {
        let axis = self.overlays.esdf_slice_axis.min(2) as usize;
        let lo = self.spec.esdf.origin[axis];
        let hi = lo + self.spec.esdf.voxel_size * (self.spec.esdf.dims[axis] as f32);
        (lo, hi)
    }

    pub fn dispatch_compile_and_plan(&mut self, dirty: DirtyFlags) {
        self.pending_debounce = None;
        if !dirty.needs_rules_recompile() {
            self.dispatch_plan(dirty);
            return;
        }
        let gen = self.worker.next_generation();
        self.solver_status = SolverStatus::Running;
        self.solver_progress = 0.1;
        self.solver_message = "Compiling workcell...".to_string();
        self.worker.submit(WorkerCommand::CompileAndPlan {
            gen,
            spec: Box::new(self.spec.clone()),
            dirty,
        });
    }

    pub fn dispatch_plan(&mut self, _dirty: DirtyFlags) {
        self.pending_debounce = None;
        let gen = self.worker.next_generation();
        self.solver_status = SolverStatus::Running;
        self.solver_progress = 0.35;
        self.solver_message = "Planning trajectory...".to_string();
        self.worker.submit(WorkerCommand::ReplanOnly {
            gen,
            workcell: Arc::clone(&self.workcell),
            spec: Box::new(self.spec.clone()),
        });
    }

    pub fn schedule_debounced(&mut self, dirty: DirtyFlags) {
        let merged = match self.pending_debounce {
            Some((_, prev)) => prev.union(dirty),
            None => dirty,
        };
        self.pending_debounce = Some((Instant::now() + Duration::from_millis(85), merged));
    }

    pub fn cancel_plan(&mut self) {
        self.worker.cancel_pending();
        self.solver_status = SolverStatus::Idle;
        self.solver_progress = 0.0;
        self.solver_message = "Cancelled".to_string();
    }

    /// Processes any queued debounce timer and advances timeline playback in wall-clock time.
    pub fn tick_timers(&mut self, ctx: &egui::Context) {
        let now = Instant::now();
        if let Some((deadline, dirty)) = self.pending_debounce {
            if now >= deadline {
                self.pending_debounce = None;
                self.dispatch_compile_and_plan(dirty);
            } else {
                ctx.request_repaint_after(deadline.saturating_duration_since(now));
            }
        }

        let dt = now
            .saturating_duration_since(self.last_playback_tick)
            .as_secs_f64()
            .min(0.1);
        self.last_playback_tick = now;

        if self.playing {
            let Some(traj) = self.trajectory.as_ref().map(Arc::clone) else {
                return;
            };
            if traj.frames.len() >= 2 {
                let dur = traj.metrics.duration_s.max(1e-3);
                let speed = SPEED_FACTORS[self.speed_index];
                self.playback_time += dt * speed;
                if self.playback_time > dur {
                    if self.looping {
                        self.playback_time %= dur;
                    } else {
                        self.playback_time = dur;
                        self.playing = false;
                    }
                }
                let frac = (self.playback_time / dur).clamp(0.0, 1.0);
                let idx = ((frac * ((traj.frames.len() - 1) as f64)).round() as usize)
                    .min(traj.frames.len() - 1);
                if idx != self.current_frame_idx || self.current_q != traj.frames[idx].q {
                    self.current_frame_idx = idx;
                    self.current_q = traj.frames[idx].q.clone();
                    self.refresh_pose_and_scene();
                }
            }
            if self.playing {
                ctx.request_repaint();
            }
        }
    }

    /// Drains notifications from [`PlannerWorker`] and updates scene/trajectory state.
    pub fn drain_worker_events(&mut self) {
        let latest = self.worker.latest_generation();
        while let Some(evt) = self.worker.try_recv() {
            match evt {
                WorkerEvent::Status {
                    gen,
                    phase,
                    progress,
                } if gen >= latest => {
                    self.solver_status = SolverStatus::Running;
                    self.solver_progress = progress;
                    self.solver_message = phase.to_string();
                }
                WorkerEvent::Compiled {
                    gen,
                    workcell,
                    compile_ms: _,
                } if gen >= latest => {
                    self.workcell = workcell;
                    self.env_scratch = self.workcell.env.clone();
                    self.workspace = self.workcell.env.create_workspace(&self.workcell.model);
                    if self.current_q.len() != self.workcell.model.dof() {
                        self.current_q = self.workcell.q_start.0.clone();
                    }
                    if let Some(lid) = self.selected_link_id {
                        if lid.0 as usize >= self.workcell.model.num_links() {
                            self.selected_link_id = None;
                        }
                    }
                    if let Some(idx) = self.selected_obstacle_idx {
                        if idx >= self.spec.obstacles.len() {
                            self.selected_obstacle_idx = None;
                        }
                    }
                    if let Some(ref pname) = self.selected_payload_name {
                        if !self.workcell.payload_by_name.contains_key(pname) {
                            self.selected_payload_name = None;
                        }
                    }
                    if !self.workcell.arms.is_empty() {
                        self.selected_arm_idx = self.selected_arm_idx.min(self.workcell.arms.len() - 1);
                        self.selected_target_arm =
                            self.selected_target_arm.min(self.workcell.arms.len() - 1);
                    }
                    self.clamp_esdf_slice_coord();
                    self.refresh_pose_and_scene();
                }
                WorkerEvent::PlanReady { gen, trajectory } if gen >= latest => {
                    let msg = format!(
                        "Converged in {:.1} ms ({} frames, min clr {:.1} mm)",
                        trajectory.metrics.solve_time_ms,
                        trajectory.frames.len(),
                        trajectory.metrics.min_clearance_mm
                    );
                    self.solver_status = if trajectory.metrics.convex_verified {
                        SolverStatus::Converged
                    } else {
                        SolverStatus::Warning
                    };
                    self.solver_progress = 1.0;
                    self.solver_message = msg;
                    self.trajectory = Some(trajectory);
                    self.current_frame_idx = 0;
                    self.playback_time = 0.0;
                    if let Some(f) = self.trajectory.as_ref().and_then(|t| t.frames.first()) {
                        self.current_q = f.q.clone();
                    }
                    self.refresh_pose_and_scene();
                }
                WorkerEvent::Error { gen, message } if gen >= latest => {
                    self.solver_status = SolverStatus::Failed;
                    self.solver_progress = 1.0;
                    self.solver_message = format!("Error: {message}");
                }
                _ => {}
            }
        }
    }

    pub fn refresh_pose_and_scene(&mut self) {
        let dyn_obs: Vec<(ObstacleId, nalgebra::Isometry3<f64>)> = self
            .trajectory
            .as_ref()
            .and_then(|t| t.frames.get(self.current_frame_idx))
            .map_or_else(Vec::new, |f| f.dynamic_obstacles.clone());

        let snap = evaluate_frame_snapshot(
            &self.workcell,
            &self.current_q,
            &dyn_obs,
            self.overlays.max_witnesses,
            if dyn_obs.is_empty() {
                None
            } else {
                Some(&mut self.env_scratch)
            },
            &mut self.workspace,
        );

        let mut highlights = build_scene_highlights(
            &self.workcell,
            &snap,
            self.selected_link_id,
            self.selected_payload_name.as_deref(),
            self.selected_obstacle_idx,
            self.acm_selected_pair.as_ref(),
        );
        highlights.imported_frames = self.imported_frame_markers.clone();
        self.render_scene = extract_render_scene_with_highlights(
            &self.workcell,
            &self.spec,
            &snap,
            self.trajectory.as_deref(),
            self.camera.eye_position(),
            &self.overlays,
            &highlights,
        );
        self.last_snapshot = Some(snap);
        self.viewport_dirty = true;
    }

    pub fn load_preset(&mut self, idx: usize) {
        let idx = idx.min(PresetId::ALL.len() - 1);
        let preset = PresetId::ALL[idx];
        self.active_preset_index = idx;
        self.spec = preset_workcell(preset);
        self.playing = false;
        self.selected_arm_idx = 0;
        self.selected_target_arm = 0;
        self.selected_tree_key = "arm:0".to_string();
        self.selected_link_id = None;
        self.selected_payload_name = None;
        self.selected_obstacle_idx = None;
        self.acm_selected_pair = None;
        self.clear_imported_state_only();
        self.dispatch_compile_and_plan(DirtyFlags::TOPO);
    }

    pub fn set_camera_preset(&mut self, preset: &str) {
        match preset {
            "iso" => self.camera.set_iso(),
            "top" => self.camera.set_top(),
            "front" => self.camera.set_front(),
            "side" => self.camera.set_side(),
            "workpiece" => self.camera.focus_point(Vector3::new(0.0, 0.0, 0.42), 0.48),
            _ => self.camera.set_iso(),
        }
        self.viewport_dirty = true;
    }

    pub fn pick_ray(&mut self, norm_x: f32, norm_y: f32) {
        let Some(snap) = self.last_snapshot.as_ref() else {
            return;
        };
        let aspect = (self.viewport_logical_size.0 as f32)
            / (self.viewport_logical_size.1.max(1) as f32);
        let (origin, dir) = self.camera.screen_ray(norm_x, norm_y, aspect);
        let hit = pick_entity_along_ray(&self.workcell, snap, origin, dir);

        self.selected_link_id = None;
        self.selected_payload_name = None;
        self.selected_obstacle_idx = None;

        self.selected_info_summary = match hit {
            Some(PickedEntity::Link(lid)) => {
                let (link_name, r_idx) = {
                    let link = self.workcell.model.link(lid);
                    (link.name.clone(), link.robot.0 as usize)
                };
                let arm_idx = r_idx.min(self.spec.arms.len().saturating_sub(1));
                self.selected_link_id = Some(lid);
                self.selected_arm_idx = arm_idx;
                self.selected_tree_key = format!("link:{}", lid.0);
                self.expanded_keys.insert("group:robots".to_string());
                self.expanded_keys.insert(format!("arm:{arm_idx}"));
                format!("Picked link {link_name} (Arm {arm_idx})")
            }
            Some(PickedEntity::Payload(pid_idx)) => {
                if let Some(p) = self.workcell.env.attached_payloads().get(pid_idx).cloned() {
                    let r_idx = self.workcell.model.link(p.parent_link).robot.0 as usize;
                    let arm_idx = r_idx.min(self.spec.arms.len().saturating_sub(1));
                    self.selected_payload_name = Some(p.id.clone());
                    self.selected_arm_idx = arm_idx;
                    self.selected_tree_key = format!("tool:{arm_idx}");
                    self.expanded_keys.insert("group:robots".to_string());
                    self.expanded_keys.insert(format!("arm:{arm_idx}"));
                    format!("Picked tool {} (Arm {arm_idx})", p.id)
                } else {
                    "No entity at cursor".to_string()
                }
            }
            Some(PickedEntity::Obstacle(oid)) => {
                let found = self
                    .workcell
                    .obstacles
                    .iter()
                    .enumerate()
                    .find(|(_, o)| o.obstacle_id == oid)
                    .map(|(idx, obs)| (idx, obs.label.clone(), obs.uid.clone()));
                if let Some((idx, label, uid)) = found {
                    self.selected_obstacle_idx = Some(idx);
                    self.selected_tree_key = format!("obs:{idx}");
                    self.expanded_keys.insert("group:obstacles".to_string());
                    format!("Picked obstacle {label} ({uid})")
                } else {
                    "No entity at cursor".to_string()
                }
            }
            None => "No entity hit by 3D pick ray".to_string(),
        };
        self.refresh_pose_and_scene();
    }

    pub fn toggle_play(&mut self) {
        self.playing = !self.playing;
        self.last_playback_tick = Instant::now();
    }

    pub fn step_frame(&mut self, delta: i32) {
        let total = self.trajectory.as_ref().map_or(1, |t| t.frames.len());
        if total == 0 {
            return;
        }
        let next = (self.current_frame_idx as i32 + delta).rem_euclid(total as i32) as usize;
        self.current_frame_idx = next;
        if let Some((q, time)) = self
            .trajectory
            .as_ref()
            .and_then(|t| t.frames.get(next))
            .map(|f| (f.q.clone(), f.time))
        {
            self.current_q = q;
            self.playback_time = time;
        }
        self.refresh_pose_and_scene();
    }

    pub fn seek_fraction(&mut self, frac: f32) {
        let target_frame = self.trajectory.as_ref().and_then(|traj| {
            let n = traj.frames.len();
            if n > 0 {
                let idx = ((frac.clamp(0.0, 1.0) * ((n - 1) as f32)).round() as usize).min(n - 1);
                Some((idx, traj.frames[idx].q.clone(), traj.frames[idx].time))
            } else {
                None
            }
        });
        if let Some((idx, q, time)) = target_frame {
            self.current_frame_idx = idx;
            self.current_q = q;
            self.playback_time = time;
            self.refresh_pose_and_scene();
        }
    }

    pub fn add_arm_preset(&mut self, is_4dof: bool) {
        let idx = self.spec.arms.len();
        let angle = (idx as f64) * 1.1;
        let (source, flange, q0) = if is_4dof {
            (UrdfSource::Builtin4Dof, "wrist_link", vec![0.0, 0.35, 0.55, 0.2])
        } else {
            (
                UrdfSource::Builtin6Dof,
                "flange_link",
                vec![0.0, -0.35, 1.35, 0.0, 0.70, 0.0],
            )
        };
        self.spec.arms.push(ArmSpec {
            uid: format!("arm{idx}"),
            label: format!("Arm {idx}"),
            urdf_source: source,
            base_pose: PoseSpec::from_xyz_yaw(
                0.56 * angle.cos(),
                0.56 * angle.sin(),
                0.0,
                angle + std::f64::consts::PI,
            ),
            flange_link_raw: flange.to_string(),
            default_tcp_z: 0.14,
            tool: ToolPreset::ParallelGripper,
            color_rgba: arm_palette_color(idx),
            q_start: q0,
            priority: 1.0,
            target: ArmTargetSpec::None,
        });
        self.selected_arm_idx = idx;
        self.expanded_keys.insert(format!("arm:{idx}"));
        self.dispatch_compile_and_plan(DirtyFlags::TOPO);
    }

    pub fn remove_arm(&mut self, idx: usize) {
        if self.spec.arms.len() > 1 && idx < self.spec.arms.len() {
            self.spec.arms.remove(idx);
            self.selected_arm_idx = self.selected_arm_idx.min(self.spec.arms.len() - 1);
            self.selected_target_arm = self.selected_target_arm.min(self.spec.arms.len() - 1);
            self.dispatch_compile_and_plan(DirtyFlags::TOPO);
        }
    }

    pub fn arrange_arms_ring(&mut self, radius: f64) {
        let k = self.spec.arms.len().max(1);
        for (i, arm) in self.spec.arms.iter_mut().enumerate() {
            let th = std::f64::consts::TAU * (i as f64) / (k as f64);
            arm.base_pose = PoseSpec::from_xyz_yaw(
                radius * th.cos(),
                radius * th.sin(),
                0.0,
                th + std::f64::consts::PI,
            );
        }
        self.dispatch_compile_and_plan(DirtyFlags::TOPO);
    }

    pub fn face_arm_origin(&mut self, robot_id: usize) {
        if let Some(arm) = self.spec.arms.get_mut(robot_id) {
            let yaw = (-arm.base_pose.xyz[1]).atan2(-arm.base_pose.xyz[0]);
            arm.base_pose.rpy[2] = yaw;
        }
        self.dispatch_compile_and_plan(DirtyFlags::TOPO);
    }

    pub fn add_obstacle(&mut self, kind: PrimitiveKind, is_wp: bool) {
        let idx = self.spec.obstacles.len();
        let shape = match kind {
            PrimitiveKind::Cuboid => ShapePrimitiveSpec::Cuboid {
                half_extents: [0.06, 0.06, 0.05],
            },
            PrimitiveKind::Cylinder => ShapePrimitiveSpec::CylinderZ {
                half_height: 0.07,
                radius: 0.05,
            },
            PrimitiveKind::Capsule => ShapePrimitiveSpec::CapsuleZ {
                half_height: 0.06,
                radius: 0.04,
            },
            PrimitiveKind::Sphere => ShapePrimitiveSpec::Sphere { radius: 0.08 },
        };
        self.spec.obstacles.push(ObstacleSpec {
            uid: format!("obs_{idx}"),
            label: if is_wp {
                format!("Workpiece {idx}")
            } else {
                format!("Obstacle {idx}")
            },
            shape,
            pose: PoseSpec::translation(0.20, -0.25 + 0.15 * ((idx % 4) as f64), 0.45),
            role: if is_wp {
                ObstacleRole::Workpiece
            } else {
                ObstacleRole::StaticObstacle
            },
            color_rgba: if is_wp {
                [0.95, 0.68, 0.22, 1.0]
            } else {
                [0.96, 0.42, 0.35, 1.0]
            },
            safety_margin: None,
            activation_distance: None,
            carve_esdf: true,
        });
        self.selected_link_id = None;
        self.selected_payload_name = None;
        self.selected_obstacle_idx = Some(idx);
        self.selected_tree_key = format!("obs:{idx}");
        self.expanded_keys.insert("group:obstacles".to_string());
        self.dispatch_compile_and_plan(DirtyFlags::OBSTACLES);
    }

    pub fn add_workpiece_box(&mut self) {
        let idx = self.spec.obstacles.len();
        self.spec.obstacles.push(ObstacleSpec {
            uid: format!("wp_box_{idx}"),
            label: format!("Workpiece Box {idx}"),
            shape: ShapePrimitiveSpec::Cuboid {
                half_extents: [0.06, 0.06, 0.05],
            },
            pose: PoseSpec::translation(0.0, 0.0, 0.42),
            role: ObstacleRole::Workpiece,
            color_rgba: [0.95, 0.68, 0.22, 1.0],
            safety_margin: Some(0.004),
            activation_distance: Some(0.020),
            carve_esdf: true,
        });
        self.selected_link_id = None;
        self.selected_payload_name = None;
        self.selected_obstacle_idx = Some(idx);
        self.selected_tree_key = format!("obs:{idx}");
        self.expanded_keys.insert("group:obstacles".to_string());
        self.dispatch_compile_and_plan(DirtyFlags::OBSTACLES);
    }

    pub fn remove_obstacle(&mut self, idx: usize) {
        if idx < self.spec.obstacles.len() {
            self.spec.obstacles.remove(idx);
            self.selected_obstacle_idx = if self.spec.obstacles.is_empty() {
                None
            } else {
                Some(idx.min(self.spec.obstacles.len() - 1))
            };
            self.dispatch_compile_and_plan(DirtyFlags::OBSTACLES);
        }
    }

    pub fn toggle_tool_touch_link(&mut self, r_idx: usize, link_name: &str, exempt: bool) {
        let current_names: Vec<String> = self
            .workcell
            .arms
            .get(r_idx)
            .and_then(|arm_c| {
                let pid = arm_c.payload_id?;
                let pay = self.workcell.env.attached_payload(pid)?;
                Some(
                    pay.touch_links
                        .iter()
                        .map(|&lid| self.workcell.model.link(lid).name.clone())
                        .collect(),
                )
            })
            .unwrap_or_default();

        if let Some(arm) = self.spec.arms.get_mut(r_idx) {
            if let Some(mut custom) = arm.tool.expand(&arm.uid, arm.default_tcp_z) {
                let mut names = custom.touch_link_names.take().unwrap_or(current_names);
                if exempt {
                    if !names.iter().any(|n| n == link_name) {
                        names.push(link_name.to_string());
                    }
                } else {
                    names.retain(|n| n != link_name);
                }
                custom.touch_link_names = Some(names);
                arm.tool = ToolPreset::Custom(Box::new(custom));
                self.dispatch_compile_and_plan(DirtyFlags::TOPO);
            }
        }
    }

    pub fn copy_current_tcp_to_target(&mut self, r_idx: usize) {
        let tcp_iso = self
            .last_snapshot
            .as_ref()
            .and_then(|s| s.tcp_poses.get(r_idx))
            .copied();
        if let (Some(iso), Some(arm)) = (tcp_iso, self.spec.arms.get_mut(r_idx)) {
            let pose = PoseSpec::from_isometry(&iso);
            match &mut arm.target {
                ArmTargetSpec::ApproachPose { final_pose, .. } => {
                    *final_pose = pose;
                }
                ArmTargetSpec::DirectPose(p) => {
                    *p = pose;
                }
                _ => {
                    arm.target = ArmTargetSpec::DirectPose(pose);
                }
            }
            self.refresh_pose_and_scene();
            self.schedule_debounced(DirtyFlags::PLAN);
        }
    }

    pub fn preview_ik_targets(&mut self) {
        let mut tasks = Vec::new();
        for (arm_s, arm_c) in self.spec.arms.iter().zip(self.workcell.arms.iter()) {
            let pose = match &arm_s.target {
                ArmTargetSpec::ApproachPose { final_pose, .. } => final_pose,
                ArmTargetSpec::DirectPose(pose) => pose,
                _ => continue,
            };
            tasks.push(TaskPose::new(arm_c.tcp_frame, pose.to_isometry()));
        }
        if tasks.is_empty() {
            self.solver_status = SolverStatus::Warning;
            self.solver_progress = 1.0;
            self.solver_message = "No SE(3) targets in active scenario".to_string();
            return;
        }
        match BatchIkSolver::default().solve_collision_free(
            &CpuSimdBatchBackend,
            &self.workcell.model,
            &self.workcell.flat_model,
            &self.workcell.env,
            Some(&self.workcell.esdf),
            &self.current_q,
            &tasks,
        ) {
            Ok(cfg) => {
                self.playing = false;
                self.current_q = cfg.0;
                self.solver_status = SolverStatus::Converged;
                self.solver_progress = 1.0;
                self.solver_message = format!("IK Preview solved ({} targets, 32 seeds)", tasks.len());
                self.refresh_pose_and_scene();
            }
            Err(err) => {
                self.solver_status = SolverStatus::Warning;
                self.solver_progress = 1.0;
                self.solver_message = format!("IK Preview failed: {err}");
            }
        }
    }

    pub fn import_urdf_dialog(&mut self) {
        if let Some(path) = rfd::FileDialog::new()
            .add_filter("URDF Robot", &["urdf", "xml"])
            .pick_file()
        {
            let mut srdf_path = path.clone();
            srdf_path.set_extension("srdf");
            let srdf_opt = srdf_path.exists().then_some(srdf_path);
            let idx = self.spec.arms.len();
            let th = (idx as f64) * 1.1;
            self.spec.arms.push(ArmSpec {
                uid: format!("arm{idx}"),
                label: path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("Imported Arm")
                    .to_string(),
                urdf_source: UrdfSource::FilePath {
                    urdf_path: path,
                    srdf_path: srdf_opt,
                },
                base_pose: PoseSpec::from_xyz_yaw(
                    0.56 * th.cos(),
                    0.56 * th.sin(),
                    0.0,
                    th + std::f64::consts::PI,
                ),
                flange_link_raw: "flange_link".to_string(),
                default_tcp_z: 0.12,
                tool: ToolPreset::None,
                color_rgba: arm_palette_color(idx),
                q_start: Vec::new(),
                priority: 1.0,
                target: ArmTargetSpec::None,
            });
            self.selected_arm_idx = idx;
            self.dispatch_compile_and_plan(DirtyFlags::TOPO);
        }
    }

    pub fn export_legacy_json(&self) {
        if let Some(path) = rfd::FileDialog::new()
            .set_file_name("unrust_scenarios.json")
            .add_filter("JSON", &["json"])
            .save_file()
        {
            let scens = generate_scenarios();
            let _ = std::fs::write(path, scenarios_to_json(&scens));
        }
    }

    pub fn export_legacy_html(&self) {
        if let Some(path) = rfd::FileDialog::new()
            .set_file_name("unrust_visualizer.html")
            .add_filter("HTML", &["html"])
            .save_file()
        {
            let scens = generate_scenarios();
            let _ = std::fs::write(path, render_visualizer_html(&scens));
        }
    }

    pub fn open_world_dialog(&mut self) {
        let Some(path) = rfd::FileDialog::new()
            .add_filter("Workcell World", &["toml", "json"])
            .pick_file()
        else {
            return;
        };
        match WorldSpec::load_file(&path) {
            Ok(world) => {
                self.spec = world_spec_to_workcell_spec(&world);
                self.playing = false;
                self.selected_arm_idx = 0;
                self.selected_target_arm = 0;
                self.selected_tree_key = "arm:0".to_string();
                self.selected_link_id = None;
                self.selected_payload_name = None;
                self.selected_obstacle_idx = None;
                self.acm_selected_pair = None;
                self.clear_imported_state_only();
                self.solver_status = SolverStatus::Running;
                self.solver_progress = 0.2;
                self.solver_message = format!("Loaded world {}", path.display());
                self.dispatch_compile_and_plan(DirtyFlags::TOPO);
            }
            Err(err) => {
                self.solver_status = SolverStatus::Failed;
                self.solver_message = format!("Failed to load world: {err}");
            }
        }
    }

    pub fn save_world_dialog(&mut self) {
        let Some(path) = rfd::FileDialog::new()
            .set_file_name("workcell.toml")
            .add_filter("Workcell World (TOML/JSON)", &["toml", "json"])
            .save_file()
        else {
            return;
        };
        let world = workcell_spec_to_world_spec(&self.spec);
        match world.save_file(&path) {
            Ok(()) => {
                self.solver_message = format!("Saved world to {}", path.display());
            }
            Err(err) => {
                self.solver_status = SolverStatus::Failed;
                self.solver_message = format!("Failed to save world: {err}");
            }
        }
    }

    pub fn import_trajectories_dialog(&mut self) {
        let Some(path) = rfd::FileDialog::new()
            .add_filter("Trajectories (TOML/JSON)", &["toml", "json"])
            .pick_file()
        else {
            return;
        };
        match TrajectorySetSpec::load_file(&path) {
            Ok(set) => {
                let label = path
                    .file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or("imported")
                    .to_string();
                self.load_imported_trajectory_set(set, label);
            }
            Err(err) => {
                self.solver_status = SolverStatus::Failed;
                self.solver_message = format!("Trajectory import failed: {err}");
            }
        }
    }

    pub fn load_sample_trajectories(&mut self) {
        let sample = sample_trajectory_set_for_workcell(&self.spec, &self.workcell, &self.current_q);
        self.load_imported_trajectory_set(sample, "Sample Multi-Robot Batch".to_string());
    }

    pub fn export_sample_trajectories_dialog(&mut self) {
        let Some(path) = rfd::FileDialog::new()
            .set_file_name("trajectories_sample.toml")
            .add_filter("Trajectories (TOML/JSON)", &["toml", "json"])
            .save_file()
        else {
            return;
        };
        let sample = sample_trajectory_set_for_workcell(&self.spec, &self.workcell, &self.current_q);
        match sample.save_file(&path) {
            Ok(()) => {
                self.solver_message = format!("Exported sample trajectories to {}", path.display());
            }
            Err(err) => {
                self.solver_status = SolverStatus::Failed;
                self.solver_message = format!("Failed to export sample trajectories: {err}");
            }
        }
    }

    pub fn load_imported_trajectory_set(&mut self, set: TrajectorySetSpec, source_label: String) {
        self.imported_trajectory_set = Some(set);
        self.imported_source_label = Some(source_label);
        self.selected_imported_node = None;
        self.apply_imported_trajectories();
    }

    pub fn apply_imported_trajectories(&mut self) {
        let Some(set) = self.imported_trajectory_set.as_ref() else {
            return;
        };
        let resolved = match resolve_trajectory_set_for_workcell(
            set,
            &self.spec,
            &self.workcell,
            &self.workcell.q_start.0,
        ) {
            Ok(r) => r,
            Err(err) => {
                self.imported_resolved = None;
                self.imported_frame_markers.clear();
                self.solver_status = SolverStatus::Failed;
                self.solver_message = format!("Trajectory resolve failed: {err}");
                self.refresh_pose_and_scene();
                return;
            }
        };

        self.imported_frame_markers =
            build_imported_frame_markers(&resolved, self.selected_imported_node.as_deref());
        self.imported_resolved = Some(resolved.clone());

        match evaluate_imported_trajectory_set(
            &self.spec,
            &self.workcell,
            &resolved,
            &self.workcell.q_start.0,
            self.imported_cbf_filter,
        ) {
            Ok(traj) => {
                let n_wps: usize = resolved
                    .robots
                    .iter()
                    .flat_map(|r| r.trajectories.iter())
                    .map(|t| t.waypoints.len())
                    .sum();
                let min_clr = traj.metrics.min_clearance_mm;
                let err_mm = traj.metrics.max_pos_error_mm;
                if let Some(first) = traj.frames.first() {
                    self.current_q = first.q.clone();
                }
                self.trajectory = Some(Arc::new(traj));
                self.playing = false;
                self.playback_time = 0.0;
                self.current_frame_idx = 0;
                self.solver_status = SolverStatus::Converged;
                self.solver_progress = 1.0;
                self.solver_message = format!(
                    "Imported `{}` ({} waypoints, err {:.1} mm, clr {:.1} mm)",
                    resolved.name, n_wps, err_mm, min_clr
                );
            }
            Err(err) => {
                self.solver_status = SolverStatus::Warning;
                self.solver_progress = 1.0;
                self.solver_message = format!("Preview markers loaded; eval: {err}");
            }
        }
        self.refresh_pose_and_scene();
    }

    pub fn select_imported_node(&mut self, node_key: Option<String>) {
        self.selected_imported_node = node_key;
        if let Some(ref resolved) = self.imported_resolved {
            self.imported_frame_markers =
                build_imported_frame_markers(resolved, self.selected_imported_node.as_deref());
        }
        self.refresh_pose_and_scene();
    }

    pub fn clear_imported_trajectories(&mut self) {
        self.clear_imported_state_only();
        self.refresh_pose_and_scene();
        self.dispatch_plan(DirtyFlags::PLAN);
    }

    fn clear_imported_state_only(&mut self) {
        self.imported_trajectory_set = None;
        self.imported_resolved = None;
        self.imported_source_label = None;
        self.imported_frame_markers.clear();
        self.selected_imported_node = None;
    }

    pub fn reset_acm_from_srdf(&mut self) {
        self.spec.rules.acm_overrides.clear();
        self.spec.rules.pair_overrides.clear();
        self.acm_selected_pair = None;
        self.dispatch_compile_and_plan(DirtyFlags::RULES);
    }

    pub fn select_acm_pair(&mut self, rk: String, ck: String) {
        self.acm_selected_pair = Some((rk.clone(), ck.clone()));
        let live_map = self.live_distance_lookup();
        let cell = self.describe_acm_cell(&rk, &ck, &live_map);
        self.acm_edit_margin_mm = cell.margin_mm;
        self.acm_edit_act_mm = cell.activation_mm;
        self.refresh_pose_and_scene();
    }

    pub fn toggle_acm_cell_quick(&mut self, rk: &str, ck: &str) {
        let live_map = self.live_distance_lookup();
        let cell = self.describe_acm_cell(rk, ck, &live_map);
        let enable = matches!(
            cell.rule,
            AcmRuleKind::DisabledDynamic
                | AcmRuleKind::DisabledDefault
                | AcmRuleKind::DisabledNever
        );
        let wc = Arc::clone(&self.workcell);
        apply_cell_rule_to_spec(
            &mut self.spec,
            &wc,
            rk,
            ck,
            enable,
            if enable && cell.margin_mm > 0.0 {
                Some((cell.margin_mm as f64) * 0.001)
            } else {
                None
            },
            None,
        );
        self.dispatch_compile_and_plan(DirtyFlags::RULES);
    }

    pub fn apply_acm_pair_rule(
        &mut self,
        rk: &str,
        ck: &str,
        enabled: bool,
        margin: Option<f64>,
        activation: Option<f64>,
    ) {
        let wc = Arc::clone(&self.workcell);
        apply_cell_rule_to_spec(&mut self.spec, &wc, rk, ck, enabled, margin, activation);
        self.dispatch_compile_and_plan(DirtyFlags::RULES);
    }

    pub fn build_acm_view(&self) -> AcmMatrixView {
        let mut row_entities: Vec<AcmHeaderInfo> = Vec::new();
        let mut col_entities: Vec<AcmHeaderInfo> = Vec::new();

        match self.acm_slice {
            AcmSliceMode::InterArm => {
                for l in self.workcell.model.links() {
                    let accent = self
                        .workcell
                        .arms
                        .get(l.robot.0 as usize)
                        .map_or([0.5, 0.5, 0.5, 1.0], |a| a.color_rgba);
                    let item = AcmHeaderInfo {
                        key: format!("L:{}", l.id.0),
                        short_label: short_name(&l.name),
                        full_label: l.name.clone(),
                        accent,
                    };
                    row_entities.push(item.clone());
                    col_entities.push(item);
                }
            }
            AcmSliceMode::SelfCollision => {
                let arm_idx = self
                    .selected_arm_idx
                    .min(self.workcell.arms.len().saturating_sub(1));
                if let Some(arm) = self.workcell.arms.get(arm_idx) {
                    for &lid in &arm.handle.links {
                        let l = self.workcell.model.link(lid);
                        let item = AcmHeaderInfo {
                            key: format!("L:{}", lid.0),
                            short_label: short_name(&l.name),
                            full_label: l.name.clone(),
                            accent: arm.color_rgba,
                        };
                        row_entities.push(item.clone());
                        col_entities.push(item);
                    }
                }
            }
            AcmSliceMode::ToolsAndWorkpieces => {
                for p in self.workcell.env.attached_payloads() {
                    let accent = self
                        .workcell
                        .arms
                        .get(self.workcell.model.link(p.parent_link).robot.0 as usize)
                        .map_or([0.65, 0.55, 0.98, 1.0], |a| a.color_rgba);
                    let item = AcmHeaderInfo {
                        key: format!("P:{}", p.id),
                        short_label: short_name(&p.id),
                        full_label: p.id.clone(),
                        accent,
                    };
                    row_entities.push(item.clone());
                    col_entities.push(item);
                }
                for obs in &self.workcell.obstacles {
                    col_entities.push(AcmHeaderInfo {
                        key: format!("O:{}", obs.obstacle_id.0),
                        short_label: short_name(&obs.uid),
                        full_label: obs.label.clone(),
                        accent: obs.color_rgba,
                    });
                }
                for arm in &self.workcell.arms {
                    let lid = arm.flange_link;
                    let l = self.workcell.model.link(lid);
                    col_entities.push(AcmHeaderInfo {
                        key: format!("L:{}", lid.0),
                        short_label: short_name(&l.name),
                        full_label: l.name.clone(),
                        accent: arm.color_rgba,
                    });
                }
            }
            AcmSliceMode::LinksAndObstacles => {
                for l in self.workcell.model.links() {
                    let accent = self
                        .workcell
                        .arms
                        .get(l.robot.0 as usize)
                        .map_or([0.5, 0.5, 0.5, 1.0], |a| a.color_rgba);
                    row_entities.push(AcmHeaderInfo {
                        key: format!("L:{}", l.id.0),
                        short_label: short_name(&l.name),
                        full_label: l.name.clone(),
                        accent,
                    });
                }
                for obs in &self.workcell.obstacles {
                    col_entities.push(AcmHeaderInfo {
                        key: format!("O:{}", obs.obstacle_id.0),
                        short_label: short_name(&obs.uid),
                        full_label: obs.label.clone(),
                        accent: obs.color_rgba,
                    });
                }
            }
        }

        let filter_q = self.acm_name_filter.trim().to_ascii_lowercase();
        if !filter_q.is_empty() {
            let matches_filter = |hdr: &AcmHeaderInfo| {
                hdr.short_label.to_ascii_lowercase().contains(&filter_q)
                    || hdr.full_label.to_ascii_lowercase().contains(&filter_q)
            };
            row_entities.retain(matches_filter);
            col_entities.retain(matches_filter);
        }

        let live_map = self.live_distance_lookup();
        let mut raw_cols = col_entities;
        let mut raw_rows: Vec<(AcmHeaderInfo, Vec<AcmCellInfo>)> =
            Vec::with_capacity(row_entities.len());

        for r_hdr in row_entities {
            let mut row_cells = Vec::with_capacity(raw_cols.len());
            for c_hdr in &raw_cols {
                row_cells.push(self.describe_acm_cell(&r_hdr.key, &c_hdr.key, &live_map));
            }
            raw_rows.push((r_hdr, row_cells));
        }

        if self.acm_only_active_or_custom {
            let is_active = |c: &AcmCellInfo| {
                c.row_key != c.col_key
                    && matches!(c.rule, AcmRuleKind::Enabled | AcmRuleKind::CustomMargin)
            };
            raw_rows.retain(|(_, cells)| cells.iter().any(is_active));
            let keep_cols: Vec<bool> = (0..raw_cols.len())
                .map(|ci| {
                    raw_rows
                        .iter()
                        .any(|(_, cells)| cells.get(ci).is_some_and(is_active))
                })
                .collect();
            raw_cols = raw_cols
                .into_iter()
                .enumerate()
                .filter_map(|(ci, h)| keep_cols[ci].then_some(h))
                .collect();
            for (_, cells) in &mut raw_rows {
                let retained: Vec<AcmCellInfo> = cells
                    .drain(..)
                    .enumerate()
                    .filter_map(|(ci, c)| keep_cols[ci].then_some(c))
                    .collect();
                *cells = retained;
            }
        }

        let (row_headers, cells): (Vec<_>, Vec<_>) = raw_rows.into_iter().unzip();

        let selected_pair = self.acm_selected_pair.as_ref().map(|(rk, ck)| {
            let cell = self.describe_acm_cell(rk, ck, &live_map);
            SelectedAcmPairInfo {
                valid: cell.rule != AcmRuleKind::Diagonal,
                row_key: rk.clone(),
                col_key: ck.clone(),
                pair_title: format!("{} <-> {}", self.entity_label(rk), self.entity_label(ck)),
                pair_kind_badge: self.pair_kind_badge(rk, ck),
                rule: cell.rule,
                locked: matches!(
                    cell.rule,
                    AcmRuleKind::DisabledAdjacent | AcmRuleKind::TouchExempt
                ),
                margin_mm: cell.margin_mm,
                activation_mm: cell.activation_mm,
                live_dist_mm: cell.live_dist_mm,
            }
        });

        AcmMatrixView {
            col_headers: raw_cols,
            row_headers,
            cells,
            selected_pair,
        }
    }

    pub fn live_distance_lookup(&self) -> HashMap<(String, String), f64> {
        let mut map = HashMap::new();
        let Some(ref snap) = self.last_snapshot else {
            return map;
        };
        for &(pair, dist) in &snap.all_distances_below_cutoff {
            let (ka, kb) = colliding_pair_to_keys(&self.workcell, &pair);
            map.insert((ka.clone(), kb.clone()), dist);
            map.insert((kb, ka), dist);
        }
        map
    }

    pub fn describe_acm_cell(
        &self,
        rk: &str,
        ck: &str,
        live_map: &HashMap<(String, String), f64>,
    ) -> AcmCellInfo {
        let selected = self
            .acm_selected_pair
            .as_ref()
            .is_some_and(|(a, b)| (a == rk && b == ck) || (a == ck && b == rk));
        let live_dist_mm = live_map
            .get(&(rk.to_string(), ck.to_string()))
            .map_or(-1000.0_f32, |d| (*d * 1000.0) as f32);

        if rk == ck {
            return AcmCellInfo {
                row_key: rk.to_string(),
                col_key: ck.to_string(),
                rule: AcmRuleKind::Diagonal,
                margin_mm: 0.0,
                activation_mm: 0.0,
                label: "·".to_string(),
                live_dist_mm,
                near_violation: false,
                selected,
            };
        }

        let (rule, margin_m, act_m, label) = match (parse_entity(rk), parse_entity(ck)) {
            (Some(UiEntity::Link(la)), Some(UiEntity::Link(lb))) => {
                let r = self
                    .workcell
                    .env
                    .acm()
                    .map_or(CollisionRule::Enabled, |acm| acm.rule(LinkId(la), LinkId(lb)));
                let pair = CollidingPair::LinkToLink {
                    link_a: LinkId(la),
                    link_b: LinkId(lb),
                };
                let act = self
                    .workcell
                    .env
                    .effective_pair_activation(&pair, self.spec.rules.default_body_activation);
                match r {
                    CollisionRule::Enabled => (AcmRuleKind::Enabled, 0.0, act, "DEF".to_string()),
                    CollisionRule::CustomMargin(m) => (
                        AcmRuleKind::CustomMargin,
                        m,
                        act,
                        format!("{}mm", (m * 1000.0).round() as i32),
                    ),
                    CollisionRule::DisabledAdjacent => {
                        (AcmRuleKind::DisabledAdjacent, 0.0, 0.0, "ADJ".to_string())
                    }
                    CollisionRule::DisabledNeverColliding => {
                        (AcmRuleKind::DisabledNever, 0.0, 0.0, "NEV".to_string())
                    }
                    CollisionRule::DisabledDefault => {
                        (AcmRuleKind::DisabledDefault, 0.0, 0.0, "OFF".to_string())
                    }
                    CollisionRule::DisabledDynamic => {
                        (AcmRuleKind::DisabledDynamic, 0.0, 0.0, "DYN".to_string())
                    }
                }
            }
            (Some(UiEntity::Payload(pa)), Some(UiEntity::Payload(pb))) => {
                let enabled = self.workcell.env.is_payload_pair_enabled(&pa, &pb);
                let pid_a = self.workcell.payload_by_name.get(&pa).copied();
                let pid_b = self.workcell.payload_by_name.get(&pb).copied();
                if let (Some(ia), Some(ib)) = (pid_a, pid_b) {
                    let link_a = self
                        .workcell
                        .env
                        .attached_payload(ia)
                        .map_or(LinkId(0), |p| p.parent_link);
                    let link_b = self
                        .workcell
                        .env
                        .attached_payload(ib)
                        .map_or(LinkId(0), |p| p.parent_link);
                    let pair = CollidingPair::PayloadToPayload {
                        payload_a: ia,
                        parent_link_a: link_a,
                        payload_b: ib,
                        parent_link_b: link_b,
                    };
                    let m = self
                        .workcell
                        .env
                        .effective_pair_margin(&pair, self.spec.rules.tool_tool_margin);
                    let act = self
                        .workcell
                        .env
                        .effective_pair_activation(&pair, self.spec.rules.tool_tool_activation);
                    if !enabled {
                        (AcmRuleKind::DisabledDynamic, m, act, "DYN".to_string())
                    } else {
                        (
                            AcmRuleKind::CustomMargin,
                            m,
                            act,
                            format!("{}mm", (m * 1000.0).round() as i32),
                        )
                    }
                } else {
                    (AcmRuleKind::Enabled, 0.0, 0.0, "DEF".to_string())
                }
            }
            (Some(UiEntity::Payload(p)), Some(UiEntity::Obstacle(o)))
            | (Some(UiEntity::Obstacle(o)), Some(UiEntity::Payload(p))) => {
                let enabled = self
                    .workcell
                    .env
                    .is_payload_obstacle_enabled(&p, ObstacleId(o));
                if let Some(&pid) = self.workcell.payload_by_name.get(&p) {
                    let parent_link = self
                        .workcell
                        .env
                        .attached_payload(pid)
                        .map_or(LinkId(0), |pay| pay.parent_link);
                    let pair = CollidingPair::PayloadToObstacle {
                        payload: pid,
                        parent_link,
                        obstacle: ObstacleId(o),
                    };
                    let m = self
                        .workcell
                        .env
                        .effective_pair_margin(&pair, self.spec.rules.tool_workpiece_margin);
                    let act = self
                        .workcell
                        .env
                        .effective_pair_activation(&pair, self.spec.rules.tool_workpiece_activation);
                    if !enabled {
                        (AcmRuleKind::DisabledDynamic, m, act, "DYN".to_string())
                    } else {
                        (
                            AcmRuleKind::CustomMargin,
                            m,
                            act,
                            format!("{}mm", (m * 1000.0).round() as i32),
                        )
                    }
                } else {
                    (AcmRuleKind::Enabled, 0.0, 0.0, "DEF".to_string())
                }
            }
            (Some(UiEntity::Payload(p)), Some(UiEntity::Link(l)))
            | (Some(UiEntity::Link(l)), Some(UiEntity::Payload(p))) => {
                let pid_opt = self.workcell.payload_by_name.get(&p).copied();
                let is_touch = pid_opt
                    .and_then(|pid| self.workcell.env.attached_payload(pid))
                    .is_some_and(|pay| pay.touch_links.contains(&LinkId(l)));
                if is_touch {
                    (AcmRuleKind::TouchExempt, 0.0, 0.0, "TCH".to_string())
                } else if !self.workcell.env.is_payload_link_enabled(&p, LinkId(l)) {
                    (AcmRuleKind::DisabledDynamic, 0.0, 0.0, "DYN".to_string())
                } else if let Some(pid) = pid_opt {
                    let parent_link = self
                        .workcell
                        .env
                        .attached_payload(pid)
                        .map_or(LinkId(0), |pay| pay.parent_link);
                    let pair = CollidingPair::PayloadToLink {
                        payload: pid,
                        parent_link,
                        link: LinkId(l),
                    };
                    let m = self
                        .workcell
                        .env
                        .effective_pair_margin(&pair, self.spec.rules.tool_other_link_margin);
                    let act = self
                        .workcell
                        .env
                        .effective_pair_activation(&pair, self.spec.rules.tool_other_link_activation);
                    (
                        AcmRuleKind::CustomMargin,
                        m,
                        act,
                        format!("{}mm", (m * 1000.0).round() as i32),
                    )
                } else {
                    (AcmRuleKind::Enabled, 0.0, 0.0, "DEF".to_string())
                }
            }
            (Some(UiEntity::Link(l)), Some(UiEntity::Obstacle(o)))
            | (Some(UiEntity::Obstacle(o)), Some(UiEntity::Link(l))) => {
                if !self
                    .workcell
                    .env
                    .is_link_obstacle_enabled(LinkId(l), ObstacleId(o))
                {
                    (AcmRuleKind::DisabledDynamic, 0.0, 0.0, "DYN".to_string())
                } else {
                    (
                        AcmRuleKind::Enabled,
                        0.0,
                        self.spec.rules.default_body_activation,
                        "DEF".to_string(),
                    )
                }
            }
            _ => (AcmRuleKind::Diagonal, 0.0, 0.0, "·".to_string()),
        };

        let margin_mm = (margin_m * 1000.0) as f32;
        let near_violation = matches!(rule, AcmRuleKind::Enabled | AcmRuleKind::CustomMargin)
            && live_dist_mm > -900.0
            && live_dist_mm <= margin_mm + 5.0;

        AcmCellInfo {
            row_key: rk.to_string(),
            col_key: ck.to_string(),
            rule,
            margin_mm,
            activation_mm: (act_m * 1000.0) as f32,
            label,
            live_dist_mm,
            near_violation,
            selected,
        }
    }

    fn entity_label(&self, key: &str) -> String {
        match parse_entity(key) {
            Some(UiEntity::Link(id)) => self.workcell.model.link(LinkId(id)).name.clone(),
            Some(UiEntity::Payload(id)) => id,
            Some(UiEntity::Obstacle(id)) => self
                .workcell
                .obstacles
                .get(id as usize)
                .map_or_else(|| format!("obs_{id}"), |o| o.uid.clone()),
            None => key.to_string(),
        }
    }

    fn pair_kind_badge(&self, rk: &str, ck: &str) -> &'static str {
        match (parse_entity(rk), parse_entity(ck)) {
            (Some(UiEntity::Link(_)), Some(UiEntity::Link(_))) => "Link <-> Link",
            (Some(UiEntity::Payload(_)), Some(UiEntity::Payload(_))) => "Tool <-> Tool",
            (Some(UiEntity::Payload(_)), Some(UiEntity::Obstacle(_)))
            | (Some(UiEntity::Obstacle(_)), Some(UiEntity::Payload(_))) => "Tool <-> Obstacle",
            (Some(UiEntity::Payload(_)), Some(UiEntity::Link(_)))
            | (Some(UiEntity::Link(_)), Some(UiEntity::Payload(_))) => "Tool <-> Link",
            _ => "Link <-> Obstacle",
        }
    }
}

pub fn apply_cell_rule_to_spec(
    spec: &mut WorkcellSpec,
    workcell: &BuiltWorkcell,
    rk: &str,
    ck: &str,
    enabled: bool,
    margin: Option<f64>,
    activation: Option<f64>,
) {
    let entity = |k: &str| match parse_entity(k) {
        Some(UiEntity::Link(id)) => {
            Some(crate::scene::EntityRef::Link(workcell.model.link(LinkId(id)).name.clone()))
        }
        Some(UiEntity::Payload(name)) => Some(crate::scene::EntityRef::Payload(name)),
        Some(UiEntity::Obstacle(id)) => workcell
            .obstacles
            .get(id as usize)
            .map(|o| crate::scene::EntityRef::Obstacle(o.uid.clone())),
        None => None,
    };

    if let (Some(a), Some(b)) = (entity(rk), entity(ck)) {
        spec.rules
            .pair_overrides
            .retain(|p| !((p.a == a && p.b == b) || (p.a == b && p.b == a)));
        spec.rules.pair_overrides.push(crate::scene::PairRuleOverrideSpec {
            a,
            b,
            enabled,
            margin,
            activation_distance: activation,
        });
    }
}

pub fn colliding_pair_to_keys(workcell: &BuiltWorkcell, pair: &CollidingPair) -> (String, String) {
    match *pair {
        CollidingPair::LinkToLink { link_a, link_b } => {
            (format!("L:{}", link_a.0), format!("L:{}", link_b.0))
        }
        CollidingPair::LinkToObstacle { link, obstacle } => {
            (format!("L:{}", link.0), format!("O:{}", obstacle.0))
        }
        CollidingPair::PayloadToObstacle {
            payload, obstacle, ..
        } => {
            let pname = workcell
                .env
                .attached_payloads()
                .get(payload.0 as usize)
                .map_or_else(String::new, |p| p.id.clone());
            (format!("P:{pname}"), format!("O:{}", obstacle.0))
        }
        CollidingPair::PayloadToLink { payload, link, .. } => {
            let pname = workcell
                .env
                .attached_payloads()
                .get(payload.0 as usize)
                .map_or_else(String::new, |p| p.id.clone());
            (format!("P:{pname}"), format!("L:{}", link.0))
        }
        CollidingPair::PayloadToPayload {
            payload_a,
            payload_b,
            ..
        } => {
            let pa = workcell
                .env
                .attached_payloads()
                .get(payload_a.0 as usize)
                .map_or_else(String::new, |p| p.id.clone());
            let pb = workcell
                .env
                .attached_payloads()
                .get(payload_b.0 as usize)
                .map_or_else(String::new, |p| p.id.clone());
            (format!("P:{pa}"), format!("P:{pb}"))
        }
    }
}

fn entity_world_point(
    workcell: &BuiltWorkcell,
    snap: &FrameSnapshot,
    key: &str,
) -> Option<[f64; 3]> {
    match parse_entity(key)? {
        UiEntity::Link(id) => snap
            .link_poses
            .get(id as usize)
            .map(|iso| iso.translation.vector.into()),
        UiEntity::Payload(name) => workcell
            .payload_by_name
            .get(&name)
            .and_then(|pid| snap.payload_poses.get(pid.0 as usize))
            .map(|iso| iso.translation.vector.into()),
        UiEntity::Obstacle(oid) => snap
            .obstacle_poses
            .iter()
            .find(|(id, _)| id.0 == oid)
            .map(|(_, iso)| iso.translation.vector.into())
            .or_else(|| {
                workcell
                    .obstacles
                    .iter()
                    .find(|o| o.obstacle_id.0 == oid)
                    .map(|o| o.base_pose.translation.vector.into())
            }),
    }
}

fn push_unique<T: PartialEq>(list: &mut Vec<T>, val: T) {
    if !list.contains(&val) {
        list.push(val);
    }
}

/// Builds 3D viewport highlights from selected entity state and the active ACM pair.
pub(crate) fn build_scene_highlights(
    workcell: &BuiltWorkcell,
    snap: &FrameSnapshot,
    selected_link: Option<LinkId>,
    selected_payload: Option<&str>,
    selected_obstacle: Option<usize>,
    acm_pair: Option<&(String, String)>,
) -> SceneHighlights {
    let mut highlights = SceneHighlights::default();
    if let Some(lid) = selected_link {
        push_unique(&mut highlights.highlighted_links, lid);
    }
    if let Some(pname) = selected_payload {
        push_unique(&mut highlights.highlighted_payloads, pname.to_string());
    }
    if let Some(oid) = selected_obstacle
        .and_then(|i| workcell.obstacles.get(i).map(|o| o.obstacle_id))
    {
        push_unique(&mut highlights.highlighted_obstacles, oid);
    }

    if let Some((rk, ck)) = acm_pair {
        for key in [rk.as_str(), ck.as_str()] {
            match parse_entity(key) {
                Some(UiEntity::Link(id)) => {
                    push_unique(&mut highlights.highlighted_links, LinkId(id));
                }
                Some(UiEntity::Payload(name)) => {
                    push_unique(&mut highlights.highlighted_payloads, name);
                }
                Some(UiEntity::Obstacle(oid)) => {
                    push_unique(&mut highlights.highlighted_obstacles, ObstacleId(oid));
                }
                None => {}
            }
        }
        if rk != ck {
            if let Some(w) = snap.witnesses.iter().find(|w| {
                let (ka, kb) = colliding_pair_to_keys(workcell, &w.pair);
                (ka == *rk && kb == *ck) || (ka == *ck && kb == *rk)
            }) {
                highlights.highlighted_pair_segment = Some((
                    [
                        w.witness_a_world.x,
                        w.witness_a_world.y,
                        w.witness_a_world.z,
                    ],
                    [
                        w.witness_b_world.x,
                        w.witness_b_world.y,
                        w.witness_b_world.z,
                    ],
                ));
            } else if let (Some(pa), Some(pb)) = (
                entity_world_point(workcell, snap, rk),
                entity_world_point(workcell, snap, ck),
            ) {
                highlights.highlighted_pair_segment = Some((pa, pb));
            }
        }
    }

    highlights
}

pub fn short_name(s: &str) -> String {
    let trimmed = s.trim_end_matches("_link");
    if trimmed.chars().count() <= 8 {
        trimmed.to_string()
    } else {
        trimmed.chars().take(8).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::{evaluate_frame_snapshot, preset_workcell, PresetId};

    #[test]
    fn scene_highlights_populates_pair_segment_for_tool_workpiece() {
        let spec = preset_workcell(PresetId::ThreeArmCloseProximity);
        let workcell = spec.compile(1).expect("three-arm preset compiles");
        let mut ws = workcell.env.create_workspace(&workcell.model);
        let snap = evaluate_frame_snapshot(&workcell, &workcell.q_start.0, &[], 5, None, &mut ws);

        let tool_name = workcell.arms[0]
            .payload_name
            .as_ref()
            .expect("arm0 has tool payload")
            .clone();
        let obs_id = workcell.obstacles[0].obstacle_id;
        let pair = (format!("P:{tool_name}"), format!("O:{}", obs_id.0));

        let hl = build_scene_highlights(&workcell, &snap, None, None, None, Some(&pair));
        assert_eq!(hl.highlighted_payloads, vec![tool_name]);
        assert_eq!(hl.highlighted_obstacles, vec![obs_id]);
        assert!(hl.highlighted_pair_segment.is_some());
    }

    #[test]
    fn apply_cell_rule_deduplicates_pair_overrides() {
        let mut spec = preset_workcell(PresetId::ThreeArmCloseProximity);
        let workcell = spec.compile(1).expect("preset compiles");
        spec.rules.pair_overrides.clear();

        let p0 = workcell.arms[0].payload_name.as_ref().unwrap();
        let obs0 = format!("O:{}", workcell.obstacles[0].obstacle_id.0);
        let key_p = format!("P:{p0}");

        apply_cell_rule_to_spec(&mut spec, &workcell, &key_p, &obs0, true, Some(0.01), Some(0.05));
        apply_cell_rule_to_spec(&mut spec, &workcell, &obs0, &key_p, true, Some(0.02), Some(0.08));

        assert_eq!(spec.rules.pair_overrides.len(), 1);
        assert_eq!(spec.rules.pair_overrides[0].margin, Some(0.02));
        assert_eq!(spec.rules.pair_overrides[0].activation_distance, Some(0.08));
    }

    #[test]
    fn custom_tool_touch_link_names_override_default_depth() {
        let mut spec = preset_workcell(PresetId::ThreeArmCloseProximity);
        let mut custom = spec.arms[0]
            .tool
            .expand(&spec.arms[0].uid, spec.arms[0].default_tcp_z)
            .expect("arm0 tool expands");
        custom.touch_link_names = Some(vec!["arm0_flange_link".to_string()]);
        spec.arms[0].tool = ToolPreset::Custom(Box::new(custom));

        let workcell = spec.compile(2).expect("compiled with custom touch_link_names");
        let pid = workcell.arms[0].payload_id.expect("payload exists");
        let payload = workcell.env.attached_payload(pid).expect("payload attached");
        assert_eq!(payload.touch_links.len(), 1);
        assert_eq!(
            workcell.model.link(payload.touch_links[0]).name,
            "arm0_flange_link"
        );
    }

    #[test]
    fn studio_state_initializes_filters_acm_and_steps_frames_headless() {
        let mut st = StudioState::new(None, "Headless Test · SIMD".to_string());
        assert_eq!(st.spec.arms.len(), 3);
        assert!(!st.render_scene.cubes.is_empty() || !st.render_scene.cylinders.is_empty());

        let deadline = Instant::now() + Duration::from_secs(10);
        while Instant::now() < deadline && st.trajectory.is_none() {
            st.drain_worker_events();
            std::thread::sleep(Duration::from_millis(10));
        }
        assert!(st.trajectory.is_some());

        st.acm_slice = AcmSliceMode::ToolsAndWorkpieces;
        let matrix = st.build_acm_view();
        assert!(!matrix.row_headers.is_empty());
        assert!(!matrix.col_headers.is_empty());

        st.step_frame(1);
        assert!(st.current_frame_idx >= 1 || st.trajectory.as_ref().unwrap().frames.len() == 1);
    }
}
