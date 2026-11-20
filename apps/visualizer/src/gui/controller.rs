//! UI-thread Studio controller binding Slint globals/callbacks to `WorkcellSpec`,
//! `PlannerWorker`, telemetry plot rasterizers, and 3D `SceneRenderer`.

use crate::gui::viewport_bridge::ViewportRenderState;
use crate::gui::{
    AcmCellItem, AcmHeaderItem, AcmMatrixRow, AcmRuleKind, AcmSliceMode, AcmState, ArmSummary,
    ArmTargetPose, JointSliderItem, MainWindow, ObstacleItem, PlannerState, PrimitiveKind,
    SceneNodeKind, SceneState, SceneTreeNode, SelectedAcmPair, SolverStatus, TimelineAlertMarker,
    TimelineState, ToolPresetKind, TouchLinkToggle, ViewportState,
};
use crate::plots::{
    render_joint_vel_plot, render_min_clearance_plot, render_pose_error_plot,
    render_tool_clearance_plot, PlotImage,
};
use crate::render::{
    extract_render_scene_with_highlights, pick_entity_along_ray, PickedEntity, SceneHighlights,
};
use crate::scene::{
    arm_palette_color, evaluate_frame_snapshot, preset_workcell, ArmSpec, ArmTargetSpec,
    BuiltWorkcell, DirtyFlags, FrameSnapshot, ObstacleRole, ObstacleSpec, PlaybackTrajectory,
    PoseSpec, PresetId, ShapePrimitiveSpec, ToolPreset, TrajectoryStage, UrdfSource, WorkcellSpec,
};
use crate::worker::{PlannerWorker, WorkerCommand, WorkerEvent};
use crate::{generate_scenarios, render_visualizer_html, scenarios_to_json};
use nalgebra::Vector3;
use slint::{Color, ComponentHandle, Image, ModelRc, Rgba8Pixel, SharedPixelBuffer, SharedString, VecModel};
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;
use std::sync::Arc;
use std::time::Duration;
use unrust_it::{
    BatchIkSolver, CollidingPair, CollisionEnvironment, CollisionRule, CollisionWorkspace,
    CpuSimdBatchBackend, LinkId, ObstacleId, TaskPose,
};

const SPEED_FACTORS: [f64; 5] = [0.1, 0.25, 0.5, 1.0, 2.0];

pub struct AppController {
    ui_weak: slint::Weak<MainWindow>,
    spec: WorkcellSpec,
    workcell: Arc<BuiltWorkcell>,
    trajectory: Option<Arc<PlaybackTrajectory>>,
    workspace: CollisionWorkspace,
    env_scratch: CollisionEnvironment,
    current_q: Vec<f64>,
    current_frame_idx: usize,
    playing: bool,
    playback_time: f64,
    speed_index: usize,
    selected_arm_idx: usize,
    selected_node_id: i32,
    selected_link_id: Option<LinkId>,
    selected_payload_name: Option<String>,
    selected_obstacle_idx: Option<usize>,
    expanded_keys: HashSet<String>,
    acm_slice: AcmSliceMode,
    acm_name_filter: String,
    acm_only_active_or_custom: bool,
    acm_selected_pair: Option<(String, String)>,
    plot_dims: (u32, u32),
    worker: PlannerWorker,
    viewport_state: Rc<RefCell<ViewportRenderState>>,
    debounce_timer: Option<slint::Timer>,
    playback_timer: Option<slint::Timer>,
    last_snapshot: Option<FrameSnapshot>,
}

impl AppController {
    /// Constructs the Studio controller and binds all Slint globals, callbacks, and timers.
    pub fn init(window: &MainWindow, viewport_state: Rc<RefCell<ViewportRenderState>>) -> Rc<RefCell<Self>> {
        let ui_weak = window.as_weak();
        let wake_weak = ui_weak.clone();
        let worker = PlannerWorker::spawn(Some(Arc::new(move || {
            let weak = wake_weak.clone();
            let _ = slint::invoke_from_event_loop(move || {
                if let Some(ui) = weak.upgrade() {
                    ui.invoke_drain_worker_events();
                }
            });
        })));

        let spec = preset_workcell(PresetId::ThreeArmCloseProximity);
        let workcell = Arc::new(spec.compile(1).expect("default preset compiles"));
        let env_scratch = workcell.env.clone();
        let workspace = workcell.env.create_workspace(&workcell.model);
        let current_q = workcell.q_start.0.clone();

        let mut expanded_keys = HashSet::new();
        expanded_keys.insert("group:robots".to_string());
        expanded_keys.insert("group:obstacles".to_string());
        expanded_keys.insert("arm:0".to_string());

        let ctrl = Rc::new(RefCell::new(Self {
            ui_weak: ui_weak.clone(),
            spec,
            workcell,
            trajectory: None,
            workspace,
            env_scratch,
            current_q,
            current_frame_idx: 0,
            playing: false,
            playback_time: 0.0,
            speed_index: 3,
            selected_arm_idx: 0,
            selected_node_id: 1,
            selected_link_id: None,
            selected_payload_name: None,
            selected_obstacle_idx: None,
            expanded_keys,
            acm_slice: AcmSliceMode::InterArm,
            acm_name_filter: String::new(),
            acm_only_active_or_custom: false,
            acm_selected_pair: None,
            plot_dims: (320, 110),
            worker,
            viewport_state,
            debounce_timer: None,
            playback_timer: None,
            last_snapshot: None,
        }));

        bind_callbacks(window, &ctrl);
        start_playback_timer(&ctrl);

        {
            let mut c = ctrl.borrow_mut();
            c.sync_adapter_label_ui();
            c.sync_esdf_bounds_ui();
            c.sync_scene_ui();
            c.sync_planner_ui();
            c.refresh_pose_and_scene();
            c.sync_acm_ui();
            c.dispatch_plan(DirtyFlags::PLAN);
        }

        ctrl
    }

    fn dispatch_compile_and_plan(&mut self, dirty: DirtyFlags) {
        if !dirty.needs_rules_recompile() {
            self.dispatch_plan(dirty);
            return;
        }
        let gen = self.worker.next_generation();
        self.set_worker_status(SolverStatus::Running, 0.1, "Compiling workcell...");
        self.worker.submit(WorkerCommand::CompileAndPlan {
            gen,
            spec: Box::new(self.spec.clone()),
            dirty,
        });
    }

    fn dispatch_plan(&mut self, _dirty: DirtyFlags) {
        let gen = self.worker.next_generation();
        self.set_worker_status(SolverStatus::Running, 0.35, "Planning trajectory...");
        self.worker.submit(WorkerCommand::ReplanOnly {
            gen,
            workcell: Arc::clone(&self.workcell),
            spec: Box::new(self.spec.clone()),
        });
    }

    fn schedule_debounced_compile_and_plan(ctrl: &Rc<RefCell<Self>>, dirty: DirtyFlags) {
        let timer = slint::Timer::default();
        let weak_ctrl = Rc::downgrade(ctrl);
        timer.start(slint::TimerMode::SingleShot, Duration::from_millis(85), move || {
            if let Some(c) = weak_ctrl.upgrade() {
                c.borrow_mut().dispatch_compile_and_plan(dirty);
            }
        });
        ctrl.borrow_mut().debounce_timer = Some(timer);
    }

    fn set_worker_status(&self, status: SolverStatus, progress: f32, msg: &str) {
        if let Some(ui) = self.ui_weak.upgrade() {
            let p = ui.global::<PlannerState>();
            p.set_status(status);
            p.set_progress(progress);
            p.set_status_message(SharedString::from(msg));
        }
    }

    fn drain_worker_events(&mut self) {
        let latest = self.worker.latest_generation();
        while let Some(evt) = self.worker.try_recv() {
            match evt {
                WorkerEvent::Status {
                    gen,
                    phase,
                    progress,
                } if gen >= latest => {
                    self.set_worker_status(SolverStatus::Running, progress, phase);
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
                    self.sync_esdf_bounds_ui();
                    self.sync_scene_ui();
                    self.sync_planner_ui();
                    self.refresh_pose_and_scene();
                    self.sync_acm_ui();
                }
                WorkerEvent::PlanReady { gen, trajectory } if gen >= latest => {
                    let msg = format!(
                        "Converged in {:.1} ms ({} frames, min clr {:.1} mm)",
                        trajectory.metrics.solve_time_ms,
                        trajectory.frames.len(),
                        trajectory.metrics.min_clearance_mm
                    );
                    let status = if trajectory.metrics.convex_verified {
                        SolverStatus::Converged
                    } else {
                        SolverStatus::Warning
                    };
                    self.trajectory = Some(trajectory);
                    self.current_frame_idx = 0;
                    self.playback_time = 0.0;
                    if let Some(f) = self.trajectory.as_ref().and_then(|t| t.frames.first()) {
                        self.current_q = f.q.clone();
                    }
                    self.set_worker_status(status, 1.0, &msg);
                    self.sync_diagnostics();
                    self.regenerate_plots();
                    self.refresh_pose_and_scene();
                    self.sync_acm_ui();
                }
                WorkerEvent::Error { gen, message } if gen >= latest => {
                    self.set_worker_status(SolverStatus::Failed, 1.0, &format!("Error: {message}"));
                }
                _ => {}
            }
        }
    }

    fn refresh_pose_and_scene(&mut self) {
        let dyn_obs: Vec<(ObstacleId, nalgebra::Isometry3<f64>)> = self
            .trajectory
            .as_ref()
            .and_then(|t| t.frames.get(self.current_frame_idx))
            .map_or_else(Vec::new, |f| f.dynamic_obstacles.clone());

        let max_wit = self.viewport_state.borrow().overlays.max_witnesses;
        let snap = evaluate_frame_snapshot(
            &self.workcell,
            &self.current_q,
            &dyn_obs,
            max_wit,
            if dyn_obs.is_empty() {
                None
            } else {
                Some(&mut self.env_scratch)
            },
            &mut self.workspace,
        );

        let highlights = build_scene_highlights(
            &self.workcell,
            &snap,
            self.selected_link_id,
            self.selected_payload_name.as_deref(),
            self.selected_obstacle_idx,
            self.acm_selected_pair.as_ref(),
        );
        let scene_data = {
            let vp = self.viewport_state.borrow();
            extract_render_scene_with_highlights(
                &self.workcell,
                &self.spec,
                &snap,
                self.trajectory.as_deref(),
                vp.camera.eye_position(),
                &vp.overlays,
                &highlights,
            )
        };
        {
            let mut vp = self.viewport_state.borrow_mut();
            vp.scene = scene_data;
            vp.dirty = true;
        }

        if let Some(ui) = self.ui_weak.upgrade() {
            ui.window().request_redraw();

            // Update Viewport HUD + Timeline Readouts
            let v = ui.global::<ViewportState>();
            v.set_hud_title(SharedString::from(self.spec.title.as_str()));
            let clr_mm = snap.min_clearance * 1000.0;
            let (safety_lvl, label) = if clr_mm < 2.0 {
                (2, format!("COLLISION · {clr_mm:.1} mm"))
            } else if clr_mm < 15.0 {
                (1, format!("TIGHT · {clr_mm:.1} mm"))
            } else {
                (0, format!("SAFE · {clr_mm:.1} mm"))
            };
            v.set_hud_safety_level(safety_lvl);
            v.set_hud_safety_label(SharedString::from(label));

            let tl = ui.global::<TimelineState>();
            tl.set_min_clearance_text(SharedString::from(format!("{clr_mm:.1} mm")));
            tl.set_min_clearance_severity(safety_lvl);

            if let Some(traj) = &self.trajectory {
                let n = traj.frames.len();
                tl.set_total_frames(n as i32);
                tl.set_current_frame(self.current_frame_idx as i32);
                let frac = if n > 1 {
                    (self.current_frame_idx as f32) / ((n - 1) as f32)
                } else {
                    0.0
                };
                tl.set_playhead_fraction(frac);
                let dur = traj.metrics.duration_s;
                let s1_frac = traj
                    .metrics
                    .stage1_end_time
                    .map_or(0.0, |t| if dur > 1e-6 { (t / dur) as f32 } else { 0.0 });
                tl.set_stage1_end_fraction(s1_frac);

                if let Some(f) = traj.frames.get(self.current_frame_idx) {
                    tl.set_time_text(SharedString::from(format!("{:.2} / {:.2} s", f.time, dur)));
                    tl.set_stage_badge(SharedString::from(match f.stage {
                        TrajectoryStage::SingleStage => "Planned",
                        TrajectoryStage::Stage1Coarse => "Stage 1",
                        TrajectoryStage::Stage2Refinement => "Stage 2",
                    }));
                    let max_vel = f.qd.iter().map(|d| d.abs()).fold(0.0_f64, f64::max);
                    tl.set_max_vel_text(SharedString::from(format!("{max_vel:.2} rad/s")));
                    let tool_clr = match (f.min_tool_clearance, f.min_tool_workpiece_clearance) {
                        (Some(a), Some(b)) => Some(a.min(b)),
                        (Some(a), None) => Some(a),
                        (None, Some(b)) => Some(b),
                        (None, None) => None,
                    };
                    tl.set_tool_clearance_text(SharedString::from(
                        tool_clr.map_or_else(|| "-- mm".to_string(), |d| format!("{:.1} mm", d * 1000.0)),
                    ));
                    tl.set_pose_error_text(SharedString::from(format!(
                        "{:.1} mm",
                        f.pose_error_mm.max(f.grasp_error_mm)
                    )));
                }
            }
        }

        self.last_snapshot = Some(snap);
        self.sync_joint_sliders_ui();
    }

    fn sync_adapter_label_ui(&self) {
        let Some(ui) = self.ui_weak.upgrade() else {
            return;
        };
        let gfx = if cfg!(feature = "gui") {
            if cfg!(target_os = "macos") {
                "Metal"
            } else if cfg!(target_os = "windows") {
                "DX12"
            } else {
                "Vulkan"
            }
        } else {
            "Readback"
        };
        let compute = if cfg!(feature = "gpu-compute") {
            "GPU"
        } else {
            "SIMD"
        };
        ui.global::<PlannerState>()
            .set_gpu_adapter_label(SharedString::from(format!("{gfx} · {compute}")));
    }

    fn sync_esdf_bounds_ui(&self) {
        let Some(ui) = self.ui_weak.upgrade() else {
            return;
        };
        let v = ui.global::<ViewportState>();
        let axis = v.get_esdf_slice_axis().clamp(0, 2) as usize;
        let lo = self.spec.esdf.origin[axis];
        let hi = lo + self.spec.esdf.voxel_size * (self.spec.esdf.dims[axis] as f32);
        v.set_esdf_slice_min_z(lo);
        v.set_esdf_slice_max_z(hi);
        let clamped = v.get_esdf_slice_z().clamp(lo, hi);
        v.set_esdf_slice_z(clamped);
    }

    fn sync_scene_ui(&self) {
        let Some(ui) = self.ui_weak.upgrade() else {
            return;
        };
        let sc = ui.global::<SceneState>();

        let arms: Vec<ArmSummary> = self
            .spec
            .arms
            .iter()
            .zip(self.workcell.arms.iter())
            .enumerate()
            .map(|(idx, (arm_s, arm_c))| {
                let payload = arm_c
                    .payload_id
                    .and_then(|pid| self.workcell.env.attached_payload(pid));
                ArmSummary {
                    robot_id: idx as i32,
                    name: SharedString::from(arm_s.label.as_str()),
                    prefix: SharedString::from(arm_c.prefix.as_str()),
                    source_label: SharedString::from(match &arm_s.urdf_source {
                        UrdfSource::Builtin4Dof => "Built-in 4-DOF",
                        UrdfSource::Builtin6Dof => "Built-in 6-DOF",
                        UrdfSource::InlineXml { .. } => "Imported XML",
                        UrdfSource::FilePath { .. } => "URDF File",
                    }),
                    dof: arm_c.q_range.len() as i32,
                    link_count: arm_c.handle.links.len() as i32,
                    accent: rgba_to_slint_color(arm_s.color_rgba),
                    base_x: arm_s.base_pose.xyz[0] as f32,
                    base_y: arm_s.base_pose.xyz[1] as f32,
                    base_z: arm_s.base_pose.xyz[2] as f32,
                    base_yaw_deg: arm_s.base_pose.rpy[2].to_degrees() as f32,
                    tool_preset: tool_to_slint(&arm_s.tool),
                    tool_name: SharedString::from(arm_c.payload_name.as_deref().unwrap_or("None")),
                    tool_margin_mm: (payload
                        .and_then(|p| p.safety_margin)
                        .unwrap_or(self.spec.rules.tool_tool_margin)
                        * 1000.0) as f32,
                    tool_activation_mm: (payload
                        .and_then(|p| p.activation_distance)
                        .unwrap_or(self.spec.rules.tool_tool_activation)
                        * 1000.0) as f32,
                    tool_mass_kg: payload.map_or(0.0, |p| p.mass) as f32,
                    tcp_z_offset_mm: (arm_s.default_tcp_z * 1000.0) as f32,
                    priority_weight: arm_s.priority as f32,
                }
            })
            .collect();

        let sel_idx = self.selected_arm_idx.min(arms.len().saturating_sub(1));
        if let Some(sel) = arms.get(sel_idx).cloned() {
            sc.set_selected_arm_index(sel_idx as i32);
            sc.set_selected_arm(sel);
        } else {
            sc.set_selected_arm_index(-1);
        }

        let touch_toggles: Vec<TouchLinkToggle> = self
            .workcell
            .arms
            .get(sel_idx)
            .and_then(|arm_c| {
                let pid = arm_c.payload_id?;
                let pay = self.workcell.env.attached_payload(pid)?;
                Some(
                    arm_c
                        .handle
                        .links
                        .iter()
                        .map(|&lid| TouchLinkToggle {
                            link_id: lid.0 as i32,
                            name: SharedString::from(self.workcell.model.link(lid).name.as_str()),
                            exempt: pay.touch_links.contains(&lid),
                        })
                        .collect(),
                )
            })
            .unwrap_or_default();
        sc.set_selected_arm_touch_links(ModelRc::new(VecModel::from(touch_toggles)));

        if let Some(obs_idx) = self
            .selected_obstacle_idx
            .filter(|&i| i < self.spec.obstacles.len())
        {
            let obs = &self.spec.obstacles[obs_idx];
            sc.set_has_selected_obstacle(true);
            sc.set_selected_obstacle(ObstacleItem {
                obstacle_id: obs_idx as i32,
                name: SharedString::from(obs.label.as_str()),
                kind: shape_to_primitive_kind(&obs.shape),
                is_workpiece: matches!(obs.role, ObstacleRole::Workpiece),
                carve_esdf: obs.carve_esdf,
                pos_x: obs.pose.xyz[0] as f32,
                pos_y: obs.pose.xyz[1] as f32,
                pos_z: obs.pose.xyz[2] as f32,
                margin_mm: (obs
                    .safety_margin
                    .unwrap_or(self.spec.rules.default_body_margin)
                    * 1000.0) as f32,
                activation_mm: (obs
                    .activation_distance
                    .unwrap_or(self.spec.rules.default_body_activation)
                    * 1000.0) as f32,
            });
        } else {
            sc.set_has_selected_obstacle(false);
        }

        sc.set_arms(ModelRc::new(VecModel::from(arms)));
        sc.set_total_dof(self.workcell.model.dof() as i32);
        sc.set_total_links(self.workcell.model.num_links() as i32);
        sc.set_total_tools(self.workcell.env.attached_payloads().len() as i32);

        // Build hierarchical indented SceneTreeNode model
        let mut nodes = Vec::new();
        let mut next_id = 0_i32;
        let has_entity_selection = self.selected_link_id.is_some()
            || self.selected_payload_name.is_some()
            || self.selected_obstacle_idx.is_some();

        let robots_open = self.expanded_keys.contains("group:robots");
        nodes.push(SceneTreeNode {
            node_id: next_id,
            kind: SceneNodeKind::GroupHeader,
            entity_id: 0,
            robot_id: -1,
            depth: 0,
            label: SharedString::from(format!("Robots ({})", self.workcell.arms.len())),
            badge: SharedString::from(format!("{} DOF", self.workcell.model.dof())),
            accent: Color::from_rgb_u8(56, 189, 248),
            expanded: robots_open,
            has_children: !self.workcell.arms.is_empty(),
            visible: true,
            selected: !has_entity_selection && self.selected_node_id == next_id,
        });
        next_id += 1;

        if robots_open {
            for (idx, arm) in self.workcell.arms.iter().enumerate() {
                let arm_key = format!("arm:{idx}");
                let arm_open = self.expanded_keys.contains(&arm_key);
                let col = rgba_to_slint_color(arm.color_rgba);
                let id = next_id;
                next_id += 1;
                nodes.push(SceneTreeNode {
                    node_id: id,
                    kind: SceneNodeKind::Arm,
                    entity_id: idx as i32,
                    robot_id: idx as i32,
                    depth: 1,
                    label: SharedString::from(arm.label.as_str()),
                    badge: SharedString::from(format!("{} DOF", arm.q_range.len())),
                    accent: col,
                    expanded: arm_open,
                    has_children: true,
                    visible: true,
                    selected: !has_entity_selection && self.selected_node_id == id,
                });

                if arm_open {
                    for &lid in &arm.handle.links {
                        let link = self.workcell.model.link(lid);
                        let sph_cnt = self
                            .workcell
                            .env
                            .link_geometry(lid)
                            .map_or(0, |g| g.local_spheres.len());
                        let nid = next_id;
                        next_id += 1;
                        nodes.push(SceneTreeNode {
                            node_id: nid,
                            kind: SceneNodeKind::Link,
                            entity_id: lid.0 as i32,
                            robot_id: idx as i32,
                            depth: 2,
                            label: SharedString::from(link.name.as_str()),
                            badge: SharedString::from(format!("{sph_cnt} sph")),
                            accent: col,
                            expanded: false,
                            has_children: false,
                            visible: true,
                            selected: self.selected_link_id == Some(lid),
                        });
                    }
                    if let Some(ref pname) = arm.payload_name {
                        let nid = next_id;
                        next_id += 1;
                        nodes.push(SceneTreeNode {
                            node_id: nid,
                            kind: SceneNodeKind::Tool,
                            entity_id: arm.payload_id.map_or(0, |p| p.0 as i32),
                            robot_id: idx as i32,
                            depth: 2,
                            label: SharedString::from(pname.as_str()),
                            badge: SharedString::from("TOOL"),
                            accent: Color::from_rgb_u8(167, 139, 250),
                            expanded: false,
                            has_children: false,
                            visible: true,
                            selected: self.selected_payload_name.as_deref() == Some(pname.as_str()),
                        });
                    }
                }
            }
        }

        let obs_open = self.expanded_keys.contains("group:obstacles");
        let obs_hdr_id = next_id;
        next_id += 1;
        nodes.push(SceneTreeNode {
            node_id: obs_hdr_id,
            kind: SceneNodeKind::GroupHeader,
            entity_id: 1,
            robot_id: -1,
            depth: 0,
            label: SharedString::from(format!("Obstacles ({})", self.workcell.obstacles.len())),
            badge: SharedString::default(),
            accent: Color::from_rgb_u8(245, 158, 11),
            expanded: obs_open,
            has_children: !self.workcell.obstacles.is_empty(),
            visible: true,
            selected: !has_entity_selection && self.selected_node_id == obs_hdr_id,
        });

        if obs_open {
            for (idx, obs) in self.workcell.obstacles.iter().enumerate() {
                let nid = next_id;
                next_id += 1;
                let badge = match obs.role {
                    ObstacleRole::Workpiece => "WP",
                    ObstacleRole::Pedestal => "FIX",
                    ObstacleRole::DynamicSweep { .. } => "DYN",
                    ObstacleRole::StaticObstacle => "OBS",
                };
                nodes.push(SceneTreeNode {
                    node_id: nid,
                    kind: SceneNodeKind::Obstacle,
                    entity_id: idx as i32,
                    robot_id: -1,
                    depth: 1,
                    label: SharedString::from(obs.label.as_str()),
                    badge: SharedString::from(badge),
                    accent: rgba_to_slint_color(obs.color_rgba),
                    expanded: false,
                    has_children: false,
                    visible: true,
                    selected: self.selected_obstacle_idx == Some(idx),
                });
            }
        }

        sc.set_tree_nodes(ModelRc::new(VecModel::from(nodes)));
    }

    fn sync_planner_ui(&self) {
        let Some(ui) = self.ui_weak.upgrade() else {
            return;
        };
        let ps = ui.global::<PlannerState>();
        ps.set_safety_margin_mm((self.spec.rules.default_body_margin * 1000.0) as f32);
        ps.set_cbf_activation_mm((self.spec.rules.default_body_activation * 1000.0) as f32);
        match &self.spec.planner {
            crate::scene::PlannerSpec::TwoStageProximity { servo, mppi, .. } => {
                ps.set_cbf_gamma(servo.cbf_gamma as f32);
                ps.set_tangential_curl_gain(servo.tangential_curl_gain as f32);
                ps.set_mppi_rollouts(mppi.num_rollouts as i32);
                ps.set_mppi_horizon(mppi.horizon as i32);
                ps.set_mppi_iterations(mppi.num_iterations as i32);
            }
            crate::scene::PlannerSpec::ThreeArmCrossing { servo, .. }
            | crate::scene::PlannerSpec::ClosedChainCbfServo { servo, .. } => {
                ps.set_cbf_gamma(servo.cbf_gamma as f32);
                ps.set_tangential_curl_gain(servo.tangential_curl_gain as f32);
            }
            crate::scene::PlannerSpec::BatchMppiEsdf { mppi, .. } => {
                ps.set_mppi_rollouts(mppi.num_rollouts as i32);
                ps.set_mppi_horizon(mppi.horizon as i32);
                ps.set_mppi_iterations(mppi.num_iterations as i32);
            }
            _ => {}
        }
        self.sync_targets_ui();
        self.sync_joint_sliders_ui();
    }

    fn sync_targets_ui(&self) {
        let Some(ui) = self.ui_weak.upgrade() else {
            return;
        };
        let ps = ui.global::<PlannerState>();
        let mut list = Vec::new();
        for (idx, (arm_s, arm_c)) in self
            .spec
            .arms
            .iter()
            .zip(self.workcell.arms.iter())
            .enumerate()
        {
            let (pose, standoff_mm) = match &arm_s.target {
                ArmTargetSpec::ApproachPose {
                    final_pose,
                    standoff_local,
                } => (final_pose, (standoff_local[2].abs() * 1000.0) as f32),
                ArmTargetSpec::DirectPose(pose) => (pose, 0.0),
                _ => continue,
            };
            let tcp_frame_name = self
                .workcell
                .model
                .frame_graph()
                .node(arm_c.tcp_frame)
                .map_or("tcp", |n| n.name.as_str());
            list.push(ArmTargetPose {
                robot_id: idx as i32,
                arm_name: SharedString::from(arm_s.label.as_str()),
                tcp_frame_name: SharedString::from(tcp_frame_name),
                accent: rgba_to_slint_color(arm_s.color_rgba),
                x: pose.xyz[0] as f32,
                y: pose.xyz[1] as f32,
                z: pose.xyz[2] as f32,
                roll_deg: pose.rpy[0].to_degrees() as f32,
                pitch_deg: pose.rpy[1].to_degrees() as f32,
                yaw_deg: pose.rpy[2].to_degrees() as f32,
                standoff_mm,
            });
        }
        let cur_sel = ps.get_selected_target_arm();
        if !list.is_empty() && !list.iter().any(|t| t.robot_id == cur_sel) {
            ps.set_selected_target_arm(list[0].robot_id);
        }
        ps.set_arm_targets(ModelRc::new(VecModel::from(list)));
    }

    fn sync_joint_sliders_ui(&self) {
        let Some(ui) = self.ui_weak.upgrade() else {
            return;
        };
        let mut sliders = Vec::with_capacity(self.workcell.model.dof());
        let limits = self.workcell.model.dof_limits();
        for arm in &self.workcell.arms {
            let col = rgba_to_slint_color(arm.color_rgba);
            for (local_i, global_q) in arm.q_range.clone().enumerate() {
                let name = arm
                    .joint_names
                    .get(local_i)
                    .cloned()
                    .unwrap_or_else(|| format!("q{global_q}"));
                let lim = limits[global_q];
                sliders.push(JointSliderItem {
                    q_index: global_q as i32,
                    robot_id: arm.robot_id.0 as i32,
                    name: SharedString::from(name),
                    accent: col,
                    value: self.current_q.get(global_q).copied().unwrap_or(0.0) as f32,
                    velocity: 0.0,
                    min_val: lim.position.min as f32,
                    max_val: lim.position.max as f32,
                    max_vel: lim.max_velocity as f32,
                });
            }
        }
        ui.global::<PlannerState>()
            .set_joints(ModelRc::new(VecModel::from(sliders)));
    }

    fn sync_diagnostics(&self) {
        let Some(ui) = self.ui_weak.upgrade() else {
            return;
        };
        let Some(traj) = &self.trajectory else {
            return;
        };
        let ps = ui.global::<PlannerState>();
        ps.set_diag_solve_ms(SharedString::from(format!("{:.1} ms", traj.metrics.solve_time_ms)));
        ps.set_diag_pose_err(SharedString::from(format!("{:.2} mm", traj.metrics.max_pos_error_mm)));
        ps.set_diag_min_body_clr(SharedString::from(format!("{:.1} mm", traj.metrics.min_clearance_mm)));
        ps.set_diag_min_tool_clr(SharedString::from(
            traj.metrics
                .min_tool_clearance_mm
                .map_or_else(|| "--".to_string(), |v| format!("{v:.1} mm")),
        ));

        // Alert ticks on scrubber
        let n = traj.frames.len().max(2);
        let mut markers = Vec::new();
        for (idx, f) in traj.frames.iter().enumerate() {
            let clr_mm = f.min_clearance * 1000.0;
            if clr_mm < 5.0 {
                markers.push(TimelineAlertMarker {
                    fraction: (idx as f32) / ((n - 1) as f32),
                    is_collision: clr_mm < 1.5,
                });
            }
        }
        ui.global::<TimelineState>()
            .set_alert_markers(ModelRc::new(VecModel::from(markers)));
    }

    fn regenerate_plots(&self) {
        let Some(ui) = self.ui_weak.upgrade() else {
            return;
        };
        let Some(traj) = &self.trajectory else {
            return;
        };
        let (w, h) = (self.plot_dims.0.max(120), self.plot_dims.1.max(48));
        let p1 = render_min_clearance_plot(traj, self.spec.rules.default_body_margin, w, h);
        let p2 = render_tool_clearance_plot(
            traj,
            self.spec.rules.tool_tool_margin,
            self.spec.rules.tool_workpiece_margin,
            w,
            h,
        );
        let p3 = render_pose_error_plot(traj, 1.8, w, h);
        let p4 = render_joint_vel_plot(&self.workcell, traj, w, h);

        let tl = ui.global::<TimelineState>();
        tl.set_plot_min_clearance(plot_to_slint_image(&p1));
        tl.set_plot_tool_clearance(plot_to_slint_image(&p2));
        tl.set_plot_pose_error(plot_to_slint_image(&p3));
        tl.set_plot_joint_vel(plot_to_slint_image(&p4));
    }

    fn sync_acm_ui(&mut self) {
        let Some(ui) = self.ui_weak.upgrade() else {
            return;
        };

        let mut row_entities: Vec<(String, AcmHeaderItem)> = Vec::new();
        let mut col_entities: Vec<(String, AcmHeaderItem)> = Vec::new();

        match self.acm_slice {
            AcmSliceMode::InterArm => {
                for l in self.workcell.model.links() {
                    let col = rgba_to_slint_color(
                        self.workcell
                            .arms
                            .get(l.robot.0 as usize)
                            .map_or([0.5, 0.5, 0.5, 1.0], |a| a.color_rgba),
                    );
                    let key = format!("L:{}", l.id.0);
                    let item = AcmHeaderItem {
                        entity_key: SharedString::from(key.as_str()),
                        short_label: SharedString::from(short_name(&l.name)),
                        full_label: SharedString::from(l.name.as_str()),
                        accent: col,
                    };
                    row_entities.push((key.clone(), item.clone()));
                    col_entities.push((key, item));
                }
            }
            AcmSliceMode::SelfCollision => {
                let arm_idx = self
                    .selected_arm_idx
                    .min(self.workcell.arms.len().saturating_sub(1));
                if let Some(arm) = self.workcell.arms.get(arm_idx) {
                    let col = rgba_to_slint_color(arm.color_rgba);
                    for &lid in &arm.handle.links {
                        let l = self.workcell.model.link(lid);
                        let key = format!("L:{}", lid.0);
                        let item = AcmHeaderItem {
                            entity_key: SharedString::from(key.as_str()),
                            short_label: SharedString::from(short_name(&l.name)),
                            full_label: SharedString::from(l.name.as_str()),
                            accent: col,
                        };
                        row_entities.push((key.clone(), item.clone()));
                        col_entities.push((key, item));
                    }
                }
            }
            AcmSliceMode::ToolsAndWorkpieces => {
                for p in self.workcell.env.attached_payloads() {
                    let key = format!("P:{}", p.id);
                    let col = rgba_to_slint_color(
                        self.workcell
                            .arms
                            .get(self.workcell.model.link(p.parent_link).robot.0 as usize)
                            .map_or([0.65, 0.55, 0.98, 1.0], |a| a.color_rgba),
                    );
                    let item = AcmHeaderItem {
                        entity_key: SharedString::from(key.as_str()),
                        short_label: SharedString::from(short_name(&p.id)),
                        full_label: SharedString::from(p.id.as_str()),
                        accent: col,
                    };
                    row_entities.push((key.clone(), item.clone()));
                    col_entities.push((key, item));
                }
                for obs in &self.workcell.obstacles {
                    let key = format!("O:{}", obs.obstacle_id.0);
                    col_entities.push((
                        key.clone(),
                        AcmHeaderItem {
                            entity_key: SharedString::from(key),
                            short_label: SharedString::from(short_name(&obs.uid)),
                            full_label: SharedString::from(obs.label.as_str()),
                            accent: rgba_to_slint_color(obs.color_rgba),
                        },
                    ));
                }
                for arm in &self.workcell.arms {
                    let lid = arm.flange_link;
                    let l = self.workcell.model.link(lid);
                    let key = format!("L:{}", lid.0);
                    col_entities.push((
                        key.clone(),
                        AcmHeaderItem {
                            entity_key: SharedString::from(key),
                            short_label: SharedString::from(short_name(&l.name)),
                            full_label: SharedString::from(l.name.as_str()),
                            accent: rgba_to_slint_color(arm.color_rgba),
                        },
                    ));
                }
            }
            AcmSliceMode::LinksAndObstacles | AcmSliceMode::FullFiltered => {
                for l in self.workcell.model.links() {
                    let key = format!("L:{}", l.id.0);
                    let col = rgba_to_slint_color(
                        self.workcell
                            .arms
                            .get(l.robot.0 as usize)
                            .map_or([0.5, 0.5, 0.5, 1.0], |a| a.color_rgba),
                    );
                    row_entities.push((
                        key.clone(),
                        AcmHeaderItem {
                            entity_key: SharedString::from(key),
                            short_label: SharedString::from(short_name(&l.name)),
                            full_label: SharedString::from(l.name.as_str()),
                            accent: col,
                        },
                    ));
                }
                for obs in &self.workcell.obstacles {
                    let key = format!("O:{}", obs.obstacle_id.0);
                    col_entities.push((
                        key.clone(),
                        AcmHeaderItem {
                            entity_key: SharedString::from(key),
                            short_label: SharedString::from(short_name(&obs.uid)),
                            full_label: SharedString::from(obs.label.as_str()),
                            accent: rgba_to_slint_color(obs.color_rgba),
                        },
                    ));
                }
            }
        }

        let filter_q = self.acm_name_filter.trim().to_ascii_lowercase();
        if !filter_q.is_empty() {
            let matches_filter = |hdr: &AcmHeaderItem| {
                hdr.short_label.to_ascii_lowercase().contains(&filter_q)
                    || hdr.full_label.to_ascii_lowercase().contains(&filter_q)
            };
            row_entities.retain(|(_, h)| matches_filter(h));
            col_entities.retain(|(_, h)| matches_filter(h));
        }

        let live_map = self.live_distance_lookup();
        let mut raw_cols: Vec<AcmHeaderItem> = col_entities.iter().map(|(_, h)| h.clone()).collect();
        let mut raw_rows: Vec<(AcmHeaderItem, Vec<AcmCellItem>)> =
            Vec::with_capacity(row_entities.len());

        for (rk, r_hdr) in &row_entities {
            let mut cells = Vec::with_capacity(col_entities.len());
            for (ck, _) in &col_entities {
                cells.push(self.describe_acm_cell(rk, ck, &live_map));
            }
            raw_rows.push((r_hdr.clone(), cells));
        }

        if self.acm_only_active_or_custom {
            let is_active = |c: &AcmCellItem| {
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
                let retained: Vec<AcmCellItem> = cells
                    .drain(..)
                    .enumerate()
                    .filter_map(|(ci, c)| keep_cols[ci].then_some(c))
                    .collect();
                *cells = retained;
            }
        }

        let rows: Vec<AcmMatrixRow> = raw_rows
            .into_iter()
            .map(|(header, cells)| AcmMatrixRow {
                header,
                cells: ModelRc::new(VecModel::from(cells)),
            })
            .collect();

        let acm_state = ui.global::<AcmState>();
        acm_state.set_col_headers(ModelRc::new(VecModel::from(raw_cols)));
        acm_state.set_rows(ModelRc::new(VecModel::from(rows)));

        if let Some((ref rk, ref ck)) = self.acm_selected_pair {
            let cell = self.describe_acm_cell(rk, ck, &live_map);
            acm_state.set_selected_pair(SelectedAcmPair {
                valid: cell.rule != AcmRuleKind::Diagonal,
                row_key: SharedString::from(rk.as_str()),
                col_key: SharedString::from(ck.as_str()),
                pair_title: SharedString::from(format!("{} <-> {}", self.entity_label(rk), self.entity_label(ck))),
                pair_kind_badge: SharedString::from(self.pair_kind_badge(rk, ck)),
                rule: cell.rule,
                locked_adjacent: cell.rule == AcmRuleKind::DisabledAdjacent
                    || cell.rule == AcmRuleKind::TouchExempt,
                margin_mm: cell.margin_mm,
                activation_mm: cell.activation_mm,
                live_dist_mm: cell.live_dist_mm,
            });
        } else {
            acm_state.set_selected_pair(SelectedAcmPair::default());
        }
    }

    fn live_distance_lookup(&self) -> HashMap<(String, String), f64> {
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

    fn describe_acm_cell(
        &self,
        rk: &str,
        ck: &str,
        live_map: &HashMap<(String, String), f64>,
    ) -> AcmCellItem {
        let selected = self
            .acm_selected_pair
            .as_ref()
            .is_some_and(|(a, b)| (a == rk && b == ck) || (a == ck && b == rk));
        let live_dist_mm = live_map
            .get(&(rk.to_string(), ck.to_string()))
            .map_or(-1000.0_f32, |d| (*d * 1000.0) as f32);

        if rk == ck {
            return AcmCellItem {
                row_key: SharedString::from(rk),
                col_key: SharedString::from(ck),
                rule: AcmRuleKind::Diagonal,
                margin_mm: 0.0,
                activation_mm: 0.0,
                label: SharedString::from("·"),
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
                    (AcmRuleKind::Enabled, 0.0, self.spec.rules.default_body_activation, "DEF".to_string())
                }
            }
            _ => (AcmRuleKind::Diagonal, 0.0, 0.0, "·".to_string()),
        };

        let margin_mm = (margin_m * 1000.0) as f32;
        let near_violation = matches!(rule, AcmRuleKind::Enabled | AcmRuleKind::CustomMargin)
            && live_dist_mm > -900.0
            && live_dist_mm <= margin_mm + 5.0;

        AcmCellItem {
            row_key: SharedString::from(rk),
            col_key: SharedString::from(ck),
            rule,
            margin_mm,
            activation_mm: (act_m * 1000.0) as f32,
            label: SharedString::from(label),
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
            (Some(UiEntity::Link(_)), Some(UiEntity::Link(_))) => "Link ↔ Link",
            (Some(UiEntity::Payload(_)), Some(UiEntity::Payload(_))) => "Tool ↔ Tool",
            (Some(UiEntity::Payload(_)), Some(UiEntity::Obstacle(_)))
            | (Some(UiEntity::Obstacle(_)), Some(UiEntity::Payload(_))) => "Tool ↔ Obstacle",
            (Some(UiEntity::Payload(_)), Some(UiEntity::Link(_)))
            | (Some(UiEntity::Link(_)), Some(UiEntity::Payload(_))) => "Tool ↔ Link",
            _ => "Link ↔ Obstacle",
        }
    }
}

enum UiEntity {
    Link(u16),
    Payload(String),
    Obstacle(u32),
}

fn parse_entity(key: &str) -> Option<UiEntity> {
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

fn bind_callbacks(window: &MainWindow, ctrl: &Rc<RefCell<AppController>>) {
    // Worker event wakeup
    {
        let c = Rc::clone(ctrl);
        window.on_drain_worker_events(move || {
            c.borrow_mut().drain_worker_events();
        });
    }

    // Viewport callbacks
    {
        let vp = window.global::<ViewportState>();
        let c = Rc::clone(ctrl);
        vp.on_resize_viewport(move |w, h| {
            let b = c.borrow();
            let resized = {
                let mut st = b.viewport_state.borrow_mut();
                let new_size = (w.max(1) as u32, h.max(1) as u32);
                if st.logical_size != new_size {
                    st.logical_size = new_size;
                    st.dirty = true;
                    true
                } else {
                    false
                }
            };
            if resized {
                if let Some(ui) = b.ui_weak.upgrade() {
                    ui.window().request_redraw();
                }
            }
        });

        let c = Rc::clone(ctrl);
        vp.on_orbit_drag(move |dx, dy| {
            let b = c.borrow();
            {
                let mut st = b.viewport_state.borrow_mut();
                st.camera.orbit(dx, dy);
                st.dirty = true;
            }
            if let Some(ui) = b.ui_weak.upgrade() {
                ui.window().request_redraw();
            }
        });

        let c = Rc::clone(ctrl);
        vp.on_pan_drag(move |dx, dy| {
            let b = c.borrow();
            {
                let mut st = b.viewport_state.borrow_mut();
                st.camera.pan(dx, dy);
                st.dirty = true;
            }
            if let Some(ui) = b.ui_weak.upgrade() {
                ui.window().request_redraw();
            }
        });

        let c = Rc::clone(ctrl);
        vp.on_zoom_wheel(move |delta| {
            let b = c.borrow();
            {
                let mut st = b.viewport_state.borrow_mut();
                st.camera.zoom(delta);
                st.dirty = true;
            }
            if let Some(ui) = b.ui_weak.upgrade() {
                ui.window().request_redraw();
            }
        });

        let c = Rc::clone(ctrl);
        vp.on_set_camera_preset(move |name| {
            let b = c.borrow();
            {
                let mut st = b.viewport_state.borrow_mut();
                match name.as_str() {
                    "iso" => st.camera.set_iso(),
                    "top" => st.camera.set_top(),
                    "front" => st.camera.set_front(),
                    "side" => st.camera.set_side(),
                    "workpiece" => st.camera.focus_point(Vector3::new(0.0, 0.0, 0.42), 0.48),
                    _ => st.camera.set_iso(),
                }
                st.dirty = true;
            }
            if let Some(ui) = b.ui_weak.upgrade() {
                ui.window().request_redraw();
            }
        });

        let c = Rc::clone(ctrl);
        vp.on_layers_changed(move || {
            let mut b = c.borrow_mut();
            if let Some(ui) = b.ui_weak.upgrade() {
                let v = ui.global::<ViewportState>();
                let axis = v.get_esdf_slice_axis().clamp(0, 2) as usize;
                let lo = b.spec.esdf.origin[axis];
                let hi = lo + b.spec.esdf.voxel_size * (b.spec.esdf.dims[axis] as f32);
                v.set_esdf_slice_min_z(lo);
                v.set_esdf_slice_max_z(hi);
                let coord = v.get_esdf_slice_z().clamp(lo, hi);
                v.set_esdf_slice_z(coord);

                let mut vp_st = b.viewport_state.borrow_mut();
                vp_st.overlays.show_links = v.get_show_links();
                vp_st.overlays.show_tool_convex = v.get_show_tool_convex();
                vp_st.overlays.show_tier1_spheres = v.get_show_tier1_spheres();
                vp_st.overlays.show_esdf_slice = v.get_show_esdf_slice();
                vp_st.overlays.esdf_slice_axis = axis as u8;
                vp_st.overlays.esdf_slice_coord = coord;
                vp_st.overlays.esdf_opacity = v.get_esdf_opacity().clamp(0.1, 0.95);
                vp_st.overlays.show_mppi_ribbons = v.get_show_mppi_ribbons();
                vp_st.overlays.show_witnesses = v.get_show_witnesses();
                vp_st.overlays.witnesses_xray = v.get_witnesses_xray();
                vp_st.overlays.max_witnesses = v.get_max_witnesses().clamp(1, 16) as usize;
                vp_st.overlays.show_tf_frames = v.get_show_tf_frames();
                vp_st.overlays.show_target_ghosts = v.get_show_target_ghosts();
                vp_st.overlays.show_floor_grid = v.get_show_floor_grid();
            }
            b.refresh_pose_and_scene();
        });

        let c = Rc::clone(ctrl);
        vp.on_pick_ray(move |nx, ny| {
            let mut b = c.borrow_mut();
            let hit = {
                let Some(snap) = b.last_snapshot.as_ref() else {
                    return;
                };
                let (origin, dir) = {
                    let vp_st = b.viewport_state.borrow();
                    let aspect = (vp_st.logical_size.0 as f32) / (vp_st.logical_size.1.max(1) as f32);
                    vp_st.camera.screen_ray(nx, ny, aspect)
                };
                pick_entity_along_ray(&b.workcell, snap, origin, dir)
            };

            b.selected_link_id = None;
            b.selected_payload_name = None;
            b.selected_obstacle_idx = None;
            let summary = match hit {
                Some(PickedEntity::Link(lid)) => {
                    let (link_name, r_idx) = {
                        let link = b.workcell.model.link(lid);
                        (link.name.clone(), link.robot.0 as usize)
                    };
                    let arm_idx = r_idx.min(b.spec.arms.len().saturating_sub(1));
                    b.selected_link_id = Some(lid);
                    b.selected_arm_idx = arm_idx;
                    b.expanded_keys.insert("group:robots".to_string());
                    b.expanded_keys.insert(format!("arm:{arm_idx}"));
                    format!("Picked link {link_name} (Arm {arm_idx})")
                }
                Some(PickedEntity::Payload(pid_idx)) => {
                    if let Some(p) = b.workcell.env.attached_payloads().get(pid_idx).cloned() {
                        let r_idx = b.workcell.model.link(p.parent_link).robot.0 as usize;
                        let arm_idx = r_idx.min(b.spec.arms.len().saturating_sub(1));
                        b.selected_payload_name = Some(p.id.clone());
                        b.selected_arm_idx = arm_idx;
                        b.expanded_keys.insert("group:robots".to_string());
                        b.expanded_keys.insert(format!("arm:{arm_idx}"));
                        format!("Picked tool {} (Arm {arm_idx})", p.id)
                    } else {
                        "No entity at cursor".to_string()
                    }
                }
                Some(PickedEntity::Obstacle(oid)) => {
                    let found = b
                        .workcell
                        .obstacles
                        .iter()
                        .enumerate()
                        .find(|(_, o)| o.obstacle_id == oid)
                        .map(|(idx, obs)| (idx, obs.label.clone(), obs.uid.clone()));
                    if let Some((idx, label, uid)) = found {
                        b.selected_obstacle_idx = Some(idx);
                        b.expanded_keys.insert("group:obstacles".to_string());
                        format!("Picked obstacle {label} ({uid})")
                    } else {
                        "No entity at cursor".to_string()
                    }
                }
                None => "No entity hit by 3D pick ray".to_string(),
            };
            if let Some(ui) = b.ui_weak.upgrade() {
                ui.global::<SceneState>()
                    .set_selected_info_summary(SharedString::from(summary));
            }
            b.sync_scene_ui();
            b.refresh_pose_and_scene();
        });
    }

    // Timeline callbacks
    {
        let tl = window.global::<TimelineState>();
        let c = Rc::clone(ctrl);
        tl.on_toggle_play(move || {
            let mut b = c.borrow_mut();
            b.playing = !b.playing;
            if let Some(ui) = b.ui_weak.upgrade() {
                ui.global::<TimelineState>().set_playing(b.playing);
            }
        });

        let c = Rc::clone(ctrl);
        tl.on_step_frame(move |delta| {
            let mut b = c.borrow_mut();
            let total = b.trajectory.as_ref().map_or(1, |t| t.frames.len());
            if total == 0 {
                return;
            }
            let next = (b.current_frame_idx as i32 + delta).rem_euclid(total as i32) as usize;
            b.current_frame_idx = next;
            if let Some((q, time)) = b
                .trajectory
                .as_ref()
                .and_then(|t| t.frames.get(next))
                .map(|f| (f.q.clone(), f.time))
            {
                b.current_q = q;
                b.playback_time = time;
            }
            b.refresh_pose_and_scene();
        });

        let c = Rc::clone(ctrl);
        tl.on_seek_fraction(move |frac| {
            let mut b = c.borrow_mut();
            let target_frame = b.trajectory.as_ref().and_then(|traj| {
                let n = traj.frames.len();
                if n > 0 {
                    let idx = ((frac.clamp(0.0, 1.0) * ((n - 1) as f32)).round() as usize)
                        .min(n - 1);
                    Some((idx, traj.frames[idx].q.clone(), traj.frames[idx].time))
                } else {
                    None
                }
            });
            if let Some((idx, q, time)) = target_frame {
                b.current_frame_idx = idx;
                b.current_q = q;
                b.playback_time = time;
                b.refresh_pose_and_scene();
            }
        });

        let c = Rc::clone(ctrl);
        tl.on_set_speed_index(move |idx| {
            c.borrow_mut().speed_index = (idx.max(0) as usize).min(SPEED_FACTORS.len() - 1);
        });

        let c = Rc::clone(ctrl);
        tl.on_resize_plots(move |w, h| {
            let mut b = c.borrow_mut();
            let dims = (w.max(120) as u32, h.max(48) as u32);
            if b.plot_dims != dims {
                b.plot_dims = dims;
                b.regenerate_plots();
            }
        });
    }

    // Planner callbacks
    {
        let ps = window.global::<PlannerState>();
        let c = Rc::clone(ctrl);
        ps.on_load_scenario_preset(move |idx| {
            let preset = *PresetId::ALL.get(idx as usize).unwrap_or(&PresetId::ThreeArmCloseProximity);
            let mut b = c.borrow_mut();
            b.spec = preset_workcell(preset);
            b.playing = false;
            b.selected_arm_idx = 0;
            b.selected_link_id = None;
            b.selected_payload_name = None;
            b.selected_obstacle_idx = None;
            b.acm_selected_pair = None;
            if let Some(ui) = b.ui_weak.upgrade() {
                ui.global::<TimelineState>().set_playing(false);
            }
            b.dispatch_compile_and_plan(DirtyFlags::TOPO);
        });

        let c = Rc::clone(ctrl);
        ps.on_run_plan(move || {
            c.borrow_mut().dispatch_compile_and_plan(DirtyFlags::RULES);
        });

        let c = Rc::clone(ctrl);
        ps.on_cancel_plan(move || {
            let b = c.borrow();
            b.worker.cancel_pending();
            b.set_worker_status(SolverStatus::Idle, 0.0, "Cancelled");
        });

        let c_prio = Rc::clone(ctrl);
        ps.on_set_arm_priority(move |robot_id, weight| {
            {
                let mut b = c_prio.borrow_mut();
                if let Some(arm) = b.spec.arms.get_mut(robot_id as usize) {
                    arm.priority = weight as f64;
                }
            }
            AppController::schedule_debounced_compile_and_plan(&c_prio, DirtyFlags::PLAN);
        });

        let c_cbf = Rc::clone(ctrl);
        ps.on_set_cbf_params(move |margin_mm, act_mm, gamma, curl| {
            {
                let mut b = c_cbf.borrow_mut();
                b.spec.rules.default_body_margin = (margin_mm as f64) * 0.001;
                b.spec.rules.default_body_activation = (act_mm as f64) * 0.001;
                match &mut b.spec.planner {
                    crate::scene::PlannerSpec::TwoStageProximity { servo, .. }
                    | crate::scene::PlannerSpec::ThreeArmCrossing { servo, .. }
                    | crate::scene::PlannerSpec::ClosedChainCbfServo { servo, .. } => {
                        servo.safety_margin = (margin_mm as f64) * 0.001;
                        servo.cbf_activation_distance = (act_mm as f64) * 0.001;
                        servo.cbf_gamma = gamma as f64;
                        servo.tangential_curl_gain = curl as f64;
                    }
                    _ => {}
                }
            }
            AppController::schedule_debounced_compile_and_plan(&c_cbf, DirtyFlags::RULES);
        });

        let c_mppi = Rc::clone(ctrl);
        ps.on_set_mppi_params(move |rollouts, horizon, iters| {
            {
                let mut b = c_mppi.borrow_mut();
                match &mut b.spec.planner {
                    crate::scene::PlannerSpec::TwoStageProximity { mppi, .. }
                    | crate::scene::PlannerSpec::BatchMppiEsdf { mppi, .. } => {
                        mppi.num_rollouts = rollouts.max(8) as usize;
                        mppi.horizon = horizon.max(4) as usize;
                        mppi.num_iterations = iters.max(1) as usize;
                    }
                    _ => {}
                }
            }
            AppController::schedule_debounced_compile_and_plan(&c_mppi, DirtyFlags::PLAN);
        });

        let c_jog = Rc::clone(ctrl);
        ps.on_set_joint_value(move |q_idx, val| {
            let mut b = c_jog.borrow_mut();
            let idx = q_idx as usize;
            if idx < b.current_q.len() {
                b.playing = false;
                b.current_q[idx] = val as f64;
                b.refresh_pose_and_scene();
            }
        });

        let c_tgt = Rc::clone(ctrl);
        ps.on_update_arm_target(
            move |robot_id, x, y, z, roll_deg, pitch_deg, yaw_deg, standoff_mm| {
                {
                    let mut b = c_tgt.borrow_mut();
                    if let Some(arm) = b.spec.arms.get_mut(robot_id as usize) {
                        let new_pose = PoseSpec {
                            xyz: [x as f64, y as f64, z as f64],
                            rpy: [
                                (roll_deg as f64).to_radians(),
                                (pitch_deg as f64).to_radians(),
                                (yaw_deg as f64).to_radians(),
                            ],
                        };
                        match &mut arm.target {
                            ArmTargetSpec::ApproachPose {
                                final_pose,
                                standoff_local,
                            } => {
                                *final_pose = new_pose;
                                standoff_local[2] = -(standoff_mm.max(0.0) as f64) * 0.001;
                            }
                            ArmTargetSpec::DirectPose(pose) => {
                                if standoff_mm > 0.5 {
                                    arm.target = ArmTargetSpec::ApproachPose {
                                        final_pose: new_pose,
                                        standoff_local: [0.0, 0.0, -(standoff_mm as f64) * 0.001],
                                    };
                                } else {
                                    *pose = new_pose;
                                }
                            }
                            _ => {
                                arm.target = if standoff_mm > 0.5 {
                                    ArmTargetSpec::ApproachPose {
                                        final_pose: new_pose,
                                        standoff_local: [0.0, 0.0, -(standoff_mm as f64) * 0.001],
                                    }
                                } else {
                                    ArmTargetSpec::DirectPose(new_pose)
                                };
                            }
                        }
                    }
                    b.sync_targets_ui();
                    b.refresh_pose_and_scene();
                }
                AppController::schedule_debounced_compile_and_plan(&c_tgt, DirtyFlags::PLAN);
            },
        );

        let c_copy = Rc::clone(ctrl);
        ps.on_copy_current_tcp_to_target(move |robot_id| {
            {
                let mut b = c_copy.borrow_mut();
                let r_idx = robot_id as usize;
                let tcp_iso = b
                    .last_snapshot
                    .as_ref()
                    .and_then(|s| s.tcp_poses.get(r_idx))
                    .copied();
                if let (Some(iso), Some(arm)) = (tcp_iso, b.spec.arms.get_mut(r_idx)) {
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
                    b.sync_targets_ui();
                    b.refresh_pose_and_scene();
                }
            }
            AppController::schedule_debounced_compile_and_plan(&c_copy, DirtyFlags::PLAN);
        });

        let c_ik = Rc::clone(ctrl);
        ps.on_preview_ik_targets(move || {
            let mut b = c_ik.borrow_mut();
            let mut tasks = Vec::new();
            for (arm_s, arm_c) in b.spec.arms.iter().zip(b.workcell.arms.iter()) {
                let pose = match &arm_s.target {
                    ArmTargetSpec::ApproachPose { final_pose, .. } => final_pose,
                    ArmTargetSpec::DirectPose(pose) => pose,
                    _ => continue,
                };
                tasks.push(TaskPose::new(arm_c.tcp_frame, pose.to_isometry()));
            }
            if tasks.is_empty() {
                b.set_worker_status(SolverStatus::Warning, 1.0, "No SE(3) targets in active scenario");
                return;
            }
            match BatchIkSolver::default().solve_collision_free(
                &CpuSimdBatchBackend,
                &b.workcell.model,
                &b.workcell.flat_model,
                &b.workcell.env,
                Some(&b.workcell.esdf),
                &b.current_q,
                &tasks,
            ) {
                Ok(cfg) => {
                    b.playing = false;
                    if let Some(ui) = b.ui_weak.upgrade() {
                        ui.global::<TimelineState>().set_playing(false);
                    }
                    b.current_q = cfg.0;
                    b.set_worker_status(
                        SolverStatus::Converged,
                        1.0,
                        &format!("IK Preview solved ({} targets, 32 seeds)", tasks.len()),
                    );
                    b.refresh_pose_and_scene();
                    b.sync_acm_ui();
                }
                Err(err) => {
                    b.set_worker_status(SolverStatus::Warning, 1.0, &format!("IK Preview failed: {err}"));
                }
            }
        });
    }

    // Scene & N-Arm Manager callbacks
    {
        let sc = window.global::<SceneState>();

        let c = Rc::clone(ctrl);
        sc.on_select_tree_node(move |node_id| {
            let mut b = c.borrow_mut();
            b.selected_node_id = node_id;
            b.selected_link_id = None;
            b.selected_payload_name = None;
            b.selected_obstacle_idx = None;
            if let Some(ui) = b.ui_weak.upgrade() {
                let nodes = ui.global::<SceneState>().get_tree_nodes();
                if let Some(n) = slint::Model::iter(&nodes).find(|n| n.node_id == node_id) {
                    if n.robot_id >= 0 {
                        b.selected_arm_idx = (n.robot_id as usize).min(b.spec.arms.len().saturating_sub(1));
                    }
                    match n.kind {
                        SceneNodeKind::Link => {
                            b.selected_link_id = Some(LinkId(n.entity_id as u16));
                        }
                        SceneNodeKind::Tool => {
                            b.selected_payload_name = Some(n.label.to_string());
                        }
                        SceneNodeKind::Obstacle => {
                            b.selected_obstacle_idx = Some(n.entity_id as usize);
                        }
                        _ => {}
                    }
                    let summary = format!("{} [{}]", n.label, n.badge);
                    ui.global::<SceneState>().set_selected_info_summary(SharedString::from(summary));
                }
            }
            b.sync_scene_ui();
            b.refresh_pose_and_scene();
        });

        let c = Rc::clone(ctrl);
        sc.on_toggle_tree_expand(move |node_id| {
            let mut b = c.borrow_mut();
            if let Some(ui) = b.ui_weak.upgrade() {
                let nodes = ui.global::<SceneState>().get_tree_nodes();
                if let Some(n) = slint::Model::iter(&nodes).find(|n| n.node_id == node_id) {
                    let key = match n.kind {
                        SceneNodeKind::GroupHeader if n.entity_id == 0 => "group:robots".to_string(),
                        SceneNodeKind::GroupHeader => "group:obstacles".to_string(),
                        SceneNodeKind::Arm => format!("arm:{}", n.robot_id),
                        _ => return,
                    };
                    if !b.expanded_keys.remove(&key) {
                        b.expanded_keys.insert(key);
                    }
                }
            }
            b.sync_scene_ui();
        });

        let c = Rc::clone(ctrl);
        sc.on_add_arm_preset(move |kind| {
            let mut b = c.borrow_mut();
            let idx = b.spec.arms.len();
            let angle = (idx as f64) * 1.1;
            let (source, flange, q0) = if kind == "4dof" {
                (UrdfSource::Builtin4Dof, "wrist_link", vec![0.0, 0.35, 0.55, 0.2])
            } else {
                (UrdfSource::Builtin6Dof, "flange_link", vec![0.0, -0.35, 1.35, 0.0, 0.70, 0.0])
            };
            b.spec.arms.push(ArmSpec {
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
            b.dispatch_compile_and_plan(DirtyFlags::TOPO);
        });

        let c = Rc::clone(ctrl);
        sc.on_remove_arm(move |robot_id| {
            let mut b = c.borrow_mut();
            let idx = robot_id as usize;
            if b.spec.arms.len() > 1 && idx < b.spec.arms.len() {
                b.spec.arms.remove(idx);
                b.selected_arm_idx = b.selected_arm_idx.min(b.spec.arms.len() - 1);
                b.dispatch_compile_and_plan(DirtyFlags::TOPO);
            }
        });

        let c = Rc::clone(ctrl);
        sc.on_arrange_arms_ring(move |radius| {
            let mut b = c.borrow_mut();
            let k = b.spec.arms.len().max(1);
            for (i, arm) in b.spec.arms.iter_mut().enumerate() {
                let th = std::f64::consts::TAU * (i as f64) / (k as f64);
                arm.base_pose = PoseSpec::from_xyz_yaw(
                    (radius as f64) * th.cos(),
                    (radius as f64) * th.sin(),
                    0.0,
                    th + std::f64::consts::PI,
                );
            }
            b.dispatch_compile_and_plan(DirtyFlags::TOPO);
        });

        let c = Rc::clone(ctrl);
        sc.on_update_arm_base_pose(move |robot_id, x, y, z, yaw_deg| {
            {
                let mut b = c.borrow_mut();
                if let Some(arm) = b.spec.arms.get_mut(robot_id as usize) {
                    arm.base_pose = PoseSpec::from_xyz_yaw(
                        x as f64,
                        y as f64,
                        z as f64,
                        (yaw_deg as f64).to_radians(),
                    );
                }
            }
            AppController::schedule_debounced_compile_and_plan(&c, DirtyFlags::TOPO);
        });

        let c = Rc::clone(ctrl);
        sc.on_face_arm_origin(move |robot_id| {
            let mut b = c.borrow_mut();
            if let Some(arm) = b.spec.arms.get_mut(robot_id as usize) {
                let yaw = (-arm.base_pose.xyz[1]).atan2(-arm.base_pose.xyz[0]);
                arm.base_pose.rpy[2] = yaw;
            }
            b.dispatch_compile_and_plan(DirtyFlags::TOPO);
        });

        let c = Rc::clone(ctrl);
        sc.on_set_arm_tool_preset(move |robot_id, preset| {
            let mut b = c.borrow_mut();
            if let Some(arm) = b.spec.arms.get_mut(robot_id as usize) {
                arm.tool = match preset {
                    ToolPresetKind::None => ToolPreset::None,
                    ToolPresetKind::ParallelGripper => ToolPreset::ParallelGripper,
                    ToolPresetKind::Screwdriver => ToolPreset::Screwdriver,
                    ToolPresetKind::Probe => ToolPreset::InspectionProbe,
                    ToolPresetKind::Custom => ToolPreset::ParallelGripper,
                };
            }
            b.dispatch_compile_and_plan(DirtyFlags::TOPO);
        });

        let c = Rc::clone(ctrl);
        sc.on_add_obstacle(move |kind, is_wp| {
            let mut b = c.borrow_mut();
            let idx = b.spec.obstacles.len();
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
            b.spec.obstacles.push(ObstacleSpec {
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
            b.selected_link_id = None;
            b.selected_payload_name = None;
            b.selected_obstacle_idx = Some(idx);
            b.expanded_keys.insert("group:obstacles".to_string());
            b.dispatch_compile_and_plan(DirtyFlags::OBSTACLES);
        });

        let c = Rc::clone(ctrl);
        sc.on_add_workpiece_box(move || {
            let mut b = c.borrow_mut();
            let idx = b.spec.obstacles.len();
            b.spec.obstacles.push(ObstacleSpec {
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
            b.selected_link_id = None;
            b.selected_payload_name = None;
            b.selected_obstacle_idx = Some(idx);
            b.expanded_keys.insert("group:obstacles".to_string());
            b.dispatch_compile_and_plan(DirtyFlags::OBSTACLES);
        });

        let c = Rc::clone(ctrl);
        sc.on_remove_obstacle(move |obs_idx| {
            let mut b = c.borrow_mut();
            let idx = obs_idx as usize;
            if idx < b.spec.obstacles.len() {
                b.spec.obstacles.remove(idx);
                b.selected_obstacle_idx = if b.spec.obstacles.is_empty() {
                    None
                } else {
                    Some(idx.min(b.spec.obstacles.len() - 1))
                };
                b.dispatch_compile_and_plan(DirtyFlags::OBSTACLES);
            }
        });

        let c_obs = Rc::clone(ctrl);
        sc.on_update_obstacle(
            move |obs_idx, is_wp, carve_esdf, x, y, z, margin_mm, act_mm| {
                {
                    let mut b = c_obs.borrow_mut();
                    if let Some(obs) = b.spec.obstacles.get_mut(obs_idx as usize) {
                        if matches!(obs.role, ObstacleRole::Workpiece) != is_wp {
                            obs.role = if is_wp {
                                ObstacleRole::Workpiece
                            } else {
                                ObstacleRole::StaticObstacle
                            };
                        }
                        obs.carve_esdf = carve_esdf;
                        obs.pose.xyz = [x as f64, y as f64, z as f64];
                        obs.safety_margin = Some((margin_mm.max(0.0) as f64) * 0.001);
                        obs.activation_distance = Some((act_mm.max(0.0) as f64) * 0.001);
                    }
                    b.sync_scene_ui();
                }
                AppController::schedule_debounced_compile_and_plan(&c_obs, DirtyFlags::OBSTACLES);
            },
        );

        let c_tool = Rc::clone(ctrl);
        sc.on_update_arm_tool_params(move |robot_id, margin_mm, act_mm| {
            {
                let mut b = c_tool.borrow_mut();
                if let Some(arm) = b.spec.arms.get_mut(robot_id as usize) {
                    if let Some(mut custom) = arm.tool.expand(&arm.uid, arm.default_tcp_z) {
                        custom.safety_margin = Some((margin_mm.max(0.0) as f64) * 0.001);
                        custom.activation_distance = Some((act_mm.max(0.0) as f64) * 0.001);
                        arm.tool = ToolPreset::Custom(Box::new(custom));
                    }
                }
            }
            AppController::schedule_debounced_compile_and_plan(&c_tool, DirtyFlags::TOPO);
        });

        let c_tl = Rc::clone(ctrl);
        sc.on_toggle_tool_touch_link(move |robot_id, link_name, exempt| {
            let mut b = c_tl.borrow_mut();
            let r_idx = robot_id as usize;
            let current_names: Vec<String> = b
                .workcell
                .arms
                .get(r_idx)
                .and_then(|arm_c| {
                    let pid = arm_c.payload_id?;
                    let pay = b.workcell.env.attached_payload(pid)?;
                    Some(
                        pay.touch_links
                            .iter()
                            .map(|&lid| b.workcell.model.link(lid).name.clone())
                            .collect(),
                    )
                })
                .unwrap_or_default();
            if let Some(arm) = b.spec.arms.get_mut(r_idx) {
                if let Some(mut custom) = arm.tool.expand(&arm.uid, arm.default_tcp_z) {
                    let mut names = custom.touch_link_names.take().unwrap_or(current_names);
                    let target_name = link_name.to_string();
                    if exempt {
                        if !names.contains(&target_name) {
                            names.push(target_name);
                        }
                    } else {
                        names.retain(|n| n != &target_name);
                    }
                    custom.touch_link_names = Some(names);
                    arm.tool = ToolPreset::Custom(Box::new(custom));
                    b.dispatch_compile_and_plan(DirtyFlags::TOPO);
                }
            }
        });

        let c = Rc::clone(ctrl);
        sc.on_import_urdf_dialog(move || {
            if let Some(path) = rfd::FileDialog::new()
                .add_filter("URDF Robot", &["urdf", "xml"])
                .pick_file()
            {
                let mut srdf_path = path.clone();
                srdf_path.set_extension("srdf");
                let srdf_opt = srdf_path.exists().then_some(srdf_path);
                let mut b = c.borrow_mut();
                let idx = b.spec.arms.len();
                let th = (idx as f64) * 1.1;
                b.spec.arms.push(ArmSpec {
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
                b.dispatch_compile_and_plan(DirtyFlags::TOPO);
            }
        });

        sc.on_export_legacy_json(|| {
            if let Some(path) = rfd::FileDialog::new()
                .set_file_name("unrust_scenarios.json")
                .add_filter("JSON", &["json"])
                .save_file()
            {
                let scens = generate_scenarios();
                let _ = std::fs::write(path, scenarios_to_json(&scens));
            }
        });

        sc.on_export_legacy_html(|| {
            if let Some(path) = rfd::FileDialog::new()
                .set_file_name("unrust_visualizer.html")
                .add_filter("HTML", &["html"])
                .save_file()
            {
                let scens = generate_scenarios();
                let _ = std::fs::write(path, render_visualizer_html(&scens));
            }
        });
    }

    // ACM Matrix callbacks
    {
        let acm = window.global::<AcmState>();
        let c = Rc::clone(ctrl);
        acm.on_set_slice_mode(move |mode| {
            let mut b = c.borrow_mut();
            b.acm_slice = mode;
            b.sync_acm_ui();
        });

        let c = Rc::clone(ctrl);
        acm.on_select_cell(move |rk, ck| {
            let mut b = c.borrow_mut();
            b.acm_selected_pair = Some((rk.to_string(), ck.to_string()));
            b.sync_acm_ui();
            b.refresh_pose_and_scene();
        });

        let c = Rc::clone(ctrl);
        acm.on_set_filters(move |name_filter, only_active| {
            let mut b = c.borrow_mut();
            b.acm_name_filter = name_filter.to_string();
            b.acm_only_active_or_custom = only_active;
            b.sync_acm_ui();
        });

        let c = Rc::clone(ctrl);
        acm.on_reset_acm_from_srdf(move || {
            let mut b = c.borrow_mut();
            b.spec.rules.acm_overrides.clear();
            b.spec.rules.pair_overrides.clear();
            b.acm_selected_pair = None;
            b.dispatch_compile_and_plan(DirtyFlags::RULES);
        });

        let c = Rc::clone(ctrl);
        acm.on_toggle_cell_quick(move |rk, ck| {
            let mut b = c.borrow_mut();
            let cell = {
                let live = b.live_distance_lookup();
                b.describe_acm_cell(&rk, &ck, &live)
            };
            let enable = matches!(
                cell.rule,
                AcmRuleKind::DisabledDynamic
                    | AcmRuleKind::DisabledDefault
                    | AcmRuleKind::DisabledNever
            );
            let wc = Arc::clone(&b.workcell);
            apply_cell_rule_to_spec(
                &mut b.spec,
                &wc,
                &rk,
                &ck,
                enable,
                if enable && cell.margin_mm > 0.0 {
                    Some((cell.margin_mm as f64) * 0.001)
                } else {
                    None
                },
                None,
            );
            b.dispatch_compile_and_plan(DirtyFlags::RULES);
        });

        let c = Rc::clone(ctrl);
        acm.on_apply_selected_pair_rule(move |rk, ck, rule, margin_mm, act_mm| {
            let mut b = c.borrow_mut();
            let enabled = matches!(rule, AcmRuleKind::Enabled | AcmRuleKind::CustomMargin);
            let margin = if matches!(rule, AcmRuleKind::CustomMargin) {
                Some((margin_mm as f64) * 0.001)
            } else {
                None
            };
            let wc = Arc::clone(&b.workcell);
            apply_cell_rule_to_spec(
                &mut b.spec,
                &wc,
                &rk,
                &ck,
                enabled,
                margin,
                Some((act_mm as f64) * 0.001),
            );
            b.dispatch_compile_and_plan(DirtyFlags::RULES);
        });

        let c = Rc::clone(ctrl);
        acm.on_bulk_set_inter_arm_margin(move |mm| {
            let mut b = c.borrow_mut();
            b.spec.rules.default_body_margin = (mm as f64) * 0.001;
            b.dispatch_compile_and_plan(DirtyFlags::RULES);
        });

        let c = Rc::clone(ctrl);
        acm.on_bulk_set_tool_tool_margin(move |mm| {
            let mut b = c.borrow_mut();
            b.spec.rules.tool_tool_margin = (mm as f64) * 0.001;
            b.dispatch_compile_and_plan(DirtyFlags::RULES);
        });

        let c = Rc::clone(ctrl);
        acm.on_bulk_set_tool_workpiece_margin(move |mm| {
            let mut b = c.borrow_mut();
            b.spec.rules.tool_workpiece_margin = (mm as f64) * 0.001;
            b.dispatch_compile_and_plan(DirtyFlags::RULES);
        });

        let c = Rc::clone(ctrl);
        acm.on_bulk_set_tool_link_margin(move |mm| {
            let mut b = c.borrow_mut();
            b.spec.rules.tool_other_link_margin = (mm as f64) * 0.001;
            b.dispatch_compile_and_plan(DirtyFlags::RULES);
        });
    }
}

fn start_playback_timer(ctrl: &Rc<RefCell<AppController>>) {
    let timer = slint::Timer::default();
    let weak_ctrl = Rc::downgrade(ctrl);
    timer.start(slint::TimerMode::Repeated, Duration::from_millis(16), move || {
        let Some(c) = weak_ctrl.upgrade() else {
            return;
        };
        let mut b = c.borrow_mut();
        if !b.playing {
            return;
        }
        let Some(traj) = b.trajectory.clone() else {
            return;
        };
        if traj.frames.len() < 2 {
            return;
        }
        let dur = traj.metrics.duration_s.max(1e-3);
        let speed = SPEED_FACTORS[b.speed_index];
        b.playback_time += 0.016 * speed;
        let looping = b
            .ui_weak
            .upgrade()
            .is_none_or(|ui| ui.global::<TimelineState>().get_looping());
        if b.playback_time > dur {
            if looping {
                b.playback_time %= dur;
            } else {
                b.playback_time = dur;
                b.playing = false;
                if let Some(ui) = b.ui_weak.upgrade() {
                    ui.global::<TimelineState>().set_playing(false);
                }
            }
        }
        let frac = (b.playback_time / dur).clamp(0.0, 1.0);
        let idx = ((frac * ((traj.frames.len() - 1) as f64)).round() as usize)
            .min(traj.frames.len() - 1);
        b.current_frame_idx = idx;
        b.current_q = traj.frames[idx].q.clone();
        b.refresh_pose_and_scene();
    });
    ctrl.borrow_mut().playback_timer = Some(timer);
}

fn apply_cell_rule_to_spec(
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

fn colliding_pair_to_keys(workcell: &BuiltWorkcell, pair: &CollidingPair) -> (String, String) {
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

fn entity_world_point(workcell: &BuiltWorkcell, snap: &FrameSnapshot, key: &str) -> Option<[f64; 3]> {
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
    if let Some(oid) = selected_obstacle.and_then(|i| workcell.obstacles.get(i).map(|o| o.obstacle_id))
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

fn shape_to_primitive_kind(shape: &ShapePrimitiveSpec) -> PrimitiveKind {
    match shape {
        ShapePrimitiveSpec::Cuboid { .. } => PrimitiveKind::Cuboid,
        ShapePrimitiveSpec::Sphere { .. } => PrimitiveKind::Sphere,
        ShapePrimitiveSpec::CylinderZ { .. } => PrimitiveKind::Cylinder,
        ShapePrimitiveSpec::CapsuleZ { .. } => PrimitiveKind::Capsule,
    }
}

fn rgba_to_slint_color(c: [f32; 4]) -> Color {
    Color::from_rgb_u8(
        (c[0].clamp(0.0, 1.0) * 255.0).round() as u8,
        (c[1].clamp(0.0, 1.0) * 255.0).round() as u8,
        (c[2].clamp(0.0, 1.0) * 255.0).round() as u8,
    )
}

fn plot_to_slint_image(p: &PlotImage) -> Image {
    let buf = SharedPixelBuffer::<Rgba8Pixel>::clone_from_slice(&p.rgba, p.width, p.height);
    Image::from_rgba8(buf)
}

fn tool_to_slint(t: &ToolPreset) -> ToolPresetKind {
    match t {
        ToolPreset::None => ToolPresetKind::None,
        ToolPreset::ParallelGripper => ToolPresetKind::ParallelGripper,
        ToolPreset::Screwdriver => ToolPresetKind::Screwdriver,
        ToolPreset::InspectionProbe => ToolPresetKind::Probe,
        ToolPreset::Custom(_) => ToolPresetKind::Custom,
    }
}

fn short_name(s: &str) -> String {
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
        assert_eq!(workcell.model.link(payload.touch_links[0]).name, "arm0_flange_link");
    }
}
