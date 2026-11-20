//! Extracts `parry3d` shapes, arm links, attached tools, Tier-1 spheres, signed-distance witnesses,
//! MPPI rollout ribbons, coordinate frames, and 2D ESDF slices into GPU instance & line buffers.

use crate::scene::{ArmTargetSpec, BuiltWorkcell, FrameSnapshot, PlaybackTrajectory, WorkcellSpec};
use bytemuck::{Pod, Zeroable};
use nalgebra::{Isometry3, Translation3, UnitQuaternion, Vector3};
use parry3d::shape::{SharedShape, TypedShape};
use std::f64::consts::PI;
use unrust_it::collision::{isometry_to_parry, parry_to_point, parry_to_vector};
use unrust_it::{parry_to_isometry, CollidingPair, EsdfGridF32, LinkId, ObstacleId};

/// Per-instance GPU transform (`model_mat`) + RGBA color (`80` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct SolidInstance {
    pub model_mat: [[f32; 4]; 4],
    pub color: [f32; 4],
}

/// Colored line vertex for grids, coordinate triads, rollout ribbons, and witnesses (`28` bytes).
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct LineVertex {
    pub pos: [f32; 3],
    pub color: [f32; 4],
}

/// Interactive viewport layer toggles and slice configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct OverlayFlags {
    pub show_links: bool,
    pub show_tool_convex: bool,
    pub show_tier1_spheres: bool,
    pub show_esdf_slice: bool,
    /// `0 = X`, `1 = Y`, `2 = Z`.
    pub esdf_slice_axis: u8,
    pub esdf_slice_coord: f32,
    pub esdf_activation_dist: f32,
    pub esdf_opacity: f32,
    pub show_mppi_ribbons: bool,
    pub show_witnesses: bool,
    pub witnesses_xray: bool,
    pub max_witnesses: usize,
    pub show_tf_frames: bool,
    pub show_target_ghosts: bool,
    pub show_floor_grid: bool,
}

impl Default for OverlayFlags {
    fn default() -> Self {
        Self {
            show_links: true,
            show_tool_convex: true,
            show_tier1_spheres: false,
            show_esdf_slice: false,
            esdf_slice_axis: 2,
            esdf_slice_coord: 0.42,
            esdf_activation_dist: 0.12,
            esdf_opacity: 0.72,
            show_mppi_ribbons: true,
            show_witnesses: true,
            witnesses_xray: true,
            max_witnesses: 5,
            show_tf_frames: true,
            show_target_ghosts: true,
            show_floor_grid: true,
        }
    }
}

/// Extracted 2D slice of `EsdfGridF32` ready for `R32Float` GPU texture upload.
#[derive(Debug, Clone)]
pub struct EsdfSliceData {
    pub width: u32,
    pub height: u32,
    pub values: Vec<f32>,
    pub corner_origin: [f32; 3],
    pub axis_u: [f32; 3],
    pub axis_v: [f32; 3],
    pub activation_dist: f32,
    pub opacity: f32,
}

/// Complete extracted 3D scene ready for [`crate::render::SceneRenderer::render`].
#[derive(Debug, Clone, Default)]
pub struct RenderSceneData {
    pub cubes: Vec<SolidInstance>,
    pub cylinders: Vec<SolidInstance>,
    pub spheres: Vec<SolidInstance>,
    pub translucent_spheres: Vec<SolidInstance>,
    pub grid_and_frame_lines: Vec<LineVertex>,
    pub overlay_lines: Vec<LineVertex>,
    pub xray_lines: Vec<LineVertex>,
    pub esdf_slice: Option<EsdfSliceData>,
}

/// Imported waypoint frame / subframe marker rendered as a 3D coordinate triad,
/// node sphere, and hierarchy / sequence connectors.
#[derive(Debug, Clone, PartialEq)]
pub struct ImportedFrameMarker {
    pub world_pose: Isometry3<f64>,
    pub parent_world_pose: Option<Isometry3<f64>>,
    pub prev_sequence_pos: Option<[f64; 3]>,
    pub depth: usize,
    pub execute_ik: bool,
    pub selected: bool,
    pub color_rgba: [f32; 4],
}

/// Interactive 3D selection & ACM-pair highlight specification.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SceneHighlights {
    pub highlighted_links: Vec<LinkId>,
    pub highlighted_payloads: Vec<String>,
    pub highlighted_obstacles: Vec<ObstacleId>,
    pub highlighted_pair_segment: Option<([f64; 3], [f64; 3])>,
    pub imported_frames: Vec<ImportedFrameMarker>,
}

/// Entity intersected by a 3D viewport picking ray.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PickedEntity {
    Link(LinkId),
    Payload(usize),
    Obstacle(ObstacleId),
}

/// Intersects a world-space ray `(ray_origin, ray_dir)` against links, payloads, and obstacles.
pub fn pick_entity_along_ray(
    workcell: &BuiltWorkcell,
    snapshot: &FrameSnapshot,
    ray_origin: Vector3<f32>,
    ray_dir: Vector3<f32>,
) -> Option<PickedEntity> {
    let dir = ray_dir.normalize();
    let mut best_t = f32::INFINITY;
    let mut best: Option<PickedEntity> = None;

    for link in workcell.model.links() {
        let Some(pose) = snapshot.link_poses.get(link.id.index()) else {
            continue;
        };
        if let Some(geom) = workcell.env.link_geometry(link.id) {
            for sph in &geom.local_spheres {
                let c = pose.transform_point(&sph.center).coords.cast::<f32>();
                let r = (sph.radius as f32).max(0.015);
                if let Some(t) = ray_sphere_t(ray_origin, dir, c, r) {
                    if t < best_t {
                        best_t = t;
                        best = Some(PickedEntity::Link(link.id));
                    }
                }
            }
        }
    }

    for (idx, payload) in workcell.env.attached_payloads().iter().enumerate() {
        let Some(pose) = snapshot.payload_poses.get(idx) else {
            continue;
        };
        for sph in &payload.geometry.local_spheres {
            let c = pose.transform_point(&sph.center).coords.cast::<f32>();
            let r = (sph.radius as f32).max(0.015);
            if let Some(t) = ray_sphere_t(ray_origin, dir, c, r) {
                if t < best_t {
                    best_t = t;
                    best = Some(PickedEntity::Payload(idx));
                }
            }
        }
    }

    for (idx, _obs_meta) in workcell.obstacles.iter().enumerate() {
        let Some(raw_obs) = workcell.env.obstacles.get(idx) else {
            continue;
        };
        let obs_pose = snapshot
            .obstacle_poses
            .iter()
            .find(|(oid, _)| oid.0 as usize == idx)
            .map_or(raw_obs.pose, |(_, p)| *p);
        let aabb = raw_obs.shape.compute_aabb(&isometry_to_parry(&obs_pose));
        let c = parry_to_point(aabb.center()).coords.cast::<f32>();
        let r = parry_to_vector(aabb.half_extents()).norm().max(0.02) as f32;
        if let Some(t) = ray_sphere_t(ray_origin, dir, c, r) {
            if t < best_t {
                best_t = t;
                best = Some(PickedEntity::Obstacle(ObstacleId(idx as u32)));
            }
        }
    }

    best
}

#[inline]
fn ray_sphere_t(orig: Vector3<f32>, dir: Vector3<f32>, center: Vector3<f32>, r: f32) -> Option<f32> {
    let oc = orig - center;
    let b = oc.dot(&dir);
    let c = oc.norm_squared() - r * r;
    let disc = b * b - c;
    if disc < 0.0 {
        return None;
    }
    let sqrt_d = disc.sqrt();
    let t0 = -b - sqrt_d;
    if t0 > 0.0 {
        Some(t0)
    } else {
        let t1 = -b + sqrt_d;
        (t1 > 0.0).then_some(t1)
    }
}

/// Extracts all visible 3D layers from `workcell`, `spec`, `snapshot`, and optional `trajectory`.
pub fn extract_render_scene(
    workcell: &BuiltWorkcell,
    spec: &WorkcellSpec,
    snapshot: &FrameSnapshot,
    trajectory: Option<&PlaybackTrajectory>,
    camera_eye: Vector3<f32>,
    overlays: &OverlayFlags,
) -> RenderSceneData {
    extract_render_scene_with_highlights(
        workcell,
        spec,
        snapshot,
        trajectory,
        camera_eye,
        overlays,
        &SceneHighlights::default(),
    )
}

/// Extracts all visible 3D layers while highlighting selected entities and/or an active ACM pair segment.
pub fn extract_render_scene_with_highlights(
    workcell: &BuiltWorkcell,
    spec: &WorkcellSpec,
    snapshot: &FrameSnapshot,
    trajectory: Option<&PlaybackTrajectory>,
    camera_eye: Vector3<f32>,
    overlays: &OverlayFlags,
    highlights: &SceneHighlights,
) -> RenderSceneData {
    let mut out = RenderSceneData::default();
    let mut translucent_with_dist: Vec<(f32, SolidInstance)> = Vec::new();
    let hl_cyan = [0.35, 0.94, 1.0, 1.0];

    if overlays.show_floor_grid {
        push_floor_grid(&mut out.grid_and_frame_lines, 1.0, 0.2);
        push_axis_triad(&mut out.grid_and_frame_lines, &Isometry3::identity(), 0.22);
    }

    // 1. Robot links
    if overlays.show_links {
        let model = &workcell.model;
        for link in model.links() {
            let Some(link_pose) = snapshot.link_poses.get(link.id.index()) else {
                continue;
            };
            let arm_idx = link.robot.0 as usize;
            let base_color = workcell
                .arms
                .get(arm_idx)
                .map_or([0.22, 0.74, 0.97, 1.0], |a| a.color_rgba);
            // Subtle distal brightening along arm links for clearer visual articulation
            let mut link_tint = tint_color(base_color, 0.92 + 0.04 * ((link.id.index() % 5) as f32));
            if highlights.highlighted_links.contains(&link.id) {
                link_tint = mix_color(link_tint, hl_cyan, 0.58);
            }

            if let Some(geom) = workcell.env.link_geometry(link.id) {
                for (sub_pose, sub_shape) in &geom.convex_shapes {
                    let world = *link_pose * *sub_pose;
                    push_parry_shape_opaque(&mut out, &world, sub_shape, link_tint);
                }
            }
        }
    }

    // 2. Attached compound tools / payloads
    if overlays.show_tool_convex {
        for (idx, payload) in workcell.env.attached_payloads().iter().enumerate() {
            let Some(payload_pose) = snapshot.payload_poses.get(idx) else {
                continue;
            };
            let robot_idx = workcell.model.link(payload.parent_link).robot.0 as usize;
            let arm_color = workcell
                .arms
                .get(robot_idx)
                .map_or([0.75, 0.80, 0.88, 1.0], |a| a.color_rgba);
            // Blend arm color toward brushed metallic titanium so tools stand out crisply from links
            let mut tool_color = mix_color(arm_color, [0.84, 0.89, 0.96, 1.0], 0.45);
            if highlights.highlighted_payloads.iter().any(|p| p == &payload.id) {
                tool_color = mix_color(tool_color, hl_cyan, 0.58);
            }
            for (sub_pose, sub_shape) in &payload.geometry.convex_shapes {
                let world = *payload_pose * *sub_pose;
                push_parry_shape_opaque(&mut out, &world, sub_shape, tool_color);
            }
        }
    }

    // 3. Static & dynamic obstacles / pedestal / workpiece
    for (idx, obs_meta) in workcell.obstacles.iter().enumerate() {
        let oid = ObstacleId(idx as u32);
        let mut color = obs_meta.color_rgba;
        if highlights.highlighted_obstacles.contains(&oid) {
            color = mix_color(color, hl_cyan, 0.58);
        }
        let maybe_env_pose = snapshot
            .obstacle_poses
            .iter()
            .find(|(id, _)| id.0 as usize == idx)
            .map(|(_, p)| *p);
        if let (Some(obs_pose), Some(raw_obs)) = (maybe_env_pose, workcell.env.obstacles.get(idx)) {
            push_parry_shape_opaque(&mut out, &obs_pose, &raw_obs.shape, color);
        } else {
            let (shape_pose, shape) = obs_meta.shape.to_parry_shape_and_pose();
            push_parry_shape_opaque(
                &mut out,
                &(obs_meta.base_pose * shape_pose),
                &shape,
                color,
            );
        }
    }

    // 4. Translucent Tier-1 collision spheres (robot links + attached tools)
    if overlays.show_tier1_spheres {
        for link in workcell.model.links() {
            let Some(link_pose) = snapshot.link_poses.get(link.id.index()) else {
                continue;
            };
            let arm_idx = link.robot.0 as usize;
            let mut color = workcell
                .arms
                .get(arm_idx)
                .map_or([0.22, 0.74, 0.97, 1.0], |a| a.color_rgba);
            color[3] = 0.34;
            if let Some(geom) = workcell.env.link_geometry(link.id) {
                for sph in &geom.local_spheres {
                    let wp = link_pose.transform_point(&sph.center);
                    push_translucent_sphere(
                        &mut translucent_with_dist,
                        camera_eye,
                        wp.coords,
                        sph.radius as f32,
                        color,
                    );
                }
            }
        }
        for (idx, payload) in workcell.env.attached_payloads().iter().enumerate() {
            let Some(payload_pose) = snapshot.payload_poses.get(idx) else {
                continue;
            };
            let robot_idx = workcell.model.link(payload.parent_link).robot.0 as usize;
            let mut color = workcell
                .arms
                .get(robot_idx)
                .map_or([0.85, 0.90, 0.95, 1.0], |a| a.color_rgba);
            color = mix_color(color, [0.95, 0.96, 1.0, 1.0], 0.35);
            color[3] = 0.36;
            for sph in &payload.geometry.local_spheres {
                let wp = payload_pose.transform_point(&sph.center);
                push_translucent_sphere(
                    &mut translucent_with_dist,
                    camera_eye,
                    wp.coords,
                    sph.radius as f32,
                    color,
                );
            }
        }
    }

    // 5. Coordinate frames (TCPs) & target/standoff ghosts
    if overlays.show_tf_frames {
        for tcp_pose in &snapshot.tcp_poses {
            push_axis_triad(&mut out.grid_and_frame_lines, tcp_pose, 0.055);
        }
    }
    if overlays.show_target_ghosts {
        for (arm_idx, arm_spec) in spec.arms.iter().enumerate() {
            let arm_col = workcell
                .arms
                .get(arm_idx)
                .map_or([0.22, 0.74, 0.97, 1.0], |a| a.color_rgba);
            match &arm_spec.target {
                ArmTargetSpec::ApproachPose {
                    final_pose,
                    standoff_local,
                } => {
                    let final_iso = final_pose.to_isometry();
                    let pre_iso = final_iso
                        * Translation3::new(
                            standoff_local[0],
                            standoff_local[1],
                            standoff_local[2],
                        );
                    push_axis_triad(&mut out.overlay_lines, &final_iso, 0.045);
                    push_line(
                        &mut out.overlay_lines,
                        pre_iso.translation.vector,
                        final_iso.translation.vector,
                        [arm_col[0], arm_col[1], arm_col[2], 0.75],
                    );
                    let mut ghost_col = arm_col;
                    ghost_col[3] = 0.42;
                    push_translucent_sphere(
                        &mut translucent_with_dist,
                        camera_eye,
                        final_iso.translation.vector,
                        0.008,
                        ghost_col,
                    );
                }
                ArmTargetSpec::DirectPose(pose) => {
                    let final_iso = pose.to_isometry();
                    push_axis_triad(&mut out.overlay_lines, &final_iso, 0.045);
                    let mut ghost_col = arm_col;
                    ghost_col[3] = 0.42;
                    push_translucent_sphere(
                        &mut translucent_with_dist,
                        camera_eye,
                        final_iso.translation.vector,
                        0.009,
                        ghost_col,
                    );
                }
                _ => {}
            }
        }
    }

    // 6. MPPI rollout ribbons & nominal TCP trails
    if overlays.show_mppi_ribbons {
        if let Some(traj) = trajectory {
            for trail in traj.tcp_trails.iter().enumerate() {
                let (arm_idx, pts) = trail;
                let base_col = workcell
                    .arms
                    .get(arm_idx)
                    .map_or([0.22, 0.74, 0.97, 1.0], |a| a.color_rgba);
                let c = [base_col[0], base_col[1], base_col[2], 0.85];
                for seg in pts.windows(2) {
                    out.overlay_lines.push(LineVertex {
                        pos: seg[0],
                        color: c,
                    });
                    out.overlay_lines.push(LineVertex {
                        pos: seg[1],
                        color: c,
                    });
                }
            }
            for rib in &traj.rollout_ribbons {
                let base_col = workcell
                    .arms
                    .get(rib.arm_index)
                    .map_or([0.22, 0.74, 0.97, 1.0], |a| a.color_rgba);
                let c = if rib.is_elite {
                    [0.98, 0.78, 0.18, 0.78]
                } else {
                    [base_col[0], base_col[1], base_col[2], 0.24]
                };
                for seg in rib.points.windows(2) {
                    out.overlay_lines.push(LineVertex {
                        pos: seg[0],
                        color: c,
                    });
                    out.overlay_lines.push(LineVertex {
                        pos: seg[1],
                        color: c,
                    });
                }
            }
        }
    }

    // 7. SignedDistanceWitness pairs (3D rods + endpoint spheres + optional X-ray overlay)
    if overlays.show_witnesses {
        let default_margin = spec.rules.default_body_margin;
        for w in snapshot.witnesses.iter().take(overlays.max_witnesses) {
            let eff_margin = workcell.env.effective_pair_margin(&w.pair, default_margin);
            let is_inter_arm = matches!(
                w.pair,
                CollidingPair::LinkToLink { .. }
                    | CollidingPair::PayloadToLink { .. }
                    | CollidingPair::PayloadToPayload { .. }
            );
            let color = if w.signed_distance <= eff_margin + 0.002 {
                [0.96, 0.26, 0.37, 1.0] // Red violation / tight margin
            } else if is_inter_arm || w.signed_distance < 0.05 {
                [0.96, 0.68, 0.14, 1.0] // Amber inter-arm / close proximity
            } else {
                [0.98, 0.45, 0.60, 1.0] // Rose obstacle witness
            };

            let pa = w.witness_a_world.coords;
            let pb = w.witness_b_world.coords;
            // Solid endpoint spheres (r = 4.5 mm) + connecting rod (r = 2.2 mm) for HiDPI clarity
            out.spheres.push(SolidInstance {
                model_mat: translation_scale_mat4(pa, [0.0045, 0.0045, 0.0045]),
                color,
            });
            out.spheres.push(SolidInstance {
                model_mat: translation_scale_mat4(pb, [0.0045, 0.0045, 0.0045]),
                color,
            });
            push_cylinder_between(&mut out.cylinders, pa, pb, 0.0022, color);

            if overlays.witnesses_xray {
                push_line(&mut out.xray_lines, pa, pb, color);
            }
        }
    }

    // 7b. Selected ACM Pair Highlight Witness Rod (Split 3D | ACM visual link)
    if let Some((a, b)) = highlights.highlighted_pair_segment {
        let pa = Vector3::new(a[0], a[1], a[2]);
        let pb = Vector3::new(b[0], b[1], b[2]);
        let col = [0.22, 0.96, 0.88, 1.0];
        out.spheres.push(SolidInstance {
            model_mat: translation_scale_mat4(pa, [0.0065, 0.0065, 0.0065]),
            color: col,
        });
        out.spheres.push(SolidInstance {
            model_mat: translation_scale_mat4(pb, [0.0065, 0.0065, 0.0065]),
            color: col,
        });
        push_cylinder_between(&mut out.cylinders, pa, pb, 0.0035, col);
        push_line(&mut out.xray_lines, pa, pb, col);
    }

    // 7c. Imported Trajectory Frames & Hierarchical Subframes
    for marker in &highlights.imported_frames {
        let axis_len = if marker.selected {
            0.068
        } else if marker.depth == 0 {
            0.048
        } else {
            0.032
        };
        push_axis_triad(&mut out.overlay_lines, &marker.world_pose, axis_len);

        let pos = marker.world_pose.translation.vector;
        let radius = if marker.selected {
            0.0085
        } else if marker.depth == 0 {
            0.0058
        } else {
            0.0040
        };
        let mut sphere_col = marker.color_rgba;
        if !marker.execute_ik {
            sphere_col[3] = sphere_col[3].min(0.60);
        }
        out.spheres.push(SolidInstance {
            model_mat: translation_scale_mat4(pos, [radius, radius, radius]),
            color: sphere_col,
        });

        if let Some(parent_iso) = marker.parent_world_pose {
            let hierarchy_col = [
                marker.color_rgba[0],
                marker.color_rgba[1],
                marker.color_rgba[2],
                0.45,
            ];
            push_line(
                &mut out.overlay_lines,
                parent_iso.translation.vector,
                pos,
                hierarchy_col,
            );
        }

        if let Some(prev) = marker.prev_sequence_pos {
            let seq_col = [
                marker.color_rgba[0],
                marker.color_rgba[1],
                marker.color_rgba[2],
                0.90,
            ];
            push_line(
                &mut out.overlay_lines,
                Vector3::new(prev[0], prev[1], prev[2]),
                pos,
                seq_col,
            );
        }
    }

    // 8. 3D ESDF 2D Slice
    if overlays.show_esdf_slice {
        out.esdf_slice = extract_esdf_slice(
            &workcell.esdf,
            overlays.esdf_slice_axis,
            overlays.esdf_slice_coord,
            overlays.esdf_activation_dist,
            overlays.esdf_opacity,
        );
    }

    // Sort translucent spheres strictly back-to-front relative to camera_eye
    translucent_with_dist.sort_by(|a, b| b.0.total_cmp(&a.0));
    out.translucent_spheres = translucent_with_dist
        .into_iter()
        .map(|(_, inst)| inst)
        .collect();

    out
}

/// Extracts a 2D world-aligned slice (`X`, `Y`, or `Z`) from `esdf`.
pub fn extract_esdf_slice(
    esdf: &EsdfGridF32,
    axis: u8,
    coord: f32,
    activation_dist: f32,
    opacity: f32,
) -> Option<EsdfSliceData> {
    let [nx, ny, nz] = esdf.dims;
    if nx == 0 || ny == 0 || nz == 0 || esdf.distances.len() < nx * ny * nz {
        return None;
    }
    let vs = esdf.voxel_size;
    let inv_vs = esdf.inv_voxel_size;
    let orig = esdf.origin;
    let stride_y = nx;
    let stride_z = nx * ny;

    match axis {
        0 => {
            // Fixed X, U = Y, V = Z
            let ix = ((coord - orig[0]) * inv_vs - 0.5)
                .round()
                .clamp(0.0, (nx - 1) as f32) as usize;
            let x_w = orig[0] + (ix as f32 + 0.5) * vs;
            let mut values = Vec::with_capacity(ny * nz);
            for iz in 0..nz {
                for iy in 0..ny {
                    values.push(esdf.distances[iz * stride_z + iy * stride_y + ix]);
                }
            }
            Some(EsdfSliceData {
                width: ny as u32,
                height: nz as u32,
                values,
                corner_origin: [x_w, orig[1], orig[2]],
                axis_u: [0.0, ny as f32 * vs, 0.0],
                axis_v: [0.0, 0.0, nz as f32 * vs],
                activation_dist,
                opacity,
            })
        }
        1 => {
            // Fixed Y, U = X, V = Z
            let iy = ((coord - orig[1]) * inv_vs - 0.5)
                .round()
                .clamp(0.0, (ny - 1) as f32) as usize;
            let y_w = orig[1] + (iy as f32 + 0.5) * vs;
            let mut values = Vec::with_capacity(nx * nz);
            for iz in 0..nz {
                let row = iz * stride_z + iy * stride_y;
                values.extend_from_slice(&esdf.distances[row..row + nx]);
            }
            Some(EsdfSliceData {
                width: nx as u32,
                height: nz as u32,
                values,
                corner_origin: [orig[0], y_w, orig[2]],
                axis_u: [nx as f32 * vs, 0.0, 0.0],
                axis_v: [0.0, 0.0, nz as f32 * vs],
                activation_dist,
                opacity,
            })
        }
        _ => {
            // Fixed Z, U = X, V = Y
            let iz = ((coord - orig[2]) * inv_vs - 0.5)
                .round()
                .clamp(0.0, (nz - 1) as f32) as usize;
            let z_w = orig[2] + (iz as f32 + 0.5) * vs;
            let start = iz * stride_z;
            let values = esdf.distances[start..start + stride_z].to_vec();
            Some(EsdfSliceData {
                width: nx as u32,
                height: ny as u32,
                values,
                corner_origin: [orig[0], orig[1], z_w],
                axis_u: [nx as f32 * vs, 0.0, 0.0],
                axis_v: [0.0, ny as f32 * vs, 0.0],
                activation_dist,
                opacity,
            })
        }
    }
}

fn push_parry_shape_opaque(
    out: &mut RenderSceneData,
    world_pose: &Isometry3<f64>,
    shape: &SharedShape,
    color: [f32; 4],
) {
    match shape.as_typed_shape() {
        TypedShape::Ball(b) => {
            let r = b.radius.max(1e-4) as f32;
            out.spheres.push(SolidInstance {
                model_mat: iso_scale_to_mat4(world_pose, [r, r, r]),
                color,
            });
        }
        TypedShape::Cuboid(c) => {
            let h = c.half_extents;
            out.cubes.push(SolidInstance {
                model_mat: iso_scale_to_mat4(
                    world_pose,
                    [
                        (h.x.max(1e-4)) as f32,
                        (h.y.max(1e-4)) as f32,
                        (h.z.max(1e-4)) as f32,
                    ],
                ),
                color,
            });
        }
        TypedShape::Cylinder(c) => {
            let r = c.radius.max(1e-4) as f32;
            let hh = c.half_height.max(1e-4) as f32;
            out.cylinders.push(SolidInstance {
                model_mat: iso_scale_to_mat4(world_pose, [r, hh, r]),
                color,
            });
        }
        TypedShape::Capsule(c) => {
            let r = c.radius.max(1e-4) as f32;
            let pa = world_pose.transform_point(&parry_to_point(c.segment.a)).coords;
            let pb = world_pose.transform_point(&parry_to_point(c.segment.b)).coords;
            out.spheres.push(SolidInstance {
                model_mat: translation_scale_mat4(pa, [r, r, r]),
                color,
            });
            out.spheres.push(SolidInstance {
                model_mat: translation_scale_mat4(pb, [r, r, r]),
                color,
            });
            push_cylinder_between(&mut out.cylinders, pa, pb, r, color);
        }
        TypedShape::Compound(comp) => {
            for (sub_pose, sub_shape) in comp.shapes() {
                let combined = *world_pose * parry_to_isometry(sub_pose);
                push_parry_shape_opaque(out, &combined, sub_shape, color);
            }
        }
        _ => {
            let parry_pose = isometry_to_parry(world_pose);
            let aabb = shape.compute_aabb(&parry_pose);
            let center = parry_to_point(aabb.center());
            let ext = parry_to_vector(aabb.half_extents());
            out.cubes.push(SolidInstance {
                model_mat: translation_scale_mat4(
                    center.coords,
                    [
                        (ext.x.max(1e-4)) as f32,
                        (ext.y.max(1e-4)) as f32,
                        (ext.z.max(1e-4)) as f32,
                    ],
                ),
                color,
            });
        }
    }
}

fn push_cylinder_between(
    cylinders: &mut Vec<SolidInstance>,
    pa: Vector3<f64>,
    pb: Vector3<f64>,
    radius: f32,
    color: [f32; 4],
) {
    let diff = pb - pa;
    let len = diff.norm();
    if len <= 1e-6 {
        return;
    }
    let dir = diff / len;
    let mid = 0.5 * (pa + pb);
    let rot = UnitQuaternion::rotation_between(&Vector3::y(), &dir)
        .unwrap_or_else(|| UnitQuaternion::from_axis_angle(&Vector3::x_axis(), PI));
    let iso = Isometry3::from_parts(Translation3::from(mid), rot);
    cylinders.push(SolidInstance {
        model_mat: iso_scale_to_mat4(&iso, [radius, (0.5 * len) as f32, radius]),
        color,
    });
}

fn push_translucent_sphere(
    out: &mut Vec<(f32, SolidInstance)>,
    camera_eye: Vector3<f32>,
    pos_f64: Vector3<f64>,
    radius: f32,
    color: [f32; 4],
) {
    let pos = pos_f64.cast::<f32>();
    let dist_sq = (pos - camera_eye).norm_squared();
    let r = radius.max(1e-4);
    out.push((
        dist_sq,
        SolidInstance {
            model_mat: translation_scale_mat4(pos_f64, [r, r, r]),
            color,
        },
    ));
}

fn push_floor_grid(lines: &mut Vec<LineVertex>, half_extent: f32, step: f32) {
    let n = (half_extent / step).round() as i32;
    let extent = (n as f32) * step;
    for i in -n..=n {
        let c = (i as f32) * step;
        let is_major = i == 0 || i % 5 == 0;
        let col = if i == 0 {
            [0.24, 0.30, 0.40, 0.85]
        } else if is_major {
            [0.16, 0.21, 0.29, 0.75]
        } else {
            [0.12, 0.16, 0.22, 0.55]
        };
        lines.push(LineVertex {
            pos: [-extent, c, 0.0],
            color: col,
        });
        lines.push(LineVertex {
            pos: [extent, c, 0.0],
            color: col,
        });
        lines.push(LineVertex {
            pos: [c, -extent, 0.0],
            color: col,
        });
        lines.push(LineVertex {
            pos: [c, extent, 0.0],
            color: col,
        });
    }
}

fn push_axis_triad(lines: &mut Vec<LineVertex>, pose: &Isometry3<f64>, length: f64) {
    let o = pose.translation.vector;
    let x = o + pose.rotation * (Vector3::x() * length);
    let y = o + pose.rotation * (Vector3::y() * length);
    let z = o + pose.rotation * (Vector3::z() * length);
    push_line(lines, o, x, [0.95, 0.28, 0.35, 0.95]);
    push_line(lines, o, y, [0.22, 0.86, 0.52, 0.95]);
    push_line(lines, o, z, [0.25, 0.62, 0.98, 0.95]);
}

#[inline]
fn push_line(lines: &mut Vec<LineVertex>, a: Vector3<f64>, b: Vector3<f64>, color: [f32; 4]) {
    lines.push(LineVertex {
        pos: [a.x as f32, a.y as f32, a.z as f32],
        color,
    });
    lines.push(LineVertex {
        pos: [b.x as f32, b.y as f32, b.z as f32],
        color,
    });
}

#[inline]
fn iso_scale_to_mat4(iso: &Isometry3<f64>, scale: [f32; 3]) -> [[f32; 4]; 4] {
    let m = iso.to_homogeneous().cast::<f32>();
    let s = nalgebra::Matrix4::new_nonuniform_scaling(&Vector3::new(scale[0], scale[1], scale[2]));
    (m * s).into()
}

#[inline]
fn translation_scale_mat4(p: Vector3<f64>, scale: [f32; 3]) -> [[f32; 4]; 4] {
    let [sx, sy, sz] = scale;
    [
        [sx, 0.0, 0.0, 0.0],
        [0.0, sy, 0.0, 0.0],
        [0.0, 0.0, sz, 0.0],
        [p.x as f32, p.y as f32, p.z as f32, 1.0],
    ]
}

#[inline]
fn tint_color(c: [f32; 4], factor: f32) -> [f32; 4] {
    [
        (c[0] * factor).clamp(0.0, 1.0),
        (c[1] * factor).clamp(0.0, 1.0),
        (c[2] * factor).clamp(0.0, 1.0),
        c[3],
    ]
}

#[inline]
fn mix_color(a: [f32; 4], b: [f32; 4], t: f32) -> [f32; 4] {
    let k = 1.0 - t;
    [
        a[0] * k + b[0] * t,
        a[1] * k + b[1] * t,
        a[2] * k + b[2] * t,
        a[3] * k + b[3] * t,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::OrbitCamera;
    use crate::scene::{evaluate_frame_snapshot, preset_workcell, PresetId};

    #[test]
    fn raycast_picks_central_workpiece_in_three_arm_cell() {
        let spec = preset_workcell(PresetId::ThreeArmCloseProximity);
        let wc = spec.compile(1).unwrap();
        let mut ws = wc.env.create_workspace(&wc.model);
        let snap = evaluate_frame_snapshot(&wc, wc.q_start.as_slice(), &[], 5, None, &mut ws);

        // Shoot ray straight toward central workpiece center (0.0, 0.0, 0.42)
        let origin = Vector3::new(0.25, 0.0, 0.42);
        let dir = Vector3::new(-1.0, 0.0, 0.0);
        let picked = pick_entity_along_ray(&wc, &snap, origin, dir);
        let wp_id = *wc.obstacle_by_uid.get("workpiece").unwrap();
        assert_eq!(picked, Some(PickedEntity::Obstacle(wp_id)));
    }

    #[test]
    fn extract_render_scene_with_highlights_emits_pair_segment_geometry() {
        let spec = preset_workcell(PresetId::ThreeArmCloseProximity);
        let wc = spec.compile(1).unwrap();
        let mut ws = wc.env.create_workspace(&wc.model);
        let snap = evaluate_frame_snapshot(&wc, wc.q_start.as_slice(), &[], 5, None, &mut ws);
        let cam = OrbitCamera::default();
        let overlays = OverlayFlags::default();

        let base = extract_render_scene(&wc, &spec, &snap, None, cam.eye_position(), &overlays);
        let hl = SceneHighlights {
            highlighted_links: vec![wc.arms[0].flange_link],
            highlighted_payloads: vec!["arm1_screwdriver".to_string()],
            highlighted_obstacles: vec![*wc.obstacle_by_uid.get("workpiece").unwrap()],
            highlighted_pair_segment: Some(([0.0, 0.0, 0.40], [0.05, 0.02, 0.44])),
            ..Default::default()
        };
        let with_hl = extract_render_scene_with_highlights(
            &wc,
            &spec,
            &snap,
            None,
            cam.eye_position(),
            &overlays,
            &hl,
        );
        assert_eq!(with_hl.spheres.len(), base.spheres.len() + 2);
        assert_eq!(with_hl.cylinders.len(), base.cylinders.len() + 1);
        assert_eq!(with_hl.xray_lines.len(), base.xray_lines.len() + 2);
    }
}
