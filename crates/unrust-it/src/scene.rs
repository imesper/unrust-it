//! Unified [`WorldScene`] owner synchronizing [`MultiRobotModel`], [`CollisionEnvironment`],
//! generational [`ObstacleHandle`]s, and dirty-region [`EsdfGridF32`] re-carving.

use crate::batch::{EsdfGridF32, FlatRobotModelF32};
use crate::collision::{
    isometry_to_parry, AllowedCollisionMatrix, CollisionEnvironment, CollisionError, LinkGeometry,
    Obstacle,
};
use crate::kinematics::TransformBuffer;
use crate::robot::{
    AttachedPayload, LinkId, MultiRobotModel, ObstacleId, PayloadId, RobotError,
};
use nalgebra::Isometry3;
use parry3d::shape::SharedShape;
use std::collections::HashMap;
use thiserror::Error;

/// Generational handle referencing a named obstacle managed by [`WorldScene`].
///
/// Handles become stale as soon as their obstacle is removed, preventing silent
/// mutations of unrelated or recycled slots.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ObstacleHandle {
    id: ObstacleId,
    generation: u64,
}

impl ObstacleHandle {
    /// Returns the underlying [`ObstacleId`] for direct queries in [`CollisionEnvironment`].
    #[inline]
    pub const fn obstacle_id(&self) -> ObstacleId {
        self.id
    }

    /// Returns the monotonically increasing generation token of this handle.
    #[inline]
    pub const fn generation(&self) -> u64 {
        self.generation
    }
}

/// Configuration options attached when adding or upserting an obstacle in [`WorldScene`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ObstacleOptions {
    /// Whether this obstacle is carved into the scene's [`EsdfGridF32`] (default `true`).
    pub carve_esdf: bool,
    /// Optional default obstacle safety margin in meters applied in [`CollisionEnvironment`].
    pub safety_margin: Option<f64>,
    /// Optional default obstacle CBF activation distance in meters applied in [`CollisionEnvironment`].
    pub activation_distance: Option<f64>,
}

impl Default for ObstacleOptions {
    fn default() -> Self {
        Self {
            carve_esdf: true,
            safety_margin: None,
            activation_distance: None,
        }
    }
}

impl ObstacleOptions {
    /// Sets whether this obstacle is carved into the ESDF voxel grid.
    #[inline]
    pub const fn with_carve_esdf(mut self, carve_esdf: bool) -> Self {
        self.carve_esdf = carve_esdf;
        self
    }

    /// Sets the obstacle's default safety margin (meters).
    #[inline]
    pub const fn with_margin(mut self, margin: f64) -> Self {
        self.safety_margin = Some(margin);
        self
    }

    /// Sets the obstacle's default CBF activation distance (meters).
    #[inline]
    pub const fn with_activation(mut self, activation: f64) -> Self {
        self.activation_distance = Some(activation);
        self
    }
}

/// Configuration for the volumetric 3D [`EsdfGridF32`] owned by [`WorldScene`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EsdfConfig {
    /// World-space minimum corner `[ox, oy, oz]` in meters.
    pub origin: [f32; 3],
    /// Isotropic voxel resolution in meters (must be `> 0.0`).
    pub voxel_size: f32,
    /// Voxel grid dimensions `[nx, ny, nz]` (each must be `> 0`).
    pub dims: [usize; 3],
    /// Positive truncation distance in meters for uncarved free-space voxels (must be `> 0.0`).
    pub truncation: f32,
}

impl Default for EsdfConfig {
    fn default() -> Self {
        Self {
            origin: [-1.0, -1.0, 0.0],
            voxel_size: 0.025,
            dims: [80, 80, 64],
            truncation: 1.0,
        }
    }
}

impl EsdfConfig {
    fn validate(&self) -> Result<(), SceneError> {
        if !(self.voxel_size > 0.0 && self.voxel_size.is_finite()) {
            return Err(SceneError::InvalidEsdfConfig(
                "voxel_size must be finite and > 0.0",
            ));
        }
        if !(self.truncation > 0.0 && self.truncation.is_finite()) {
            return Err(SceneError::InvalidEsdfConfig(
                "truncation must be finite and > 0.0",
            ));
        }
        if self.dims.contains(&0) {
            return Err(SceneError::InvalidEsdfConfig(
                "all ESDF grid dimensions must be > 0",
            ));
        }
        Ok(())
    }
}

/// Errors produced by synchronized [`WorldScene`] operations.
#[derive(Debug, Error)]
pub enum SceneError {
    /// Attempted to add a new obstacle with a name already in use.
    #[error("obstacle named `{0}` already exists in WorldScene")]
    DuplicateObstacleName(String),
    /// Lookup by name failed because no active obstacle carries that name.
    #[error("no active obstacle named `{0}` in WorldScene")]
    UnknownObstacleName(String),
    /// Handle refers to a removed obstacle or an older generation.
    #[error("stale or invalid ObstacleHandle (id={}, gen={})", .0.id.0, .0.generation)]
    StaleOrInvalidHandle(ObstacleHandle),
    /// Invalid [`EsdfConfig`] supplied to constructor or `enable_esdf`.
    #[error("invalid ESDF configuration: {0}")]
    InvalidEsdfConfig(&'static str),
    /// Wrapped robot kinematic or payload error.
    #[error("robot model error: {0}")]
    Robot(#[from] RobotError),
    /// Wrapped collision environment error.
    #[error("collision environment error: {0}")]
    Collision(#[from] CollisionError),
}

#[derive(Debug, Clone)]
struct ObstacleEntry {
    name: String,
    handle: ObstacleHandle,
    carve_esdf: bool,
    /// Inflated world-space AABB last carved into `esdf`, used for exact dirty-region clearing.
    carved_aabb: Option<([f32; 3], [f32; 3])>,
}

/// Single authoritative owner synchronizing [`MultiRobotModel`], [`CollisionEnvironment`]
/// (Tier-2 BVH + per-pair rules + payloads), and dirty-region [`EsdfGridF32`] re-carving.
#[derive(Debug, Clone)]
pub struct WorldScene {
    model: MultiRobotModel,
    env: CollisionEnvironment,
    esdf: Option<EsdfGridF32>,
    esdf_truncation: f32,
    by_name: HashMap<String, ObstacleHandle>,
    entries: Vec<Option<ObstacleEntry>>,
    next_generation: u64,
    scene_revision: u64,
    esdf_revision: u64,
    defer_sync: bool,
    bvh_dirty: bool,
    pending_dirty_regions: Vec<([f32; 3], [f32; 3])>,
    pending_carve_ids: Vec<ObstacleId>,
}

impl WorldScene {
    /// Creates a [`WorldScene`] wrapping `model` and `env` without allocating an [`EsdfGridF32`].
    ///
    /// Pre-existing active obstacles in `env` are registered under `"obstacle_{id}"` with
    /// `carve_esdf = true`.
    pub fn new(model: MultiRobotModel, env: CollisionEnvironment) -> Self {
        let mut by_name = HashMap::new();
        let mut entries = vec![None; env.obstacles.len()];
        let mut next_generation = 1_u64;

        for (id, _) in env.active_obstacles() {
            let name = format!("obstacle_{}", id.0);
            let handle = ObstacleHandle {
                id,
                generation: next_generation,
            };
            next_generation += 1;
            by_name.insert(name.clone(), handle);
            entries[id.0 as usize] = Some(ObstacleEntry {
                name,
                handle,
                carve_esdf: true,
                carved_aabb: None,
            });
        }

        Self {
            model,
            env,
            esdf: None,
            esdf_truncation: 1.0,
            by_name,
            entries,
            next_generation,
            scene_revision: 1,
            esdf_revision: 0,
            defer_sync: false,
            bvh_dirty: false,
            pending_dirty_regions: Vec::new(),
            pending_carve_ids: Vec::new(),
        }
    }

    /// Creates a [`WorldScene`] with an initialized [`EsdfGridF32`] and immediately carves all
    /// pre-existing active obstacles.
    pub fn with_esdf(
        model: MultiRobotModel,
        env: CollisionEnvironment,
        cfg: EsdfConfig,
    ) -> Result<Self, SceneError> {
        let mut scene = Self::new(model, env);
        scene.enable_esdf(cfg)?;
        Ok(scene)
    }

    /// Allocates (or replaces) the owned [`EsdfGridF32`] and performs a full carve of all
    /// active obstacles marked `carve_esdf`.
    pub fn enable_esdf(&mut self, cfg: EsdfConfig) -> Result<(), SceneError> {
        cfg.validate()?;
        self.esdf_truncation = cfg.truncation;
        self.esdf = Some(EsdfGridF32::new(
            cfg.origin,
            cfg.voxel_size,
            cfg.dims,
            cfg.truncation,
        ));
        self.rebuild_esdf_full();
        Ok(())
    }

    /// Releases the owned [`EsdfGridF32`], clearing any tracked carved bounds.
    pub fn disable_esdf(&mut self) {
        self.esdf = None;
        self.pending_dirty_regions.clear();
        self.pending_carve_ids.clear();
        for entry in self.entries.iter_mut().flatten() {
            entry.carved_aabb = None;
        }
        self.esdf_revision += 1;
    }

    /// Re-clears the entire owned [`EsdfGridF32`] (if enabled) and re-carves all active
    /// `carve_esdf` obstacles cleanly.
    pub fn rebuild_esdf_full(&mut self) {
        let Some(ref mut esdf) = self.esdf else {
            return;
        };
        let trunc = self.esdf_truncation;
        let pad = trunc + 1.5 * esdf.voxel_size;
        esdf.clear(trunc);
        self.pending_dirty_regions.clear();
        self.pending_carve_ids.clear();

        for (id, obs) in self.env.active_obstacles() {
            if let Some(Some(entry)) = self.entries.get_mut(id.0 as usize) {
                if entry.carve_esdf {
                    let box_w = inflated_obstacle_aabb(&obs.shape, &obs.pose, pad);
                    if let Some((min_idx, max_idx)) =
                        esdf.voxel_region_for_world_aabb(box_w.0, box_w.1)
                    {
                        esdf.carve_shape_in_region(&obs.shape, &obs.pose, min_idx, max_idx);
                    }
                    entry.carved_aabb = Some(box_w);
                } else {
                    entry.carved_aabb = None;
                }
            }
        }
        self.esdf_revision += 1;
    }

    /// Immutable reference to the owned [`MultiRobotModel`].
    #[inline]
    pub fn model(&self) -> &MultiRobotModel {
        &self.model
    }

    /// Mutable reference to the owned [`MultiRobotModel`].
    #[inline]
    pub fn model_mut(&mut self) -> &mut MultiRobotModel {
        self.scene_revision += 1;
        &mut self.model
    }

    /// Immutable reference to the synchronized [`CollisionEnvironment`].
    #[inline]
    pub fn env(&self) -> &CollisionEnvironment {
        &self.env
    }

    /// Immutable reference to the synchronized [`EsdfGridF32`], if enabled.
    #[inline]
    pub fn esdf(&self) -> Option<&EsdfGridF32> {
        self.esdf.as_ref()
    }

    /// Positive truncation distance of the owned ESDF grid in meters.
    #[inline]
    pub fn esdf_truncation(&self) -> f32 {
        self.esdf_truncation
    }

    /// Monotonically incremented whenever topology, obstacles, payloads, or collision rules change.
    #[inline]
    pub fn scene_revision(&self) -> u64 {
        self.scene_revision
    }

    /// Monotonically incremented whenever the owned [`EsdfGridF32`] content changes.
    #[inline]
    pub fn esdf_revision(&self) -> u64 {
        self.esdf_revision
    }

    /// Looks up the active [`ObstacleHandle`] for `name`.
    #[inline]
    pub fn obstacle_handle(&self, name: &str) -> Option<ObstacleHandle> {
        self.by_name.get(name).copied()
    }

    /// Returns the name of an active [`ObstacleHandle`], or `None` if stale.
    #[inline]
    pub fn obstacle_name(&self, handle: ObstacleHandle) -> Option<&str> {
        self.validate_handle(handle).ok().map(|e| e.name.as_str())
    }

    /// Returns the underlying `parry3d` [`Obstacle`] for an active [`ObstacleHandle`].
    #[inline]
    pub fn obstacle(&self, handle: ObstacleHandle) -> Option<&Obstacle> {
        self.validate_handle(handle).ok()?;
        self.env.obstacle(handle.id)
    }

    /// Convenience lookup for an active obstacle by name.
    #[inline]
    pub fn obstacle_by_name(&self, name: &str) -> Option<&Obstacle> {
        self.obstacle_handle(name).and_then(|h| self.obstacle(h))
    }

    /// Returns `true` if `handle` currently points to an active obstacle in this scene.
    #[inline]
    pub fn contains_handle(&self, handle: ObstacleHandle) -> bool {
        self.validate_handle(handle).is_ok()
    }

    /// Number of active named obstacles in this scene.
    #[inline]
    pub fn num_obstacles(&self) -> usize {
        self.by_name.len()
    }

    /// Iterates over all active `(name, ObstacleHandle, &Obstacle)` entries.
    pub fn obstacles(&self) -> impl Iterator<Item = (&str, ObstacleHandle, &Obstacle)> + '_ {
        self.entries.iter().filter_map(|slot| {
            let entry = slot.as_ref()?;
            let obs = self.env.obstacle(entry.handle.id)?;
            Some((entry.name.as_str(), entry.handle, obs))
        })
    }

    /// Compiles an `f32` batched rollout model from the current `MultiRobotModel` and `CollisionEnvironment`.
    #[inline]
    pub fn compile_flat_model(&self) -> FlatRobotModelF32 {
        FlatRobotModelF32::compile(&self.model, &self.env)
    }

    /// Allocates a thread-local [`crate::collision::CollisionWorkspace`] sized for the current scene.
    #[inline]
    pub fn create_workspace(&self) -> crate::collision::CollisionWorkspace {
        self.env.create_workspace(&self.model)
    }

    /// Allocates a thread-local [`TransformBuffer`] sized for the owned `MultiRobotModel`.
    #[inline]
    pub fn create_transform_buffer(&self) -> TransformBuffer {
        self.model.create_transform_buffer()
    }

    /// Pushes the current [`EsdfGridF32`] (if enabled) into a [`crate::batch::ResidentGpuBatchBackend`].
    #[cfg(feature = "gpu")]
    pub fn sync_to_resident_gpu(&self, backend: &crate::batch::ResidentGpuBatchBackend) {
        if let Some(esdf) = &self.esdf {
            backend.update_esdf(esdf.clone());
        }
    }

    /// Adds a named obstacle with default options (`carve_esdf = true`), syncing BVH and ESDF.
    pub fn add_obstacle(
        &mut self,
        name: impl Into<String>,
        shape: SharedShape,
        pose: Isometry3<f64>,
    ) -> Result<ObstacleHandle, SceneError> {
        self.add_obstacle_with_options(name, shape, pose, ObstacleOptions::default())
    }

    /// Adds a named obstacle with custom [`ObstacleOptions`], syncing BVH and ESDF.
    pub fn add_obstacle_with_options(
        &mut self,
        name: impl Into<String>,
        shape: SharedShape,
        pose: Isometry3<f64>,
        options: ObstacleOptions,
    ) -> Result<ObstacleHandle, SceneError> {
        let name = name.into();
        if self.by_name.contains_key(&name) {
            return Err(SceneError::DuplicateObstacleName(name));
        }

        let id = self.env.add_obstacle_deferred(shape, pose);
        if let Some(m) = options.safety_margin {
            self.env.set_obstacle_margin(id, m);
        }
        if let Some(act) = options.activation_distance {
            self.env.set_obstacle_activation(id, act);
        }

        let gen = self.alloc_generation();
        let handle = ObstacleHandle {
            id,
            generation: gen,
        };
        let idx = id.0 as usize;
        if self.entries.len() <= idx {
            self.entries.resize(idx + 1, None);
        }

        let carved_aabb = self.compute_carved_aabb(id);
        self.entries[idx] = Some(ObstacleEntry {
            name: name.clone(),
            handle,
            carve_esdf: options.carve_esdf,
            carved_aabb: if options.carve_esdf { carved_aabb } else { None },
        });
        self.by_name.insert(name, handle);
        self.bvh_dirty = true;
        self.scene_revision += 1;

        if options.carve_esdf && self.esdf.is_some() {
            Self::push_unique_id(&mut self.pending_carve_ids, id);
        }
        if !self.defer_sync {
            self.flush_sync();
        }
        Ok(handle)
    }

    /// Creates a named obstacle if absent, or atomically updates its shape/pose if already present.
    pub fn upsert_obstacle(
        &mut self,
        name: impl Into<String>,
        shape: SharedShape,
        pose: Isometry3<f64>,
    ) -> Result<ObstacleHandle, SceneError> {
        let name = name.into();
        if let Some(handle) = self.by_name.get(&name).copied() {
            self.update_obstacle(handle, shape, pose)?;
            Ok(handle)
        } else {
            self.add_obstacle(name, shape, pose)
        }
    }

    /// Updates the pose of an active obstacle by handle, syncing BVH and dirty ESDF regions.
    pub fn update_obstacle_pose(
        &mut self,
        handle: ObstacleHandle,
        pose: Isometry3<f64>,
    ) -> Result<(), SceneError> {
        let idx = self.validate_handle(handle)?.handle.id.0 as usize;
        let old_box = self.entries[idx].as_mut().unwrap().carved_aabb.take();
        let carve = self.entries[idx].as_ref().unwrap().carve_esdf;

        self.env.set_obstacle_pose_deferred(handle.id, pose);
        self.finish_obstacle_edit(handle.id, idx, carve, old_box);
        Ok(())
    }

    /// Updates the pose of an active obstacle by name, syncing BVH and dirty ESDF regions.
    pub fn update_obstacle_pose_by_name(
        &mut self,
        name: &str,
        pose: Isometry3<f64>,
    ) -> Result<(), SceneError> {
        let handle = self
            .by_name
            .get(name)
            .copied()
            .ok_or_else(|| SceneError::UnknownObstacleName(name.to_string()))?;
        self.update_obstacle_pose(handle, pose)
    }

    /// Replaces the geometry shape of an active obstacle by handle, syncing BVH and dirty ESDF regions.
    pub fn update_obstacle_shape(
        &mut self,
        handle: ObstacleHandle,
        shape: SharedShape,
    ) -> Result<(), SceneError> {
        let idx = self.validate_handle(handle)?.handle.id.0 as usize;
        let old_box = self.entries[idx].as_mut().unwrap().carved_aabb.take();
        let carve = self.entries[idx].as_ref().unwrap().carve_esdf;

        self.env.set_obstacle_shape_deferred(handle.id, shape);
        self.finish_obstacle_edit(handle.id, idx, carve, old_box);
        Ok(())
    }

    /// Atomically updates both shape and pose of an active obstacle, syncing BVH and ESDF.
    pub fn update_obstacle(
        &mut self,
        handle: ObstacleHandle,
        shape: SharedShape,
        pose: Isometry3<f64>,
    ) -> Result<(), SceneError> {
        let idx = self.validate_handle(handle)?.handle.id.0 as usize;
        let old_box = self.entries[idx].as_mut().unwrap().carved_aabb.take();
        let carve = self.entries[idx].as_ref().unwrap().carve_esdf;

        self.env.update_obstacle_deferred(handle.id, shape, pose);
        self.finish_obstacle_edit(handle.id, idx, carve, old_box);
        Ok(())
    }

    /// Enables or disables ESDF carving for an existing obstacle and synchronizes affected voxels.
    pub fn set_obstacle_carve_esdf(
        &mut self,
        handle: ObstacleHandle,
        carve_esdf: bool,
    ) -> Result<(), SceneError> {
        let idx = self.validate_handle(handle)?.handle.id.0 as usize;
        let prev = self.entries[idx].as_ref().unwrap().carve_esdf;
        if prev == carve_esdf {
            return Ok(());
        }
        let old_box = self.entries[idx].as_mut().unwrap().carved_aabb.take();
        self.entries[idx].as_mut().unwrap().carve_esdf = carve_esdf;
        self.scene_revision += 1;

        if self.esdf.is_some() {
            if carve_esdf {
                let new_box = self.compute_carved_aabb(handle.id);
                self.entries[idx].as_mut().unwrap().carved_aabb = new_box;
                Self::push_unique_id(&mut self.pending_carve_ids, handle.id);
            } else {
                if let Some(b) = old_box {
                    self.pending_dirty_regions.push(b);
                }
                self.pending_carve_ids.retain(|&id| id != handle.id);
            }
            if !self.defer_sync {
                self.flush_sync();
            }
        }
        Ok(())
    }

    /// Removes an active obstacle by handle, tombstoning it cleanly in `CollisionEnvironment`,
    /// restoring cleared ESDF voxels, and invalidating `handle`.
    pub fn remove_obstacle(&mut self, handle: ObstacleHandle) -> Result<(), SceneError> {
        let idx = self.validate_handle(handle)?.handle.id.0 as usize;
        let entry = self.entries[idx].take().unwrap();
        self.by_name.remove(&entry.name);
        self.env.remove_obstacle_deferred(handle.id);
        self.bvh_dirty = true;
        self.scene_revision += 1;

        self.pending_carve_ids.retain(|&id| id != handle.id);
        if self.esdf.is_some() {
            if let Some(old_box) = entry.carved_aabb {
                self.pending_dirty_regions.push(old_box);
            }
        }
        if !self.defer_sync {
            self.flush_sync();
        }
        Ok(())
    }

    /// Removes an active obstacle by name, synchronizing BVH and ESDF.
    pub fn remove_obstacle_by_name(&mut self, name: &str) -> Result<(), SceneError> {
        let handle = self
            .by_name
            .get(name)
            .copied()
            .ok_or_else(|| SceneError::UnknownObstacleName(name.to_string()))?;
        self.remove_obstacle(handle)
    }

    /// Defers BVH rebuilds and dirty-region ESDF updates across multiple edits, flushing once
    /// when `f` completes so overlapping obstacle edits are cleared before any re-carving.
    pub fn batch_update<F, R>(&mut self, f: F) -> R
    where
        F: FnOnce(&mut Self) -> R,
    {
        let was_deferred = self.defer_sync;
        self.defer_sync = true;
        let guard = BatchDeferGuard {
            scene: self,
            prev_deferred: was_deferred,
        };
        f(guard.scene)
    }

    /// Attaches a rigid payload to `env` and advances `scene_revision`.
    pub fn attach_payload(&mut self, payload: AttachedPayload) -> PayloadId {
        self.scene_revision += 1;
        self.env.attach_payload(payload)
    }

    /// Detaches an attached payload by `id` from `env` and advances `scene_revision`.
    pub fn detach_payload(&mut self, id: &str) -> Option<AttachedPayload> {
        let removed = self.env.detach_payload(id);
        if removed.is_some() {
            self.scene_revision += 1;
        }
        removed
    }

    /// Replaces the link-to-link [`AllowedCollisionMatrix`] on `env`.
    pub fn set_acm(&mut self, acm: AllowedCollisionMatrix) {
        self.env.set_acm(acm);
        self.scene_revision += 1;
    }

    /// Registers link collision geometry on `env`.
    pub fn set_link_geometry(&mut self, link: LinkId, geometry: LinkGeometry) {
        self.env.set_link_geometry(link, geometry);
        self.scene_revision += 1;
    }

    /// Sets a handle-validated default safety margin (meters) on `handle`.
    pub fn set_obstacle_margin(
        &mut self,
        handle: ObstacleHandle,
        margin: f64,
    ) -> Result<(), SceneError> {
        self.validate_handle(handle)?;
        self.env.set_obstacle_margin(handle.id, margin);
        self.scene_revision += 1;
        Ok(())
    }

    /// Sets a handle-validated default CBF activation distance (meters) on `handle`.
    pub fn set_obstacle_activation(
        &mut self,
        handle: ObstacleHandle,
        activation: f64,
    ) -> Result<(), SceneError> {
        self.validate_handle(handle)?;
        self.env.set_obstacle_activation(handle.id, activation);
        self.scene_revision += 1;
        Ok(())
    }

    /// Sets a custom safety margin between `link` and `handle`.
    pub fn set_link_obstacle_margin(
        &mut self,
        link: LinkId,
        handle: ObstacleHandle,
        margin: f64,
    ) -> Result<(), SceneError> {
        self.validate_handle(handle)?;
        self.env.set_link_obstacle_margin(link, handle.id, margin);
        self.scene_revision += 1;
        Ok(())
    }

    /// Sets a custom CBF activation distance between `link` and `handle`.
    pub fn set_link_obstacle_activation(
        &mut self,
        link: LinkId,
        handle: ObstacleHandle,
        activation: f64,
    ) -> Result<(), SceneError> {
        self.validate_handle(handle)?;
        self.env
            .set_link_obstacle_activation(link, handle.id, activation);
        self.scene_revision += 1;
        Ok(())
    }

    /// Enables or disables collision checks between `link` and `handle`.
    pub fn set_link_obstacle_enabled(
        &mut self,
        link: LinkId,
        handle: ObstacleHandle,
        enabled: bool,
    ) -> Result<(), SceneError> {
        self.validate_handle(handle)?;
        self.env.set_link_obstacle_enabled(link, handle.id, enabled);
        self.scene_revision += 1;
        Ok(())
    }

    /// Sets a custom safety margin between `payload_id` and `handle`.
    pub fn set_payload_obstacle_margin(
        &mut self,
        payload_id: &str,
        handle: ObstacleHandle,
        margin: f64,
    ) -> Result<(), SceneError> {
        self.validate_handle(handle)?;
        self.env
            .set_payload_obstacle_margin(payload_id, handle.id, margin);
        self.scene_revision += 1;
        Ok(())
    }

    /// Sets a custom CBF activation distance between `payload_id` and `handle`.
    pub fn set_payload_obstacle_activation(
        &mut self,
        payload_id: &str,
        handle: ObstacleHandle,
        activation: f64,
    ) -> Result<(), SceneError> {
        self.validate_handle(handle)?;
        self.env
            .set_payload_obstacle_activation(payload_id, handle.id, activation);
        self.scene_revision += 1;
        Ok(())
    }

    /// Enables or disables collision checks between `payload_id` and `handle`.
    pub fn set_payload_obstacle_enabled(
        &mut self,
        payload_id: &str,
        handle: ObstacleHandle,
        enabled: bool,
    ) -> Result<(), SceneError> {
        self.validate_handle(handle)?;
        self.env
            .set_payload_obstacle_enabled(payload_id, handle.id, enabled);
        self.scene_revision += 1;
        Ok(())
    }

    #[inline]
    fn alloc_generation(&mut self) -> u64 {
        let g = self.next_generation;
        self.next_generation += 1;
        g
    }

    fn validate_handle(&self, handle: ObstacleHandle) -> Result<&ObstacleEntry, SceneError> {
        let Some(Some(entry)) = self.entries.get(handle.id.0 as usize) else {
            return Err(SceneError::StaleOrInvalidHandle(handle));
        };
        if entry.handle != handle || !self.env.is_obstacle_active(handle.id) {
            return Err(SceneError::StaleOrInvalidHandle(handle));
        }
        Ok(entry)
    }

    fn push_unique_id(list: &mut Vec<ObstacleId>, id: ObstacleId) {
        if !list.contains(&id) {
            list.push(id);
        }
    }

    fn compute_carved_aabb(&self, id: ObstacleId) -> Option<([f32; 3], [f32; 3])> {
        let esdf = self.esdf.as_ref()?;
        let obs = self.env.obstacle(id)?;
        let pad = self.esdf_truncation + 1.5 * esdf.voxel_size;
        Some(inflated_obstacle_aabb(&obs.shape, &obs.pose, pad))
    }

    fn finish_obstacle_edit(
        &mut self,
        id: ObstacleId,
        idx: usize,
        carve: bool,
        old_box: Option<([f32; 3], [f32; 3])>,
    ) {
        self.bvh_dirty = true;
        self.scene_revision += 1;

        if self.esdf.is_some() {
            if let Some(old_b) = old_box {
                self.pending_dirty_regions.push(old_b);
            }
            if carve {
                let new_box = self.compute_carved_aabb(id);
                if let Some(Some(entry)) = self.entries.get_mut(idx) {
                    entry.carved_aabb = new_box;
                }
                Self::push_unique_id(&mut self.pending_carve_ids, id);
            }
        }
        if !self.defer_sync {
            self.flush_sync();
        }
    }

    fn flush_sync(&mut self) {
        if self.bvh_dirty {
            self.env.rebuild_bvh();
            self.bvh_dirty = false;
        }

        let Some(ref mut esdf) = self.esdf else {
            self.pending_dirty_regions.clear();
            self.pending_carve_ids.clear();
            return;
        };

        let dirty_regions = std::mem::take(&mut self.pending_dirty_regions);
        let carve_ids = std::mem::take(&mut self.pending_carve_ids);
        if dirty_regions.is_empty() && carve_ids.is_empty() {
            return;
        }

        // Pass 1: Clear ALL dirty voxel boxes before any re-carving so overlapping boxes
        // never erase shapes re-carved by neighboring dirty boxes.
        let mut voxel_boxes: Vec<([usize; 3], [usize; 3])> = Vec::with_capacity(dirty_regions.len());
        for (min_w, max_w) in &dirty_regions {
            if let Some((min_idx, max_idx)) = esdf.voxel_region_for_world_aabb(*min_w, *max_w) {
                esdf.clear_region(min_idx, max_idx, self.esdf_truncation);
                voxel_boxes.push((min_idx, max_idx));
            }
        }

        // Pass 2: Re-carve every active `carve_esdf` obstacle whose carved voxel box overlaps
        // each cleared dirty voxel box.
        if !voxel_boxes.is_empty() {
            type VoxelBox = ([usize; 3], [usize; 3]);
            let active_carved: Vec<(ObstacleId, VoxelBox)> = self
                .entries
                .iter()
                .filter_map(|slot| {
                    let entry = slot.as_ref()?;
                    if !entry.carve_esdf || !self.env.is_obstacle_active(entry.handle.id) {
                        return None;
                    }
                    let (min_w, max_w) = entry.carved_aabb?;
                    let vox = esdf.voxel_region_for_world_aabb(min_w, max_w)?;
                    Some((entry.handle.id, vox))
                })
                .collect();

            for (d_min, d_max) in &voxel_boxes {
                for &(oid, (e_min, e_max)) in &active_carved {
                    if voxel_boxes_overlap(*d_min, *d_max, e_min, e_max) {
                        if let Some(obs) = self.env.obstacle(oid) {
                            esdf.carve_shape_in_region(&obs.shape, &obs.pose, *d_min, *d_max);
                        }
                    }
                }
            }
        }

        // Pass 3: Carve newly added / moved / reshaped obstacles within their own inflated bounds.
        for oid in carve_ids {
            let Some(Some(entry)) = self.entries.get(oid.0 as usize) else {
                continue;
            };
            if !entry.carve_esdf {
                continue;
            }
            let Some((min_w, max_w)) = entry.carved_aabb else {
                continue;
            };
            let Some(obs) = self.env.obstacle(oid) else {
                continue;
            };
            if let Some((min_idx, max_idx)) = esdf.voxel_region_for_world_aabb(min_w, max_w) {
                esdf.carve_shape_in_region(&obs.shape, &obs.pose, min_idx, max_idx);
            }
        }

        self.esdf_revision += 1;
    }
}

struct BatchDeferGuard<'a> {
    scene: &'a mut WorldScene,
    prev_deferred: bool,
}

impl Drop for BatchDeferGuard<'_> {
    fn drop(&mut self) {
        self.scene.defer_sync = self.prev_deferred;
        if !self.prev_deferred {
            self.scene.flush_sync();
        }
    }
}

#[inline]
fn inflated_obstacle_aabb(
    shape: &SharedShape,
    pose: &Isometry3<f64>,
    pad: f32,
) -> ([f32; 3], [f32; 3]) {
    let aabb = shape.compute_aabb(&isometry_to_parry(pose));
    (
        [
            aabb.mins.x as f32 - pad,
            aabb.mins.y as f32 - pad,
            aabb.mins.z as f32 - pad,
        ],
        [
            aabb.maxs.x as f32 + pad,
            aabb.maxs.y as f32 + pad,
            aabb.maxs.z as f32 + pad,
        ],
    )
}

#[inline]
fn voxel_boxes_overlap(
    a_min: [usize; 3],
    a_max: [usize; 3],
    b_min: [usize; 3],
    b_max: [usize; 3],
) -> bool {
    a_min[0] <= b_max[0]
        && a_max[0] >= b_min[0]
        && a_min[1] <= b_max[1]
        && a_max[1] >= b_min[1]
        && a_min[2] <= b_max[2]
        && a_max[2] >= b_min[2]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::batch::{BatchMppiOptimizer, CpuSimdBatchBackend, MppiConfig};
    use crate::collision::{AllowedCollisionMatrix, GeometryTier, LinkGeometry};
    use crate::robot::{JointLimit, RobotId};
    use approx::assert_relative_eq;
    use nalgebra::{Point3, UnitVector3, Vector3};
    use parry3d::math::Vector as ParryVec;
    use parry3d::shape::{Ball, Cuboid};
    use std::f64::consts::PI;

    fn build_two_arm_test_model() -> MultiRobotModel {
        let mut model = MultiRobotModel::new();
        let limit = JointLimit::new(-PI, PI);
        let segs = [
            (
                UnitVector3::new_normalize(Vector3::z()),
                Vector3::new(0.0, 0.0, 0.2),
                limit,
                0.05,
            ),
            (
                UnitVector3::new_normalize(Vector3::y()),
                Vector3::new(0.0, 0.0, 0.38),
                limit,
                0.045,
            ),
            (
                UnitVector3::new_normalize(Vector3::y()),
                Vector3::new(0.0, 0.0, 0.35),
                limit,
                0.04,
            ),
        ];
        model
            .add_serial_arm(
                RobotId(0),
                "left",
                Isometry3::translation(0.0, 0.35, 0.0),
                &segs,
                Isometry3::translation(0.0, 0.0, 0.2),
            )
            .unwrap();
        model
            .add_serial_arm(
                RobotId(1),
                "right",
                Isometry3::translation(0.0, -0.35, 0.0),
                &segs,
                Isometry3::translation(0.0, 0.0, 0.2),
            )
            .unwrap();
        model
    }

    fn build_full_esdf_reference(scene: &WorldScene, cfg: EsdfConfig) -> EsdfGridF32 {
        let mut ref_grid = EsdfGridF32::new(cfg.origin, cfg.voxel_size, cfg.dims, cfg.truncation);
        for (_, handle, obs) in scene.obstacles() {
            let idx = handle.obstacle_id().0 as usize;
            if scene.entries[idx].as_ref().is_some_and(|e| e.carve_esdf) {
                ref_grid.carve_shape(&obs.shape, &obs.pose);
            }
        }
        ref_grid
    }

    fn assert_esdf_exact(scene: &WorldScene, cfg: EsdfConfig, label: &str) {
        let actual = scene.esdf().expect("ESDF should be enabled");
        let expected = build_full_esdf_reference(scene, cfg);
        assert_eq!(actual.distances.len(), expected.distances.len(), "{label}");
        for (i, (&a, &b)) in actual
            .distances
            .iter()
            .zip(expected.distances.iter())
            .enumerate()
        {
            assert!(
                (a - b).abs() <= 1e-6,
                "{label}: ESDF mismatch at voxel {i}: actual={a}, expected={b}"
            );
        }
    }

    #[test]
    fn world_scene_add_move_shape_remove_syncs_bvh_and_esdf_exact() {
        let model = build_two_arm_test_model();
        let mut env = CollisionEnvironment::new();
        env.set_acm(AllowedCollisionMatrix::from_model(&model));
        for l in model.links() {
            env.set_link_geometry(
                l.id,
                LinkGeometry::from_capsule(
                    Point3::origin(),
                    Point3::new(0.0, 0.0, 0.25),
                    l.bounding_radius,
                ),
            );
        }

        let esdf_cfg = EsdfConfig {
            origin: [-0.7, -0.7, 0.0],
            voxel_size: 0.05,
            dims: [28, 28, 22],
            truncation: 0.45,
        };
        let mut scene = WorldScene::with_esdf(model, env, esdf_cfg).unwrap();
        let mut ws = scene.create_workspace();
        let mut witnesses = Vec::new();
        let q = [0.1, 0.3, -0.2, -0.1, 0.3, -0.2];

        // 1. Add two overlapping obstacles
        let h_ball = scene
            .add_obstacle(
                "pedestal_ball",
                SharedShape::new(Ball::new(0.11)),
                Isometry3::translation(0.28, 0.10, 0.42),
            )
            .unwrap();
        let h_box = scene
            .add_obstacle(
                "fixture_box",
                SharedShape::new(Cuboid::new(ParryVec::new(0.09, 0.08, 0.07))),
                Isometry3::translation(0.32, -0.05, 0.44),
            )
            .unwrap();
        assert_eq!(scene.num_obstacles(), 2);
        assert_esdf_exact(&scene, esdf_cfg, "after initial add");

        scene
            .env()
            .query_signed_distances(
                scene.model(),
                &q,
                1.0,
                GeometryTier::Convex,
                &mut ws,
                &mut witnesses,
            )
            .unwrap();
        assert!(!witnesses.is_empty());

        // 2. Move ball out of overlap -> old dirty region must cleanly retain box distances
        scene
            .update_obstacle_pose(h_ball, Isometry3::translation(-0.25, 0.22, 0.55))
            .unwrap();
        assert_esdf_exact(&scene, esdf_cfg, "after moving ball away from overlapping box");

        // 3. Replace box shape (Cuboid -> larger Ball)
        scene
            .update_obstacle_shape(h_box, SharedShape::new(Ball::new(0.14)))
            .unwrap();
        assert_esdf_exact(&scene, esdf_cfg, "after reshaping fixture_box");

        // 4. Remove ball -> stale handle must error, BVH must no longer emit removed obstacle
        scene.remove_obstacle(h_ball).unwrap();
        assert!(!scene.contains_handle(h_ball));
        assert!(matches!(
            scene.update_obstacle_pose(h_ball, Isometry3::identity()),
            Err(SceneError::StaleOrInvalidHandle(_))
        ));
        assert_esdf_exact(&scene, esdf_cfg, "after removing ball");

        scene
            .env()
            .query_signed_distances(
                scene.model(),
                &q,
                2.0,
                GeometryTier::Convex,
                &mut ws,
                &mut witnesses,
            )
            .unwrap();
        assert!(!witnesses.iter().any(|w| matches!(
            w.pair,
            crate::collision::CollidingPair::LinkToObstacle { obstacle, .. }
                if obstacle == h_ball.obstacle_id()
        )));

        // 5. Re-using the name allocates a strictly newer generation handle
        let h_reused = scene
            .add_obstacle(
                "pedestal_ball",
                SharedShape::new(Ball::new(0.07)),
                Isometry3::translation(0.20, 0.0, 0.35),
            )
            .unwrap();
        assert!(h_reused.generation() > h_ball.generation());
        assert_ne!(h_reused, h_ball);
        assert_esdf_exact(&scene, esdf_cfg, "after reusing obstacle name");
    }

    #[test]
    fn world_scene_batch_update_handles_overlapping_edits_exact() {
        let model = build_two_arm_test_model();
        let esdf_cfg = EsdfConfig {
            origin: [-0.6, -0.6, 0.0],
            voxel_size: 0.05,
            dims: [24, 24, 20],
            truncation: 0.40,
        };
        let mut scene =
            WorldScene::with_esdf(model, CollisionEnvironment::new(), esdf_cfg).unwrap();

        let h_a = scene
            .add_obstacle(
                "obs_a",
                SharedShape::new(Ball::new(0.12)),
                Isometry3::translation(0.0, 0.0, 0.45),
            )
            .unwrap();
        let h_b = scene
            .add_obstacle(
                "obs_b",
                SharedShape::new(Cuboid::new(ParryVec::new(0.08, 0.10, 0.09))),
                Isometry3::translation(0.12, 0.04, 0.45),
            )
            .unwrap();
        let h_c = scene
            .add_obstacle(
                "obs_c",
                SharedShape::new(Ball::new(0.09)),
                Isometry3::translation(-0.18, -0.15, 0.35),
            )
            .unwrap();

        let rev_before = scene.esdf_revision();
        scene.batch_update(|sc| {
            sc.update_obstacle_pose(h_a, Isometry3::translation(0.20, 0.12, 0.50))
                .unwrap();
            sc.update_obstacle_pose(h_b, Isometry3::translation(-0.08, 0.05, 0.45))
                .unwrap();
            sc.remove_obstacle(h_c).unwrap();
            sc.add_obstacle(
                "obs_d",
                SharedShape::new(Ball::new(0.10)),
                Isometry3::translation(0.02, -0.02, 0.44),
            )
            .unwrap();
            sc.set_obstacle_carve_esdf(h_b, false).unwrap();
            sc.set_obstacle_carve_esdf(h_b, true).unwrap();
        });
        assert_eq!(scene.esdf_revision(), rev_before + 1);
        assert_esdf_exact(&scene, esdf_cfg, "after overlapping batch_update edits");
    }

    #[test]
    fn world_scene_rules_upsert_and_mppi_avoid_moved_obstacle() {
        let model = build_two_arm_test_model();
        let mut env = CollisionEnvironment::new();
        env.set_acm(AllowedCollisionMatrix::from_model(&model));
        for l in model.links() {
            env.set_link_geometry(
                l.id,
                LinkGeometry::from_capsule(
                    Point3::origin(),
                    Point3::new(0.0, 0.0, 0.25),
                    l.bounding_radius,
                ),
            );
        }

        let esdf_cfg = EsdfConfig {
            origin: [-0.8, -0.8, 0.0],
            voxel_size: 0.05,
            dims: [32, 32, 24],
            truncation: 0.60,
        };
        let mut scene = WorldScene::with_esdf(model, env, esdf_cfg).unwrap();

        let start = [0.0, -0.3, -0.2, 0.0, -0.3, -0.2];
        let goal = [0.0, 0.6, 0.3, 0.0, 0.6, 0.3];

        // Compute midpoint TCP pose of left arm and upsert an obstacle directly onto that sweep
        let mut buf = scene.create_transform_buffer();
        let mid_q = [0.0, 0.15, 0.05, 0.0, 0.15, 0.05];
        scene.model().forward_kinematics(&mid_q, &mut buf).unwrap();
        let mid_pos = buf
            .link_pose(scene.model().links()[2].id)
            .translation
            .vector;

        let h_block = scene
            .upsert_obstacle(
                "blocker",
                SharedShape::new(Ball::new(0.11)),
                Isometry3::translation(mid_pos.x, mid_pos.y + 0.5, mid_pos.z),
            )
            .unwrap();
        // Upsert updates existing handle when called again with the same name
        let h_block2 = scene
            .upsert_obstacle(
                "blocker",
                SharedShape::new(Ball::new(0.13)),
                Isometry3::translation(mid_pos.x, mid_pos.y, mid_pos.z),
            )
            .unwrap();
        assert_eq!(h_block, h_block2);

        scene.set_obstacle_margin(h_block, 0.025).unwrap();
        assert_relative_eq!(
            scene
                .env()
                .custom_pair_margin(&crate::collision::CollidingPair::LinkToObstacle {
                    link: LinkId(2),
                    obstacle: h_block.obstacle_id(),
                })
                .unwrap(),
            0.025,
            epsilon = 1e-12
        );

        let flat = scene.compile_flat_model();
        let mppi = BatchMppiOptimizer::new(MppiConfig {
            num_rollouts: 32,
            horizon: 12,
            num_iterations: 15,
            collision_weight: 250.0,
            ..MppiConfig::default()
        });
        let traj = mppi.optimize(&CpuSimdBatchBackend, scene.model(), &flat, scene.esdf(), &start, &goal);
        assert_eq!(traj.len(), 12);

        // Verify trajectory avoids the newly moved obstacle in exact Convex checks
        let mut ws = scene.create_workspace();
        let mut witnesses = Vec::new();
        let mut min_dist = f64::INFINITY;
        for wp in &traj {
            scene
                .env()
                .query_signed_distances(
                    scene.model(),
                    wp.as_slice(),
                    0.5,
                    GeometryTier::Convex,
                    &mut ws,
                    &mut witnesses,
                )
                .unwrap();
            for w in &witnesses {
                if matches!(
                    w.pair,
                    crate::collision::CollidingPair::LinkToObstacle { obstacle, .. }
                        if obstacle == h_block.obstacle_id()
                ) {
                    min_dist = min_dist.min(w.signed_distance);
                }
            }
        }
        assert!(
            min_dist > -0.02,
            "MPPI using WorldScene.esdf() should detour cleanly around moved blocker, got min_dist={min_dist:.4}"
        );
    }

    #[test]
    fn collision_environment_remove_obstacle_preserves_ids_and_cleans_rules() {
        let mut env = CollisionEnvironment::new();
        let o0 = env.add_obstacle(
            SharedShape::new(Ball::new(0.1)),
            Isometry3::translation(0.2, 0.0, 0.4),
        );
        let o1 = env.add_obstacle(
            SharedShape::new(Ball::new(0.1)),
            Isometry3::translation(-0.2, 0.0, 0.4),
        );
        env.set_obstacle_margin(o0, 0.03);
        env.set_link_obstacle_enabled(LinkId(0), o0, false);

        assert!(env.remove_obstacle(o0));
        assert!(!env.remove_obstacle(o0));
        assert!(!env.is_obstacle_active(o0));
        assert!(env.obstacle(o0).is_none());
        assert!(env.is_obstacle_active(o1));
        assert!(env.obstacle(o1).is_some());
        assert_eq!(env.num_active_obstacles(), 1);
        // After removing o0, its rule is purged so link0-o0 falls back to default enabled
        assert!(env.is_link_obstacle_enabled(LinkId(0), o0));
    }
}
