//! URDF and MoveIt SRDF parser with automatic multi-tier collision geometry
//! decomposition (`BoundingSphere`, `LinkCapsule`, and `parry3d` convex shapes)
//! for `unrust-it`.

use nalgebra::{Isometry3, Point3, Translation3, UnitQuaternion, UnitVector3, Vector3};
use parry3d::bounding_volume::{Aabb, BoundingVolume};
use parry3d::math::{Pose as ParryPose, Vector as ParryVector};
use parry3d::shape::{Ball, Capsule, Cuboid, Cylinder, SharedShape};
use std::collections::{HashMap, HashSet, VecDeque};
use std::f64::consts::{FRAC_PI_2, PI};
use thiserror::Error;
use unrust_it::{
    AllowedCollisionMatrix, BoundingSphere, CollisionEnvironment, CollisionRule, Configuration,
    JointId, JointLimit, JointType, KinodynamicLimits, LinkCapsule, LinkGeometry, LinkId,
    MultiRobotModel, PlanningGroup, RobotError, RobotId,
};

/// Errors that can occur when parsing URDF or SRDF XML descriptions.
#[derive(Debug, Error)]
pub enum UrdfError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("XML parse error: {0}")]
    Xml(#[from] roxmltree::Error),
    #[error("missing required XML element `<{0}>`")]
    MissingElement(&'static str),
    #[error("missing required attribute `{attr}` on `<{element}>`")]
    MissingAttribute {
        element: &'static str,
        attr: &'static str,
    },
    #[error("invalid float `{value}` in `{context}`")]
    InvalidFloat { value: String, context: &'static str },
    #[error("expected 3 space-separated floats in `{context}`, got `{value}`")]
    InvalidVector3 { value: String, context: &'static str },
    #[error("unsupported URDF joint type `{0}`")]
    UnsupportedJointType(String),
    #[error("URDF kinematic graph has no root link or contains a cycle")]
    InvalidKinematicTree,
    #[error("unknown link `{0}` referenced in URDF/SRDF")]
    UnknownLink(String),
    #[error("unknown joint `{0}` referenced in SRDF")]
    UnknownJoint(String),
    #[error("unknown planning group `{0}` referenced in SRDF")]
    UnknownGroup(String),
    #[error("robot model error: {0}")]
    Robot(#[from] RobotError),
}

/// Summary handle returned after loading a URDF robot into a [`MultiRobotModel`].
#[derive(Debug, Clone)]
pub struct UrdfRobotHandle {
    /// Assigned robot identifier.
    pub robot_id: RobotId,
    /// Robot name from `<robot name="...">`.
    pub name: String,
    /// Prefix applied to all link, joint, and frame names.
    pub prefix: String,
    /// Root link ID of this robot instance.
    pub root_link: LinkId,
    /// All link IDs belonging to this robot instance in topological order.
    pub links: Vec<LinkId>,
    /// All actuated joint IDs belonging to this robot instance.
    pub actuated_joints: Vec<JointId>,
    /// Indices into the global configuration vector `q`.
    pub q_indices: Vec<usize>,
    /// Automatically registered whole-robot planning group.
    pub planning_group: PlanningGroup,
}

/// Named joint-space preset parsed from an SRDF `<group_state>` tag.
#[derive(Debug, Clone)]
pub struct SrdfGroupState {
    /// Name of the preset (e.g., `"home"`, `"ready"`).
    pub name: String,
    /// Target planning group name.
    pub group: String,
    /// Joint values mapped into a full-model [`Configuration`].
    pub configuration: Configuration,
}

/// Result of applying a MoveIt SRDF configuration to a [`MultiRobotModel`] and [`CollisionEnvironment`].
#[derive(Debug, Clone)]
pub struct SrdfData {
    /// Planning groups defined in the SRDF.
    pub groups: Vec<PlanningGroup>,
    /// Named group state presets defined in the SRDF.
    pub group_states: Vec<SrdfGroupState>,
    /// Disabled collision link pairs `(link_a, link_b, reason)`.
    pub disabled_collisions: Vec<(LinkId, LinkId, String)>,
}

#[derive(Debug, Clone)]
struct ParsedCollisionPrimitive {
    origin: Isometry3<f64>,
    spheres: Vec<BoundingSphere>,
    capsule: Option<LinkCapsule>,
    convex_pose: Isometry3<f64>,
    convex_shape: SharedShape,
}

#[derive(Debug, Clone)]
struct ParsedUrdfJoint {
    name: String,
    joint_type: JointType,
    parent_link: String,
    child_link: String,
    origin: Isometry3<f64>,
    limits: Option<KinodynamicLimits>,
}

/// Parses a single-robot URDF XML string into a fresh [`MultiRobotModel`] and [`CollisionEnvironment`].
pub fn load_urdf_str(
    urdf_xml: &str,
) -> Result<(MultiRobotModel, CollisionEnvironment, UrdfRobotHandle), UrdfError> {
    let mut model = MultiRobotModel::new();
    let mut env = CollisionEnvironment::new();
    let handle = add_urdf_to_scene(
        &mut model,
        &mut env,
        urdf_xml,
        RobotId(0),
        "",
        Isometry3::identity(),
    )?;
    Ok((model, env, handle))
}

/// Loads a single-robot URDF file from `path` into a fresh [`MultiRobotModel`] and [`CollisionEnvironment`].
pub fn load_urdf_file(
    path: impl AsRef<std::path::Path>,
) -> Result<(MultiRobotModel, CollisionEnvironment, UrdfRobotHandle), UrdfError> {
    let urdf_xml = std::fs::read_to_string(path)?;
    load_urdf_str(&urdf_xml)
}

/// Reads a URDF XML file from `path` and mounts the robot into an existing [`MultiRobotModel`] and
/// [`CollisionEnvironment`] with a name `prefix` and world `base_pose`.
pub fn add_urdf_file_to_scene(
    model: &mut MultiRobotModel,
    env: &mut CollisionEnvironment,
    path: impl AsRef<std::path::Path>,
    robot_id: RobotId,
    prefix: &str,
    base_pose: Isometry3<f64>,
) -> Result<UrdfRobotHandle, UrdfError> {
    let urdf_xml = std::fs::read_to_string(path)?;
    add_urdf_to_scene(model, env, &urdf_xml, robot_id, prefix, base_pose)
}

/// Reads a MoveIt SRDF XML file from `path` and applies it to `model` and `env`.
pub fn apply_srdf_file(
    model: &mut MultiRobotModel,
    env: &mut CollisionEnvironment,
    path: impl AsRef<std::path::Path>,
    prefix: &str,
) -> Result<SrdfData, UrdfError> {
    let srdf_xml = std::fs::read_to_string(path)?;
    apply_srdf_str(model, env, &srdf_xml, prefix)
}

/// Parses a URDF XML string and mounts the robot into an existing [`MultiRobotModel`] and
/// [`CollisionEnvironment`] with a name `prefix` and world `base_pose`.
///
/// This enables assembling multi-arm cells (e.g., `"left_"`, `"right_"`) from a single URDF file.
pub fn add_urdf_to_scene(
    model: &mut MultiRobotModel,
    env: &mut CollisionEnvironment,
    urdf_xml: &str,
    robot_id: RobotId,
    prefix: &str,
    base_pose: Isometry3<f64>,
) -> Result<UrdfRobotHandle, UrdfError> {
    let doc = roxmltree::Document::parse(urdf_xml)?;
    let robot_node = doc
        .descendants()
        .find(|n| n.has_tag_name("robot"))
        .ok_or(UrdfError::MissingElement("robot"))?;

    let raw_robot_name = robot_node.attribute("name").unwrap_or("robot");
    let group_name = if prefix.is_empty() {
        raw_robot_name.to_string()
    } else {
        prefix.trim_end_matches('_').to_string()
    };

    // 1. Parse all <link> definitions and their collision geometries
    let mut link_geoms: HashMap<String, Option<LinkGeometry>> = HashMap::new();
    let mut link_order: Vec<String> = Vec::new();

    for node in robot_node.children().filter(|n| n.has_tag_name("link")) {
        let name = node
            .attribute("name")
            .ok_or(UrdfError::MissingAttribute {
                element: "link",
                attr: "name",
            })?
            .to_string();

        let mut primitives = Vec::new();
        for coll_node in node.children().filter(|n| n.has_tag_name("collision")) {
            if let Some(prim) = parse_collision_element(coll_node)? {
                primitives.push(prim);
            }
        }
        // Fallback to <visual> geometry if no <collision> tag is present
        if primitives.is_empty() {
            for vis_node in node.children().filter(|n| n.has_tag_name("visual")) {
                if let Some(prim) = parse_collision_element(vis_node)? {
                    primitives.push(prim);
                }
            }
        }

        let geom = build_link_geometry(&primitives);
        link_order.push(name.clone());
        link_geoms.insert(name, geom);
    }

    if link_geoms.is_empty() {
        return Err(UrdfError::MissingElement("link"));
    }

    // 2. Parse all <joint> definitions
    let mut joints_by_parent: HashMap<String, Vec<ParsedUrdfJoint>> = HashMap::new();
    let mut child_links: HashSet<String> = HashSet::new();

    for node in robot_node.children().filter(|n| n.has_tag_name("joint")) {
        let parsed = parse_joint_element(node)?;
        if !link_geoms.contains_key(&parsed.parent_link) {
            return Err(UrdfError::UnknownLink(parsed.parent_link));
        }
        if !link_geoms.contains_key(&parsed.child_link) {
            return Err(UrdfError::UnknownLink(parsed.child_link));
        }
        child_links.insert(parsed.child_link.clone());
        joints_by_parent
            .entry(parsed.parent_link.clone())
            .or_default()
            .push(parsed);
    }

    // 3. Identify root link(s) (links that are never a child of any joint)
    let roots: Vec<String> = link_order
        .iter()
        .filter(|name| !child_links.contains(*name))
        .cloned()
        .collect();

    if roots.is_empty() {
        return Err(UrdfError::InvalidKinematicTree);
    }

    // Preserve any existing custom ACM pairs before adding new links
    let prev_num_links = model.num_links();
    let prev_acm = env.acm().cloned();

    let mut link_id_map: HashMap<String, LinkId> = HashMap::new();
    let mut created_links = Vec::new();
    let mut actuated_joints = Vec::new();
    let mut q_indices = Vec::new();
    let mut leaf_frames = Vec::new();

    // 4. Traverse kinematic tree in topological order (BFS from roots)
    let mut queue: VecDeque<String> = VecDeque::new();
    for root_name in &roots {
        let prefixed_link = format!("{prefix}{root_name}");
        let prefixed_joint = format!("{prefix}{root_name}_base_joint");
        let geom_opt = link_geoms.get(root_name).and_then(|g| g.clone());
        let bounding_radius = geom_opt.as_ref().map_or(0.05, compute_bounding_radius);

        let (link_id, _joint_id, frame_id) = model.add_link_and_joint(
            robot_id,
            prefixed_link,
            prefixed_joint,
            None,
            base_pose,
            JointType::Fixed,
            None,
            bounding_radius,
        )?;

        if let Some(geom) = geom_opt {
            env.set_link_geometry(link_id, geom);
        }

        link_id_map.insert(root_name.clone(), link_id);
        created_links.push(link_id);
        if !joints_by_parent.contains_key(root_name) {
            leaf_frames.push(frame_id);
        }
        queue.push_back(root_name.clone());
    }

    while let Some(parent_raw) = queue.pop_front() {
        let parent_link_id = link_id_map[&parent_raw];
        if let Some(children) = joints_by_parent.get(&parent_raw) {
            for joint in children {
                let prefixed_link = format!("{prefix}{}", joint.child_link);
                let prefixed_joint = format!("{prefix}{}", joint.name);
                let geom_opt = link_geoms.get(&joint.child_link).and_then(|g| g.clone());
                let bounding_radius = geom_opt.as_ref().map_or(0.05, compute_bounding_radius);

                let (child_link_id, joint_id, frame_id) = model.add_link_and_joint(
                    robot_id,
                    prefixed_link,
                    prefixed_joint,
                    Some(parent_link_id),
                    joint.origin,
                    joint.joint_type,
                    joint.limits,
                    bounding_radius,
                )?;

                if let Some(q_idx) = model.joint(joint_id).q_index {
                    actuated_joints.push(joint_id);
                    q_indices.push(q_idx);
                }

                if let Some(geom) = geom_opt {
                    env.set_link_geometry(child_link_id, geom);
                }

                link_id_map.insert(joint.child_link.clone(), child_link_id);
                created_links.push(child_link_id);
                if !joints_by_parent.contains_key(&joint.child_link) {
                    leaf_frames.push(frame_id);
                }
                queue.push_back(joint.child_link.clone());
            }
        }
    }

    if created_links.len() != link_geoms.len() {
        return Err(UrdfError::InvalidKinematicTree);
    }

    // 5. Rebuild AllowedCollisionMatrix for the expanded model while preserving prior rules
    let mut new_acm = AllowedCollisionMatrix::from_model(model);
    if let Some(prev) = &prev_acm {
        for i in 0..prev_num_links {
            for j in (i + 1)..prev_num_links {
                let li = LinkId(i as u16);
                let lj = LinkId(j as u16);
                let prev_rule = prev.rule(li, lj);
                if prev_rule != CollisionRule::Enabled {
                    new_acm.set_rule(li, lj, prev_rule);
                }
            }
        }
    }
    // Also disable collisions between links joined exclusively by fixed joints
    for j_spec in model.joints() {
        if matches!(j_spec.joint_type, JointType::Fixed) {
            if let Some(p_link) = j_spec.parent_link {
                new_acm.set_rule(p_link, j_spec.child_link, CollisionRule::DisabledAdjacent);
            }
        }
    }
    env.set_acm(new_acm);

    let planning_group = PlanningGroup {
        name: group_name,
        joints: actuated_joints.clone(),
        q_indices: q_indices.clone(),
        active_links: created_links.clone(),
        end_effectors: leaf_frames,
    };
    model.add_planning_group(planning_group.clone());

    Ok(UrdfRobotHandle {
        robot_id,
        name: raw_robot_name.to_string(),
        prefix: prefix.to_string(),
        root_link: created_links[0],
        links: created_links,
        actuated_joints,
        q_indices,
        planning_group,
    })
}

/// Parses a MoveIt SRDF XML string and applies `<disable_collisions>`, `<group>`, and `<group_state>`
/// definitions to `model` and `env`.
pub fn apply_srdf_str(
    model: &mut MultiRobotModel,
    env: &mut CollisionEnvironment,
    srdf_xml: &str,
    prefix: &str,
) -> Result<SrdfData, UrdfError> {
    let doc = roxmltree::Document::parse(srdf_xml)?;
    let robot_node = doc
        .descendants()
        .find(|n| n.has_tag_name("robot"))
        .ok_or(UrdfError::MissingElement("robot"))?;

    let mut acm = match env.acm() {
        Some(existing) if existing.num_links() == model.num_links() => existing.clone(),
        _ => AllowedCollisionMatrix::from_model(model),
    };

    // 1. Parse <disable_collisions link1="..." link2="..." reason="..."/>
    let mut disabled_collisions = Vec::new();
    for node in robot_node
        .children()
        .filter(|n| n.has_tag_name("disable_collisions"))
    {
        let raw_l1 = node.attribute("link1").ok_or(UrdfError::MissingAttribute {
            element: "disable_collisions",
            attr: "link1",
        })?;
        let raw_l2 = node.attribute("link2").ok_or(UrdfError::MissingAttribute {
            element: "disable_collisions",
            attr: "link2",
        })?;
        let reason = node.attribute("reason").unwrap_or("srdf").to_string();
        let rule = match reason.as_str() {
            "Adjacent" | "adjacent" => CollisionRule::DisabledAdjacent,
            "Never" | "never" => CollisionRule::DisabledNeverColliding,
            _ => CollisionRule::DisabledDefault,
        };

        let l1 = resolve_link(model, prefix, raw_l1)?;
        let l2 = resolve_link(model, prefix, raw_l2)?;
        acm.set_rule(l1, l2, rule);
        disabled_collisions.push((l1, l2, reason));
    }
    env.set_acm(acm);

    // 2. Parse <group name="...">
    let mut parsed_groups: Vec<PlanningGroup> = Vec::new();
    for node in robot_node.children().filter(|n| n.has_tag_name("group")) {
        let raw_name = node.attribute("name").ok_or(UrdfError::MissingAttribute {
            element: "group",
            attr: "name",
        })?;
        let group_name = format!("{prefix}{raw_name}");

        let mut joints = Vec::new();
        let mut q_indices = Vec::new();
        let mut active_links = Vec::new();
        let mut end_effectors = Vec::new();

        // 2a. <chain base_link="..." tip_link="..."/>
        for chain_node in node.children().filter(|n| n.has_tag_name("chain")) {
            let base_raw = chain_node
                .attribute("base_link")
                .ok_or(UrdfError::MissingAttribute {
                    element: "chain",
                    attr: "base_link",
                })?;
            let tip_raw = chain_node
                .attribute("tip_link")
                .ok_or(UrdfError::MissingAttribute {
                    element: "chain",
                    attr: "tip_link",
                })?;

            let base_id = resolve_link(model, prefix, base_raw)?;
            let tip_id = resolve_link(model, prefix, tip_raw)?;

            let mut chain_links = Vec::new();
            let mut curr = Some(tip_id);
            while let Some(lid) = curr {
                chain_links.push(lid);
                if lid == base_id {
                    break;
                }
                curr = model.link(lid).parent_link;
            }
            chain_links.reverse();

            for &lid in &chain_links {
                if !active_links.contains(&lid) {
                    active_links.push(lid);
                }
                if lid != base_id {
                    let jid = model.link(lid).parent_joint;
                    if let Some(q_idx) = model.joint(jid).q_index {
                        if !joints.contains(&jid) {
                            joints.push(jid);
                            q_indices.push(q_idx);
                        }
                    }
                }
            }

            let tip_frame = model.link(tip_id).frame;
            if !end_effectors.contains(&tip_frame) {
                end_effectors.push(tip_frame);
            }
        }

        // 2b. <joint name="..."/>
        for j_node in node.children().filter(|n| n.has_tag_name("joint")) {
            let j_raw = j_node
                .attribute("name")
                .ok_or(UrdfError::MissingAttribute {
                    element: "joint",
                    attr: "name",
                })?;
            let jid = resolve_joint(model, prefix, j_raw)?;
            let j_spec = model.joint(jid);
            if let Some(q_idx) = j_spec.q_index {
                if !joints.contains(&jid) {
                    joints.push(jid);
                    q_indices.push(q_idx);
                }
            }
            if !active_links.contains(&j_spec.child_link) {
                active_links.push(j_spec.child_link);
            }
        }

        // 2c. <link name="..."/>
        for l_node in node.children().filter(|n| n.has_tag_name("link")) {
            let l_raw = l_node
                .attribute("name")
                .ok_or(UrdfError::MissingAttribute {
                    element: "link",
                    attr: "name",
                })?;
            let lid = resolve_link(model, prefix, l_raw)?;
            if !active_links.contains(&lid) {
                active_links.push(lid);
            }
            let jid = model.link(lid).parent_joint;
            if let Some(q_idx) = model.joint(jid).q_index {
                if !joints.contains(&jid) {
                    joints.push(jid);
                    q_indices.push(q_idx);
                }
            }
        }

        // 2d. <group name="..."/> (subgroup inclusion)
        for sub_node in node.children().filter(|n| n.has_tag_name("group")) {
            let sub_raw = sub_node
                .attribute("name")
                .ok_or(UrdfError::MissingAttribute {
                    element: "group",
                    attr: "name",
                })?;
            let prefixed_sub = format!("{prefix}{sub_raw}");
            let sub_group = model
                .group_by_name(&prefixed_sub)
                .or_else(|| model.group_by_name(sub_raw))
                .cloned()
                .ok_or_else(|| UrdfError::UnknownGroup(sub_raw.to_string()))?;

            for (&j, &q_idx) in sub_group.joints.iter().zip(&sub_group.q_indices) {
                if !joints.contains(&j) {
                    joints.push(j);
                    q_indices.push(q_idx);
                }
            }
            for &l in &sub_group.active_links {
                if !active_links.contains(&l) {
                    active_links.push(l);
                }
            }
            for &ee in &sub_group.end_effectors {
                if !end_effectors.contains(&ee) {
                    end_effectors.push(ee);
                }
            }
        }

        if end_effectors.is_empty() {
            if let Some(&last_link) = active_links.last() {
                end_effectors.push(model.link(last_link).frame);
            }
        }

        let group = PlanningGroup {
            name: group_name,
            joints,
            q_indices,
            active_links,
            end_effectors,
        };
        model.add_planning_group(group.clone());
        parsed_groups.push(group);
    }

    // 3. Parse <group_state name="..." group="...">
    let mut group_states = Vec::new();
    for node in robot_node
        .children()
        .filter(|n| n.has_tag_name("group_state"))
    {
        let state_name = node
            .attribute("name")
            .ok_or(UrdfError::MissingAttribute {
                element: "group_state",
                attr: "name",
            })?
            .to_string();
        let raw_group = node.attribute("group").ok_or(UrdfError::MissingAttribute {
            element: "group_state",
            attr: "group",
        })?;
        let prefixed_group = format!("{prefix}{raw_group}");
        let group_name = if model.group_by_name(&prefixed_group).is_some() {
            prefixed_group
        } else {
            raw_group.to_string()
        };

        let mut q = vec![0.0_f64; model.dof()];
        for j_node in node.children().filter(|n| n.has_tag_name("joint")) {
            let j_raw = j_node
                .attribute("name")
                .ok_or(UrdfError::MissingAttribute {
                    element: "joint",
                    attr: "name",
                })?;
            let val_str = j_node
                .attribute("value")
                .ok_or(UrdfError::MissingAttribute {
                    element: "joint",
                    attr: "value",
                })?;
            let val = parse_f64(val_str, "group_state joint value")?;
            let jid = resolve_joint(model, prefix, j_raw)?;
            if let Some(q_idx) = model.joint(jid).q_index {
                q[q_idx] = val;
            }
        }

        group_states.push(SrdfGroupState {
            name: state_name,
            group: group_name,
            configuration: Configuration::new(q),
        });
    }

    Ok(SrdfData {
        groups: parsed_groups,
        group_states,
        disabled_collisions,
    })
}

/// Built-in canonical URDF and MoveIt SRDF presets shared across tests, benchmarks, and visualizers.
pub mod presets {
    /// 4-DOF articulated workcell arm URDF (`base_link` through `wrist_link`, cylinder + capsules).
    pub const ARM_4DOF_URDF: &str = r#"
<robot name="unrust_arm_4dof">
  <link name="base_link">
    <collision>
      <origin xyz="0 0 0.05" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.065" length="0.10"/>
      </geometry>
    </collision>
  </link>
  <link name="shoulder_link">
    <collision>
      <origin xyz="0 0 0.09" rpy="0 0 0"/>
      <geometry>
        <capsule radius="0.055" length="0.18"/>
      </geometry>
    </collision>
  </link>
  <link name="upper_arm_link">
    <collision>
      <origin xyz="0 0 0.16" rpy="0 0 0"/>
      <geometry>
        <capsule radius="0.048" length="0.32"/>
      </geometry>
    </collision>
  </link>
  <link name="forearm_link">
    <collision>
      <origin xyz="0 0 0.14" rpy="0 0 0"/>
      <geometry>
        <capsule radius="0.042" length="0.28"/>
      </geometry>
    </collision>
  </link>
  <link name="wrist_link">
    <collision>
      <origin xyz="0 0 0.08" rpy="0 0 0"/>
      <geometry>
        <capsule radius="0.035" length="0.16"/>
      </geometry>
    </collision>
  </link>

  <joint name="j1_waist" type="revolute">
    <parent link="base_link"/>
    <child link="shoulder_link"/>
    <origin xyz="0 0 0.10" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-2.8" upper="2.8" effort="90" velocity="2.2"/>
  </joint>
  <joint name="j2_shoulder" type="revolute">
    <parent link="shoulder_link"/>
    <child link="upper_arm_link"/>
    <origin xyz="0 0 0.18" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.9" upper="1.9" effort="90" velocity="2.2"/>
  </joint>
  <joint name="j3_elbow" type="revolute">
    <parent link="upper_arm_link"/>
    <child link="forearm_link"/>
    <origin xyz="0 0 0.32" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-2.4" upper="2.4" effort="70" velocity="2.5"/>
  </joint>
  <joint name="j4_wrist" type="revolute">
    <parent link="forearm_link"/>
    <child link="wrist_link"/>
    <origin xyz="0 0 0.28" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-2.4" upper="2.4" effort="50" velocity="3.0"/>
  </joint>
</robot>
"#;

    /// MoveIt SRDF configuration corresponding to [`ARM_4DOF_URDF`].
    pub const ARM_4DOF_SRDF: &str = r#"
<robot name="unrust_arm_4dof">
  <group name="arm">
    <chain base_link="base_link" tip_link="wrist_link"/>
  </group>
  <disable_collisions link1="base_link" link2="upper_arm_link" reason="Never"/>
  <disable_collisions link1="shoulder_link" link2="forearm_link" reason="Never"/>
</robot>
"#;

    /// 6-DOF precision industrial manipulator URDF (`base_link` through `flange_link`).
    pub const ARM_6DOF_URDF: &str = r#"
<robot name="unrust_arm_6dof">
  <link name="base_link">
    <collision>
      <origin xyz="0 0 0.05" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.060" length="0.10"/>
      </geometry>
    </collision>
  </link>
  <link name="shoulder_link">
    <collision>
      <origin xyz="0 0 0.08" rpy="0 0 0"/>
      <geometry>
        <capsule radius="0.048" length="0.16"/>
      </geometry>
    </collision>
  </link>
  <link name="upper_arm_link">
    <collision>
      <origin xyz="0 0 0.17" rpy="0 0 0"/>
      <geometry>
        <capsule radius="0.040" length="0.34"/>
      </geometry>
    </collision>
  </link>
  <link name="forearm_link">
    <collision>
      <origin xyz="0 0 0.15" rpy="0 0 0"/>
      <geometry>
        <capsule radius="0.034" length="0.30"/>
      </geometry>
    </collision>
  </link>
  <link name="wrist1_link">
    <collision>
      <origin xyz="0 0 0.035" rpy="0 0 0"/>
      <geometry>
        <capsule radius="0.026" length="0.07"/>
      </geometry>
    </collision>
  </link>
  <link name="wrist2_link">
    <collision>
      <origin xyz="0 0 0.025" rpy="0 0 0"/>
      <geometry>
        <capsule radius="0.022" length="0.05"/>
      </geometry>
    </collision>
  </link>
  <link name="flange_link">
    <collision>
      <origin xyz="0 0 0.015" rpy="0 0 0"/>
      <geometry>
        <cylinder radius="0.024" length="0.03"/>
      </geometry>
    </collision>
  </link>

  <joint name="j1_waist" type="revolute">
    <parent link="base_link"/>
    <child link="shoulder_link"/>
    <origin xyz="0 0 0.10" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.0" upper="3.0" effort="120" velocity="3.0"/>
  </joint>
  <joint name="j2_shoulder" type="revolute">
    <parent link="shoulder_link"/>
    <child link="upper_arm_link"/>
    <origin xyz="0 0 0.16" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-2.1" upper="2.1" effort="120" velocity="3.0"/>
  </joint>
  <joint name="j3_elbow" type="revolute">
    <parent link="upper_arm_link"/>
    <child link="forearm_link"/>
    <origin xyz="0 0 0.34" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-2.6" upper="2.6" effort="90" velocity="3.5"/>
  </joint>
  <joint name="j4_wrist_roll" type="revolute">
    <parent link="forearm_link"/>
    <child link="wrist1_link"/>
    <origin xyz="0 0 0.30" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.0" upper="3.0" effort="50" velocity="4.0"/>
  </joint>
  <joint name="j5_wrist_pitch" type="revolute">
    <parent link="wrist1_link"/>
    <child link="wrist2_link"/>
    <origin xyz="0 0 0.07" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-2.4" upper="2.4" effort="40" velocity="4.0"/>
  </joint>
  <joint name="j6_flange_roll" type="revolute">
    <parent link="wrist2_link"/>
    <child link="flange_link"/>
    <origin xyz="0 0 0.05" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" effort="30" velocity="5.0"/>
  </joint>
</robot>
"#;

    /// MoveIt SRDF configuration corresponding to [`ARM_6DOF_URDF`].
    pub const ARM_6DOF_SRDF: &str = r#"
<robot name="unrust_arm_6dof">
  <group name="arm">
    <chain base_link="base_link" tip_link="flange_link"/>
  </group>
  <disable_collisions link1="base_link" link2="upper_arm_link" reason="Never"/>
  <disable_collisions link1="shoulder_link" link2="forearm_link" reason="Never"/>
  <disable_collisions link1="upper_arm_link" link2="wrist1_link" reason="Never"/>
  <disable_collisions link1="forearm_link" link2="wrist2_link" reason="Never"/>
  <disable_collisions link1="wrist1_link" link2="flange_link" reason="Never"/>
</robot>
"#;
}

fn resolve_link(
    model: &MultiRobotModel,
    prefix: &str,
    raw_name: &str,
) -> Result<LinkId, UrdfError> {
    let prefixed = format!("{prefix}{raw_name}");
    model
        .link_by_name(&prefixed)
        .or_else(|| model.link_by_name(raw_name))
        .ok_or_else(|| UrdfError::UnknownLink(raw_name.to_string()))
}

fn resolve_joint(
    model: &MultiRobotModel,
    prefix: &str,
    raw_name: &str,
) -> Result<JointId, UrdfError> {
    let prefixed = format!("{prefix}{raw_name}");
    model
        .joints()
        .iter()
        .find(|j| j.name == prefixed || j.name == raw_name)
        .map(|j| j.id)
        .ok_or_else(|| UrdfError::UnknownJoint(raw_name.to_string()))
}

fn parse_joint_element(node: roxmltree::Node<'_, '_>) -> Result<ParsedUrdfJoint, UrdfError> {
    let name = node
        .attribute("name")
        .ok_or(UrdfError::MissingAttribute {
            element: "joint",
            attr: "name",
        })?
        .to_string();
    let type_str = node.attribute("type").ok_or(UrdfError::MissingAttribute {
        element: "joint",
        attr: "type",
    })?;

    let parent_node = node
        .children()
        .find(|n| n.has_tag_name("parent"))
        .ok_or(UrdfError::MissingElement("parent"))?;
    let parent_link = parent_node
        .attribute("link")
        .ok_or(UrdfError::MissingAttribute {
            element: "parent",
            attr: "link",
        })?
        .to_string();

    let child_node = node
        .children()
        .find(|n| n.has_tag_name("child"))
        .ok_or(UrdfError::MissingElement("child"))?;
    let child_link = child_node
        .attribute("link")
        .ok_or(UrdfError::MissingAttribute {
            element: "child",
            attr: "link",
        })?
        .to_string();

    let origin = match node.children().find(|n| n.has_tag_name("origin")) {
        Some(orig_node) => parse_origin(orig_node)?,
        None => Isometry3::identity(),
    };

    let axis_vec = match node.children().find(|n| n.has_tag_name("axis")) {
        Some(ax_node) => {
            let xyz_str = ax_node.attribute("xyz").unwrap_or("1 0 0");
            parse_vec3(xyz_str, "joint axis xyz")?
        }
        None => Vector3::x(),
    };
    let axis = UnitVector3::try_new(axis_vec, 1e-9).unwrap_or_else(Vector3::x_axis);

    let limit_node = node.children().find(|n| n.has_tag_name("limit"));
    let (joint_type, limits) = match type_str {
        "fixed" => (JointType::Fixed, None),
        "continuous" => {
            let max_vel = limit_node
                .and_then(|l| l.attribute("velocity"))
                .map(|s| parse_f64(s, "limit velocity"))
                .transpose()?
                .unwrap_or(3.0);
            let max_eff = limit_node
                .and_then(|l| l.attribute("effort"))
                .map(|s| parse_f64(s, "limit effort"))
                .transpose()?
                .unwrap_or(100.0);
            let mut lim = KinodynamicLimits::from_position(JointLimit::unbounded());
            lim.max_velocity = max_vel.max(1e-3);
            lim.max_effort = max_eff.max(0.0);
            (JointType::Continuous { axis }, Some(lim))
        }
        "revolute" | "prismatic" => {
            let lower = limit_node
                .and_then(|l| l.attribute("lower"))
                .map(|s| parse_f64(s, "limit lower"))
                .transpose()?
                .unwrap_or(-PI);
            let upper = limit_node
                .and_then(|l| l.attribute("upper"))
                .map(|s| parse_f64(s, "limit upper"))
                .transpose()?
                .unwrap_or(PI);
            let max_vel = limit_node
                .and_then(|l| l.attribute("velocity"))
                .map(|s| parse_f64(s, "limit velocity"))
                .transpose()?
                .unwrap_or(2.5);
            let max_eff = limit_node
                .and_then(|l| l.attribute("effort"))
                .map(|s| parse_f64(s, "limit effort"))
                .transpose()?
                .unwrap_or(100.0);

            let (lo, hi) = if lower <= upper {
                (lower, upper)
            } else {
                (upper, lower)
            };
            let mut lim = KinodynamicLimits::from_position(JointLimit::new(lo, hi));
            lim.max_velocity = max_vel.max(1e-3);
            lim.max_effort = max_eff.max(0.0);

            let jtype = if type_str == "revolute" {
                JointType::Revolute { axis }
            } else {
                JointType::Prismatic { axis }
            };
            (jtype, Some(lim))
        }
        other => return Err(UrdfError::UnsupportedJointType(other.to_string())),
    };

    Ok(ParsedUrdfJoint {
        name,
        joint_type,
        parent_link,
        child_link,
        origin,
        limits,
    })
}

fn parse_collision_element(
    coll_node: roxmltree::Node<'_, '_>,
) -> Result<Option<ParsedCollisionPrimitive>, UrdfError> {
    let Some(geom_node) = coll_node.children().find(|n| n.has_tag_name("geometry")) else {
        return Ok(None);
    };
    let origin = match coll_node.children().find(|n| n.has_tag_name("origin")) {
        Some(orig_node) => parse_origin(orig_node)?,
        None => Isometry3::identity(),
    };

    if let Some(sphere_node) = geom_node.children().find(|n| n.has_tag_name("sphere")) {
        let r_str = sphere_node
            .attribute("radius")
            .ok_or(UrdfError::MissingAttribute {
                element: "sphere",
                attr: "radius",
            })?;
        let r = parse_f64(r_str, "sphere radius")?.max(1e-4);
        let center = Point3::from(origin.translation.vector);
        return Ok(Some(ParsedCollisionPrimitive {
            origin,
            spheres: vec![BoundingSphere::new(center, r)],
            capsule: Some(LinkCapsule::new(center, center, r)),
            convex_pose: origin,
            convex_shape: SharedShape::new(Ball::new(r)),
        }));
    }

    if let Some(cyl_node) = geom_node.children().find(|n| n.has_tag_name("cylinder")) {
        let r_str = cyl_node
            .attribute("radius")
            .ok_or(UrdfError::MissingAttribute {
                element: "cylinder",
                attr: "radius",
            })?;
        let l_str = cyl_node
            .attribute("length")
            .ok_or(UrdfError::MissingAttribute {
                element: "cylinder",
                attr: "length",
            })?;
        let r = parse_f64(r_str, "cylinder radius")?.max(1e-4);
        let len = parse_f64(l_str, "cylinder length")?.max(0.0);
        let half_len = 0.5 * len;

        // URDF cylinders are aligned with local Z; decompose into spheres & capsule along local Z
        let p_a = origin.transform_point(&Point3::new(0.0, 0.0, -half_len));
        let p_b = origin.transform_point(&Point3::new(0.0, 0.0, half_len));
        let spheres = sample_spheres_along_segment(p_a, p_b, r);

        // parry3d::shape::Cylinder is aligned with Y, so rotate local X by +pi/2 to align Y with Z
        let z_to_y = Isometry3::from_parts(
            Translation3::identity(),
            UnitQuaternion::from_axis_angle(&Vector3::x_axis(), FRAC_PI_2),
        );
        return Ok(Some(ParsedCollisionPrimitive {
            origin,
            spheres,
            capsule: Some(LinkCapsule::new(p_a, p_b, r)),
            convex_pose: origin * z_to_y,
            convex_shape: SharedShape::new(Cylinder::new(half_len, r)),
        }));
    }

    if let Some(cap_node) = geom_node.children().find(|n| n.has_tag_name("capsule")) {
        let r_str = cap_node
            .attribute("radius")
            .ok_or(UrdfError::MissingAttribute {
                element: "capsule",
                attr: "radius",
            })?;
        let l_str = cap_node
            .attribute("length")
            .ok_or(UrdfError::MissingAttribute {
                element: "capsule",
                attr: "length",
            })?;
        let r = parse_f64(r_str, "capsule radius")?.max(1e-4);
        let len = parse_f64(l_str, "capsule length")?.max(0.0);
        let half_len = 0.5 * len;

        let p_a = origin.transform_point(&Point3::new(0.0, 0.0, -half_len));
        let p_b = origin.transform_point(&Point3::new(0.0, 0.0, half_len));
        let spheres = sample_spheres_along_segment(p_a, p_b, r);

        return Ok(Some(ParsedCollisionPrimitive {
            origin,
            spheres,
            capsule: Some(LinkCapsule::new(p_a, p_b, r)),
            convex_pose: origin,
            convex_shape: SharedShape::new(Capsule::new_z(half_len, r)),
        }));
    }

    if let Some(box_node) = geom_node.children().find(|n| n.has_tag_name("box")) {
        let size_str = box_node
            .attribute("size")
            .ok_or(UrdfError::MissingAttribute {
                element: "box",
                attr: "size",
            })?;
        let size = parse_vec3(size_str, "box size")?;
        let hx = (0.5 * size.x.abs()).max(1e-4);
        let hy = (0.5 * size.y.abs()).max(1e-4);
        let hz = (0.5 * size.z.abs()).max(1e-4);

        // Decompose the box along its longest principal axis into overlapping bounding spheres
        let (p_a, p_b, r) = if hx >= hy && hx >= hz {
            let rad = (hy * hy + hz * hz).sqrt();
            let span = (hx - 0.5 * rad).max(0.0);
            (
                origin.transform_point(&Point3::new(-span, 0.0, 0.0)),
                origin.transform_point(&Point3::new(span, 0.0, 0.0)),
                (rad * rad + (hx - span).powi(2)).sqrt(),
            )
        } else if hy >= hx && hy >= hz {
            let rad = (hx * hx + hz * hz).sqrt();
            let span = (hy - 0.5 * rad).max(0.0);
            (
                origin.transform_point(&Point3::new(0.0, -span, 0.0)),
                origin.transform_point(&Point3::new(0.0, span, 0.0)),
                (rad * rad + (hy - span).powi(2)).sqrt(),
            )
        } else {
            let rad = (hx * hx + hy * hy).sqrt();
            let span = (hz - 0.5 * rad).max(0.0);
            (
                origin.transform_point(&Point3::new(0.0, 0.0, -span)),
                origin.transform_point(&Point3::new(0.0, 0.0, span)),
                (rad * rad + (hz - span).powi(2)).sqrt(),
            )
        };

        let spheres = sample_spheres_along_segment(p_a, p_b, r);
        return Ok(Some(ParsedCollisionPrimitive {
            origin,
            spheres,
            capsule: Some(LinkCapsule::new(p_a, p_b, r)),
            convex_pose: origin,
            convex_shape: SharedShape::new(Cuboid::new(ParryVector::new(hx, hy, hz))),
        }));
    }

    Ok(None)
}

fn sample_spheres_along_segment(
    endpoint_a: Point3<f64>,
    endpoint_b: Point3<f64>,
    radius: f64,
) -> Vec<BoundingSphere> {
    let seg_vec = endpoint_b - endpoint_a;
    let len = seg_vec.norm();
    let num_spheres = ((len / radius.max(1e-3)).ceil() as usize).clamp(1, 8) + 1;
    if len <= 1e-6 {
        return vec![BoundingSphere::new(endpoint_a, radius)];
    }
    let mut spheres = Vec::with_capacity(num_spheres);
    for i in 0..num_spheres {
        let alpha = (i as f64) / ((num_spheres - 1) as f64);
        spheres.push(BoundingSphere::new(endpoint_a + seg_vec * alpha, radius));
    }
    spheres
}

fn build_link_geometry(primitives: &[ParsedCollisionPrimitive]) -> Option<LinkGeometry> {
    if primitives.is_empty() {
        return None;
    }

    let mut local_spheres = Vec::new();
    let mut convex_shapes = Vec::new();
    let mut combined_aabb: Option<Aabb> = None;

    for prim in primitives {
        local_spheres.extend_from_slice(&prim.spheres);
        convex_shapes.push((prim.convex_pose, prim.convex_shape.clone()));

        let parry_pose = ParryPose::from_parts(
            ParryVector::new(
                prim.convex_pose.translation.x,
                prim.convex_pose.translation.y,
                prim.convex_pose.translation.z,
            ),
            glamx::DQuat::from_xyzw(
                prim.convex_pose.rotation.i,
                prim.convex_pose.rotation.j,
                prim.convex_pose.rotation.k,
                prim.convex_pose.rotation.w,
            ),
        );
        let aabb = prim.convex_shape.compute_aabb(&parry_pose);
        combined_aabb = Some(match combined_aabb {
            Some(mut prev) => {
                prev.merge(&aabb);
                prev
            }
            None => aabb,
        });
    }

    let local_aabb = combined_aabb.unwrap();
    let bounding_capsule = if primitives.len() == 1 {
        primitives[0].capsule
    } else {
        let c = local_aabb.center();
        let he = local_aabb.half_extents();
        let center = Point3::new(c.x, c.y, c.z);
        let radius = (he.x * he.x + he.y * he.y + he.z * he.z).sqrt();
        Some(LinkCapsule::new(center, center, radius))
    };

    let _ = primitives[0].origin;
    Some(LinkGeometry {
        local_aabb,
        bounding_capsule,
        local_spheres,
        convex_shapes,
    })
}

fn compute_bounding_radius(geom: &LinkGeometry) -> f64 {
    let mut max_r = 0.05_f64;
    for s in &geom.local_spheres {
        let dist = s.center.coords.norm() + s.radius;
        if dist > max_r {
            max_r = dist;
        }
    }
    max_r
}

fn parse_origin(node: roxmltree::Node<'_, '_>) -> Result<Isometry3<f64>, UrdfError> {
    let xyz = match node.attribute("xyz") {
        Some(s) => parse_vec3(s, "origin xyz")?,
        None => Vector3::zeros(),
    };
    let rpy = match node.attribute("rpy") {
        Some(s) => parse_vec3(s, "origin rpy")?,
        None => Vector3::zeros(),
    };
    let rot = UnitQuaternion::from_euler_angles(rpy.x, rpy.y, rpy.z);
    Ok(Isometry3::from_parts(Translation3::from(xyz), rot))
}

fn parse_f64(s: &str, context: &'static str) -> Result<f64, UrdfError> {
    s.trim().parse::<f64>().map_err(|_| UrdfError::InvalidFloat {
        value: s.to_string(),
        context,
    })
}

fn parse_vec3(s: &str, context: &'static str) -> Result<Vector3<f64>, UrdfError> {
    let mut iter = s.split_whitespace();
    let (Some(x_s), Some(y_s), Some(z_s), None) =
        (iter.next(), iter.next(), iter.next(), iter.next())
    else {
        return Err(UrdfError::InvalidVector3 {
            value: s.to_string(),
            context,
        });
    };
    Ok(Vector3::new(
        parse_f64(x_s, context)?,
        parse_f64(y_s, context)?,
        parse_f64(z_s, context)?,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use unrust_it::{
        BatchRolloutBackend, CpuSimdBatchBackend, FlatRobotModelF32, GeometryTier,
    };

    const SAMPLE_ARM_URDF: &str = r#"
    <robot name="panda_3dof">
      <link name="base_link">
        <collision>
          <origin xyz="0 0 0.05" rpy="0 0 0"/>
          <geometry>
            <cylinder radius="0.06" length="0.10"/>
          </geometry>
        </collision>
      </link>

      <link name="link1">
        <collision>
          <origin xyz="0 0 0.15" rpy="0 0 0"/>
          <geometry>
            <capsule radius="0.05" length="0.30"/>
          </geometry>
        </collision>
      </link>

      <link name="link2">
        <collision>
          <origin xyz="0 0 0.15" rpy="0 0 0"/>
          <geometry>
            <box size="0.08 0.08 0.30"/>
          </geometry>
        </collision>
      </link>

      <link name="ee_link">
        <collision>
          <origin xyz="0 0 0.04" rpy="0 0 0"/>
          <geometry>
            <sphere radius="0.04"/>
          </geometry>
        </collision>
      </link>

      <joint name="joint1" type="revolute">
        <parent link="base_link"/>
        <child link="link1"/>
        <origin xyz="0 0 0.10" rpy="0 0 0"/>
        <axis xyz="0 0 1"/>
        <limit lower="-2.8973" upper="2.8973" effort="87" velocity="2.1750"/>
      </joint>

      <joint name="joint2" type="revolute">
        <parent link="link1"/>
        <child link="link2"/>
        <origin xyz="0 0 0.30" rpy="0 0 0"/>
        <axis xyz="0 1 0"/>
        <limit lower="-1.7628" upper="1.7628" effort="87" velocity="2.1750"/>
      </joint>

      <joint name="joint3" type="revolute">
        <parent link="link2"/>
        <child link="ee_link"/>
        <origin xyz="0 0 0.30" rpy="0 0 0"/>
        <axis xyz="0 1 0"/>
        <limit lower="-2.8973" upper="2.8973" effort="87" velocity="2.1750"/>
      </joint>
    </robot>
    "#;

    const SAMPLE_ARM_SRDF: &str = r#"
    <robot name="panda_3dof">
      <group name="manipulator">
        <chain base_link="base_link" tip_link="ee_link"/>
      </group>
      <group_state name="ready" group="manipulator">
        <joint name="joint1" value="0.25"/>
        <joint name="joint2" value="-0.50"/>
        <joint name="joint3" value="0.75"/>
      </group_state>
      <disable_collisions link1="base_link" link2="link2" reason="Never"/>
    </robot>
    "#;

    #[test]
    fn loads_dual_arm_urdf_workcell_and_applies_srdf() {
        let mut model = MultiRobotModel::new();
        let mut env = CollisionEnvironment::new();

        let left = add_urdf_to_scene(
            &mut model,
            &mut env,
            SAMPLE_ARM_URDF,
            RobotId(0),
            "left_",
            Isometry3::translation(0.0, 0.35, 0.0),
        )
        .expect("should parse left URDF arm");

        let right = add_urdf_to_scene(
            &mut model,
            &mut env,
            SAMPLE_ARM_URDF,
            RobotId(1),
            "right_",
            Isometry3::translation(0.0, -0.35, 0.0),
        )
        .expect("should parse right URDF arm");

        assert_eq!(model.dof(), 6);
        assert_eq!(model.num_links(), 8);
        assert_eq!(left.actuated_joints.len(), 3);
        assert_eq!(right.actuated_joints.len(), 3);

        // Apply MoveIt SRDF to both arms using their prefixes
        let left_srdf = apply_srdf_str(&mut model, &mut env, SAMPLE_ARM_SRDF, "left_")
            .expect("should apply left SRDF");
        let right_srdf = apply_srdf_str(&mut model, &mut env, SAMPLE_ARM_SRDF, "right_")
            .expect("should apply right SRDF");

        assert_eq!(left_srdf.groups.len(), 1);
        assert_eq!(left_srdf.groups[0].name, "left_manipulator");
        assert_eq!(left_srdf.groups[0].dof(), 3);
        assert_eq!(left_srdf.group_states.len(), 1);
        assert_relative_eq!(
            left_srdf.group_states[0].configuration.as_slice()[0],
            0.25,
            epsilon = 1e-6
        );
        assert_relative_eq!(
            right_srdf.group_states[0].configuration.as_slice()[4],
            -0.50,
            epsilon = 1e-6
        );

        // Verify SRDF <disable_collisions> updated the AllowedCollisionMatrix
        let left_base = model.link_by_name("left_base_link").unwrap();
        let left_link2 = model.link_by_name("left_link2").unwrap();
        assert!(!env.acm().unwrap().is_pair_active(left_base, left_link2));

        // Verify both Tier-1 (AnalyticSsv) and Tier-2 (Convex) collision checks work on the loaded URDF
        let q = [0.25, -0.50, 0.75, -0.25, -0.50, 0.75];
        let mut ws = env.create_workspace(&model);
        assert!(env
            .is_state_collision_free(&model, &q, GeometryTier::AnalyticSsv, &mut ws)
            .unwrap());
        assert!(env
            .is_state_collision_free(&model, &q, GeometryTier::Convex, &mut ws)
            .unwrap());

        // Verify FlatRobotModelF32 compiles cleanly from the URDF model and runs batched FK
        let flat = FlatRobotModelF32::compile(&model, &env);
        assert_eq!(flat.dof, 6);
        assert!(!flat.spheres.is_empty());

        let q_f32: Vec<f32> = q.iter().map(|&v| v as f32).collect();
        let mut out_spheres = vec![[0.0_f32; 4]; flat.spheres.len()];
        let mut out_ees = vec![[0.0_f32; 7]; flat.end_effectors.len()];
        CpuSimdBatchBackend.forward_kinematics_batch_f32(
            &flat,
            &q_f32,
            &mut out_spheres,
            &mut out_ees,
        );
        assert!(out_ees[0][2] > 0.3);
    }

    #[test]
    fn loads_urdf_and_srdf_from_files_and_builtin_presets() {
        let dir = std::env::temp_dir().join(format!(
            "unrust_it_urdf_test_{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let urdf_path = dir.join("arm6.urdf");
        let srdf_path = dir.join("arm6.srdf");
        std::fs::write(&urdf_path, presets::ARM_6DOF_URDF).unwrap();
        std::fs::write(&srdf_path, presets::ARM_6DOF_SRDF).unwrap();

        let (model_single, _, handle_single) = load_urdf_file(&urdf_path).unwrap();
        assert_eq!(model_single.dof(), 6);
        assert_eq!(handle_single.actuated_joints.len(), 6);

        let mut model = MultiRobotModel::new();
        let mut env = CollisionEnvironment::new();
        let h = add_urdf_file_to_scene(
            &mut model,
            &mut env,
            &urdf_path,
            RobotId(0),
            "r0_",
            Isometry3::translation(0.5, 0.0, 0.0),
        )
        .unwrap();
        assert_eq!(h.actuated_joints.len(), 6);
        let srdf = apply_srdf_file(&mut model, &mut env, &srdf_path, "r0_").unwrap();
        assert_eq!(srdf.groups.len(), 1);
        assert_eq!(srdf.disabled_collisions.len(), 5);

        let missing = load_urdf_file(dir.join("missing.urdf"));
        assert!(matches!(missing, Err(UrdfError::Io(_))));

        let _ = std::fs::remove_dir_all(&dir);
    }
}
