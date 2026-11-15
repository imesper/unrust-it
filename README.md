# unrust-it

An open-source path planner and collision-avoidance library for robotic
manipulators operating in 3D space.

## Repository layout

This repository is a Cargo workspace. The reusable robotics API lives in the
`unrust-it` library crate:

```text
crates/
└── unrust-it/
    ├── src/       # robot, kinematics, collision, and planner modules
    ├── examples/  # runnable library examples
    └── tests/     # integration tests
```

The workspace root intentionally contains no application code. Future
independent tools should be added as separate crates, for example:

```text
apps/
└── visualizer/   # GUI or 3D visualization executable
```

Applications should depend on `unrust-it` for shared robot, configuration,
kinematics, collision, and planning types. The core library must remain
independent of GUI, CLI, simulator, and visualization concerns.

Potential future crates include `unrust-it-visualizer`, `unrust-it-cli`,
`unrust-it-urdf`, and `unrust-it-sim`; they should be introduced only when
they provide independently reusable functionality or a separately shipped
application.

## Development

Run commands for the entire workspace from the repository root:

```sh
cargo test --workspace
cargo run -p unrust-it --example simple_reach
```
