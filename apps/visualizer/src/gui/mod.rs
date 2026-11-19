//! Native desktop engineering Studio UI (`Slint ~1.19` + zero-copy `wgpu 30` viewport).

slint::include_modules!();

pub mod controller;
pub mod viewport_bridge;

use controller::AppController;
use slint::ComponentHandle;
use std::cell::RefCell;
use std::rc::Rc;
use viewport_bridge::{attach_viewport_renderer, init_slint_backend, ViewportRenderState};

/// Launches the interactive native Slint + `wgpu 30` Desktop Engineering Studio.
pub fn run_gui() -> Result<(), Box<dyn std::error::Error>> {
    init_slint_backend().map_err(std::io::Error::other)?;
    let window = MainWindow::new()?;
    let viewport_state = Rc::new(RefCell::new(ViewportRenderState::default()));
    attach_viewport_renderer(&window, Rc::clone(&viewport_state))
        .map_err(std::io::Error::other)?;
    let _controller = AppController::init(&window, viewport_state);
    window.run()?;
    Ok(())
}
