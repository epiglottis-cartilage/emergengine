use cartilage_engine::*;
use glam::{Vec3, vec3};
use winit::event_loop::EventLoop;

fn main() -> Result<()> {
    env_logger::init();
    let events_loop = EventLoop::new().unwrap();
    let mut app = AppHandler::default();
    events_loop.run_app(&mut app).log()?;
    Ok(())
}
