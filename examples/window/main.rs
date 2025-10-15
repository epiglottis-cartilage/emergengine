fn main() {
    env_logger::init();
    let mut app = cartilage_engine::App::new().unwrap();
    app.run();
}
