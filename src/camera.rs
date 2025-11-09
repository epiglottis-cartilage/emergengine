use winit::{
    event::{ElementState, KeyEvent},
    keyboard::{KeyCode, PhysicalKey},
};

pub struct Camera {
    pub eye: glam::Vec3,
    pub target: glam::Vec3,
    pub up: glam::Vec3,
    pub aspect: f32,
    pub fovy: f32,
    pub znear: f32,
    pub zfar: f32,
}

impl Camera {
    pub fn build_view_projection_matrix(&self) -> glam::Mat4 {
        // 1.s
        let view = glam::Mat4::look_at_rh(self.eye, self.target, self.up);
        // 2.
        let proj =
            glam::Mat4::perspective_rh(self.fovy.to_radians(), self.aspect, self.znear, self.zfar);

        // 3.
        return proj * view;
    }
}
#[derive(Clone, Default)]
pub struct CameraController {
    pub speed: f32,
    pub up: bool,
    pub down: bool,
    pub forward: bool,
    pub backward: bool,
    pub left: bool,
    pub right: bool,
}

impl CameraController {
    pub fn new(speed: f32) -> Self {
        Self {
            speed,
            up: false,
            down: false,
            forward: false,
            backward: false,
            left: false,
            right: false,
        }
    }

    pub fn process_events(&mut self, event: &KeyEvent) -> bool {
        // 直接检查 KeyEvent 的状态
        let state = event.state == ElementState::Pressed;

        if event.physical_key == PhysicalKey::Code(KeyCode::Space) {
            self.up = state;
            return true;
        }

        match event.physical_key {
            PhysicalKey::Code(KeyCode::Space) => {
                self.up = state;
                true
            }
            PhysicalKey::Code(KeyCode::ShiftLeft) => {
                self.down = state;
                true
            }
            PhysicalKey::Code(KeyCode::KeyW) | PhysicalKey::Code(KeyCode::ArrowUp) => {
                self.forward = state;
                true
            }
            PhysicalKey::Code(KeyCode::KeyA) | PhysicalKey::Code(KeyCode::ArrowLeft) => {
                self.left = state;
                true
            }
            PhysicalKey::Code(KeyCode::KeyS) | PhysicalKey::Code(KeyCode::ArrowDown) => {
                self.backward = state;
                true
            }
            PhysicalKey::Code(KeyCode::KeyD) | PhysicalKey::Code(KeyCode::ArrowRight) => {
                self.right = state;
                true
            }
            _ => false,
        }
    }

    pub fn update_camera(&self, camera: &mut Camera) {
        let forward = camera.target - camera.eye;
        let forward_norm = forward.normalize();
        let forward_mag = forward.length();

        // 防止摄像机离场景中心太近时出现问题
        if self.forward && forward_mag > self.speed {
            camera.eye += forward_norm * self.speed;
        }
        if self.backward {
            camera.eye -= forward_norm * self.speed;
        }

        let right = forward_norm.cross(camera.up);

        // 在按下前进或后退键时重做半径计算
        let forward = camera.target - camera.eye;
        let forward_mag = forward.length();

        if self.right {
            // 重新调整目标和眼睛之间的距离，以便其不发生变化。
            // 因此，眼睛仍然位于目标和眼睛形成的圆圈上。
            camera.eye = camera.target - (forward + right * self.speed).normalize() * forward_mag;
        }
        if self.left {
            camera.eye = camera.target - (forward - right * self.speed).normalize() * forward_mag;
        }
        if self.up {
            camera.eye += camera.up * self.speed;
        }
        if self.down {
            camera.eye -= camera.up * self.speed;
        }
    }
}
