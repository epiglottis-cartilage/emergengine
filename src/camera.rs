use winit::{
    dpi::PhysicalPosition,
    event::{ElementState, KeyEvent},
    keyboard::{KeyCode, PhysicalKey},
};

pub struct Camera {
    pub eye: glam::Vec3,
    pub facing: glam::Vec3,
    pub up: glam::Vec3,
    pub aspect: f32,
    pub fovy: f32,
    pub znear: f32,
    pub zfar: f32,
}

impl Camera {
    pub fn build_view_projection_matrix(&self) -> glam::Mat4 {
        // 1.s
        let view = glam::Mat4::look_to_rh(self.eye, self.facing, self.up);
        // 2.
        let proj =
            glam::Mat4::perspective_rh(self.fovy.to_radians(), self.aspect, self.znear, self.zfar);

        // 3.
        return proj * view;
    }
}

#[derive(Clone, Default)]
pub struct CameraLookingAt {
    pub speed: f32,
    pub up: bool,
    pub down: bool,
    pub forward: bool,
    pub backward: bool,
    pub left: bool,
    pub right: bool,
    pub rotation: glam::Quat,
}

impl CameraLookingAt {
    pub fn new(speed: f32) -> Self {
        Self {
            speed,
            up: false,
            down: false,
            forward: false,
            backward: false,
            left: false,
            right: false,
            rotation: glam::Quat::IDENTITY,
        }
    }

    pub fn process_keyevents(&mut self, event: &KeyEvent) {
        // 直接检查 KeyEvent 的状态
        let state = event.state == ElementState::Pressed;

        match event.physical_key {
            PhysicalKey::Code(KeyCode::Space) => {
                self.up = state;
            }
            PhysicalKey::Code(KeyCode::ShiftLeft) => {
                self.down = state;
            }
            PhysicalKey::Code(KeyCode::KeyW) | PhysicalKey::Code(KeyCode::ArrowUp) => {
                self.forward = state;
            }
            PhysicalKey::Code(KeyCode::KeyA) | PhysicalKey::Code(KeyCode::ArrowLeft) => {
                self.left = state;
            }
            PhysicalKey::Code(KeyCode::KeyS) | PhysicalKey::Code(KeyCode::ArrowDown) => {
                self.backward = state;
            }
            PhysicalKey::Code(KeyCode::KeyD) | PhysicalKey::Code(KeyCode::ArrowRight) => {
                self.right = state;
            }
            _ => {}
        }
    }
    pub fn process_mouse(&mut self, delta_x: f64, delta_y: f64) {
        let yaw = delta_x as f32 / 20.;
        let pitch = delta_y as f32 / 20.;
        self.rotation = glam::Quat::from_euler(glam::EulerRot::YXZ, yaw, pitch, 0.0);
    }

    pub fn update_camera(&mut self, camera: &mut Camera) {
        if self.forward {
            camera.eye += camera.facing * self.speed;
        }
        if self.backward {
            camera.eye -= camera.facing * self.speed;
        }
        if self.up {
            camera.eye += camera.up * self.speed;
        }
        if self.down {
            camera.eye -= camera.up * self.speed;
        }
        if self.left {
            camera.eye += camera.up.cross(camera.facing) * self.speed;
        }
        if self.right {
            camera.eye += camera.facing.cross(camera.up) * self.speed;
        }
        camera.facing = self.rotation.mul_vec3(camera.facing);
        camera.up = self.rotation.mul_vec3(camera.up);
        self.rotation = glam::Quat::IDENTITY;
    }
}
