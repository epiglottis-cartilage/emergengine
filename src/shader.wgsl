struct CameraUniform {
    view_proj: mat4x4f,
};
@group(1) @binding(0) 
var<uniform> camera: CameraUniform;

struct InstanceInput {
    @location(5) model_matrix_0: vec4f,
    @location(6) model_matrix_1: vec4f,
    @location(7) model_matrix_2: vec4f,
    @location(8) model_matrix_3: vec4f,
};

struct VertexInput {
    @location(0) position: vec3f,
    @location(1) uv: vec2f,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4f,
    @location(1) uv: vec2f,
};
alias FragmentInput = VertexOutput;

@vertex
fn vs_main(
    model: VertexInput,
    instance: InstanceInput,
) -> VertexOutput {
    let model_matrix = mat4x4f(
        instance.model_matrix_0,
        instance.model_matrix_1,
        instance.model_matrix_2,
        instance.model_matrix_3,
    );
    var out: VertexOutput;
    out.uv = model.uv;
    out.clip_position = camera.view_proj * model_matrix * vec4f(model.position, 1.0);
    return out;
}

@group(0) @binding(0)
var t_diffuse: texture_2d<f32>;
@group(0) @binding(1)
var s_diffuse: sampler;

struct FragmentOutput {
    @location(0) color: vec4f,
}

@fragment
fn fs_main(in: FragmentInput) -> @location(0)  vec4f {
    let uv = vec2f(in.uv.x, 1.0 - in.uv.y);
    return textureSample(t_diffuse, s_diffuse, uv);
}