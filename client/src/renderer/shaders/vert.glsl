#version 450

layout(push_constant) uniform UniformBufferObject {
    mat4 world_to_clip;
};

layout(location = 0) in vec3 position;
layout(location = 1) in float light;
layout(location = 2) out float interp_light;

void main() {
    gl_Position = world_to_clip * vec4(position, 1.0);
    interp_light = light;
}