#version 450

const vec2 verts[4] = vec2[4](
    vec2(-0.004, 0.004),
    vec2(0.004, 0.004),
    vec2(-0.004, -0.004),
    vec2(0.004, -0.004)
);

layout(push_constant) uniform UniformBufferObject {
    mat4 aspect_correction;
};

void main() {
    gl_Position = aspect_correction * vec4(verts[gl_VertexIndex], 1.0, 1.0);
}
