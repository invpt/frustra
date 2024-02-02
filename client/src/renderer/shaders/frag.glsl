#version 450

layout(location = 0) flat in float light;

layout(location = 0) out vec4 f_color;

void main() {
    vec4 lin_color = vec4(light, light, light, 1.0);
    f_color = pow(lin_color, vec4(0.4545));
}