#version 450

layout(location = 2) in float interp_light;
layout(location = 0) out vec4 f_color;

void main() {
    //5.0 / (pos_vec.norm() + 1.0).powi(2)
    //float light = 5.0 / ((interp_light + 1.0)*(interp_light + 1.0));
    //float intensity = 0.25 * log2(6.0 * interp_light + 1.0);
    f_color = vec4(interp_light, interp_light, interp_light, 1.0);
}