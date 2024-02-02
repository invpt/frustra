#version 450

struct FaceData {
    uint voxel_position;
    float direct;
    float ambient;
};

layout(push_constant) uniform UniformBufferObject {
    mat4 world_to_clip;
};

layout(set = 0, binding = 0) buffer Faces {
    FaceData[] faces;
};

layout(location = 0) in uint bits;

layout(location = 0) flat out float light;

vec3 decode_voxel_position(uint voxel_position) {
    return vec3(float((voxel_position >> 16)), float((voxel_position >> 8) & 255), float(voxel_position & 255));
}

vec3 decode_voxel_offset(uint vtx_bits) {
    return vec3(float(vtx_bits >> 2), float((vtx_bits >> 1) & 1), float((vtx_bits) & 1));
}

void main() {
    uint face_index = bits >> 8;
    vec3 position = decode_voxel_position(faces[face_index].voxel_position) + decode_voxel_offset((bits >> 2) & 7);
    gl_Position = world_to_clip * vec4(position, 1.0);
    light = faces[face_index].direct + faces[face_index].ambient * 0.2;
}