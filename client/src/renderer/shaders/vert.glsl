#version 450

struct FaceData {
    uint bits;
    float direct;
    float ambient;
};

const vec3 verts[6][6] = vec3[6][6](
    // Top
    vec3[](vec3(0.0,1.0,1.0),vec3(0.0,1.0,0.0),vec3(1.0,1.0,0.0),vec3(0.0,1.0,1.0),vec3(1.0,1.0,0.0),vec3(1.0,1.0,1.0)),
    // Bottom
    vec3[](vec3(1.0,0.0,0.0),vec3(0.0,0.0,0.0),vec3(0.0,0.0,1.0),vec3(1.0,0.0,0.0),vec3(0.0,0.0,1.0),vec3(1.0,0.0,1.0)),
    // Left
    vec3[](vec3(0.0,0.0,1.0),vec3(0.0,0.0,0.0),vec3(0.0,1.0,0.0),vec3(0.0,0.0,1.0),vec3(0.0,1.0,0.0),vec3(0.0,1.0,1.0)),
    // Right
    vec3[](vec3(1.0,1.0,0.0),vec3(1.0,0.0,0.0),vec3(1.0,0.0,1.0),vec3(1.0,1.0,0.0),vec3(1.0,0.0,1.0),vec3(1.0,1.0,1.0)),
    // Back
    vec3[](vec3(1.0,0.0,1.0),vec3(0.0,0.0,1.0),vec3(0.0,1.0,1.0),vec3(1.0,0.0,1.0),vec3(0.0,1.0,1.0),vec3(1.0,1.0,1.0)),
    // Front
    vec3[](vec3(0.0,1.0,0.0),vec3(0.0,0.0,0.0),vec3(1.0,0.0,0.0),vec3(0.0,1.0,0.0),vec3(1.0,0.0,0.0),vec3(1.0,1.0,0.0))
);

layout(push_constant) uniform UniformBufferObject {
    mat4 world_to_clip;
};

layout(set = 0, binding = 0) buffer Faces {
    FaceData[] faces;
};

layout(location = 0) flat out float light;

uint decode_face_direction(uint bits) {
    return bits >> 24;
}

vec3 decode_voxel_position(uint bits) {
    return vec3(float((bits >> 16) & 255), float((bits >> 8) & 255), float(bits & 255));
}

void main() {
    uint face_index = gl_VertexIndex / 6;
    uint vertex_index = gl_VertexIndex % 6;

    FaceData face_data = faces[face_index];

    vec3 voxel_position = decode_voxel_position(face_data.bits);
    uint direction = decode_face_direction(face_data.bits);

    vec3 vertex_position = verts[direction][vertex_index];

    gl_Position = world_to_clip * vec4(voxel_position + vertex_position, 1.0);
    light = face_data.direct + face_data.ambient * 0.2;
}