//! Computing triangle meshes from voxel data.

use vulkano::{buffer::BufferContents, pipeline::graphics::vertex_input::Vertex};

use crate::{
    math::{CubeFace, CubeVertex},
    world::Object,
};

#[derive(Debug, Clone, BufferContents, Vertex)]
#[repr(C)]
pub struct VertexBufferItem {
    #[format(R32_UINT)]
    bits: u32,
    #[format(R32G32B32_SFLOAT)]
    position: [f32; 3],
}

impl VertexBufferItem {
    pub fn new(
        data: u32,
        face: CubeFace,
        vertex: CubeVertex,
        position: [f32; 3],
    ) -> VertexBufferItem {
        VertexBufferItem {
            bits: (data << 8) | ((face as u32) << 5) | ((vertex as u32) << 2),
            position,
        }
    }

    /// Gets this vertex's extra data.
    pub fn data(&self) -> u32 {
        self.bits >> 8
    }

    /// Sets this vertex's extra data.
    pub fn set_data(&mut self, data: u32) {
        assert!(data < 0x1000000, "cannot have data >= 2^24");

        self.bits = (self.bits & 0xFF) | (data << 8);
    }

    /// Gets the index of the face on the cube.
    pub fn face(&self) -> CubeFace {
        CubeFace::try_from_u8(((self.bits >> 5) & 0b111) as u8).unwrap()
    }

    /// Gets the index of the vertex on the cube.
    pub fn vertex(&self) -> CubeVertex {
        CubeVertex::try_from_u8(((self.bits >> 2) & 0b111) as u8).unwrap()
    }
}

/// Computes a triangle mesh in the format the renderer expects.
pub fn mesh(
    object: &Object<bool>,
    mut vertex: impl FnMut(usize, usize, usize, CubeVertex, &mut VertexBufferItem),
    mut face: impl FnMut(usize, usize, usize, CubeFace, &mut [VertexBufferItem]),
    mut voxel: impl FnMut(usize, usize, usize, &mut [VertexBufferItem]),
) -> Vec<VertexBufferItem> {
    let mut vertices = Vec::<VertexBufferItem>::new();
    for (x, y, z, data) in object.items() {
        if !*data {
            continue;
        }

        let start = vertices.len();
        for direction in CubeFace::enumerate() {
            if let None | Some(false) = object.adjacent(x, y, z, direction) {
                let start = vertices.len();
                for tri in face_tris(direction) {
                    for vtx in tri {
                        let start = vertices.len();
                        vertices.push(VertexBufferItem::new(
                            0,
                            direction,
                            vtx,
                            [
                                x as f32 + vtx.x() as u8 as f32,
                                y as f32 + vtx.y() as u8 as f32,
                                z as f32 + vtx.z() as u8 as f32,
                            ],
                        ));
                        vertex(x, y, z, vtx, &mut vertices[start]);
                    }
                }
                face(x, y, z, direction, &mut vertices[start..]);
            }
        }
        if vertices.len() > start {
            voxel(x, y, z, &mut vertices[start..]);
        }
    }

    vertices
}

/// Retrieves the triangles making up a single voxel face with the given normal direction.
fn face_tris(direction: CubeFace) -> [[CubeVertex; 3]; 2] {
    use CubeFace::*;
    use CubeVertex::*;
    let vertices = match direction {
        Up => [X0Y1Z0, X1Y1Z0, X0Y1Z1, X1Y1Z1],
        Down => [X0Y0Z0, X0Y0Z1, X1Y0Z0, X1Y0Z1],
        Left => [X0Y0Z0, X0Y1Z0, X0Y0Z1, X0Y1Z1],
        Right => [X1Y0Z0, X1Y0Z1, X1Y1Z0, X1Y1Z1],
        Backwards => [X0Y0Z1, X0Y1Z1, X1Y0Z1, X1Y1Z1],
        Forwards => [X0Y0Z0, X1Y0Z0, X0Y1Z0, X1Y1Z0],
    };

    let tris = [
        [vertices[2], vertices[0], vertices[1]],
        [vertices[2], vertices[1], vertices[3]],
    ];

    tris
}
