use std::sync::Arc;

use vulkano::{
    buffer::{Buffer, BufferCreateInfo, BufferUsage, Subbuffer},
    memory::allocator::{AllocationCreateInfo, MemoryAllocator, MemoryTypeFilter},
};

use crate::{math::CubeFace, world::Object};

use super::data::FaceData;

pub struct ObjectMesh {
    pub faces: Vec<FaceData>,
}

impl ObjectMesh {
    pub fn new(object: &Object<bool>) -> ObjectMesh {
        let mut faces = Vec::new();
        for (x, y, z, data) in object.items() {
            if !*data {
                continue;
            }

            Self::add_faces(object, &mut faces, x, y, z);
        }

        ObjectMesh { faces }
    }

    fn add_faces(object: &Object<bool>, faces: &mut Vec<FaceData>, x: u8, y: u8, z: u8) {
        for direction in CubeFace::enumerate() {
            if let None | Some(false) = object.adjacent(x, y, z, direction) {
                faces.push(FaceData::new(x, y, z, direction))
            }
        }
    }

    /// Call this method to update the mesh when a voxel has been added at (`x`, `y`, `z`).
    pub fn added(&mut self, object: &Object<bool>, x: u8, y: u8, z: u8) {
        {
            // Get rid of neighboring faces

            let (x, y, z) = (x as i16, y as i16, z as i16);

            let mut i = 0;
            while i < self.faces.len() {
                let face = &self.faces[i];
                let dx = face.x() as i16 - x;
                let dy = face.y() as i16 - y;
                let dz = face.z() as i16 - z;

                let expected_face = match (dx, dy, dz) {
                    (0, 1, 0) => CubeFace::Top,
                    (0, -1, 0) => CubeFace::Bottom,
                    (-1, 0, 0) => CubeFace::Left,
                    (1, 0, 0) => CubeFace::Right,
                    (0, 0, 1) => CubeFace::Back,
                    (0, 0, -1) => CubeFace::Front,
                    _ => {
                        i += 1;
                        continue;
                    }
                }
                .opposite();

                if face.face() == expected_face {
                    // face is no longer needed!
                    self.faces.remove(i);
                } else {
                    i += 1;
                }
            }
        }

        // Add in the new faces
        Self::add_faces(object, &mut self.faces, x, y, z);
    }

    /// Call this method to update the mesh when a voxel has been removed at (`x`, `y`, `z`).
    pub fn removed(&mut self, object: &Object<bool>, x: u8, y: u8, z: u8) {
        // Remove the old faces
        let mut i = 0;
        while i < self.faces.len() {
            let face = &self.faces[i];
            if face.x() == x && face.y() == y && face.z() == z {
                self.faces.remove(i);
            } else {
                i += 1;
            }
        }

        // Add the now-needed faces
        for direction in CubeFace::enumerate() {
            let (dx, dy, dz) = direction.as_triple();
            let (Some(nx), Some(ny), Some(nz)) = (
                x.checked_add_signed(dx),
                y.checked_add_signed(dy),
                z.checked_add_signed(dz),
            ) else {
                continue;
            };

            if object.get(nx, ny, nz) == Some(&true) {
                self.faces.push(FaceData::new(nx, ny, nz, direction.opposite()));
            }
        }
    }

    pub fn create_buffer(
        &self,
        memory_allocator: Arc<impl MemoryAllocator>,
    ) -> Subbuffer<[FaceData]> {
        Buffer::from_iter(
            memory_allocator.clone(),
            BufferCreateInfo {
                usage: BufferUsage::STORAGE_BUFFER,
                ..Default::default()
            },
            AllocationCreateInfo {
                memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                    | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
                ..Default::default()
            },
            self.faces.iter().copied(),
        )
        .unwrap()
    }

    pub fn update_buffer(
        &self,
        memory_allocator: Arc<impl MemoryAllocator>,
        buffer: &mut Subbuffer<[FaceData]>,
    ) {
        if buffer.len() >= self.faces.len() as u64 {
            let mut buffer = buffer.write().unwrap();
            buffer[..self.faces.len()].copy_from_slice(&self.faces[..])
        } else {
            *buffer = self.create_buffer(memory_allocator);
        }
    }
}
