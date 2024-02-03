//! Computing triangle meshes from voxel data.

use crate::{math::CubeFace, world::Object};

pub fn mesh(
    object: &Object<bool>,
    mut face: impl FnMut(usize, usize, usize, CubeFace),
    mut voxel: impl FnMut(usize, usize, usize),
) {
    for (x, y, z, data) in object.items() {
        if !*data {
            continue;
        }

        let mut found_face = false;
        for direction in CubeFace::enumerate() {
            if let None | Some(false) = object.adjacent(x, y, z, direction) {
                found_face = true;
                face(x, y, z, direction);
            }
        }
        if found_face {
            voxel(x, y, z);
        }
    }
}
