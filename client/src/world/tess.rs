use std::collections::HashMap;

use crate::world::{Direction, Object};

/// Tesselates an object into triangles.
///
/// Returns: (vertices, triangles), where each triangle indexes into the vertices array.
pub fn tess(object: &Object<bool>) -> (Vec<(usize, usize, usize)>, Vec<[usize; 3]>) {
    let mut vertices = HashMap::<(usize, usize, usize), usize>::new();
    let mut idx_to_vtx = Vec::<(usize, usize, usize)>::new();
    let mut tris = Vec::<[usize; 3]>::new();
    for (x, y, z, data) in object.items() {
        if !*data {
            continue;
        }

        for direction in Direction::directions() {
            if let None | Some(false) = object.adjacent(x, y, z, direction) {
                tess_face(
                    direction,
                    x,
                    y,
                    z,
                    &mut vertices,
                    &mut idx_to_vtx,
                    &mut tris,
                );
            }
        }
    }

    (idx_to_vtx, tris)
}

fn tess_face(
    direction: Direction,
    x: usize,
    y: usize,
    z: usize,
    vertices: &mut HashMap<(usize, usize, usize), usize>,
    idx_to_vtx: &mut Vec<(usize, usize, usize)>,
    tris: &mut Vec<[usize; 3]>,
) {
    let face_vertices = match direction {
        Direction::Up => [
            (x, y + 1, z),
            (x + 1, y + 1, z),
            (x, y + 1, z + 1),
            (x + 1, y + 1, z + 1),
        ],
        Direction::Down => [(x, y, z), (x, y, z + 1), (x + 1, y, z), (x + 1, y, z + 1)],
        Direction::Left => [(x, y, z), (x, y + 1, z), (x, y, z + 1), (x, y + 1, z + 1)],
        Direction::Right => [
            (x + 1, y, z),
            (x + 1, y, z + 1),
            (x + 1, y + 1, z),
            (x + 1, y + 1, z + 1),
        ],
        Direction::Backwards => [
            (x, y, z + 1),
            (x, y + 1, z + 1),
            (x + 1, y, z + 1),
            (x + 1, y + 1, z + 1),
        ],
        Direction::Forwards => [(x, y, z), (x + 1, y, z), (x, y + 1, z), (x + 1, y + 1, z)],
    };

    let mut vertex_ids = [0usize; 4];
    for (vertex, vertex_id) in face_vertices.into_iter().zip(vertex_ids.iter_mut()) {
        if let Some(id) = vertices.get(&vertex) {
            *vertex_id = *id
        } else {
            *vertex_id = idx_to_vtx.len();
            vertices.insert(vertex, idx_to_vtx.len());
            idx_to_vtx.push(vertex);
        }
    }

    tris.push([vertex_ids[2], vertex_ids[0], vertex_ids[1]]);
    tris.push([vertex_ids[2], vertex_ids[1], vertex_ids[3]]);
}
