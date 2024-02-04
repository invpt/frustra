//! Computing the lighting of a voxel scene using raytracing-esque methods.

use nalgebra::Vector3;
use rand::{rngs::ThreadRng, Rng};

use crate::world::Object;

use super::{data::FaceData, mesh::ObjectMesh};

pub fn light(object: &Object<bool>, mesh: &mut ObjectMesh) {
    for face in &mut mesh.faces {
        light_face(object, face);
    }
}

/// Precomputes lighting for the given cube face.
fn light_face(
    object: &Object<bool>,
    face: &mut FaceData,
) {
    let normal = face.face().as_vector();
    let point = Vector3::new(face.x() as f32 + 0.5, face.y() as f32 + 0.5, face.z() as f32 + 0.5) + normal;
    let iters = 4000usize;

    let sun = Vector3::new(0.6, 0.9, 0.7).normalize();
    let inv_softness = 20.0;
    let mut direct = 0.0;
    for _ in 0..iters {
        direct += sun_shadow_ray(point, normal, sun, inv_softness, object);
    }
    direct /= iters as f32;

    let inv_softness = 1.5;
    let mut ambient = 0.0;
    for _ in 0..iters {
        ambient += ambient_ray(point, normal, inv_softness, object);
    }
    ambient /= iters as f32;

    face.direct = direct;
    face.ambient = ambient;
}

/// Casts a sun shadow ray.
fn sun_shadow_ray(
    point: Vector3<f32>,
    normal: Vector3<f32>,
    sun: Vector3<f32>,
    inv_softness: f32,
    object: &Object<bool>,
) -> f32 {
    cast_ray(
        point,
        normal,
        (sun * inv_softness + random_unit_sphere(&mut rand::thread_rng())).normalize(),
        object,
    )
}

/// Casts a random ambient ray.
fn ambient_ray(
    point: Vector3<f32>,
    normal: Vector3<f32>,
    inv_softness: f32,
    object: &Object<bool>,
) -> f32 {
    let mut rng = rand::thread_rng();

    cast_ray(
        point,
        normal,
        (normal * inv_softness + random_in_hemisphere(&mut rng, &normal)).normalize(),
        object,
    )
}

fn cast_ray(
    point: Vector3<f32>,
    normal: Vector3<f32>,
    to_light: Vector3<f32>,
    object: &Object<bool>,
) -> f32 {
    let mut lit = false;
    dda(point, to_light, |x, y, z| {
        if let (Ok(x), Ok(y), Ok(z)) = (x.try_into(), y.try_into(), z.try_into()) {
            if let Some(b) = object.get(x, y, z) {
                if *b && (x, y, z) != (point.x as u8, point.y as u8, point.z as u8) {
                    TraversalAction::Stop
                } else {
                    TraversalAction::Continue
                }
            } else {
                lit = true;
                TraversalAction::Stop
            }
        } else {
            lit = true;
            TraversalAction::Stop
        }
    });

    if lit {
        to_light.dot(&normal).max(0.0)
    } else {
        0.0
    }
}

fn random_in_hemisphere(rng: &mut ThreadRng, normal: &Vector3<f32>) -> Vector3<f32> {
    let on_unit_sphere = random_unit_sphere(rng);
    if on_unit_sphere.dot(normal) > 0.0 {
        on_unit_sphere
    } else {
        -on_unit_sphere
    }
}

fn random_unit_sphere(rng: &mut ThreadRng) -> Vector3<f32> {
    loop {
        let p = Vector3::new(
            rng.gen_range(-1.0..1.0f32),
            rng.gen_range(-1.0..1.0f32),
            rng.gen_range(-1.0..1.0f32),
        );
        if p.norm_squared() <= 1.0 {
            return p;
        }
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
enum TraversalAction {
    Stop,
    Continue,
}

fn dda(
    start: Vector3<f32>,
    dir: Vector3<f32>,
    mut f: impl FnMut(isize, isize, isize) -> TraversalAction,
) {
    let mut pos = Vector3::new(
        start.x.floor() as isize,
        start.y.floor() as isize,
        start.z.floor() as isize,
    );
    let step = Vector3::new(sign(dir.x), sign(dir.y), sign(dir.z));
    let mut t_max = (pos.cast() - start).component_div(&dir);
    for x in t_max.iter_mut() {
        if !x.is_finite() {
            *x = f32::INFINITY;
        }
    }
    let mut t_delta = Vector3::new(1.0, 1.0, 1.0).component_div(&dir).abs();
    for x in t_delta.iter_mut() {
        if !x.is_finite() {
            *x = f32::NAN;
        }
    }

    while f(pos.x, pos.y, pos.z) == TraversalAction::Continue {
        if t_max.x < t_max.y {
            if t_max.x < t_max.z {
                pos.x += step.x;
                t_max.x += t_delta.x;
            } else {
                pos.z += step.z;
                t_max.z += t_delta.z;
            }
        } else {
            if t_max.y < t_max.z {
                pos.y += step.y;
                t_max.y += t_delta.y;
            } else {
                pos.z += step.z;
                t_max.z += t_delta.z;
            }
        }
    }
}

fn sign(x: f32) -> isize {
    if x < 0.0 {
        -1
    } else if x > 0.0 {
        1
    } else {
        0
    }
}
