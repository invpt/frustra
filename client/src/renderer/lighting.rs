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
fn light_face(object: &Object<bool>, face: &mut FaceData) {
    let normal = face.face().as_vector();
    let point = Vector3::new(
        face.x() as f32 + 0.5,
        face.y() as f32 + 0.5,
        face.z() as f32 + 0.5,
    ) + normal;

    let iters = 100usize;
    let sun = Vector3::new(-0.6, 0.9, -0.7).normalize();
    let inv_softness = 20.0;
    let mut direct = 0.0;
    for _ in 0..iters {
        direct += sun_shadow_ray(point, normal, sun, inv_softness, object);
    }
    direct /= iters as f32;

    let iters = 2000usize;
    let inv_softness = 0.0;
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
    if object.cast_ray(point, to_light).is_none() {
        to_light.dot(&normal).max(0.0)
    } else {
        0.0
    }
}

fn random_in_hemisphere(rng: &mut ThreadRng, normal: &Vector3<f32>) -> Vector3<f32> {
    let in_unit_sphere = random_unit_sphere(rng);
    if in_unit_sphere.dot(normal) > 0.0 {
        in_unit_sphere
    } else {
        -in_unit_sphere
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
