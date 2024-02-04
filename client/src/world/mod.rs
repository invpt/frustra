use nalgebra::Vector3;

use crate::math::CubeFace;

pub struct Object<T> {
    size: Size3,
    data: Box<[T]>,
}

impl<T> Object<T> {
    pub fn new(x: u8, y: u8, z: u8, fill: impl Fn(u8, u8, u8) -> T) -> Object<T> {
        let size = Size3 { x, y, z };
        let mut data = Vec::with_capacity(x as usize * y as usize * z as usize);

        for (x, y, z) in Self::positions_for_size(size) {
            data.push(fill(x, y, z));
        }

        Object {
            size,
            data: data.into_boxed_slice(),
        }
    }

    pub fn positions(&self) -> impl Iterator<Item = (u8, u8, u8)> {
        Self::positions_for_size(self.size)
    }

    fn positions_for_size(size: Size3) -> impl Iterator<Item = (u8, u8, u8)> {
        (0..size.y)
            .flat_map(move |y| (0..size.x).flat_map(move |x| (0..size.z).map(move |z| (x, y, z))))
    }

    pub fn items(&self) -> impl Iterator<Item = (u8, u8, u8, &T)> {
        self.positions()
            .map(|(x, y, z)| (x, y, z, self.get(x, y, z).unwrap()))
    }

    pub fn adjacent(&self, x: u8, y: u8, z: u8, direction: CubeFace) -> Option<&T> {
        let (x, y, z) = match direction {
            CubeFace::Top => (Some(x), y.checked_add(1), Some(z)),
            CubeFace::Bottom => (Some(x), y.checked_sub(1), Some(z)),
            CubeFace::Left => (x.checked_sub(1), Some(y), Some(z)),
            CubeFace::Right => (x.checked_add(1), Some(y), Some(z)),
            CubeFace::Back => (Some(x), Some(y), z.checked_add(1)),
            CubeFace::Front => (Some(x), Some(y), z.checked_sub(1)),
        };

        if let (Some(x), Some(y), Some(z)) = (x, y, z) {
            self.get(x, y, z)
        } else {
            None
        }
    }

    pub fn get(&self, x: u8, y: u8, z: u8) -> Option<&T> {
        if x >= self.size.x || y >= self.size.y || z >= self.size.z {
            return None;
        }

        self.data.get(
            y as usize * self.size.x as usize * self.size.z as usize
                + x as usize * self.size.z as usize
                + z as usize,
        )
    }

    pub fn get_mut(&mut self, x: u8, y: u8, z: u8) -> Option<&mut T> {
        if x >= self.size.x || y >= self.size.y || z >= self.size.z {
            return None;
        }

        self.data.get_mut(
            y as usize * self.size.x as usize * self.size.z as usize
                + x as usize * self.size.z as usize
                + z as usize,
        )
    }
}

impl Object<bool> {
    /// Casts a ray, returning the hit coordinate if the ray intersects with a voxel.
    pub fn cast_ray(&self, start: Vector3<f32>, dir: Vector3<f32>) -> Option<(u8, u8, u8)> {
        fn sign(x: f32) -> i8 {
            if x < 0.0 {
                -1
            } else if x > 0.0 {
                1
            } else {
                0
            }
        }

        fn fract(x: f32) -> f32 {
            if x >= 0.0 {
                x.fract()
            } else {
                1.0 + x.fract()
            }
        }

        // first, intersect with the aabb
        let Some(start) = Self::intersect_aabb(
            start,
            dir,
            Vector3::new(0.0, 0.0, 0.0),
            Vector3::new(
                self.size.x as f32 - 1.0,
                self.size.y as f32 - 1.0,
                self.size.z as f32 - 1.0,
            ),
        ) else {
            return None;
        };

        let mut pos = start.map(|x| x.floor() as u8);
        let step = dir.map(sign);
        let mut t_delta = Vector3::new(1.0, 1.0, 1.0).component_div(&dir).abs();
        for x in t_delta.iter_mut() {
            if !x.is_finite() {
                *x = f32::NAN;
            }
        }
        let mut t_max = t_delta.component_mul(&(Vector3::new(1.0, 1.0, 1.0) - start.map(fract)));
        for x in t_max.iter_mut() {
            if !x.is_finite() {
                *x = f32::INFINITY;
            }
        }

        // the < 255 checks are to make sure we don't overflow
        let mut current = self.get(pos.x, pos.y, pos.z);
        while matches!(current, Some(false)) && pos.x < 255 && pos.y < 255 && pos.z < 255 {
            if t_max.x < t_max.y {
                if t_max.x < t_max.z {
                    pos.x = pos.x.wrapping_add_signed(step.x);
                    t_max.x += t_delta.x;
                } else {
                    pos.z = pos.z.wrapping_add_signed(step.z);
                    t_max.z += t_delta.z;
                }
            } else {
                if t_max.y < t_max.z {
                    pos.y = pos.y.wrapping_add_signed(step.y);
                    t_max.y += t_delta.y;
                } else {
                    pos.z = pos.z.wrapping_add_signed(step.z);
                    t_max.z += t_delta.z;
                }
            }

            current = self.get(pos.x, pos.y, pos.z);
        }

        match current {
            Some(true) => Some((pos.x, pos.y, pos.z)),
            Some(false) | None => None,
        }
    }

    fn intersect_aabb(
        o: Vector3<f32>,
        r: Vector3<f32>,
        l: Vector3<f32>,
        h: Vector3<f32>,
    ) -> Option<Vector3<f32>> {
        if o >= l && o <= h {
            return Some(o);
        }

        let t_low = (l - o).component_div(&r);
        let t_high = (h - o).component_div(&r);
        let t_close = t_low.zip_map(&t_high, |a, b| a.min(b)).max();
        let t_far = t_low.zip_map(&t_high, |a, b| a.max(b)).min();

        if t_close < t_far && t_close >= 0.0 {
            Some(o + t_close * r)
        } else {
            None
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Size3 {
    /// Sideways (x) size.
    pub x: u8,
    /// Vertical (y) size.
    pub y: u8,
    /// Forwards (z) size.
    pub z: u8,
}
