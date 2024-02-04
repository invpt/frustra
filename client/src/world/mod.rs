

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

        self.data
            .get(y as usize * self.size.x as usize * self.size.z as usize + x as usize * self.size.z as usize + z as usize)
    }

    pub fn get_mut(&mut self, x: u8, y: u8, z: u8) -> Option<&mut T> {
        if x >= self.size.x || y >= self.size.y || z >= self.size.z {
            return None;
        }

        self.data
            .get_mut(y as usize * self.size.x as usize * self.size.z as usize + x as usize * self.size.z as usize + z as usize)
    }
}

pub struct Pos3 {
    pub x: usize,
    pub y: usize,
    pub z: usize,
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
