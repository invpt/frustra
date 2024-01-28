mod tess;

pub struct Object<T> {
    size: Size3<usize>,
    data: Box<[T]>,
}

impl<T> Object<T> {
    pub fn new(x: usize, y: usize, z: usize, fill: impl Fn(usize, usize, usize) -> T) -> Object<T> {
        let size = Size3 { x, y, z };
        let mut data = Vec::with_capacity(x * y * z);

        for (x, y, z) in Self::positions_for_size(size) {
            data.push(fill(x, y, z));
        }

        Object {
            size,
            data: data.into_boxed_slice(),
        }
    }

    pub fn positions(&self) -> impl Iterator<Item = (usize, usize, usize)> {
        Self::positions_for_size(self.size)
    }

    fn positions_for_size(size: Size3<usize>) -> impl Iterator<Item = (usize, usize, usize)> {
        (0..size.y)
            .flat_map(move |y| (0..size.x).flat_map(move |x| (0..size.z).map(move |z| (x, y, z))))
    }

    pub fn items(&self) -> impl Iterator<Item = (usize, usize, usize, &T)> {
        self.positions()
            .map(|(x, y, z)| (x, y, z, self.get(x, y, z).unwrap()))
    }

    pub fn adjacent(&self, x: usize, y: usize, z: usize, direction: Direction) -> Option<&T> {
        let (x, y, z) = match direction {
            Direction::Up => (Some(x), y.checked_add(1), Some(z)),
            Direction::Down => (Some(x), y.checked_sub(1), Some(z)),
            Direction::Left => (x.checked_sub(1), Some(y), Some(z)),
            Direction::Right => (x.checked_add(1), Some(y), Some(z)),
            Direction::Backwards => (Some(x), Some(y), z.checked_add(1)),
            Direction::Forwards => (Some(x), Some(y), z.checked_sub(1)),
        };

        if let (Some(x), Some(y), Some(z)) = (x, y, z) {
            self.get(x, y, z)
        } else {
            None
        }
    }

    pub fn get(&self, x: usize, y: usize, z: usize) -> Option<&T> {
        if x >= self.size.x || y >= self.size.y || z >= self.size.z {
            return None;
        }
        self.data
            .get(y * self.size.x * self.size.z + x * self.size.z + z)
    }

    pub fn get_mut(&mut self, x: usize, y: usize, z: usize) -> Option<&mut T> {
        if x >= self.size.x || y >= self.size.y || z >= self.size.z {
            return None;
        }
        self.data
            .get_mut(y * self.size.x * self.size.z + x * self.size.z + z)
    }
}

impl Object<bool> {
    pub fn tess(&self) -> (Vec<(usize, usize, usize)>, Vec<[usize; 3]>) {
        tess::tess(self)
    }
}

pub struct Pos3<T> {
    pub x: T,
    pub y: T,
    pub z: T,
}

#[derive(Debug, Clone, Copy)]
pub struct Size3<T> {
    /// Sideways (x) size.
    pub x: T,
    /// Vertical (y) size.
    pub y: T,
    /// Forwards (z) size.
    pub z: T,
}

#[derive(Debug, Clone, Copy)]
pub enum Direction {
    Up,
    Down,
    Left,
    Right,
    Forwards,
    Backwards,
}

impl Direction {
    pub fn directions() -> impl Iterator<Item = Direction> {
        use Direction::*;
        [Up, Down, Left, Right, Forwards, Backwards].into_iter()
    }
}
