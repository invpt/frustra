use nalgebra::Vector3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum CubeFace {
    Top = 0,
    Bottom = 1,
    Left = 2,
    Right = 3,
    Back = 4,
    Front = 5,
}

impl CubeFace {
    pub fn enumerate() -> impl Iterator<Item = CubeFace> {
        use CubeFace::*;
        [Top, Bottom, Left, Right, Front, Back].into_iter()
    }

    pub fn as_vector(self) -> Vector3<f32> {
        use CubeFace::*;
        match self {
            Front => Vector3::new(0.0, 0.0, -1.0),
            Back => Vector3::new(0.0, 0.0, 1.0),
            Top => Vector3::new(0.0, 1.0, 0.0),
            Bottom => Vector3::new(0.0, -1.0, 0.0),
            Left => Vector3::new(-1.0, 0.0, 0.0),
            Right => Vector3::new(1.0, 0.0, 0.0),
        }
    }

    pub fn as_triple(self) -> (i8, i8, i8) {
        use CubeFace::*;
        match self {
            Front => (0, 0, -1),
            Back => (0, 0, 1),
            Top => (0, 1, 0),
            Bottom => (0, -1, 0),
            Left => (-1, 0, 0),
            Right => (1, 0, 0),
        }
    }

    pub fn opposite(self) -> CubeFace {
        use CubeFace::*;
        match self {
            Top => Bottom,
            Bottom => Top,
            Left => Right,
            Right => Left,
            Front => Back,
            Back => Front,
        }
    }

    pub fn try_from_u8(value: u8) -> Option<CubeFace> {
        use CubeFace::*;
        match value {
            0 => Some(Top),
            1 => Some(Bottom),
            2 => Some(Left),
            3 => Some(Right),
            4 => Some(Back),
            5 => Some(Front),
            _ => None,
        }
    }
}
