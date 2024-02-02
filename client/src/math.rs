use nalgebra::Vector3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum CubeFace {
    Up = 0,
    Down = 1,
    Left = 2,
    Right = 3,
    Forwards = 4,
    Backwards = 5,
}

impl CubeFace {
    pub fn enumerate() -> impl Iterator<Item = CubeFace> {
        use CubeFace::*;
        [Up, Down, Left, Right, Forwards, Backwards].into_iter()
    }

    pub fn as_vector(self) -> Vector3<f32> {
        use CubeFace::*;
        match self {
            Forwards => Vector3::new(0.0, 0.0, -1.0),
            Backwards => Vector3::new(0.0, 0.0, 1.0),
            Up => Vector3::new(0.0, 1.0, 0.0),
            Down => Vector3::new(0.0, -1.0, 0.0),
            Left => Vector3::new(-1.0, 0.0, 0.0),
            Right => Vector3::new(1.0, 0.0, 0.0),
        }
    }

    pub fn try_from_u8(value: u8) -> Option<CubeFace> {
        use CubeFace::*;
        match value {
            0 => Some(Up),
            1 => Some(Down),
            2 => Some(Left),
            3 => Some(Right),
            4 => Some(Forwards),
            5 => Some(Backwards),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum CubeVertex {
    X0Y0Z0 = 0b000,
    X0Y0Z1 = 0b001,
    X0Y1Z0 = 0b010,
    X0Y1Z1 = 0b011,
    X1Y0Z0 = 0b100,
    X1Y0Z1 = 0b101,
    X1Y1Z0 = 0b110,
    X1Y1Z1 = 0b111,
}

impl CubeVertex {
    pub fn try_from_u8(value: u8) -> Option<CubeVertex> {
        use CubeVertex::*;
        match value {
            0b000 => Some(X0Y0Z0),
            0b001 => Some(X0Y0Z1),
            0b010 => Some(X0Y1Z0),
            0b011 => Some(X0Y1Z1),
            0b100 => Some(X1Y0Z1),
            0b101 => Some(X1Y0Z1),
            0b110 => Some(X1Y1Z0),
            0b111 => Some(X1Y1Z1),
            _ => None,
        }
    }

    pub fn x(self) -> bool {
        (self as u8 >> 2) & 1 != 0
    }

    pub fn y(self) -> bool {
        (self as u8 >> 1) & 1 != 0
    }

    pub fn z(self) -> bool {
        self as u8 & 1 != 0
    }
}
