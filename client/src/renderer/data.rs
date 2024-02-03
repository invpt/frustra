use vulkano::buffer::BufferContents;

use crate::math::CubeFace;

#[derive(Clone, Copy, Debug, BufferContents)]
#[repr(C)]
pub struct FaceData {
    bits: u32,
    pub direct: f32,
    pub ambient: f32,
}

impl FaceData {
    pub fn new(x: u8, y: u8, z: u8, face: CubeFace) -> FaceData {
        FaceData {
            bits: (z as u32) | ((y as u32) << 8) | ((x as u32) << 16) | ((face as u32) << 24),
            direct: 0.0,
            ambient: 0.0,
        }
    }

    pub fn x(&self) -> u8 {
        (self.bits >> 16) as u8
    }

    pub fn y(&self) -> u8 {
        (self.bits >> 8) as u8
    }

    pub fn z(&self) -> u8 {
        self.bits as u8
    }

    pub fn face(&self) -> CubeFace {
        CubeFace::try_from_u8((self.bits >> 24) as u8).unwrap()
    }
}
