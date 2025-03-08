use std::ops::{Shl, ShlAssign, Shr, ShrAssign};

#[derive(Debug, Copy, Clone)]
pub struct IVec3(pub i32, pub i32, pub i32);



impl IVec3 {
    pub fn new<T: Into<i32>>(x: T, y: T, z: T) -> Self {
        Self(x.into(), y.into(), z.into())
    }
    pub fn sqr_magnitude(&self) -> i32 {
        self.0*self.0 + self.1*self.1 + self.2*self.2
    }
}

impl std::ops::Mul<i32> for IVec3 {
    type Output = Self;

    fn mul(self, rhs: i32) -> Self::Output {
        Self{0: self.0*rhs, 1: self.1*rhs, 2: self.2*rhs}
    }
}

impl std::ops::Add for IVec3 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self{0: self.0+rhs.0, 1: self.1+rhs.1, 2: self.2+rhs.2}
    }
}

impl std::ops::Sub for IVec3 {
    type Output = IVec3;

    fn sub(self, rhs: Self) -> Self::Output {
        Self{0: self.0-rhs.0, 1: self.1-rhs.1, 2: self.2-rhs.2}
    }
}

impl Into<[i32; 3]> for IVec3 {
    fn into(self) -> [i32; 3] {
        [self.0, self.1, self.2]
    }
}
impl Into<[f64; 3]> for IVec3 {
    fn into(self) -> [f64; 3] {
        [self.0 as f64, self.1 as f64, self.2 as f64]
    }
}

impl ShrAssign<i32> for IVec3 {
    fn shr_assign(&mut self, rhs: i32) {
        self.0 >>= rhs;
        self.1 >>= rhs;
        self.2 >>= rhs;
    }
}


impl Shr<u32> for IVec3 {
    type Output = IVec3;

    fn shr(self, rhs: u32) -> Self::Output {
        Self{
            0: self.0 >> rhs,
            1: self.1 >> rhs,
            2: self.2 >> rhs,
        }
    }
}