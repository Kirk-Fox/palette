/*
Data is the Pointer data set
https://www.rit.edu/cos/colorscience/rc_useful_data.php

White Point for the data is (using C illuminant)
Xn	Yn	Zn
SC		100	118.2254189827
x, y		0.310	0.3161578637
u', v'		0.2008907213	0.4608888395

Note: The xyz and yxy conversions do not use the updated conversion formula. So they are not used.
*/

use approx::assert_relative_eq;
use lazy_static::lazy_static;
use serde_derive::Deserialize;

use palette::{
    convert::IntoColorUnclamped,
    num::IntoScalarArray,
    white_point::{Any, WhitePoint},
    Lab, LabHue, Lch, Xyz,
};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct PointerWP;
impl WhitePoint<f64> for PointerWP {
    fn get_xyz() -> Xyz<Any, f64> {
        Xyz::new(0.980722647624, 1.0, 1.182254189827)
    }
}

#[derive(Deserialize, PartialEq)]
struct PointerDataRaw {
    lch_l: f64,
    lch_c: f64,
    lch_h: f64,
    lab_l: f64,
    lab_a: f64,
    lab_b: f64,
    luv_l: f64,
    luv_u: f64,
    luv_v: f64,
}

#[derive(Copy, Clone, Debug)]
struct PointerData<T = f64> {
    lch: Lch<PointerWP, T>,
    lab: Lab<PointerWP, T>,
}

impl<T> PartialEq for PointerData<T>
where
    T: PartialEq,
    LabHue<T>: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.lch == other.lch && self.lab == other.lab
    }
}

impl From<PointerDataRaw> for PointerData {
    fn from(src: PointerDataRaw) -> PointerData {
        PointerData {
            lch: Lch::new(src.lch_l, src.lch_c, src.lch_h),
            lab: Lab::new(src.lab_l, src.lab_a, src.lab_b),
        }
    }
}

macro_rules! impl_from_color_pointer {
    ($self_ty:ident) => {
        impl<T> From<$self_ty<PointerWP, T>> for PointerData<T>
        where
            T: Copy,
            $self_ty<PointerWP, T>:
                IntoColorUnclamped<Lch<PointerWP, T>> + IntoColorUnclamped<Lab<PointerWP, T>>,
        {
            fn from(color: $self_ty<PointerWP, T>) -> PointerData<T> {
                PointerData {
                    lch: color.into_color_unclamped(),
                    lab: color.into_color_unclamped(),
                }
            }
        }
    };
}

impl_from_color_pointer!(Lch);
impl_from_color_pointer!(Lab);

impl<V> From<PointerData<V>> for [PointerData<V::Scalar>; 2]
where
    V: IntoScalarArray<2>,
    Lch<PointerWP, V>: Into<[Lch<PointerWP, V::Scalar>; 2]>,
    Lab<PointerWP, V>: Into<[Lab<PointerWP, V::Scalar>; 2]>,
{
    fn from(color_data: PointerData<V>) -> Self {
        let [lch0, lch1]: [_; 2] = color_data.lch.into();
        let [lab0, lab1]: [_; 2] = color_data.lab.into();

        [
            PointerData {
                lch: lch0,
                lab: lab0,
            },
            PointerData {
                lch: lch1,
                lab: lab1,
            },
        ]
    }
}

lazy_static! {
    static ref TEST_DATA: Vec<PointerData> = load_data();
}

fn load_data() -> Vec<PointerData> {
    let file_name = "tests/pointer_dataset/pointer_data.csv";
    let mut rdr = csv::Reader::from_path(file_name)
        .expect("csv file could not be loaded in tests for pointer data");
    let mut color_data: Vec<PointerData> = Vec::new();
    for record in rdr.deserialize() {
        let r: PointerDataRaw =
            record.expect("color data could not be decoded in tests for cie 2004 data");
        color_data.push(r.into())
    }
    color_data
}

fn check_equal(src: &PointerData, tgt: &PointerData) {
    const MAX_ERROR: f64 = 0.000000000001;
    assert_relative_eq!(src.lch, tgt.lch, epsilon = MAX_ERROR);
    assert_relative_eq!(src.lab, tgt.lab, epsilon = MAX_ERROR);
}

pub fn run_from_lch_tests() {
    for expected in TEST_DATA.iter() {
        let result = PointerData::from(expected.lch);
        check_equal(&result, expected);
    }
}
pub fn run_from_lab_tests() {
    for expected in TEST_DATA.iter() {
        let result = PointerData::from(expected.lab);
        check_equal(&result, expected);
    }
}

pub mod wide_f64x2 {
    use super::*;

    pub fn run_from_lch_tests() {
        for expected in TEST_DATA.chunks_exact(2) {
            let [result0, result1]: [PointerData; 2] =
                PointerData::from(Lch::<_, wide::f64x2>::from([
                    expected[0].lch,
                    expected[1].lch,
                ]))
                .into();
            check_equal(&result0, &expected[0]);
            check_equal(&result1, &expected[1]);
        }
    }
    pub fn run_from_lab_tests() {
        for expected in TEST_DATA.chunks_exact(2) {
            let [result0, result1]: [PointerData; 2] =
                PointerData::from(Lab::<_, wide::f64x2>::from([
                    expected[0].lab,
                    expected[1].lab,
                ]))
                .into();
            check_equal(&result0, &expected[0]);
            check_equal(&result1, &expected[1]);
        }
    }
}
