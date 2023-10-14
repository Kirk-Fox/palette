use core::ops::{Add, AddAssign, Sub, SubAssign};

use crate::num::{self, Arithmetics, FromScalarArray, IntoScalarArray, One, Real, Zero};
use crate::{angle::SignedAngle, hues::OklabHueIter};

use crate::{angle::RealAngle, clamp_assign, ok_utils, Alpha, OklabHue};
use crate::{clamp, stimulus::Stimulus, Clamp, ClampAssign, Mix, MixAssign};

use super::Okhsv;

impl_is_within_bounds! {
    Okhsv {
        saturation => [Self::min_saturation(), Self::max_saturation()+ T::from_f64(ok_utils::MAX_SRGB_SATURATION_INACCURACY)],
        value => [Self::min_value(), Self::max_value()+ T::from_f64(ok_utils::MAX_SRGB_SATURATION_INACCURACY)]
    }
    where T: Real+Arithmetics+Stimulus
}

impl<T> Clamp for Okhsv<T>
where
    T: Real + Stimulus + num::Clamp,
{
    #[inline]
    fn clamp(self) -> Self {
        Self::new(
            self.hue,
            clamp(
                self.saturation,
                Self::min_saturation(),
                Self::max_saturation(),
            ),
            clamp(self.value, Self::min_value(), Self::max_value()),
        )
    }
}

impl<T> ClampAssign for Okhsv<T>
where
    T: Real + Stimulus + num::ClampAssign,
{
    #[inline]
    fn clamp_assign(&mut self) {
        clamp_assign(
            &mut self.saturation,
            Self::min_saturation(),
            Self::max_saturation(),
        );
        clamp_assign(&mut self.value, Self::min_value(), Self::max_value());
    }
}

impl_mix_hue!(Okhsv { saturation, value });
impl_lighten!(Okhsv increase {value => [Self::min_value(), Self::max_value()]} other {hue, saturation}  where T: Real+Stimulus);
impl_saturate!(Okhsv increase {saturation => [Self::min_saturation(), Self::max_saturation()]} other {hue, value}  where T:Real+ Stimulus);
impl_hue_ops!(Okhsv, OklabHue);

impl_color_add!(Okhsv<T>, [hue, saturation, value]);
impl_color_sub!(Okhsv<T>, [hue, saturation, value]);

impl_array_casts!(Okhsv<T>, [T; 3]);
impl_simd_array_conversion_hue!(Okhsv, [saturation, value]);
impl_struct_of_array_traits_hue!(Okhsv, OklabHueIter, [saturation, value]);

impl_eq_hue!(Okhsv, OklabHue, [hue, saturation, value]);
