//! Miscellaneous algorithms.

use crate::core_crypto::prelude::*;

#[inline]
pub fn divide_ceil<Scalar>(numerator: Scalar, denominator: Scalar) -> Scalar
where
    Scalar: UnsignedInteger,
{
    // Should be a single instruction on x86 and likely other processors
    let (div, rem) = (numerator / denominator, numerator % denominator);
    div + Scalar::from(rem != Scalar::ZERO)
}

#[inline]
pub fn divide_round<Scalar: UnsignedInteger>(numerator: Scalar, denominator: Scalar) -> Scalar {
    // // Does the following without overflowing (which can happen with the addition of denom / 2)
    // (numerator + denominator / Scalar::TWO) / denominator

    // Add the half interval mapping
    // [denominator * (numerator - 1/2); denominator * (numerator + 1/2)[ to
    // [denominator * numerator; denominator * (numerator + 1)[
    // Dividing by denominator gives numerator which is what we want

    // div and rem should be computed in a single instruction on most CPUs for native types < u128
    let (div, rem) = (numerator / denominator, numerator % denominator);
    div + Scalar::from(rem >= (denominator >> 1))
}

#[track_caller]
pub fn torus_abs_diff_custom_mod<Scalar: UnsignedInteger>(
    x: Scalar,
    y: Scalar,
    modulus: Scalar,
) -> Scalar {
    let (x, y) = if y >= x { (x, y) } else { (y, x) };
    let diff = y - x;
    let x_u128: u128 = x.cast_into();
    let y_u128: u128 = y.cast_into();
    let modulus_u128: u128 = modulus.cast_into();
    let wrap_diff = Scalar::cast_from(modulus_u128 + x_u128 - y_u128);
    diff.min(wrap_diff)
}

/// Compute the smallest signed difference between two torus elements
pub fn torus_modular_diff<T: UnsignedInteger>(
    first: T,
    other: T,
    modulus: CiphertextModulus<T>,
) -> f64 {
    if modulus.is_native_modulus() {
        let bits = T::BITS as i32;
        // Using the [0; 1[ torus to reason
        // Example with first = 0.1 and other = 0.9
        // d0 = first - other = -0.8 = 0.2 mod 1
        // d1 = other - first = 0.8
        // d0 < d1 return 0.2
        // if other and first are inverted we get
        // d0 = 0.8
        // d1 = 0.2
        // d1 <= d0 return -0.2, the minus here can be seen as taking first as a reference
        // In the first example adding 0.2 to other (0.9 + 0.2 mod 1 = 0.1) gets us to first
        // In the second example adding -0.2 to other (0.1 - 0.2 mod 1 = 0.9) gets us to first
        let d0 = first.wrapping_sub(other);
        let d1 = other.wrapping_sub(first);
        if d0 < d1 {
            let d: f64 = d0.cast_into();
            d / 2_f64.powi(bits)
        } else {
            let d: f64 = d1.cast_into();
            -d / 2_f64.powi(bits)
        }
    } else {
        let custom_modulus = T::cast_from(modulus.get_custom_modulus());
        let d0 = first.wrapping_sub_custom_mod(other, custom_modulus);
        let d1 = other.wrapping_sub_custom_mod(first, custom_modulus);
        if d0 < d1 {
            let d: f64 = d0.cast_into();
            let cm_f: f64 = custom_modulus.cast_into();
            d / cm_f
        } else {
            let d: f64 = d1.cast_into();
            let cm_f: f64 = custom_modulus.cast_into();
            -d / cm_f
        }
    }
}

// Our representation of non native power of 2 moduli puts the information in the MSBs and leaves
// the LSBs empty, this is what this function is checking
pub fn check_content_respects_mod<Scalar: UnsignedInteger, Input: AsRef<[Scalar]>>(
    input: &Input,
    modulus: CiphertextModulus<Scalar>,
) -> bool {
    if modulus.is_native_modulus() {
        true
    } else if modulus.is_power_of_two() {
        // If our modulus is 2^60, the scaling is 2^4 = 00...00010000, minus 1 = 00...00001111
        // we want the bits under the mask to be 0
        let power_2_diff_mask = modulus.get_power_of_two_scaling_to_native_torus() - Scalar::ONE;
        input
            .as_ref()
            .iter()
            .all(|&x| (x & power_2_diff_mask) == Scalar::ZERO)
    } else {
        // non native, not power of two
        let scalar_modulus: Scalar = modulus.get_custom_modulus().cast_into();

        input.as_ref().iter().all(|&x| x < scalar_modulus)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_divide_funcs() {
        use rand::Rng;

        let mut rng = rand::thread_rng();

        const NB_TESTS: usize = 1_000_000_000;
        const SCALING: f64 = u64::MAX as f64;
        for _ in 0..NB_TESTS {
            let num: f64 = rng.gen();
            let mut denom = 0.0f64;
            while denom == 0.0f64 {
                denom = rng.gen();
            }

            let num = (num * SCALING).round();
            let denom = (denom * SCALING).round();

            let div_f64 = num / denom;

            let rounded = div_f64.round();
            let expected_rounded_u64: u64 = rounded as u64;

            let num_u64: u64 = num as u64;
            let denom_u64: u64 = denom as u64;

            // sanity check
            assert_eq!(num, num_u64 as f64);
            assert_eq!(denom, denom_u64 as f64);

            let rounded = divide_round(num_u64, denom_u64);

            assert_eq!(expected_rounded_u64, rounded);

            let ceiled = div_f64.ceil();
            let expected_ceiled_u64: u64 = ceiled as u64;

            let ceiled = divide_ceil(num_u64, denom_u64);
            assert_eq!(expected_ceiled_u64, ceiled);
        }
    }
}
