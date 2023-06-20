//! Module containing primitives pertaining to [`LWE trace packing keyswitch key
//! generation`](`LweTracePackingKeyswitchKey`).

use crate::core_crypto::algorithms::*;
use crate::core_crypto::commons::dispersion::DispersionParameter;
use crate::core_crypto::commons::generators::EncryptionRandomGenerator;
use crate::core_crypto::commons::traits::*;
use crate::core_crypto::entities::*;
use crate::core_crypto::prelude::polynomial_algorithms::apply_automorphism_wrapping_add_assign_custom_mod;
use crate::core_crypto::prelude::CiphertextModulus;

/// Fill a [`GLWE secret key`](`GlweSecretKey`) with an actual key derived from an
/// [`LWE secret key`](`LweSecretKey`) for use in the [`LWE trace packing keyswitch key`]
/// (`LweTracePackingKeyswitchKey`)
/// # Example
///
/// ```
/// use tfhe::core_crypto::prelude::*;
///
/// // DISCLAIMER: these toy example parameters are not guaranteed to be secure or yield correct
/// // computations
/// // Define parameters for GlweCiphertext creation
/// let glwe_size = GlweSize(3);
/// let polynomial_size = PolynomialSize(1024);
/// let lwe_dimension = LweDimension(900);
/// let ciphertext_modulus = CiphertextModulus::try_new((1 << 64) - (1 << 32) + 1).unwrap();
///
/// let mut seeder = new_seeder();
/// let mut secret_generator =
///     SecretRandomGenerator::<ActivatedRandomGenerator>::new(seeder.seed());
/// let lwe_secret_key =
///     allocate_and_generate_new_binary_lwe_secret_key(lwe_dimension, &mut secret_generator);
///
/// let mut glwe_secret_key =
///     GlweSecretKey::new_empty_key(0u64, glwe_size.to_glwe_dimension(), polynomial_size);
///
/// generate_tpksk_output_glwe_secret_key(
///     &lwe_secret_key,
///     &mut glwe_secret_key,
///     ciphertext_modulus,
/// );
///
/// let decomp_base_log = DecompositionBaseLog(2);
/// let decomp_level_count = DecompositionLevelCount(8);
/// let var_small = Variance::from_variance(2f64.powf(-80.0));
/// let mut seeder = new_seeder();
/// let seeder = seeder.as_mut();
/// let mut encryption_generator =
///     EncryptionRandomGenerator::<ActivatedRandomGenerator>::new(seeder.seed(), seeder);
///
/// let mut lwe_tpksk = LweTracePackingKeyswitchKey::new(
///     0u64,
///     decomp_base_log,
///     decomp_level_count,
///     lwe_dimension.to_lwe_size(),
///     glwe_size,
///     polynomial_size,
///     ciphertext_modulus,
/// );
///
/// generate_lwe_trace_packing_keyswitch_key(
///     &glwe_secret_key,
///     &mut lwe_tpksk,
///     var_small,
///     &mut encryption_generator,
/// );
/// ```
pub fn generate_tpksk_output_glwe_secret_key<Scalar, InputKeyCont, OutputKeyCont>(
    input_lwe_secret_key: &LweSecretKey<InputKeyCont>,
    output_glwe_secret_key: &mut GlweSecretKey<OutputKeyCont>,
    ciphertext_modulus: CiphertextModulus<Scalar>,
) where
    Scalar: UnsignedTorus,
    InputKeyCont: Container<Element = Scalar>,
    OutputKeyCont: ContainerMut<Element = Scalar>,
{
    let lwe_dimension = input_lwe_secret_key.lwe_dimension();
    let glwe_dimension = output_glwe_secret_key.glwe_dimension();
    let glwe_poly_size = output_glwe_secret_key.polynomial_size();

    assert!(
        lwe_dimension.0 <= glwe_dimension.0 * glwe_poly_size.0,
        "Mismatched between input_lwe_secret_key dimension {:?} and number of coefficients of \
        output_glwe_secret_key {:?}.",
        lwe_dimension.0,
        glwe_dimension.0 * glwe_poly_size.0
    );

    let glwe_key_container = output_glwe_secret_key.as_mut();

    for (index, lwe_key_bit) in input_lwe_secret_key.as_ref().iter().enumerate() {
        if index % glwe_poly_size.0 == 0 {
            glwe_key_container[index] = *lwe_key_bit;
        } else {
            let rem = index % glwe_poly_size.0;
            let quo = index / glwe_poly_size.0;
            let new_index = (quo + 1) * glwe_poly_size.0 - rem;
            glwe_key_container[new_index] = Scalar::ZERO.wrapping_sub_custom_mod(
                *lwe_key_bit,
                ciphertext_modulus.get_custom_modulus().cast_into(),
            );
        }
    }
}

/// Fill an [`LWE trace packing keyswitch key`](`LweTracePackingKeyswitchKey`)
/// with an actual key.
pub fn generate_lwe_trace_packing_keyswitch_key<Scalar, InputKeyCont, KSKeyCont, Gen>(
    input_glwe_secret_key: &GlweSecretKey<InputKeyCont>,
    lwe_tpksk: &mut LweTracePackingKeyswitchKey<KSKeyCont>,
    noise_parameters: impl DispersionParameter,
    generator: &mut EncryptionRandomGenerator<Gen>,
) where
    Scalar: UnsignedTorus,
    InputKeyCont: Container<Element = Scalar>,
    KSKeyCont: ContainerMut<Element = Scalar>,
    Gen: ByteRandomGenerator,
{
    assert_eq!(
        input_glwe_secret_key.glwe_dimension(),
        lwe_tpksk.output_glwe_key_dimension()
    );
    assert_eq!(
        input_glwe_secret_key.polynomial_size(),
        lwe_tpksk.polynomial_size()
    );

    // We retrieve decomposition arguments
    let glwe_dimension = lwe_tpksk.output_glwe_key_dimension();
    let decomp_level_count = lwe_tpksk.decomposition_level_count();
    let decomp_base_log = lwe_tpksk.decomposition_base_log();
    let polynomial_size = lwe_tpksk.polynomial_size();
    let ciphertext_modulus = lwe_tpksk.ciphertext_modulus();

    let automorphism_index_iter = 1..=polynomial_size.log2().0;

    let gen_iter = generator
        .fork_tpksk_to_tpksk_chunks::<Scalar>(
            decomp_level_count,
            glwe_dimension.to_glwe_size(),
            polynomial_size,
        )
        .unwrap();

    // loop over the before key blocks

    for ((auto_index, glwe_keyswitch_block), mut loop_generator) in automorphism_index_iter
        .zip(lwe_tpksk.iter_mut())
        .zip(gen_iter)
    {
        let mut auto_glwe_sk = GlweSecretKey::new_empty_key(
            Scalar::ZERO,
            input_glwe_secret_key.glwe_dimension(),
            input_glwe_secret_key.polynomial_size(),
        );
        let input_key_block_iter = input_glwe_secret_key
            .as_ref()
            .chunks_exact(polynomial_size.0);
        let auto_key_block_iter = auto_glwe_sk.as_mut().chunks_exact_mut(polynomial_size.0);
        for (auto_key_block, input_key_block) in auto_key_block_iter.zip(input_key_block_iter) {
            let mut output_poly = Polynomial::from_container(auto_key_block);
            let input_poly = Polynomial::from_container(input_key_block);
            apply_automorphism_wrapping_add_assign_custom_mod(
                &mut output_poly,
                &input_poly,
                2_usize.pow(auto_index as u32) + 1,
                ciphertext_modulus.get_custom_modulus().cast_into(),
            );
        }
        let mut glwe_ksk = GlweKeyswitchKey::from_container(
            glwe_keyswitch_block.into_container(),
            decomp_base_log,
            decomp_level_count,
            glwe_dimension.to_glwe_size(),
            polynomial_size,
            ciphertext_modulus,
        );
        generate_glwe_keyswitch_key(
            &auto_glwe_sk,
            input_glwe_secret_key,
            &mut glwe_ksk,
            noise_parameters,
            &mut loop_generator,
        );
    }
}