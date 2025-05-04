//! Utility functions used around Curdleproofs

#![allow(non_snake_case)]

use ark_bls12_381::{Fr, G1Affine, G1Projective};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize, Read, SerializationError, Write};
use ark_std::rand::RngCore;
use ark_std::{UniformRand, Zero};

use ark_ec::{AffineRepr, CurveGroup, VariableBaseMSM};
use core::iter;
use std::ops::Mul;

use crate::crs::CurdleproofsCrs;
use crate::N_BLINDERS;

/// An ergonomic MSM function
///
pub fn msm(points: &[G1Affine], scalars: &[Fr]) -> G1Projective {
    assert_eq!(points.len(), scalars.len());
    G1Projective::msm(points, scalars).expect("number of points != number of scalars")
}

/// An ergonomic MSM function that works with projective points
pub fn msm_from_projective(points: &[G1Projective], scalars: &[Fr]) -> G1Projective {
    assert_eq!(points.len(), scalars.len());
    let points_affine = G1Projective::normalize_batch(points);
    msm(&points_affine, scalars)
}

/// Generate and return `n` blinders
pub fn generate_blinders<T: RngCore>(rng: &mut T, n: usize) -> Vec<Fr> {
    iter::repeat_with(|| Fr::rand(rng)).take(n).collect()
}

/// Get a bitstring to derive the verification scalars using binary decomposition. Used to [optimize the
/// verifier](crate::notes::optimizations#ipa-verification-scalars).
///
/// TODO: This can be done more efficiently
pub fn get_verification_scalars_bitstring(n: usize, logn: usize) -> Vec<Vec<usize>> {
    // Initialize the result vector: bitstring[i] stores the list of gamma rounds applied to original index i.
    // Capacity estimation: Max length of inner vec is logn.
    let mut bitstring: Vec<Vec<usize>> = vec![Vec::with_capacity(logn); n];

    // Keep track of the original indices that are still present in the (shrinking) vector.
    // We only need to store the positions relevant for the *next* size `n_first`.
    let mut active_positions: Vec<(usize, usize)> = (0..n).map(|i| (i, i)).collect(); // Stores (original_index, current_position_k)

    // Size of the vector in the current iteration.
    let mut current_n = n;

    // --- Simulate the folding process round by round ---
    for j in 0..logn {
        // --- Calculate folding parameters for the current vector size `current_n` ---
        let n_power_of_2 = current_n.next_power_of_two();
        let n_first = n_power_of_2 >> 1; // Size of the first half (and the vector in the next round)

        // Calculate parameters related to the "folding" range (T)
        let n_fold = current_n - n_first; 
        let n_dont_fold = n_first - n_fold;
        let ih = n_dont_fold / 2;


        // --- Prepare for the next iteration ---
        let mut next_active_positions = Vec::with_capacity(n_first);

        // --- Process each active element from the *current* round ---
        for &(original_index, k) in &active_positions {
            // --- Determine if the element is folded in this round (j) ---
            if k >= n_first {
                // Record that gamma `j` was applied to this original index.
                bitstring[original_index].push(j);
            }

            // --- Calculate the position for the next round ---
            let new_pos = if k >= n_first {
                k - n_first + ih
            } else {
                k
            };

            // --- Store result for the next iteration ---
            next_active_positions.push((original_index, new_pos));
        }

        // --- Update state for the next iteration ---
        active_positions = next_active_positions; 
        current_n = n_first;
    }

    bitstring
}

/// Return the inner product of two field vectors
pub fn inner_product(a: &[Fr], b: &[Fr]) -> Fr {
    assert!(a.len() == b.len());
    let mut c: Fr = Fr::zero();
    for i in 0..a.len() {
        c += a[i] * b[i];
    }
    c
}

/// Return `vec_a` permuted
pub fn get_permutation<T: Copy>(vec_a: &[T], permutation: &[u32]) -> Vec<T> {
    permutation.iter().map(|i| vec_a[*i as usize]).collect()
}

/// Given input vectors, the permutation and the randomizer, shuffle and permute the input.  Basically, prepare
/// everything so that a shuffle proof can be created!
pub fn shuffle_permute_and_commit_input<T: RngCore>(
    crs: &CurdleproofsCrs,
    vec_R: &[G1Affine],
    vec_S: &[G1Affine],
    permutation: &[u32],
    k: &Fr,
    rng: &mut T,
) -> (Vec<G1Affine>, Vec<G1Affine>, G1Projective, Vec<Fr>) {
    let ell = crs.vec_G.len();

    // Derive shuffled outputs
    let mut vec_T: Vec<G1Affine> = vec_R.iter().map(|R| R.mul(k).into_affine()).collect();
    let mut vec_U: Vec<G1Affine> = vec_S.iter().map(|S| S.mul(k).into_affine()).collect();
    vec_T = get_permutation(&vec_T, permutation);
    vec_U = get_permutation(&vec_U, permutation);

    let range_as_fr: Vec<Fr> = (0..ell as u32).map(Fr::from).collect();
    let sigma_ell = get_permutation(&range_as_fr, permutation);

    let vec_m_blinders = generate_blinders(rng, N_BLINDERS);
    let M = msm(&crs.vec_G, &sigma_ell) + msm(&crs.vec_H, &vec_m_blinders);

    (vec_T, vec_U, M, vec_m_blinders)
}

pub(crate) fn sum_affine_points(affine_points: &[G1Affine]) -> G1Affine {
    affine_points
        .iter()
        .map(|affine| affine.into_group())
        .sum::<G1Projective>()
        .into_affine()
}

pub fn deserialize_g1projective_vec<R: Read>(
    mut r: R,
    n: usize,
) -> Result<Vec<G1Projective>, SerializationError> {
    (0..n)
        .map(|_| G1Projective::deserialize_compressed(&mut r))
        .collect()
}

pub fn serialize_g1projective_vec<W: Write>(
    v: &[G1Projective],
    mut w: W,
) -> Result<(), SerializationError> {
    for p in v {
        p.serialize_compressed(&mut w)?;
    }
    Ok(())
}
