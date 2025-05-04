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
    // Initialize gamma tracker: gamma_tracker[i] stores the list of gamma rounds applied to original index i.
    let mut bitstring: Vec<Vec<usize>> = vec![Vec::new(); n];

    // Initialize map: current_map[k] holds the list of *original indices* contributing to position k
    // in the vector of the current iteration.
    let mut current_map: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();
    let mut current_n = n; // Size of the vector in the current iteration.

    for j in 0..logn { // Iterate through gamma rounds (j=0 -> gamma0, j=1 -> gamma1, ...)
        // --- Calculate folding parameters for the current vector size `current_n` ---
        let n_power_of_2 = current_n.next_power_of_two();
        let n_first = n_power_of_2 >> 1;
        let n_fold = current_n - n_first;
        let ih = (n_first - n_fold) / 2;
        let it = ih + n_fold;

        // --- Identify T and S ranges based on *current* indices (k = 0 to current_n - 1) ---
        // --- Apply gamma j ---
        for k_source in n_first..current_n {
            // k_source is the position index in the *current* vector (current_map)
            if k_source < current_map.len() { // Safety bounds check
                // Iterate through all original indices currently mapped to this position
                for &original_index in &current_map[k_source] {
                    // Record that gamma j was applied to this original index
                    if original_index < bitstring.len() { // Safety bounds check
                        bitstring[original_index].push(j);
                    }
                }
            }
        }

        // --- Prepare the map for the next iteration ---
        // Simulate the concatenation: I_next = S_left + T_folded + S_right
        let mut next_map: Vec<Vec<usize>> = Vec::with_capacity(n_first); // Max possible size after folding

        // 1. Concatenate S_left part (original indices from positions 0 to i_start_fold_target - 1)
        for k in 0..ih {
            if k < current_map.len() { // Safety bounds check
                next_map.push(current_map[k].clone()); // Clone the list of original indices
            }
        }

        // 2. Concatenate Folded T part
        // Iterate through the target positions (left side of T)
        for k_target in ih..it {
            // Find the corresponding source position on the right side of T
            let k_source = k_target - ih + n_first;

            if k_target < current_map.len() && k_source < current_map.len() { // Safety bounds check
                // Create the new combined list of original indices for the folded position.
                // The new position (index in next_map corresponding to k_target)
                // inherits the history (original indices) from both k_target and k_source.
                let mut combined_history = current_map[k_target].clone();
                combined_history.extend(current_map[k_source].iter().cloned());
                next_map.push(combined_history);
            }
        }

        // 3. Concatenate S_right part (original indices from positions i_end_fold_target to n_first - 1)
        for k in it..n_first {
            if k < current_map.len() { // Safety bounds check
                next_map.push(current_map[k].clone()); // Clone the list of original indices
            }
        }

        // --- Update state for the next iteration ---
        current_map = next_map;
        current_n = current_map.len(); // Update the size for the next round's calculations
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
