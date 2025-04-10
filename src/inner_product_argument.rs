#![allow(non_snake_case)]

use std::ops::Mul;

use ark_bls12_381::{Fr, G1Affine, G1Projective};
use ark_ec::CurveGroup;
use ark_ff::{batch_inversion, Field, PrimeField};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize, Read, SerializationError, Write};
use ark_std::rand::RngCore;
use ark_std::{One, UniformRand, Zero};
use itertools::iterate;

use merlin::Transcript;

use crate::errors::ProofError;
use crate::msm_accumulator::MsmAccumulator;
use crate::transcript::CurdleproofsTranscript;
use crate::util::deserialize_g1projective_vec;
use crate::util::serialize_g1projective_vec;
use crate::util::{
    generate_blinders, get_verification_scalars_bitstring, inner_product, weighted_inner_product, msm, msm_from_projective,
};

/// An IPA proof object
#[derive(Clone, Debug, CanonicalSerialize, CanonicalDeserialize)]
pub struct InnerProductProof {
    B_c: G1Projective,
    B_d: G1Projective,

    vec_L_C: Vec<G1Projective>,
    vec_R_C: Vec<G1Projective>,
    vec_L_D: Vec<G1Projective>,
    vec_R_D: Vec<G1Projective>,

    c_final: Fr,
    d_final: Fr,
}

pub struct WeightedInnerProductProof {
    vec_L: Vec<G1Projective>,
    vec_R: Vec<G1Projective>,
    pub a_tag: G1Projective,
    pub b_tag: G1Projective,
    pub r_prime: Fr,
    pub s_prime: Fr,
    pub delta_prime: Fr,
}

/// Generate two blinder vectors `r` and `z` that satisfy the following constraints:
///    <r, d> + <z, c> == 0
/// ^  <r, z> == 0
///
/// We do this by solving a system of two equations over two unknowns.
fn generate_ipa_blinders<T: RngCore>(rng: &mut T, c: &Vec<Fr>, d: &[Fr]) -> (Vec<Fr>, Vec<Fr>) {
    let n = c.len();

    // Generate all the blinders but leave out two blinders from z
    let r: Vec<Fr> = generate_blinders(rng, n);
    let mut z: Vec<Fr> = generate_blinders(rng, n - 2); // leave two out

    // We have to solve a system of two linear equations over the two unknowns: z_{n-1} and z_n (the two blinders we left out)
    // Consider first equation: <r, d> + <z, c> == 0
    // <=> r_1 * d_1 + ... + r_n * d_n + z_1 * c_1 + ... + z_{n-1} * c_{n-1} + z_n * c_n == 0
    // The last two products contain the unknowns whereas all the previous is a known quantity `omega` -- let's compute it below
    let omega = inner_product(&r, d) + inner_product(&z[..n - 2], &c[..n - 2]);
    // Now let's consider the second equation: <r, z> == 0
    // <=> r_1 * z_1 + ... r_{n-1} * z_{n-1} * r_n * z_n == 0
    // Again, the last two products contain the unknowns whereas all the previous is a known quantity `delta` -- let's compute it below
    let delta = inner_product(&r[..n - 2], &z[..n - 2]);

    // Solving the first equation for z_{n-1} we get:
    //
    //   z_{n-1} = - c_{n-1}^-1 (z_n * c_n + omega)
    //
    // then plugging the above z_{n-1} into the second equation, we get:
    //
    //   z_n = (r_{n-1} * c_{n-1}^-1 * omega - delta) / (- r_{n-1} * c_{n-1}^-1 * c_n + r_{n-1})
    //
    // We compute these values below:

    let inv_c = c[n - 2].inverse().unwrap(); // save c_{n-1}^-1 for later
    let last_z = (r[n - 2] * inv_c * omega - delta)
        * (-r[n - 2] * inv_c * c[n - 1] + r[n - 1]).inverse().unwrap();
    let penultimate_z = -inv_c * (last_z * c[n - 1] + omega);

    z.push(penultimate_z);
    z.push(last_z);

    // Make sure the constraints were satisfied
    debug_assert!(inner_product(&r, d) + inner_product(&z, c) == Fr::zero());
    debug_assert!(inner_product(&r, &z) == Fr::zero());

    (r, z)
}

impl InnerProductProof {
    /// Create an inner product proof
    ///
    /// # Arguments
    ///
    /// * `crs_G_vec` - $\bm{G}$ CRS vector
    /// * `crs_G_prime_vec` - $\bm{G'}$ CRS blinder vector
    /// * `crs_H` - $H$ CRS element
    /// * `C` - commitment to `vec_c`
    /// * `D` - commitment to `vec_d`
    /// * `z` - inner product result
    /// * `vec_c` - first inner product vector (*witness*)
    /// * `vec_d` - second inner product vector (*witness*)
    #[allow(clippy::too_many_arguments)]
    pub fn new<T: RngCore>(
        mut crs_G_vec: Vec<G1Affine>,
        mut crs_G_prime_vec: Vec<G1Affine>,
        crs_H: &G1Projective,

        C: G1Projective,
        D: G1Projective,
        z: Fr,

        mut vec_c: Vec<Fr>,
        mut vec_d: Vec<Fr>,

        transcript: &mut Transcript,
        rng: &mut T,
    ) -> InnerProductProof {
        let mut n = vec_c.len();
        let lg_n = ark_std::log2(n) as usize;
        assert_eq!(vec_d.len(), n);
        assert!(n.is_power_of_two());

        let mut vec_L_C = Vec::with_capacity(lg_n);
        let mut vec_R_C = Vec::with_capacity(lg_n);
        let mut vec_L_D = Vec::with_capacity(lg_n);
        let mut vec_R_D = Vec::with_capacity(lg_n);

        // Step 1
        let (vec_r_c, vec_r_d) = generate_ipa_blinders(rng, &vec_c, &vec_d);

        let B_c = msm(&crs_G_vec, &vec_r_c);
        let B_d = msm(&crs_G_prime_vec, &vec_r_d);

        transcript.append_list(b"ipa_step1", &[&C, &D]);
        transcript.append(b"ipa_step1", &z);
        transcript.append_list(b"ipa_step1", &[&B_c, &B_d]);
        let alpha = transcript.get_and_append_challenge(b"ipa_alpha");
        let beta = transcript.get_and_append_challenge(b"ipa_beta");

        // Rewrite vectors c and d
        for i in 0..n {
            vec_c[i] = vec_r_c[i] + alpha * vec_c[i];
            vec_d[i] = vec_r_d[i] + alpha * vec_d[i];
        }
        let H = crs_H.mul(beta);

        // Step 2
        // Create slices backed by their respective vectors.  This lets us reslice as we compress the lengths of the
        // vectors in the main loop below.
        let mut slice_G = &mut crs_G_vec[..];
        let mut slice_H = &mut crs_G_prime_vec[..];
        let mut slice_c = &mut vec_c[..];
        let mut slice_d = &mut vec_d[..];

        while slice_c.len() > 1 {
            n /= 2;

            let (c_L, c_R) = slice_c.split_at_mut(n);
            let (d_L, d_R) = slice_d.split_at_mut(n);
            let (G_L, G_R) = slice_G.split_at_mut(n);
            let (H_L, H_R) = slice_H.split_at_mut(n);

            let L_C = msm(G_R, c_L) + H.mul(inner_product(c_L, d_R));
            let L_D = msm(H_L, d_R);
            let R_C = msm(G_L, c_R) + H.mul(inner_product(c_R, d_L));
            let R_D = msm(H_R, d_L);

            // Append elements to the proof
            vec_L_C.push(L_C);
            vec_L_D.push(L_D);
            vec_R_C.push(R_C);
            vec_R_D.push(R_D);

            transcript.append_list(b"ipa_loop", &[&L_C, &L_D, &R_C, &R_D]);
            let gamma = transcript.get_and_append_challenge(b"ipa_gamma");
            let gamma_inv = gamma.inverse().expect("gamma must have an inverse");

            // Fold input vectors and basis
            for i in 0..n {
                c_L[i] += gamma_inv * c_R[i];
                d_L[i] += gamma * d_R[i];
                G_L[i] = (G_L[i] + G_R[i].mul(gamma)).into_affine();
                H_L[i] = (H_L[i] + H_R[i].mul(gamma_inv)).into_affine();
            }

            // Save the rescaled vector for splitting in the next loop
            slice_c = c_L;
            slice_d = d_L;
            slice_G = G_L;
            slice_H = H_L;
        }

        InnerProductProof {
            B_c,
            B_d,
            vec_L_C,
            vec_R_C,
            vec_L_D,
            vec_R_D,
            c_final: slice_c[0],
            d_final: slice_d[0],
        }
    }

    /// Generate verification scalars for the IPA [verifier optimization](crate::notes::optimizations#ipa-verification-scalars)
    #[allow(clippy::type_complexity)]
    fn verification_scalars(
        &self,
        n: usize,
        transcript: &mut Transcript,
    ) -> Result<(Vec<Fr>, Vec<Fr>, Vec<Fr>, Vec<Fr>), ProofError> {
        let lg_n = self.vec_L_C.len();
        if lg_n >= 32 {
            return Err(ProofError::VerificationError);
        }
        if n != (1 << lg_n) {
            return Err(ProofError::VerificationError);
        }

        let verification_scalars_bitstring = get_verification_scalars_bitstring(n, lg_n);

        // 1. Recompute gamma_k,...,gamma_1 based on the proof transcript
        let mut challenges: Vec<Fr> = Vec::with_capacity(lg_n);
        for i in 0..self.vec_L_C.len() {
            transcript.append_list(
                b"ipa_loop",
                &[
                    &self.vec_L_C[i],
                    &self.vec_L_D[i],
                    &self.vec_R_C[i],
                    &self.vec_R_D[i],
                ],
            );
            challenges.push(transcript.get_and_append_challenge(b"ipa_gamma"));
        }

        // 2. Compute 1/gamma_k, ..., 1/gamma_1
        let mut challenges_inv: Vec<Fr> = challenges.clone();
        batch_inversion(&mut challenges_inv);

        // 3. Compute s values by iterating over the bitstring
        let mut vec_s: Vec<Fr> = Vec::with_capacity(n);
        for i in 0..n {
            vec_s.push(Fr::one());
            for j in 0..verification_scalars_bitstring[i].len() {
                vec_s[i] *= challenges[verification_scalars_bitstring[i][j]]
            }
        }

        // 4. Also compute 1/s vector
        let mut vec_inv_s = vec_s.clone();
        batch_inversion(&mut vec_inv_s);

        Ok((challenges, challenges_inv, vec_s, vec_inv_s))
    }

    /// Verify an inner product proof
    ///
    /// # Arguments
    ///
    /// * `crs_G_vec` - $\bm{G}$ CRS vector
    /// * `crs_G_prime_vec` - $\bm{G'}$ CRS blinder vector
    /// * `crs_H` - $H$ CRS element
    /// * `C` - commitment to witness vector `vec_c`
    /// * `D` - commitment to witness vector `vec_d`
    /// * `z` - inner product result
    /// * `vec_u` - Auxiliary vector for verifier [optimization](crate::notes::optimizations#grandproduct-verifier-optimizations)
    #[allow(clippy::too_many_arguments)]
    pub fn verify<T: RngCore>(
        &self,
        crs_G_vec: &Vec<G1Affine>,
        crs_H: &G1Projective,

        C: G1Projective, // no need for mut
        D: G1Projective,
        z: Fr,
        vec_u: Vec<Fr>,

        transcript: &mut Transcript,
        msm_accumulator: &mut MsmAccumulator,

        rng: &mut T,
    ) -> Result<(), ProofError> {
        let n = crs_G_vec.len();
        assert!(n.is_power_of_two());

        // Step 1:
        transcript.append_list(b"ipa_step1", &[&C, &D]);
        transcript.append(b"ipa_step1", &z);
        transcript.append_list(b"ipa_step1", &[&self.B_c, &self.B_d]);
        let alpha = transcript.get_and_append_challenge(b"ipa_alpha");
        let beta = transcript.get_and_append_challenge(b"ipa_beta");

        // Step 2
        let (vec_gamma, vec_gamma_inv, vec_s, vec_inv_s) =
            self.verification_scalars(n, transcript)?;

        // Get vector of c*s_i for first accumulated check
        let vec_c_times_s: Vec<Fr> = vec_s.iter().map(|s_i| self.c_final * *s_i).collect();

        let mut vec_rhs_scalars = vec_c_times_s; // collect right-hand-side scalars of first check
        vec_rhs_scalars.push(self.c_final * self.d_final * beta);
        let mut vec_G_H = crs_G_vec.clone(); // collect right-hand-side points of first check
        vec_G_H.push(crs_H.into_affine());

        // Step 3
        let H = crs_H.mul(beta);
        let C_a = self.B_c + C.mul(alpha) + H.mul(alpha * alpha * z);

        let point_lhs = msm_from_projective(&self.vec_L_C, &vec_gamma)
            + C_a
            + msm_from_projective(&self.vec_R_C, &vec_gamma_inv);

        msm_accumulator.accumulate_check(&point_lhs, &vec_rhs_scalars, &vec_G_H, rng);

        // Get vector of d*((1/s_i) * u_i) for the second accumulated check
        let vec_d_div_s: Vec<Fr> = vec_inv_s
            .into_iter()
            .zip(vec_u)
            .map(|(s_inv_i, u_i)| self.d_final * (s_inv_i * u_i))
            .collect();

        let D_a = self.B_d + D.mul(alpha);
        let point_lhs = msm_from_projective(&self.vec_L_D, &vec_gamma)
            + D_a
            + msm_from_projective(&self.vec_R_D, &vec_gamma_inv);

        msm_accumulator.accumulate_check(&point_lhs, &vec_d_div_s, crs_G_vec, rng);

        Ok(())
    }

    pub fn serialize<W: Write>(&self, mut w: W) -> Result<(), SerializationError> {
        self.B_c.serialize_compressed(&mut w)?;
        self.B_d.serialize_compressed(&mut w)?;
        serialize_g1projective_vec(&self.vec_L_C, &mut w)?;
        serialize_g1projective_vec(&self.vec_R_C, &mut w)?;
        serialize_g1projective_vec(&self.vec_L_D, &mut w)?;
        serialize_g1projective_vec(&self.vec_R_D, &mut w)?;
        self.c_final.serialize_compressed(&mut w)?;
        self.d_final.serialize_compressed(&mut w)?;
        Ok(())
    }

    pub fn deserialize<R: Read>(mut r: R, log2_n: usize) -> Result<Self, SerializationError> {
        Ok(Self {
            B_c: G1Projective::deserialize_compressed(&mut r)?,
            B_d: G1Projective::deserialize_compressed(&mut r)?,
            vec_L_C: deserialize_g1projective_vec(&mut r, log2_n)?,
            vec_R_C: deserialize_g1projective_vec(&mut r, log2_n)?,
            vec_L_D: deserialize_g1projective_vec(&mut r, log2_n)?,
            vec_R_D: deserialize_g1projective_vec(&mut r, log2_n)?,
            c_final: Fr::deserialize_compressed(&mut r)?,
            d_final: Fr::deserialize_compressed(&mut r)?,
        })
    }
}

impl WeightedInnerProductProof {
    /// Create a weighted inner product proof
    ///
    /// # Arguments
    ///
    /// * `crs_G_vec` - $\bm{G}$ CRS vector
    /// * `crs_H_vec` - $\bm{G'}$ CRS blinder vector
    /// * `crs_G` - $G$ CRS element
    /// * `crs_H` - $H$ CRS element
    /// * `P` - commitment to `vec_c`, `vec_d` and `z`
    /// * `z` - inner product result
    /// * `y` - weight scalar
    /// * `alpha` scalar
    /// * `vec_c` - first inner product vector (*witness*)
    /// * `vec_d` - second inner product vector (*witness*)
    #[allow(clippy::too_many_arguments)]
    pub fn new<T: RngCore>(
        mut crs_G_vec: Vec<G1Affine>,
        mut crs_H_vec: Vec<G1Affine>,
        crs_G: &G1Projective,
        crs_H: &G1Projective,

        P: G1Projective,
        z: Fr,

        mut vec_c: Vec<Fr>,
        mut vec_d: Vec<Fr>,
        y: Fr,
        mut alpha: Fr,

        transcript: &mut Transcript,
        rng: &mut T,
    ) -> WeightedInnerProductProof {
        let mut n = vec_c.len();
        let lg_n = ark_std::log2(n) as usize;
        assert_eq!(vec_d.len(), n);
        assert_eq!(crs_G_vec.len(), n);
        assert_eq!(crs_H_vec.len(), n);
        assert!(n.is_power_of_two());

        // Compute powers of y
        let y_inv = y.inverse().unwrap();
        let powers_y = iterate(y.clone(), |i| i.clone() * y)
            .take(n)
            .collect::<Vec<Fr>>();
        let powers_y_inv = iterate(y_inv.clone(), |i| i.clone() * y_inv.clone())
            .take(n)
            .collect::<Vec<Fr>>();

        let mut vec_L = Vec::with_capacity(lg_n);
        let mut vec_R = Vec::with_capacity(lg_n);

        // Step 1
        /* 
        We don't need blinders as bp+ is zk
        let (vec_r_c, vec_r_d) = generate_ipa_blinders(rng, &vec_c, &vec_d);

        let B_c = msm(&crs_G_vec, &vec_r_c);
        let B_d = msm(&crs_H_vec, &vec_r_d); */

        transcript.append(b"ipa_step1", &P);
        transcript.append(b"ipa_step1", &z);
        transcript.append_list(b"ipa_step1", &[&crs_G_vec, &crs_H_vec]);

        /* Not needed for bp+
        let alpha = transcript.get_and_append_challenge(b"ipa_alpha");
        let beta = transcript.get_and_append_challenge(b"ipa_beta"); */

        // Rewrite vectors c and d (NOT NEEDED FOR BP+)
        /* for i in 0..n {
            vec_c[i] = vec_r_c[i] + alpha * vec_c[i];
            vec_d[i] = vec_r_d[i] + alpha * vec_d[i];
        }
        let H = crs_H.mul(beta); */

        // Step 2
        // Create slices backed by their respective vectors.  This lets us reslice as we compress the lengths of the
        // vectors in the main loop below.
        let mut slice_G = &mut crs_G_vec[..];
        let mut slice_H = &mut crs_H_vec[..];
        let mut slice_c = &mut vec_c[..];
        let mut slice_d = &mut vec_d[..];
        let mut c_hat = vec![Fr::zero(); if n==1 {n} else {n/2}];
        let mut d_hat = vec![Fr::zero(); if n==1 {n} else {n/2}];
        let mut vec_G_hat_affine = vec![G1Affine::identity(); if n==1 {n} else {n/2}];
        let mut vec_H_hat_affine = vec![G1Affine::identity(); if n==1 {n} else {n/2}];

        while slice_c.len() > 1 {
            n /= 2;

            let (c_L, c_R) = slice_c.split_at_mut(n);
            let (d_L, d_R) = slice_d.split_at_mut(n);
            let (G_L, G_R) = slice_G.split_at_mut(n);
            let (H_L, H_R) = slice_H.split_at_mut(n);

            /* let L_C = msm(G_R, c_L) + H.mul(inner_product(c_L, d_R));
            let L_D = msm(H_L, d_R);
            let R_C = msm(G_L, c_R) + H.mul(inner_product(c_R, d_L));
            let R_D = msm(H_R, d_L); */

            // Compute variables for z_R
            let yn_c_R = (0..n)
                .map(|i| &powers_y[n - 1] * &c_R[i])
                .collect::<Vec<Fr>>();
            let yninv_cL = (0..n)
                .map(|i| &powers_y_inv[n - 1] * &c_L[i])
                .collect::<Vec<Fr>>();

            // First compute z_L
            let z_L = weighted_inner_product(c_L, d_R, y.clone());
            // Compute z_R
            let z_R = weighted_inner_product(&yn_c_R, d_L, y.clone());

            /*let gamma = transcript.get_and_append_challenge(b"ipa_gamma");
            let gamma_inv = gamma.inverse().expect("gamma must have an inverse");*/

            // Now we construct L
            // Note that no element in vectors c_L and d_R can be 0
            // since 0 is an invalid secret key!
            // L = <yninv_cL * G_R> + <d_R * H_L> + (z_L * g) + (x_L * h)
            let g_zL:G1Projective = *crs_G * z_L;
            let x_L_Fr = Fr::rand(rng);
            let h_x_L:G1Projective = *crs_H * x_L_Fr;
            let g_zL_h_xL:G1Projective = g_zL + h_x_L;
            let yninv_cL_GR = G_R.iter().zip(yninv_cL).fold(g_zL_h_xL, |acc,x| {
                if x.1 != Fr::zero() {
                    let cLi = x.1;
                    let cLi_GRi = *x.0 * cLi;
                    acc + &cLi_GRi
                } else {
                    acc
                }
            });
            let L = H_L.iter().zip(&mut *d_R).fold(yninv_cL_GR, |acc,x| {
                if x.1 != &Fr::zero() {
                    let dRi = x.1;
                    let dRi_HLi = *x.0 * dRi;
                    acc + &dRi_HLi
                } else {
                    acc
                }
            });

            // Now we construct R
            // Note that no element in vectors c_R and d_L can be 0
            // since 0 is an invalid secret key!
            //
            // R = <yn_c_R * G_R> + <d_L * H_R> + (z_R * g) + (x_R * h)
            let g_zR:G1Projective = *crs_G * z_R;
            let x_R_Fr = Fr::rand(rng);
            let h_x_R:G1Projective = *crs_H * x_R_Fr;
            let g_zR_h_xR:G1Projective = g_zR + h_x_R;
            let cR_GL = G_L.iter().zip(yn_c_R.clone()).fold(g_zR_h_xR, |acc,x| {
                if x.1 != Fr::zero() {
                    let cRi = x.1;
                    let cRi_GLi = *x.0 * cRi;
                    acc + &cRi_GLi
                } else {
                    acc
                }
            });
            let R = H_R.iter().zip(&mut *d_L).fold(cR_GL, |acc, x| {
                if x.1 != &Fr::zero() {
                    let dLi = x.1;
                    let dLi_HRi = *x.0 * dLi;
                    acc + &dLi_HRi
                } else {
                    acc
                }
            });

            // Append elements to the proof
            vec_L.push(L);
            vec_R.push(R);

            transcript.append_list(b"LR_step", &[&L, &R]);
            let e = transcript.get_and_append_challenge(b"ipa_e");
            let e_inv = e.inverse().expect("e must have an inverse");

            let e_squared = &e * &e;
            let e_inv_squared = &e_squared.inverse().expect("e_squared must have an inverse");

            // Fold input vectors and basis
            // We make c_hat
            c_hat = (0..n)
                .map(|i| {
                    let cLe = &c_L[i] * &e;
                    let cR_minuse = &yn_c_R[i] * &e_inv;
                    &cLe + &cR_minuse
                })
                .collect::<Vec<Fr>>();
            //   c = &mut c_hat[..];

            // We make c_hat
            d_hat = (0..n)
                .map(|i| {
                    let dRe = &d_R[i] * &e;
                    let dL_minuse = &d_L[i] * &e_inv;
                    &dRe + &dL_minuse
                })
                .collect::<Vec<Fr>>();
            //   d = &mut d_hat[..];

            // Now we make alpha_hat
            let e2_xL = &e_squared * &x_L_Fr;
            let einv2_xR = e_inv_squared * &x_R_Fr;
            let e2_xL_einv2_xR = &e2_xL + &einv2_xR;
            alpha = alpha + e2_xL_einv2_xR;

            // Now we make G_hat
            let e_yinv = e * powers_y_inv[n-1];
            let mut G_hat = (0..n)
                .map(|i| {
                    let GLe_inv:G1Projective = G_L[i] * e_inv;
                    let GRe_yinv:G1Projective = G_R[i] * e_yinv;
                    GRe_yinv + GLe_inv
                })
                .collect::<Vec<G1Projective>>();
            //   G = &mut G_hat[..];

            let mut H_hat = (0..n)
                .map(|i| {
                    let HLe:G1Projective = H_L[i] * e;
                    let HRe_inv:G1Projective = H_R[i] * e_inv;
                    HLe + HRe_inv
                })
                .collect::<Vec<G1Projective>>();


            vec_G_hat_affine = G_hat.iter()
                .map(|x| x.into_affine())
                .collect::<Vec<G1Affine>>();
            vec_H_hat_affine = H_hat.iter()
                .map(|x| x.into_affine())
                .collect::<Vec<G1Affine>>();
            // Save the rescaled vector for splitting in the next loop
            slice_c = c_hat.as_mut_slice();
            slice_d = d_hat.as_mut_slice();
            slice_G = vec_G_hat_affine.as_mut_slice();
            slice_H = vec_H_hat_affine.as_mut_slice();
        }

        // n should now be equal to 1, and every vector should therefore have length 1
        let r = Fr::rand(rng);
        let s = Fr::rand(rng);
        let delta = Fr::rand(rng);
        let eta = Fr::rand(rng);

        // Now we compute A
        let Gr: G1Projective = slice_G[0] * r;
        let Hs: G1Projective = slice_H[0] * s;
        let c_s = slice_c[0] * s;
        let c_sy = c_s*y;
        let d_r = slice_d[0] * r;
        let d_ry = d_r * y;
        let c_sy_d_ry = c_sy + d_ry;
        let g_c_sy_d_ry: G1Projective = *crs_G * c_sy_d_ry;
        let h_delta: G1Projective = *crs_H * delta;
        let A: G1Projective = Gr + Hs + g_c_sy_d_ry + h_delta;

        // Now we compute B
        let r_s = r * s;
        let r_sy = y * r_s;
        let g_r_sy: G1Projective = *crs_G * r_sy;
        let h_eta: G1Projective = *crs_H * eta;
        let B: G1Projective = g_r_sy + h_eta;


        transcript.append_list(b"final_A_and_B_step", &[&A, &B]);
        // compute challenge ee
        let ee = transcript.get_and_append_challenge(b"final_e");
        let ee_inv = ee.inverse().expect("ee must have an inverse");
        let ee_squared = ee * ee;

        // compute r_prime, s_prime, delta_prime
        let cee = slice_c[0] * ee;
        let dee = slice_d[0] * ee;
        let r_prime = r + cee;
        let s_prime = s + dee;

        let deltaee = delta * ee;
        let alpha_ee2 = alpha * ee_squared;
        let deltaee_alpha_ee2 = deltaee + alpha_ee2;
        let delta_prime = eta + deltaee_alpha_ee2;

        WeightedInnerProductProof {
            vec_L,
            vec_R,
            a_tag: A,
            b_tag: B,
            r_prime,
            s_prime,
            delta_prime
        }
    }

    /// Generate verification scalars for the IPA [verifier optimization](crate::notes::optimizations#ipa-verification-scalars)
    #[allow(clippy::type_complexity)]
    fn verification_scalars(
        &self,
        n: usize,
        transcript: &mut Transcript,
    ) -> Result<(Vec<Fr>, Vec<Fr>, Vec<Fr>, Vec<Fr>), ProofError> {
        let lg_n = self.vec_L.len();
        if lg_n >= 32 {
            return Err(ProofError::VerificationError);
        }
        if n != (1 << lg_n) {
            return Err(ProofError::VerificationError);
        }

        let verification_scalars_bitstring = get_verification_scalars_bitstring(n, lg_n);

        // 1. Recompute gamma_k,...,gamma_1 based on the proof transcript
        let mut challenges: Vec<Fr> = Vec::with_capacity(lg_n);
        for i in 0..self.vec_L.len() {
            transcript.append_list(
                b"ipa_loop",
                &[
                    &self.vec_L[i],
                    &self.vec_R[i],
                ],
            );
            challenges.push(transcript.get_and_append_challenge(b"ipa_gamma"));
        }

        // 2. Compute 1/gamma_k, ..., 1/gamma_1
        let mut challenges_inv: Vec<Fr> = challenges.clone();
        batch_inversion(&mut challenges_inv);

        // 3. Compute s values by iterating over the bitstring
        let mut vec_s: Vec<Fr> = Vec::with_capacity(n);
        for i in 0..n {
            vec_s.push(Fr::one());
            for j in 0..verification_scalars_bitstring[i].len() {
                vec_s[i] *= challenges[verification_scalars_bitstring[i][j]]
            }
        }

        // 4. Also compute 1/s vector
        let mut vec_inv_s = vec_s.clone();
        batch_inversion(&mut vec_inv_s);

        Ok((challenges, challenges_inv, vec_s, vec_inv_s))
    }

    /// Verify an inner product proof
    ///
    /// # Arguments
    ///
    /// * `crs_G_vec` - $\bm{G}$ CRS vector
    /// * `crs_G_prime_vec` - $\bm{G'}$ CRS blinder vector
    /// * `crs_H` - $H$ CRS element
    /// * `C` - commitment to witness vector `vec_c`
    /// * `D` - commitment to witness vector `vec_d`
    /// * `z` - inner product result
    /// * `vec_u` - Auxiliary vector for verifier [optimization](crate::notes::optimizations#grandproduct-verifier-optimizations)
    #[allow(clippy::too_many_arguments)]
    pub fn verify<T: RngCore>(
        &self,
        crs_G_vec: &Vec<G1Affine>,
        hi_tag: &Vec<G1Affine>,
        crs_G: &G1Projective,
        crs_H: &G1Projective,

        P: G1Projective, // no need for mut
        z: Fr,
        vec_u: Vec<Fr>,
        y: Fr,

        transcript: &mut Transcript,
        msm_accumulator: &mut MsmAccumulator,

        rng: &mut T,
    ) -> Result<(), ProofError> {
        let G = crs_G_vec;
        let H = hi_tag;
        let n = G.len();
        
        assert_eq!(H.len(), n);
        assert!(n.is_power_of_two());

        // Step 1:
        transcript.append(b"ipa_step1", &P);
        transcript.append(b"ipa_step1", &z);
/*        let alpha = transcript.get_and_append_challenge(b"ipa_alpha");
        let beta = transcript.get_and_append_challenge(b"ipa_beta");*/

        // Step 2
        let (vec_gamma, vec_gamma_inv, vec_s, vec_inv_s) =
            self.verification_scalars(n, transcript)?;

        // Get vector of c*s_i for first accumulated check
        let vec_c_times_s: Vec<Fr> = vec_s.iter().map(|s_i| self.c_final * *s_i).collect();

        let mut vec_rhs_scalars = vec_c_times_s; // collect right-hand-side scalars of first check
        vec_rhs_scalars.push(self.c_final * self.d_final * beta);
        let mut vec_G_H = crs_G_vec.clone(); // collect right-hand-side points of first check
        vec_G_H.push(crs_H.into_affine());

        // Step 3
        let H = crs_H.mul(beta);
        let C_a = self.B_c + C.mul(alpha) + H.mul(alpha * alpha * z);

        let point_lhs = msm_from_projective(&self.vec_c_L, &vec_gamma)
            + C_a
            + msm_from_projective(&self.vec_c_R, &vec_gamma_inv);

        msm_accumulator.accumulate_check(&point_lhs, &vec_rhs_scalars, &vec_G_H, rng);

        // Get vector of d*((1/s_i) * u_i) for the second accumulated check
        let vec_d_div_s: Vec<Fr> = vec_inv_s
            .into_iter()
            .zip(vec_u)
            .map(|(s_inv_i, u_i)| self.d_final * (s_inv_i * u_i))
            .collect();

        let D_a = self.B_d + D.mul(alpha);
        let point_lhs = msm_from_projective(&self.vec_d_L, &vec_gamma)
            + D_a
            + msm_from_projective(&self.vec_d_R, &vec_gamma_inv);

        msm_accumulator.accumulate_check(&point_lhs, &vec_d_div_s, crs_G_vec, rng);

        Ok(())
    }

    pub fn serialize<W: Write>(&self, mut w: W) -> Result<(), SerializationError> {
        self.B_c.serialize_compressed(&mut w)?;
        self.B_d.serialize_compressed(&mut w)?;
        serialize_g1projective_vec(&self.vec_c_L, &mut w)?;
        serialize_g1projective_vec(&self.vec_c_R, &mut w)?;
        serialize_g1projective_vec(&self.vec_d_L, &mut w)?;
        serialize_g1projective_vec(&self.vec_d_R, &mut w)?;
        self.c_final.serialize_compressed(&mut w)?;
        self.d_final.serialize_compressed(&mut w)?;
        Ok(())
    }

    pub fn deserialize<R: Read>(mut r: R, log2_n: usize) -> Result<Self, SerializationError> {
        Ok(Self {
            B_c: G1Projective::deserialize_compressed(&mut r)?,
            B_d: G1Projective::deserialize_compressed(&mut r)?,
            vec_c_L: deserialize_g1projective_vec(&mut r, log2_n)?,
            vec_c_R: deserialize_g1projective_vec(&mut r, log2_n)?,
            vec_d_L: deserialize_g1projective_vec(&mut r, log2_n)?,
            vec_d_R: deserialize_g1projective_vec(&mut r, log2_n)?,
            c_final: Fr::deserialize_compressed(&mut r)?,
            d_final: Fr::deserialize_compressed(&mut r)?,
        })
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use ark_std::rand::{rngs::StdRng, Rng, SeedableRng};
    use ark_std::UniformRand;
    use core::iter;
    use sha2::digest::Update;
    use sha2::Sha256;
    use crate::msm_accumulator::MsmAccumulator;

    #[test]
    fn test_inner_product_argument() {
        let mut rng = StdRng::seed_from_u64(0u64);
        let mut transcript_prover = merlin::Transcript::new(b"IPA");

        let n = 128;

        let crs_G_vec: Vec<G1Affine> =
            iter::repeat_with(|| G1Projective::rand(&mut rng).into_affine())
                .take(n)
                .collect();
        // There is actually a relationship between crs_G_vec and crs_G_prime_vec because of the grandproduct optimization
        // We generate a `vec_u` which has the discrete logs of every crs_G_prime element with respect to crs_G
        let vec_u = generate_blinders(&mut rng, n);
        let crs_G_prime_vec: Vec<G1Affine> = crs_G_vec
            .iter()
            .zip(&vec_u)
            .map(|(G_i, u_i)| G_i.mul(*u_i).into_affine())
            .collect();
        let crs_H = G1Projective::rand(&mut rng);

        // Generate some random vectors
        let vec_b: Vec<Fr> = iter::repeat_with(|| rng.gen()).take(n).collect();
        let vec_c: Vec<Fr> = iter::repeat_with(|| rng.gen()).take(n).collect();

        let z = inner_product(&vec_b, &vec_c);

        // Create commitments
        let B = msm(&crs_G_vec, &vec_b);
        let C = msm(&crs_G_prime_vec, &vec_c);

        let proof = InnerProductProof::new(
            crs_G_vec.clone(),
            crs_G_prime_vec.clone(),
            &crs_H,
            B.clone(),
            C.clone(),
            z,
            vec_b.clone(),
            vec_c.clone(),
            &mut transcript_prover,
            &mut rng,
        );

        // Reset the FS
        let mut transcript_verifier = merlin::Transcript::new(b"IPA");
        let mut msm_accumulator = MsmAccumulator::new();

        assert!(proof
            .verify(
                &crs_G_vec,
                &crs_H,
                B,
                C,
                z,
                vec_u.clone(),
                &mut transcript_verifier,
                &mut msm_accumulator,
                &mut rng,
            )
            .is_ok());

        assert!(msm_accumulator.verify().is_ok());

        ////////////////////////////////////////////////////
        // Let's also try a basic bad proof test where we provide the wrong inner product result to the verifeir
        let mut transcript_verifier = merlin::Transcript::new(b"IPA");
        let mut msm_accumulator = MsmAccumulator::new();

        assert!(proof
            .verify(
                &crs_G_vec,
                &crs_H,
                B,
                C,
                z + Fr::one(),
                vec_u,
                &mut transcript_verifier,
                &mut msm_accumulator,
                &mut rng,
            )
            .is_ok());

        assert!(msm_accumulator.verify().is_err());
    }

    #[test]
    fn test_weighted_inner_product_argument() {
        let mut rng = StdRng::seed_from_u64(0u64);
        let mut transcript_prover = merlin::Transcript::new(b"IPA");

        let n = 128;

        let crs_G_vec: Vec<G1Affine> =
            iter::repeat_with(|| G1Projective::rand(&mut rng).into_affine())
                .take(n)
                .collect();
        // There is actually a relationship between crs_G_vec and crs_G_prime_vec because of the grandproduct optimization
        // We generate a `vec_u` which has the discrete logs of every crs_G_prime element with respect to crs_G
        let vec_u = generate_blinders(&mut rng, n);
        let crs_H_vec: Vec<G1Affine> = crs_G_vec
            .iter()
            .zip(&vec_u)
            .map(|(G_i, u_i)| G_i.mul(*u_i).into_affine())
            .collect();
        let crs_H = G1Projective::rand(&mut rng);
        let crs_G = G1Projective::rand(&mut rng);

        // Generate some random vectors
        let vec_c: Vec<Fr> = iter::repeat_with(|| rng.gen()).take(n).collect();
        let vec_d: Vec<Fr> = iter::repeat_with(|| rng.gen()).take(n).collect();

        let z = inner_product(&vec_c, &vec_d);

        let y_scalar = Fr::rand(&mut rng);
        let y_inv = y_scalar.inverse().unwrap();
        let powers_y = iterate(y_scalar.clone(), |i| i.clone() * y_scalar)
            .take(n)
            .collect::<Vec<Fr>>();
        let powers_y_inv = iterate(y_inv.clone(), |i| i.clone() * y_inv.clone())
            .take(n)
            .collect::<Vec<Fr>>();

        let alpha = Fr::rand(&mut rng);

        let hi_tag = (0..n)
            .map(|i| crs_H_vec[i] * powers_y_inv[i])
            .collect::<Vec<G1Projective>>();

        // P = <a * G> + <b_L * H_R> + c * g + alpha*h
        let g_z: G1Projective = crs_G * z; // Todo: Implement
        let h_alpha: G1Projective = crs_H * alpha;
        let gz_halpha: G1Projective = g_z + h_alpha;
        let c_G: G1Projective = (0..n)
            .map(|i| {
                crs_G_vec[i] * vec_c[i]
            })
            .fold(gz_halpha, |acc, x| {
                acc + x
            });
        
        let P = (0..n)
            .map(|i| {
                hi_tag[i] * vec_d[i]
            })
            .fold(c_G, |acc, x| {
                acc + x
            });

        let proof = WeightedInnerProductProof::new(
            crs_G_vec.clone(),
            crs_H_vec.clone(),
            &crs_H,
            &crs_G,
            P.clone(),
            z,
            vec_c.clone(),
            vec_d.clone(),
            y_scalar,
            alpha,
            &mut transcript_prover,
            &mut rng,
        );

        // Reset the FS
        let mut transcript_verifier = merlin::Transcript::new(b"IPA");
        let mut msm_accumulator = MsmAccumulator::new();

        assert!(proof
            .verify(
                &crs_G_vec,
                &hi_tag.iter().map(|i| i.into_affine()).collect(),
                &crs_G,
                &crs_H,
                P,
                z,
                vec_u.clone(),
                y_scalar,
                &mut transcript_verifier,
                &mut msm_accumulator,
                &mut rng,
            )
            .is_ok());

        assert!(msm_accumulator.verify().is_ok());

        ////////////////////////////////////////////////////
        // Let's also try a basic bad proof test where we provide the wrong inner product result to the verifeir
        let mut transcript_verifier = merlin::Transcript::new(b"IPA");
        let mut msm_accumulator = MsmAccumulator::new();

        /*assert!(proof
            .verify(
                &crs_G_vec,
                &crs_H,
                B,
                C,
                z + Fr::one(),
                vec_u,
                &mut transcript_verifier,
                &mut msm_accumulator,
                &mut rng,
            )
            .is_ok());*/

        assert!(msm_accumulator.verify().is_err());
    }

    #[test]
    fn test_inner_product() {
        let a = vec![
            Fr::from(1u64),
            Fr::from(2u64),
            Fr::from(3u64),
            Fr::from(4u64),
        ];
        let b = vec![
            Fr::from(2u64),
            Fr::from(3u64),
            Fr::from(4u64),
            Fr::from(5u64),
        ];
        assert_eq!(Fr::from(40u64), inner_product(&a, &b));
    }

    #[test]
    fn test_weighted_inner_product() {
        let a = vec![
            Fr::from(1u64),
            Fr::from(2u64),
            Fr::from(3u64),
            Fr::from(4u64),
        ];
        let b = vec![
            Fr::from(2u64),
            Fr::from(3u64),
            Fr::from(4u64),
            Fr::from(5u64),
        ];
        let y = Fr::from(2u64);
        assert_eq!(Fr::from(444u64), weighted_inner_product(&a, &b, y));
    }
}

