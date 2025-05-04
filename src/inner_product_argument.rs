#![allow(non_snake_case)]
use std::ops::Mul;

use ark_bls12_381::{Fr, G1Affine, G1Projective};
use ark_ec::CurveGroup;
use ark_ff::{batch_inversion, Field};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize, Read, SerializationError, Write};
use ark_std::rand::RngCore;
use ark_std::{One, Zero};

use merlin::Transcript;

use crate::errors::ProofError;
use crate::msm_accumulator::MsmAccumulator;
use crate::transcript::CurdleproofsTranscript;
use crate::util::deserialize_g1projective_vec;
use crate::util::serialize_g1projective_vec;
use crate::util::{
    generate_blinders, get_verification_scalars_bitstring, inner_product, msm, msm_from_projective,
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
        let lg_n = n.next_power_of_two().trailing_zeros() as usize;
        assert_eq!(vec_d.len(), n);
        // assert!(n.is_power_of_two());

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

        // Find out where to divide the vectors

        let n_first = n.next_power_of_two() >> 1; // bitshifting
        let n_fold = n - n_first;
        let i_start = (n_first - n_fold) >> 1;
        let i_end = i_start + n_fold;

        // Step 2
        // Create slices backed by their respective vectors.  This lets us reslice as we compress the lengths of the
        // vectors in the main loop below.
        let mut slice_G = &mut crs_G_vec[..];
        let mut slice_G_prime = &mut crs_G_prime_vec[..];
        let mut slice_c = &mut vec_c[..];
        let mut slice_d = &mut vec_d[..];
        
        // SIPA STEPS
        // Handle n = 1
        if n == 1 {
            return InnerProductProof {
                B_c,
                B_d,
                vec_L_C,
                vec_R_C,
                vec_L_D,
                vec_R_D,
                c_final: slice_c[0],
                d_final: slice_d[0],
            };
        }

        // Comment from SP:
        // If it's the first or second iteration, unroll the Hprime = H*y_inv scalar mults
        // into multiscalar muls, for performance.
        
        if n != 1 {
            // Split corresponding to the SP formula
            let (c_first, c_R) = slice_c.split_at_mut(n_first);
            let c_L = &mut c_first[i_start..i_end];
            let (d_first, d_R) = slice_d.split_at_mut(n_first);
            let d_L = &mut d_first[i_start..i_end];
            let (G_first, G_R) = slice_G.split_at_mut(n_first);
            let G_L = &mut G_first[i_start..i_end];
            let (G_prime_first, G_prime_R) = slice_G_prime.split_at_mut(n_first);
            let G_prime_L = &mut G_prime_first[i_start..i_end];

            
            let L_C = msm(G_R, c_L) + H.mul(inner_product(c_L, d_R));
            let L_D = msm(G_prime_L, d_R);
            let R_C = msm(G_L, c_R) + H.mul(inner_product(c_R, d_L));
            let R_D = msm(G_prime_R, d_L);

            // Append elements to the proof
            vec_L_C.push(L_C);
            vec_L_D.push(L_D);
            vec_R_C.push(R_C);
            vec_R_D.push(R_D);

            transcript.append_list(b"ipa_loop", &[&L_C, &L_D, &R_C, &R_D]);
            let gamma = transcript.get_and_append_challenge(b"ipa_gamma");
            let gamma_inv = gamma.inverse().expect("gamma must have an inverse");

            // Fold input vectors and basis
            for i in 0..n_fold {
                c_L[i] += gamma_inv * c_R[i];
                d_L[i] += gamma * d_R[i];
                G_L[i] = (G_L[i] + G_R[i].mul(gamma)).into_affine();
                G_prime_L[i] = (G_prime_L[i] + G_prime_R[i].mul(gamma_inv)).into_affine();
            }



            n = n_first;
            slice_c = c_first;
            slice_d = d_first;
            slice_G = G_first;
            slice_G_prime = G_prime_first;
        }

        if n != 1 {
            n = n / 2;
            let (c_L, c_R) = slice_c.split_at_mut(n);
            let (d_L, d_R) = slice_d.split_at_mut(n);
            let (G_L, G_R) = slice_G.split_at_mut(n);
            let (G_prime_L, G_prime_R) = slice_G_prime.split_at_mut(n);
            
            let L_C = msm(G_R, c_L) + H.mul(inner_product(c_L, d_R));
            let L_D = msm(G_prime_L, d_R);
            let R_C = msm(G_L, c_R) + H.mul(inner_product(c_R, d_L));
            let R_D = msm(G_prime_R, d_L);

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
                G_prime_L[i] = (G_prime_L[i] + G_prime_R[i].mul(gamma_inv)).into_affine();
            }

            // Save the rescaled vector for splitting in the next loop
            slice_c = c_L;
            slice_d = d_L;
            slice_G = G_L;
            slice_G_prime = G_prime_L;

        }
        

        while n != 1 {
            n = n / 2;

            let (c_L, c_R) = slice_c.split_at_mut(n);
            let (d_L, d_R) = slice_d.split_at_mut(n);
            let (G_L, G_R) = slice_G.split_at_mut(n);
            let (G_prime_L, G_prime_R) = slice_G_prime.split_at_mut(n);
            
            let L_C = msm(G_R, c_L) + H.mul(inner_product(c_L, d_R));
            let L_D = msm(G_prime_L, d_R);
            let R_C = msm(G_L, c_R) + H.mul(inner_product(c_R, d_L));
            let R_D = msm(G_prime_R, d_L);

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
                G_prime_L[i] = (G_prime_L[i] + G_prime_R[i].mul(gamma_inv)).into_affine();
            }

            // Save the rescaled vector for splitting in the next loop
            slice_c = c_L;
            slice_d = d_L;
            slice_G = G_L;
            slice_G_prime = G_prime_L;
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
        //assert!(n.is_power_of_two());

        
        // Step 1:
        transcript.append_list(b"ipa_step1", &[&C, &D]);
        transcript.append(b"ipa_step1", &z);
        transcript.append_list(b"ipa_step1", &[&self.B_c, &self.B_d]);
        let alpha = transcript.get_and_append_challenge(b"ipa_alpha");
        let beta = transcript.get_and_append_challenge(b"ipa_beta");

        //let C = C - msm(&self.GS, &self.cS);
        //let D = D - msm(&self.GPrimeS, &self.dS);

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

#[cfg(test)]
mod tests {
    use super::*;
    use ark_std::rand::{rngs::StdRng, Rng, SeedableRng};
    use ark_std::UniformRand;
    use core::iter;

    use crate::msm_accumulator::MsmAccumulator;
    
    fn testhelper(n_param: usize) {
        let mut rng = StdRng::seed_from_u64(0u64);
        let mut transcript_prover = merlin::Transcript::new(b"IPA");

        let n = n_param;

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

    #[test] fn test2()   { testhelper(2); }
    #[test] fn test3()   { testhelper(3); }
    #[test] fn test4()   { testhelper(4); }
    #[test] fn test5()   { testhelper(5); }
    #[test] fn test6()   { testhelper(6); }
    #[test] fn test7()   { testhelper(7); }
    #[test] fn test8()   { testhelper(8); }
    #[test] fn test9()   { testhelper(9); }
    #[test] fn test10()  { testhelper(10); }
    #[test] fn test11()  { testhelper(11); }
    #[test] fn test12()  { testhelper(12); }
    #[test] fn test13()  { testhelper(13); }
    #[test] fn test14()  { testhelper(14); }
    #[test] fn test15()  { testhelper(15); }
    #[test] fn test16()  { testhelper(16); }
    #[test] fn test17()  { testhelper(17); }
    #[test] fn test18()  { testhelper(18); }
    #[test] fn test19()  { testhelper(19); }
    #[test] fn test20()  { testhelper(20); }
    #[test] fn test21()  { testhelper(21); }
    #[test] fn test22()  { testhelper(22); }
    #[test] fn test23()  { testhelper(23); }
    #[test] fn test24()  { testhelper(24); }
    #[test] fn test25()  { testhelper(25); }
    #[test] fn test26()  { testhelper(26); }
    #[test] fn test27()  { testhelper(27); }
    #[test] fn test28()  { testhelper(28); }
    #[test] fn test29()  { testhelper(29); }
    #[test] fn test30()  { testhelper(30); }
    #[test] fn test31()  { testhelper(31); }
    #[test] fn test32()  { testhelper(32); }
    #[test] fn test33()  { testhelper(33); }
    #[test] fn test34()  { testhelper(34); }
    #[test] fn test35()  { testhelper(35); }
    #[test] fn test36()  { testhelper(36); }
    #[test] fn test37()  { testhelper(37); }
    #[test] fn test38()  { testhelper(38); }
    #[test] fn test39()  { testhelper(39); }
    #[test] fn test40()  { testhelper(40); }
    #[test] fn test41()  { testhelper(41); }
    #[test] fn test42()  { testhelper(42); }
    #[test] fn test43()  { testhelper(43); }
    #[test] fn test44()  { testhelper(44); }
    #[test] fn test45()  { testhelper(45); }
    #[test] fn test46()  { testhelper(46); }
    #[test] fn test47()  { testhelper(47); }
    #[test] fn test48()  { testhelper(48); }
    #[test] fn test49()  { testhelper(49); }
    #[test] fn test50()  { testhelper(50); }
    #[test] fn test51()  { testhelper(51); }
    #[test] fn test52()  { testhelper(52); }
    #[test] fn test53()  { testhelper(53); }
    #[test] fn test54()  { testhelper(54); }
    #[test] fn test55()  { testhelper(55); }
    #[test] fn test56()  { testhelper(56); }
    #[test] fn test57()  { testhelper(57); }
    #[test] fn test58()  { testhelper(58); }
    #[test] fn test59()  { testhelper(59); }
    #[test] fn test60()  { testhelper(60); }
    #[test] fn test61()  { testhelper(61); }
    #[test] fn test62()  { testhelper(62); }
    #[test] fn test63()  { testhelper(63); }
    #[test] fn test64()  { testhelper(64); }
    #[test] fn test65()  { testhelper(65); }
    #[test] fn test66()  { testhelper(66); }
    #[test] fn test67()  { testhelper(67); }
    #[test] fn test68()  { testhelper(68); }
    #[test] fn test69()  { testhelper(69); }
    #[test] fn test70()  { testhelper(70); }
    #[test] fn test71()  { testhelper(71); }
    #[test] fn test72()  { testhelper(72); }
    #[test] fn test73()  { testhelper(73); }
    #[test] fn test74()  { testhelper(74); }
    #[test] fn test75()  { testhelper(75); }
    #[test] fn test76()  { testhelper(76); }
    #[test] fn test77()  { testhelper(77); }
    #[test] fn test78()  { testhelper(78); }
    #[test] fn test79()  { testhelper(79); }
    #[test] fn test80()  { testhelper(80); }
    #[test] fn test81()  { testhelper(81); }
    #[test] fn test82()  { testhelper(82); }
    #[test] fn test83()  { testhelper(83); }
    #[test] fn test84()  { testhelper(84); }
    #[test] fn test85()  { testhelper(85); }
    #[test] fn test86()  { testhelper(86); }
    #[test] fn test87()  { testhelper(87); }
    #[test] fn test88()  { testhelper(88); }
    #[test] fn test89()  { testhelper(89); }
    #[test] fn test90()  { testhelper(90); }
    #[test] fn test91()  { testhelper(91); }
    #[test] fn test92()  { testhelper(92); }
    #[test] fn test93()  { testhelper(93); }
    #[test] fn test94()  { testhelper(94); }
    #[test] fn test95()  { testhelper(95); }
    #[test] fn test96()  { testhelper(96); }
    #[test] fn test97()  { testhelper(97); }
    #[test] fn test98()  { testhelper(98); }
    #[test] fn test99()  { testhelper(99); }
    #[test] fn test100() { testhelper(100); }
    #[test] fn test101() { testhelper(101); }
    #[test] fn test102() { testhelper(102); }
    #[test] fn test103() { testhelper(103); }
    #[test] fn test104() { testhelper(104); }
    #[test] fn test105() { testhelper(105); }
    #[test] fn test106() { testhelper(106); }
    #[test] fn test107() { testhelper(107); }
    #[test] fn test108() { testhelper(108); }
    #[test] fn test109() { testhelper(109); }
    #[test] fn test110() { testhelper(110); }
    #[test] fn test111() { testhelper(111); }
    #[test] fn test112() { testhelper(112); }
    #[test] fn test113() { testhelper(113); }
    #[test] fn test114() { testhelper(114); }
    #[test] fn test115() { testhelper(115); }
    #[test] fn test116() { testhelper(116); }
    #[test] fn test117() { testhelper(117); }
    #[test] fn test118() { testhelper(118); }
    #[test] fn test119() { testhelper(119); }
    #[test] fn test120() { testhelper(120); }
    #[test] fn test121() { testhelper(121); }
    #[test] fn test122() { testhelper(122); }
    #[test] fn test123() { testhelper(123); }
    #[test] fn test124() { testhelper(124); }
    #[test] fn test125() { testhelper(125); }
    #[test] fn test126() { testhelper(126); }
    #[test] fn test127() { testhelper(127); }
    #[test] fn test128() { testhelper(128); }

    #[test]
    fn test_inner_product_argument_n_62() {
        let mut rng = StdRng::seed_from_u64(0u64);
        let mut transcript_prover = merlin::Transcript::new(b"IPA");

        let n = 62;

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
}
