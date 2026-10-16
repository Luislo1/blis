/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin
   Copyright (C) 2021, The University of Tokyo

   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:
	- Redistributions of source code must retain the above copyright
	  notice, this list of conditions and the following disclaimer.
	- Redistributions in binary form must reproduce the above copyright
	  notice, this list of conditions and the following disclaimer in the
	  documentation and/or other materials provided with the distribution.
	- Neither the name(s) of the copyright holder(s) nor the names of its
	  contributors may be used to endorse or promote products derived
	  from this software without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
   HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
   SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
   LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
   DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
   THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
   (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

#include <arm_sme.h>
#include "blis.h"

__arm_new( "za" ) __arm_locally_streaming void bli_sgemm_armsme_int_SVLx4SVL
     (
             dim_t      m,
             dim_t      n,
             dim_t      k,
       const void*      alpha,
       const void*      a,
       const void*      b,
       const void*      beta,
             void*      c, inc_t rs_c, inc_t cs_c,
       const auxinfo_t* data,
       const cntx_t*    cntx
     ) 
{
	uint64_t SVL = svcntsw();

	// GEMM_UKR_SETUP_CT_AMBI( s, SVL, 4 * SVL, false );

	float* a_ = (float*)a;
	float* b_ = (float*)b;

	const void* a_next = bli_auxinfo_next_a( data );
	const void* b_next = bli_auxinfo_next_b( data );

	float* c_ = (float*)c;

	svzero_za();

	uint64_t k_;
	uint64_t k_iter = k / 4;
	uint64_t k_left = k % 4;

	for ( k_ = 0; k_ < k_iter; k_++ )
	{
		// Loads
		svfloat32x4_t zL00 = svld1_f32_x4( svptrue_c32(),
			(float32_t*)( &a_[0] ) );

		svfloat32x4_t zR00 = svld1_f32_x4( svptrue_c32(),
			(float32_t*)( &b_[0] ) );

		svmopa_za32_m( 0, svptrue_b32(), svptrue_b32(), svget4( zL00, 0 ),
			svget4( zR00, 0 ) );
		svmopa_za32_m( 1, svptrue_b32(), svptrue_b32(), svget4( zL00, 0 ),
			svget4( zR00, 1 ) );

		svfloat32x4_t zR01 = svld1_f32_x4( svptrue_c32(),
			(float32_t*)( &b_[( 4 * SVL )] ) );

		svmopa_za32_m( 2, svptrue_b32(), svptrue_b32(), svget4( zL00, 0 ),
			svget4( zR00, 2 ) );
		svmopa_za32_m( 3, svptrue_b32(), svptrue_b32(), svget4( zL00, 0 ),
			svget4( zR00, 3 ) );

		svmopa_za32_m( 0, svptrue_b32(), svptrue_b32(), svget4( zL00, 1 ),
			svget4( zR01, 0 ) );
		svmopa_za32_m( 1, svptrue_b32(), svptrue_b32(), svget4( zL00, 1 ),
			svget4( zR01, 1 ) );

		svfloat32x4_t zR02 = svld1_f32_x4( svptrue_c32(),
			(float32_t*)( &b_[2 * ( 4 * SVL )] ) );

		svmopa_za32_m( 2, svptrue_b32(), svptrue_b32(), svget4( zL00, 1 ),
			svget4( zR01, 2 ) );
		svmopa_za32_m( 3, svptrue_b32(), svptrue_b32(), svget4( zL00, 1 ),
			svget4( zR01, 3 ) );

		svmopa_za32_m( 0, svptrue_b32(), svptrue_b32(), svget4( zL00, 2 ),
			svget4( zR02, 0 ) );
		svmopa_za32_m( 1, svptrue_b32(), svptrue_b32(), svget4( zL00, 2 ),
			svget4( zR02, 1 ) );

		svfloat32x4_t zR03 = svld1_f32_x4( svptrue_c32(),
			(float32_t*)( &b_[3 * ( 4 * SVL )] ) );

		svmopa_za32_m( 2, svptrue_b32(), svptrue_b32(), svget4( zL00, 2 ),
			svget4( zR02, 2 ) );
		svmopa_za32_m( 3, svptrue_b32(), svptrue_b32(), svget4( zL00, 2 ),
			svget4( zR02, 3 ) );

		svmopa_za32_m( 0, svptrue_b32(), svptrue_b32(), svget4( zL00, 3 ),
			svget4( zR03, 0 ) );
		svprfb( svptrue_b32(), (float*)&a_next, 0 );
		svmopa_za32_m( 1, svptrue_b32(), svptrue_b32(), svget4( zL00, 3 ),
			svget4( zR03, 1 ) );

		svprfb( svptrue_b32(), (float*)&b_next, 0 );
		svmopa_za32_m( 2, svptrue_b32(), svptrue_b32(), svget4( zL00, 3 ),
			svget4( zR03, 2 ) );
		svmopa_za32_m( 3, svptrue_b32(), svptrue_b32(), svget4( zL00, 3 ),
			svget4( zR03, 3 ) );

		a_ += ( 4 * SVL );
		b_ += ( 4 * 4 * SVL );
	}

	for ( k_ = 0; k_ < k_left; k_ += 1 )
	{
		svfloat32_t zL00 = svld1_f32( svptrue_b32(), (float32_t*)( &a_[0] ) );
		svfloat32x4_t zR00 = svld1_f32_x4( svptrue_c32(),
			(float32_t*)( &b_[0] ) );

		svmopa_za32_m( 0, svptrue_b32(), svptrue_b32(), zL00,
			svget4( zR00, 0 ) );
		svmopa_za32_m( 1, svptrue_b32(), svptrue_b32(), zL00,
			svget4( zR00, 1 ) );

		svmopa_za32_m( 2, svptrue_b32(), svptrue_b32(), zL00,
			svget4( zR00, 2 ) );
		svmopa_za32_m( 3, svptrue_b32(), svptrue_b32(), zL00,
			svget4( zR00, 3 ) );

		a_ += ( SVL );
		b_ += ( 4 * SVL );
	}


// EPILOGUE
    float beta_ = *(float *)beta;
    float alpha_ = *(float *)alpha;

    svbool_t pg = svptrue_b32();
    svcount_t pg_c = svptrue_c32();

    // Predicates for edge masking
    svbool_t pg_N = svwhilelt_b32_u64(0, (uint64_t)m);
    svbool_t pg_M0 = svwhilelt_b32_u64(0, (uint64_t)n);
    svbool_t pg_M1 = svwhilelt_b32_u64(SVL, (uint64_t)n);
    svbool_t pg_M2 = svwhilelt_b32_u64(2 * SVL, (uint64_t)n);
    svbool_t pg_M3 = svwhilelt_b32_u64(3 * SVL, (uint64_t)n);

    if ( rs_c == 1 ) 
    {
        // ==========================================
        // C is ROW-MAJOR (or general stride)
        // ==========================================
        if (n == 4 * SVL && m == SVL) 
        {
            // FAST PATH
            float *c_ptr_0 = c_ + 0 * SVL * cs_c;
            float *c_ptr_1 = c_ + 1 * SVL * cs_c;
            float *c_ptr_2 = c_ + 2 * SVL * cs_c;
            float *c_ptr_3 = c_ + 3 * SVL * cs_c;

            if (beta_ == 0 )
            {
                for ( uint64_t tcol = 0; tcol < SVL; tcol += 4 )
                {
                    // Read 4 columns at once out of each ZA tile, mapping them to rows of C
                    svfloat32x4_t zq0 = svread_ver_za32_f32_vg4( 0, tcol );
                    svfloat32x4_t zq1 = svread_ver_za32_f32_vg4( 1, tcol );
                    svfloat32x4_t zq2 = svread_ver_za32_f32_vg4( 2, tcol );
                    svfloat32x4_t zq3 = svread_ver_za32_f32_vg4( 3, tcol );

                    // Row 0..3 of Tile 0
                    svst1_f32( pg, c_ptr_0 + 0 * cs_c, svmul_n_f32_x( pg, svget4(zq0, 0), alpha_ ) );
                    svst1_f32( pg, c_ptr_0 + 1 * cs_c, svmul_n_f32_x( pg, svget4(zq0, 1), alpha_ ) );
                    svst1_f32( pg, c_ptr_0 + 2 * cs_c, svmul_n_f32_x( pg, svget4(zq0, 2), alpha_ ) );
                    svst1_f32( pg, c_ptr_0 + 3 * cs_c, svmul_n_f32_x( pg, svget4(zq0, 3), alpha_ ) );

                    // Row 0..3 of Tile 1
                    svst1_f32( pg, c_ptr_1 + 0 * cs_c, svmul_n_f32_x( pg, svget4(zq1, 0), alpha_ ) );
                    svst1_f32( pg, c_ptr_1 + 1 * cs_c, svmul_n_f32_x( pg, svget4(zq1, 1), alpha_ ) );
                    svst1_f32( pg, c_ptr_1 + 2 * cs_c, svmul_n_f32_x( pg, svget4(zq1, 2), alpha_ ) );
                    svst1_f32( pg, c_ptr_1 + 3 * cs_c, svmul_n_f32_x( pg, svget4(zq1, 3), alpha_ ) );

                    // Row 0..3 of Tile 2
                    svst1_f32( pg, c_ptr_2 + 0 * cs_c, svmul_n_f32_x( pg, svget4(zq2, 0), alpha_ ) );
                    svst1_f32( pg, c_ptr_2 + 1 * cs_c, svmul_n_f32_x( pg, svget4(zq2, 1), alpha_ ) );
                    svst1_f32( pg, c_ptr_2 + 2 * cs_c, svmul_n_f32_x( pg, svget4(zq2, 2), alpha_ ) );
                    svst1_f32( pg, c_ptr_2 + 3 * cs_c, svmul_n_f32_x( pg, svget4(zq2, 3), alpha_ ) );

                    // Row 0..3 of Tile 3
                    svst1_f32( pg, c_ptr_3 + 0 * cs_c, svmul_n_f32_x( pg, svget4(zq3, 0), alpha_ ) );
                    svst1_f32( pg, c_ptr_3 + 1 * cs_c, svmul_n_f32_x( pg, svget4(zq3, 1), alpha_ ) );
                    svst1_f32( pg, c_ptr_3 + 2 * cs_c, svmul_n_f32_x( pg, svget4(zq3, 2), alpha_ ) );
                    svst1_f32( pg, c_ptr_3 + 3 * cs_c, svmul_n_f32_x( pg, svget4(zq3, 3), alpha_ ) );
                    
                    c_ptr_0 += 4 * cs_c;
                    c_ptr_1 += 4 * cs_c;
                    c_ptr_2 += 4 * cs_c;
                    c_ptr_3 += 4 * cs_c;
                }
            }
            else {
                for ( uint64_t tcol = 0; tcol < SVL; tcol += 4 )
                {
                    svfloat32x4_t zq0 = svread_ver_za32_f32_vg4( 0, tcol );
                    svfloat32x4_t zq1 = svread_ver_za32_f32_vg4( 1, tcol );
                    svfloat32x4_t zq2 = svread_ver_za32_f32_vg4( 2, tcol );
                    svfloat32x4_t zq3 = svread_ver_za32_f32_vg4( 3, tcol );

                    // Tile 0
                    {
                        svfloat32_t z0 = svmul_n_f32_x( pg, svget4(zq0, 0), alpha_ );
                        svfloat32_t z1 = svmul_n_f32_x( pg, svget4(zq0, 1), alpha_ );
                        svfloat32_t z2 = svmul_n_f32_x( pg, svget4(zq0, 2), alpha_ );
                        svfloat32_t z3 = svmul_n_f32_x( pg, svget4(zq0, 3), alpha_ );
                        z0 = svmla_n_f32_x( pg, z0, svld1_f32( pg, c_ptr_0 + 0 * cs_c ), beta_ );
                        z1 = svmla_n_f32_x( pg, z1, svld1_f32( pg, c_ptr_0 + 1 * cs_c ), beta_ );
                        z2 = svmla_n_f32_x( pg, z2, svld1_f32( pg, c_ptr_0 + 2 * cs_c ), beta_ );
                        z3 = svmla_n_f32_x( pg, z3, svld1_f32( pg, c_ptr_0 + 3 * cs_c ), beta_ );
                        svst1_f32( pg, c_ptr_0 + 0 * cs_c, z0 );
                        svst1_f32( pg, c_ptr_0 + 1 * cs_c, z1 );
                        svst1_f32( pg, c_ptr_0 + 2 * cs_c, z2 );
                        svst1_f32( pg, c_ptr_0 + 3 * cs_c, z3 );
                    }
                    // Tile 1
                    {
                        svfloat32_t z0 = svmul_n_f32_x( pg, svget4(zq1, 0), alpha_ );
                        svfloat32_t z1 = svmul_n_f32_x( pg, svget4(zq1, 1), alpha_ );
                        svfloat32_t z2 = svmul_n_f32_x( pg, svget4(zq1, 2), alpha_ );
                        svfloat32_t z3 = svmul_n_f32_x( pg, svget4(zq1, 3), alpha_ );
                        z0 = svmla_n_f32_x( pg, z0, svld1_f32( pg, c_ptr_1 + 0 * cs_c ), beta_ );
                        z1 = svmla_n_f32_x( pg, z1, svld1_f32( pg, c_ptr_1 + 1 * cs_c ), beta_ );
                        z2 = svmla_n_f32_x( pg, z2, svld1_f32( pg, c_ptr_1 + 2 * cs_c ), beta_ );
                        z3 = svmla_n_f32_x( pg, z3, svld1_f32( pg, c_ptr_1 + 3 * cs_c ), beta_ );
                        svst1_f32( pg, c_ptr_1 + 0 * cs_c, z0 );
                        svst1_f32( pg, c_ptr_1 + 1 * cs_c, z1 );
                        svst1_f32( pg, c_ptr_1 + 2 * cs_c, z2 );
                        svst1_f32( pg, c_ptr_1 + 3 * cs_c, z3 );
                    }
                    // Tile 2
                    {
                        svfloat32_t z0 = svmul_n_f32_x( pg, svget4(zq2, 0), alpha_ );
                        svfloat32_t z1 = svmul_n_f32_x( pg, svget4(zq2, 1), alpha_ );
                        svfloat32_t z2 = svmul_n_f32_x( pg, svget4(zq2, 2), alpha_ );
                        svfloat32_t z3 = svmul_n_f32_x( pg, svget4(zq2, 3), alpha_ );
                        z0 = svmla_n_f32_x( pg, z0, svld1_f32( pg, c_ptr_2 + 0 * cs_c ), beta_ );
                        z1 = svmla_n_f32_x( pg, z1, svld1_f32( pg, c_ptr_2 + 1 * cs_c ), beta_ );
                        z2 = svmla_n_f32_x( pg, z2, svld1_f32( pg, c_ptr_2 + 2 * cs_c ), beta_ );
                        z3 = svmla_n_f32_x( pg, z3, svld1_f32( pg, c_ptr_2 + 3 * cs_c ), beta_ );
                        svst1_f32( pg, c_ptr_2 + 0 * cs_c, z0 );
                        svst1_f32( pg, c_ptr_2 + 1 * cs_c, z1 );
                        svst1_f32( pg, c_ptr_2 + 2 * cs_c, z2 );
                        svst1_f32( pg, c_ptr_2 + 3 * cs_c, z3 );
                    }
                    // Tile 3
                    {
                        svfloat32_t z0 = svmul_n_f32_x( pg, svget4(zq3, 0), alpha_ );
                        svfloat32_t z1 = svmul_n_f32_x( pg, svget4(zq3, 1), alpha_ );
                        svfloat32_t z2 = svmul_n_f32_x( pg, svget4(zq3, 2), alpha_ );
                        svfloat32_t z3 = svmul_n_f32_x( pg, svget4(zq3, 3), alpha_ );
                        z0 = svmla_n_f32_x( pg, z0, svld1_f32( pg, c_ptr_3 + 0 * cs_c ), beta_ );
                        z1 = svmla_n_f32_x( pg, z1, svld1_f32( pg, c_ptr_3 + 1 * cs_c ), beta_ );
                        z2 = svmla_n_f32_x( pg, z2, svld1_f32( pg, c_ptr_3 + 2 * cs_c ), beta_ );
                        z3 = svmla_n_f32_x( pg, z3, svld1_f32( pg, c_ptr_3 + 3 * cs_c ), beta_ );
                        svst1_f32( pg, c_ptr_3 + 0 * cs_c, z0 );
                        svst1_f32( pg, c_ptr_3 + 1 * cs_c, z1 );
                        svst1_f32( pg, c_ptr_3 + 2 * cs_c, z2 );
                        svst1_f32( pg, c_ptr_3 + 3 * cs_c, z3 );
                    }
                    
                    c_ptr_0 += 4 * cs_c;
                    c_ptr_1 += 4 * cs_c;
                    c_ptr_2 += 4 * cs_c;
                    c_ptr_3 += 4 * cs_c;
                }
            }
        }
        else 
        {
            // EDGE PATH
            float *c_ptr_0 = c_ + 0 * SVL * cs_c;
            float *c_ptr_1 = c_ + 1 * SVL * cs_c;
            float *c_ptr_2 = c_ + 2 * SVL * cs_c;
            float *c_ptr_3 = c_ + 3 * SVL * cs_c;

            if (beta_ == 0) {
                for ( uint64_t tcol = 0; tcol < SVL; tcol += 4 ) {
                    svfloat32x4_t zq0 = svread_ver_za32_f32_vg4( 0, tcol );
                    
                    if (tcol + 0 < n) svst1_f32( pg_N, c_ptr_0 + 0 * cs_c, svmul_n_f32_x( pg_N, svget4(zq0, 0), alpha_ ) );
                    if (tcol + 1 < n) svst1_f32( pg_N, c_ptr_0 + 1 * cs_c, svmul_n_f32_x( pg_N, svget4(zq0, 1), alpha_ ) );
                    if (tcol + 2 < n) svst1_f32( pg_N, c_ptr_0 + 2 * cs_c, svmul_n_f32_x( pg_N, svget4(zq0, 2), alpha_ ) );
                    if (tcol + 3 < n) svst1_f32( pg_N, c_ptr_0 + 3 * cs_c, svmul_n_f32_x( pg_N, svget4(zq0, 3), alpha_ ) );

                    if (n > SVL) {
                        svfloat32x4_t zq1 = svread_ver_za32_f32_vg4( 1, tcol );
                        if (tcol + 0 + SVL < n) svst1_f32( pg_N, c_ptr_1 + 0 * cs_c, svmul_n_f32_x( pg_N, svget4(zq1, 0), alpha_ ) );
                        if (tcol + 1 + SVL < n) svst1_f32( pg_N, c_ptr_1 + 1 * cs_c, svmul_n_f32_x( pg_N, svget4(zq1, 1), alpha_ ) );
                        if (tcol + 2 + SVL < n) svst1_f32( pg_N, c_ptr_1 + 2 * cs_c, svmul_n_f32_x( pg_N, svget4(zq1, 2), alpha_ ) );
                        if (tcol + 3 + SVL < n) svst1_f32( pg_N, c_ptr_1 + 3 * cs_c, svmul_n_f32_x( pg_N, svget4(zq1, 3), alpha_ ) );
                    }
                    if (n > 2 * SVL) {
                        svfloat32x4_t zq2 = svread_ver_za32_f32_vg4( 2, tcol );
                        if (tcol + 0 + 2*SVL < n) svst1_f32( pg_N, c_ptr_2 + 0 * cs_c, svmul_n_f32_x( pg_N, svget4(zq2, 0), alpha_ ) );
                        if (tcol + 1 + 2*SVL < n) svst1_f32( pg_N, c_ptr_2 + 1 * cs_c, svmul_n_f32_x( pg_N, svget4(zq2, 1), alpha_ ) );
                        if (tcol + 2 + 2*SVL < n) svst1_f32( pg_N, c_ptr_2 + 2 * cs_c, svmul_n_f32_x( pg_N, svget4(zq2, 2), alpha_ ) );
                        if (tcol + 3 + 2*SVL < n) svst1_f32( pg_N, c_ptr_2 + 3 * cs_c, svmul_n_f32_x( pg_N, svget4(zq2, 3), alpha_ ) );
                    }
                    if (n > 3 * SVL) {
                        svfloat32x4_t zq3 = svread_ver_za32_f32_vg4( 3, tcol );
                        if (tcol + 0 + 3*SVL < n) svst1_f32( pg_N, c_ptr_3 + 0 * cs_c, svmul_n_f32_x( pg_N, svget4(zq3, 0), alpha_ ) );
                        if (tcol + 1 + 3*SVL < n) svst1_f32( pg_N, c_ptr_3 + 1 * cs_c, svmul_n_f32_x( pg_N, svget4(zq3, 1), alpha_ ) );
                        if (tcol + 2 + 3*SVL < n) svst1_f32( pg_N, c_ptr_3 + 2 * cs_c, svmul_n_f32_x( pg_N, svget4(zq3, 2), alpha_ ) );
                        if (tcol + 3 + 3*SVL < n) svst1_f32( pg_N, c_ptr_3 + 3 * cs_c, svmul_n_f32_x( pg_N, svget4(zq3, 3), alpha_ ) );
                    }

                    c_ptr_0 += 4 * cs_c;
                    c_ptr_1 += 4 * cs_c;
                    c_ptr_2 += 4 * cs_c;
                    c_ptr_3 += 4 * cs_c;
                }
            }
            else {
                for ( uint64_t tcol = 0; tcol < SVL; tcol += 4 ) {
                    svfloat32x4_t zq0 = svread_ver_za32_f32_vg4( 0, tcol );
                    
                    if (tcol + 0 < n) { svfloat32_t z0 = svmul_n_f32_x( pg_N, svget4(zq0, 0), alpha_ ); z0 = svmla_n_f32_x( pg_N, z0, svld1_f32( pg_N, c_ptr_0 + 0 * cs_c ), beta_ ); svst1_f32( pg_N, c_ptr_0 + 0 * cs_c, z0 ); }
                    if (tcol + 1 < n) { svfloat32_t z0 = svmul_n_f32_x( pg_N, svget4(zq0, 1), alpha_ ); z0 = svmla_n_f32_x( pg_N, z0, svld1_f32( pg_N, c_ptr_0 + 1 * cs_c ), beta_ ); svst1_f32( pg_N, c_ptr_0 + 1 * cs_c, z0 ); }
                    if (tcol + 2 < n) { svfloat32_t z0 = svmul_n_f32_x( pg_N, svget4(zq0, 2), alpha_ ); z0 = svmla_n_f32_x( pg_N, z0, svld1_f32( pg_N, c_ptr_0 + 2 * cs_c ), beta_ ); svst1_f32( pg_N, c_ptr_0 + 2 * cs_c, z0 ); }
                    if (tcol + 3 < n) { svfloat32_t z0 = svmul_n_f32_x( pg_N, svget4(zq0, 3), alpha_ ); z0 = svmla_n_f32_x( pg_N, z0, svld1_f32( pg_N, c_ptr_0 + 3 * cs_c ), beta_ ); svst1_f32( pg_N, c_ptr_0 + 3 * cs_c, z0 ); }

                    if (n > SVL) {
                        svfloat32x4_t zq1 = svread_ver_za32_f32_vg4( 1, tcol );
                        if (tcol + 0 + SVL < n) { svfloat32_t z1 = svmul_n_f32_x( pg_N, svget4(zq1, 0), alpha_ ); z1 = svmla_n_f32_x( pg_N, z1, svld1_f32( pg_N, c_ptr_1 + 0 * cs_c ), beta_ ); svst1_f32( pg_N, c_ptr_1 + 0 * cs_c, z1 ); }
                        if (tcol + 1 + SVL < n) { svfloat32_t z1 = svmul_n_f32_x( pg_N, svget4(zq1, 1), alpha_ ); z1 = svmla_n_f32_x( pg_N, z1, svld1_f32( pg_N, c_ptr_1 + 1 * cs_c ), beta_ ); svst1_f32( pg_N, c_ptr_1 + 1 * cs_c, z1 ); }
                        if (tcol + 2 + SVL < n) { svfloat32_t z1 = svmul_n_f32_x( pg_N, svget4(zq1, 2), alpha_ ); z1 = svmla_n_f32_x( pg_N, z1, svld1_f32( pg_N, c_ptr_1 + 2 * cs_c ), beta_ ); svst1_f32( pg_N, c_ptr_1 + 2 * cs_c, z1 ); }
                        if (tcol + 3 + SVL < n) { svfloat32_t z1 = svmul_n_f32_x( pg_N, svget4(zq1, 3), alpha_ ); z1 = svmla_n_f32_x( pg_N, z1, svld1_f32( pg_N, c_ptr_1 + 3 * cs_c ), beta_ ); svst1_f32( pg_N, c_ptr_1 + 3 * cs_c, z1 ); }
                    }
                    if (n > 2 * SVL) {
                        svfloat32x4_t zq2 = svread_ver_za32_f32_vg4( 2, tcol );
                        if (tcol + 0 + 2*SVL < n) { svfloat32_t z2 = svmul_n_f32_x( pg_N, svget4(zq2, 0), alpha_ ); z2 = svmla_n_f32_x( pg_N, z2, svld1_f32( pg_N, c_ptr_2 + 0 * cs_c ), beta_ ); svst1_f32( pg_N, c_ptr_2 + 0 * cs_c, z2 ); }
                        if (tcol + 1 + 2*SVL < n) { svfloat32_t z2 = svmul_n_f32_x( pg_N, svget4(zq2, 1), alpha_ ); z2 = svmla_n_f32_x( pg_N, z2, svld1_f32( pg_N, c_ptr_2 + 1 * cs_c ), beta_ ); svst1_f32( pg_N, c_ptr_2 + 1 * cs_c, z2 ); }
                        if (tcol + 2 + 2*SVL < n) { svfloat32_t z2 = svmul_n_f32_x( pg_N, svget4(zq2, 2), alpha_ ); z2 = svmla_n_f32_x( pg_N, z2, svld1_f32( pg_N, c_ptr_2 + 2 * cs_c ), beta_ ); svst1_f32( pg_N, c_ptr_2 + 2 * cs_c, z2 ); }
                        if (tcol + 3 + 2*SVL < n) { svfloat32_t z2 = svmul_n_f32_x( pg_N, svget4(zq2, 3), alpha_ ); z2 = svmla_n_f32_x( pg_N, z2, svld1_f32( pg_N, c_ptr_2 + 3 * cs_c ), beta_ ); svst1_f32( pg_N, c_ptr_2 + 3 * cs_c, z2 ); }
                    }
                    if (n > 3 * SVL) {
                        svfloat32x4_t zq3 = svread_ver_za32_f32_vg4( 3, tcol );
                        if (tcol + 0 + 3*SVL < n) { svfloat32_t z3 = svmul_n_f32_x( pg_N, svget4(zq3, 0), alpha_ ); z3 = svmla_n_f32_x( pg_N, z3, svld1_f32( pg_N, c_ptr_3 + 0 * cs_c ), beta_ ); svst1_f32( pg_N, c_ptr_3 + 0 * cs_c, z3 ); }
                        if (tcol + 1 + 3*SVL < n) { svfloat32_t z3 = svmul_n_f32_x( pg_N, svget4(zq3, 1), alpha_ ); z3 = svmla_n_f32_x( pg_N, z3, svld1_f32( pg_N, c_ptr_3 + 1 * cs_c ), beta_ ); svst1_f32( pg_N, c_ptr_3 + 1 * cs_c, z3 ); }
                        if (tcol + 2 + 3*SVL < n) { svfloat32_t z3 = svmul_n_f32_x( pg_N, svget4(zq3, 2), alpha_ ); z3 = svmla_n_f32_x( pg_N, z3, svld1_f32( pg_N, c_ptr_3 + 2 * cs_c ), beta_ ); svst1_f32( pg_N, c_ptr_3 + 2 * cs_c, z3 ); }
                        if (tcol + 3 + 3*SVL < n) { svfloat32_t z3 = svmul_n_f32_x( pg_N, svget4(zq3, 3), alpha_ ); z3 = svmla_n_f32_x( pg_N, z3, svld1_f32( pg_N, c_ptr_3 + 3 * cs_c ), beta_ ); svst1_f32( pg_N, c_ptr_3 + 3 * cs_c, z3 ); }
                    }

                    c_ptr_0 += 4 * cs_c;
                    c_ptr_1 += 4 * cs_c;
                    c_ptr_2 += 4 * cs_c;
                    c_ptr_3 += 4 * cs_c;
                }
            }
        }
    }
    else 
    {
        // ==========================================
        // C is COL-MAJOR
        // ==========================================
        if (n == 4 * SVL && m == SVL) 
        {
            // FAST PATH
            float *c_ptr = c_;

            if (beta_ == 0 )
            {
                for ( uint64_t tcol = 0; tcol < SVL; tcol += 4 )
                {
                    // Read 4 rows at once out of each ZA tile, mapping them to columns of C
                    svfloat32x4_t zq0 = svread_hor_za32_f32_vg4( 0, tcol );
                    svfloat32x4_t zq1 = svread_hor_za32_f32_vg4( 1, tcol );
                    svfloat32x4_t zq2 = svread_hor_za32_f32_vg4( 2, tcol );
                    svfloat32x4_t zq3 = svread_hor_za32_f32_vg4( 3, tcol );

                    svfloat32x4_t z40 = svcreate4( svmul_n_f32_x( pg, svget4(zq0, 0), alpha_ ), svmul_n_f32_x( pg, svget4(zq1, 0), alpha_ ), svmul_n_f32_x( pg, svget4(zq2, 0), alpha_ ), svmul_n_f32_x( pg, svget4(zq3, 0), alpha_ ) );
                    svst1_f32_x4( pg_c, c_ptr + 0 * rs_c, z40 );

                    svfloat32x4_t z41 = svcreate4( svmul_n_f32_x( pg, svget4(zq0, 1), alpha_ ), svmul_n_f32_x( pg, svget4(zq1, 1), alpha_ ), svmul_n_f32_x( pg, svget4(zq2, 1), alpha_ ), svmul_n_f32_x( pg, svget4(zq3, 1), alpha_ ) );
                    svst1_f32_x4( pg_c, c_ptr + 1 * rs_c, z41 );

                    svfloat32x4_t z42 = svcreate4( svmul_n_f32_x( pg, svget4(zq0, 2), alpha_ ), svmul_n_f32_x( pg, svget4(zq1, 2), alpha_ ), svmul_n_f32_x( pg, svget4(zq2, 2), alpha_ ), svmul_n_f32_x( pg, svget4(zq3, 2), alpha_ ) );
                    svst1_f32_x4( pg_c, c_ptr + 2 * rs_c, z42 );

                    svfloat32x4_t z43 = svcreate4( svmul_n_f32_x( pg, svget4(zq0, 3), alpha_ ), svmul_n_f32_x( pg, svget4(zq1, 3), alpha_ ), svmul_n_f32_x( pg, svget4(zq2, 3), alpha_ ), svmul_n_f32_x( pg, svget4(zq3, 3), alpha_ ) );
                    svst1_f32_x4( pg_c, c_ptr + 3 * rs_c, z43 );

                    c_ptr += 4 * rs_c;
                }
            }
            else {
                for ( uint64_t tcol = 0; tcol < SVL; tcol += 4 )
                {
                    svfloat32x4_t zq0 = svread_hor_za32_f32_vg4( 0, tcol );
                    svfloat32x4_t zq1 = svread_hor_za32_f32_vg4( 1, tcol );
                    svfloat32x4_t zq2 = svread_hor_za32_f32_vg4( 2, tcol );
                    svfloat32x4_t zq3 = svread_hor_za32_f32_vg4( 3, tcol );

                    // col tcol + 0
                    svfloat32x4_t z_c0 = svld1_f32_x4( pg_c, c_ptr + 0 * rs_c );
                    svfloat32_t z0_0 = svmla_n_f32_x( pg, svmul_n_f32_x( pg, svget4(zq0, 0), alpha_ ), svget4(z_c0, 0), beta_ );
                    svfloat32_t z1_0 = svmla_n_f32_x( pg, svmul_n_f32_x( pg, svget4(zq1, 0), alpha_ ), svget4(z_c0, 1), beta_ );
                    svfloat32_t z2_0 = svmla_n_f32_x( pg, svmul_n_f32_x( pg, svget4(zq2, 0), alpha_ ), svget4(z_c0, 2), beta_ );
                    svfloat32_t z3_0 = svmla_n_f32_x( pg, svmul_n_f32_x( pg, svget4(zq3, 0), alpha_ ), svget4(z_c0, 3), beta_ );
                    svst1_f32_x4( pg_c, c_ptr + 0 * rs_c, svcreate4(z0_0, z1_0, z2_0, z3_0) );

                    // col tcol + 1
                    svfloat32x4_t z_c1 = svld1_f32_x4( pg_c, c_ptr + 1 * rs_c );
                    svfloat32_t z0_1 = svmla_n_f32_x( pg, svmul_n_f32_x( pg, svget4(zq0, 1), alpha_ ), svget4(z_c1, 0), beta_ );
                    svfloat32_t z1_1 = svmla_n_f32_x( pg, svmul_n_f32_x( pg, svget4(zq1, 1), alpha_ ), svget4(z_c1, 1), beta_ );
                    svfloat32_t z2_1 = svmla_n_f32_x( pg, svmul_n_f32_x( pg, svget4(zq2, 1), alpha_ ), svget4(z_c1, 2), beta_ );
                    svfloat32_t z3_1 = svmla_n_f32_x( pg, svmul_n_f32_x( pg, svget4(zq3, 1), alpha_ ), svget4(z_c1, 3), beta_ );
                    svst1_f32_x4( pg_c, c_ptr + 1 * rs_c, svcreate4(z0_1, z1_1, z2_1, z3_1) );

                    // col tcol + 2
                    svfloat32x4_t z_c2 = svld1_f32_x4( pg_c, c_ptr + 2 * rs_c );
                    svfloat32_t z0_2 = svmla_n_f32_x( pg, svmul_n_f32_x( pg, svget4(zq0, 2), alpha_ ), svget4(z_c2, 0), beta_ );
                    svfloat32_t z1_2 = svmla_n_f32_x( pg, svmul_n_f32_x( pg, svget4(zq1, 2), alpha_ ), svget4(z_c2, 1), beta_ );
                    svfloat32_t z2_2 = svmla_n_f32_x( pg, svmul_n_f32_x( pg, svget4(zq2, 2), alpha_ ), svget4(z_c2, 2), beta_ );
                    svfloat32_t z3_2 = svmla_n_f32_x( pg, svmul_n_f32_x( pg, svget4(zq3, 2), alpha_ ), svget4(z_c2, 3), beta_ );
                    svst1_f32_x4( pg_c, c_ptr + 2 * rs_c, svcreate4(z0_2, z1_2, z2_2, z3_2) );

                    // col tcol + 3
                    svfloat32x4_t z_c3 = svld1_f32_x4( pg_c, c_ptr + 3 * rs_c );
                    svfloat32_t z0_3 = svmla_n_f32_x( pg, svmul_n_f32_x( pg, svget4(zq0, 3), alpha_ ), svget4(z_c3, 0), beta_ );
                    svfloat32_t z1_3 = svmla_n_f32_x( pg, svmul_n_f32_x( pg, svget4(zq1, 3), alpha_ ), svget4(z_c3, 1), beta_ );
                    svfloat32_t z2_3 = svmla_n_f32_x( pg, svmul_n_f32_x( pg, svget4(zq2, 3), alpha_ ), svget4(z_c3, 2), beta_ );
                    svfloat32_t z3_3 = svmla_n_f32_x( pg, svmul_n_f32_x( pg, svget4(zq3, 3), alpha_ ), svget4(z_c3, 3), beta_ );
                    svst1_f32_x4( pg_c, c_ptr + 3 * rs_c, svcreate4(z0_3, z1_3, z2_3, z3_3) );

                    c_ptr += 4 * rs_c;
                }
            }
        }
        else 
        {
            // EDGE PATH
            float *c_ptr = c_;

            if (beta_ == 0) {
                for ( uint64_t tcol = 0; tcol < SVL; tcol += 4 ) {
                    if (tcol >= m) break;

                    svfloat32x4_t zq0 = svread_hor_za32_f32_vg4( 0, tcol );
                    svfloat32x4_t zq1, zq2, zq3;
                    
                    // Do not load underlying tiles if n strictly avoids them
                    if (n > SVL) zq1 = svread_hor_za32_f32_vg4( 1, tcol );
                    if (n > 2 * SVL) zq2 = svread_hor_za32_f32_vg4( 2, tcol );
                    if (n > 3 * SVL) zq3 = svread_hor_za32_f32_vg4( 3, tcol );

                    if (tcol + 0 < m) {
                        svst1_f32( pg_M0, c_ptr + 0 * rs_c + 0 * SVL, svmul_n_f32_x( pg_M0, svget4(zq0, 0), alpha_ ) );
                        if (n > SVL) svst1_f32( pg_M1, c_ptr + 0 * rs_c + 1 * SVL, svmul_n_f32_x( pg_M1, svget4(zq1, 0), alpha_ ) );
                        if (n > 2 * SVL) svst1_f32( pg_M2, c_ptr + 0 * rs_c + 2 * SVL, svmul_n_f32_x( pg_M2, svget4(zq2, 0), alpha_ ) );
                        if (n > 3 * SVL) svst1_f32( pg_M3, c_ptr + 0 * rs_c + 3 * SVL, svmul_n_f32_x( pg_M3, svget4(zq3, 0), alpha_ ) );
                    }
                    if (tcol + 1 < m) {
                        svst1_f32( pg_M0, c_ptr + 1 * rs_c + 0 * SVL, svmul_n_f32_x( pg_M0, svget4(zq0, 1), alpha_ ) );
                        if (n > SVL) svst1_f32( pg_M1, c_ptr + 1 * rs_c + 1 * SVL, svmul_n_f32_x( pg_M1, svget4(zq1, 1), alpha_ ) );
                        if (n > 2 * SVL) svst1_f32( pg_M2, c_ptr + 1 * rs_c + 2 * SVL, svmul_n_f32_x( pg_M2, svget4(zq2, 1), alpha_ ) );
                        if (n > 3 * SVL) svst1_f32( pg_M3, c_ptr + 1 * rs_c + 3 * SVL, svmul_n_f32_x( pg_M3, svget4(zq3, 1), alpha_ ) );
                    }
                    if (tcol + 2 < m) {
                        svst1_f32( pg_M0, c_ptr + 2 * rs_c + 0 * SVL, svmul_n_f32_x( pg_M0, svget4(zq0, 2), alpha_ ) );
                        if (n > SVL) svst1_f32( pg_M1, c_ptr + 2 * rs_c + 1 * SVL, svmul_n_f32_x( pg_M1, svget4(zq1, 2), alpha_ ) );
                        if (n > 2 * SVL) svst1_f32( pg_M2, c_ptr + 2 * rs_c + 2 * SVL, svmul_n_f32_x( pg_M2, svget4(zq2, 2), alpha_ ) );
                        if (n > 3 * SVL) svst1_f32( pg_M3, c_ptr + 2 * rs_c + 3 * SVL, svmul_n_f32_x( pg_M3, svget4(zq3, 2), alpha_ ) );
                    }
                    if (tcol + 3 < m) {
                        svst1_f32( pg_M0, c_ptr + 3 * rs_c + 0 * SVL, svmul_n_f32_x( pg_M0, svget4(zq0, 3), alpha_ ) );
                        if (n > SVL) svst1_f32( pg_M1, c_ptr + 3 * rs_c + 1 * SVL, svmul_n_f32_x( pg_M1, svget4(zq1, 3), alpha_ ) );
                        if (n > 2 * SVL) svst1_f32( pg_M2, c_ptr + 3 * rs_c + 2 * SVL, svmul_n_f32_x( pg_M2, svget4(zq2, 3), alpha_ ) );
                        if (n > 3 * SVL) svst1_f32( pg_M3, c_ptr + 3 * rs_c + 3 * SVL, svmul_n_f32_x( pg_M3, svget4(zq3, 3), alpha_ ) );
                    }

                    c_ptr += 4 * rs_c;
                }
            }
            else {
                for ( uint64_t tcol = 0; tcol < SVL; tcol += 4 ) {
                    if (tcol >= m) break;

                    svfloat32x4_t zq0 = svread_hor_za32_f32_vg4( 0, tcol );
                    svfloat32x4_t zq1, zq2, zq3;

                    if (n > SVL) zq1 = svread_hor_za32_f32_vg4( 1, tcol );
                    if (n > 2 * SVL) zq2 = svread_hor_za32_f32_vg4( 2, tcol );
                    if (n > 3 * SVL) zq3 = svread_hor_za32_f32_vg4( 3, tcol );

                    if (tcol + 0 < m) {
                        svfloat32_t z0 = svmul_n_f32_x( pg_M0, svget4(zq0, 0), alpha_ );
                        z0 = svmla_n_f32_x( pg_M0, z0, svld1_f32( pg_M0, c_ptr + 0 * rs_c + 0 * SVL ), beta_ );
                        svst1_f32( pg_M0, c_ptr + 0 * rs_c + 0 * SVL, z0 );

                        if (n > SVL) {
                            svfloat32_t z1 = svmul_n_f32_x( pg_M1, svget4(zq1, 0), alpha_ );
                            z1 = svmla_n_f32_x( pg_M1, z1, svld1_f32( pg_M1, c_ptr + 0 * rs_c + 1 * SVL ), beta_ );
                            svst1_f32( pg_M1, c_ptr + 0 * rs_c + 1 * SVL, z1 );
                        }
                        if (n > 2 * SVL) {
                            svfloat32_t z2 = svmul_n_f32_x( pg_M2, svget4(zq2, 0), alpha_ );
                            z2 = svmla_n_f32_x( pg_M2, z2, svld1_f32( pg_M2, c_ptr + 0 * rs_c + 2 * SVL ), beta_ );
                            svst1_f32( pg_M2, c_ptr + 0 * rs_c + 2 * SVL, z2 );
                        }
                        if (n > 3 * SVL) {
                            svfloat32_t z3 = svmul_n_f32_x( pg_M3, svget4(zq3, 0), alpha_ );
                            z3 = svmla_n_f32_x( pg_M3, z3, svld1_f32( pg_M3, c_ptr + 0 * rs_c + 3 * SVL ), beta_ );
                            svst1_f32( pg_M3, c_ptr + 0 * rs_c + 3 * SVL, z3 );
                        }
                    }
                    if (tcol + 1 < m) {
                        svfloat32_t z0 = svmul_n_f32_x( pg_M0, svget4(zq0, 1), alpha_ );
                        z0 = svmla_n_f32_x( pg_M0, z0, svld1_f32( pg_M0, c_ptr + 1 * rs_c + 0 * SVL ), beta_ );
                        svst1_f32( pg_M0, c_ptr + 1 * rs_c + 0 * SVL, z0 );

                        if (n > SVL) {
                            svfloat32_t z1 = svmul_n_f32_x( pg_M1, svget4(zq1, 1), alpha_ );
                            z1 = svmla_n_f32_x( pg_M1, z1, svld1_f32( pg_M1, c_ptr + 1 * rs_c + 1 * SVL ), beta_ );
                            svst1_f32( pg_M1, c_ptr + 1 * rs_c + 1 * SVL, z1 );
                        }
                        if (n > 2 * SVL) {
                            svfloat32_t z2 = svmul_n_f32_x( pg_M2, svget4(zq2, 1), alpha_ );
                            z2 = svmla_n_f32_x( pg_M2, z2, svld1_f32( pg_M2, c_ptr + 1 * rs_c + 2 * SVL ), beta_ );
                            svst1_f32( pg_M2, c_ptr + 1 * rs_c + 2 * SVL, z2 );
                        }
                        if (n > 3 * SVL) {
                            svfloat32_t z3 = svmul_n_f32_x( pg_M3, svget4(zq3, 1), alpha_ );
                            z3 = svmla_n_f32_x( pg_M3, z3, svld1_f32( pg_M3, c_ptr + 1 * rs_c + 3 * SVL ), beta_ );
                            svst1_f32( pg_M3, c_ptr + 1 * rs_c + 3 * SVL, z3 );
                        }
                    }
                    if (tcol + 2 < m) {
                        svfloat32_t z0 = svmul_n_f32_x( pg_M0, svget4(zq0, 2), alpha_ );
                        z0 = svmla_n_f32_x( pg_M0, z0, svld1_f32( pg_M0, c_ptr + 2 * rs_c + 0 * SVL ), beta_ );
                        svst1_f32( pg_M0, c_ptr + 2 * rs_c + 0 * SVL, z0 );

                        if (n > SVL) {
                            svfloat32_t z1 = svmul_n_f32_x( pg_M1, svget4(zq1, 2), alpha_ );
                            z1 = svmla_n_f32_x( pg_M1, z1, svld1_f32( pg_M1, c_ptr + 2 * rs_c + 1 * SVL ), beta_ );
                            svst1_f32( pg_M1, c_ptr + 2 * rs_c + 1 * SVL, z1 );
                        }
                        if (n > 2 * SVL) {
                            svfloat32_t z2 = svmul_n_f32_x( pg_M2, svget4(zq2, 2), alpha_ );
                            z2 = svmla_n_f32_x( pg_M2, z2, svld1_f32( pg_M2, c_ptr + 2 * rs_c + 2 * SVL ), beta_ );
                            svst1_f32( pg_M2, c_ptr + 2 * rs_c + 2 * SVL, z2 );
                        }
                        if (n > 3 * SVL) {
                            svfloat32_t z3 = svmul_n_f32_x( pg_M3, svget4(zq3, 2), alpha_ );
                            z3 = svmla_n_f32_x( pg_M3, z3, svld1_f32( pg_M3, c_ptr + 2 * rs_c + 3 * SVL ), beta_ );
                            svst1_f32( pg_M3, c_ptr + 2 * rs_c + 3 * SVL, z3 );
                        }
                    }
                    if (tcol + 3 < m) {
                        svfloat32_t z0 = svmul_n_f32_x( pg_M0, svget4(zq0, 3), alpha_ );
                        z0 = svmla_n_f32_x( pg_M0, z0, svld1_f32( pg_M0, c_ptr + 3 * rs_c + 0 * SVL ), beta_ );
                        svst1_f32( pg_M0, c_ptr + 3 * rs_c + 0 * SVL, z0 );

                        if (n > SVL) {
                            svfloat32_t z1 = svmul_n_f32_x( pg_M1, svget4(zq1, 3), alpha_ );
                            z1 = svmla_n_f32_x( pg_M1, z1, svld1_f32( pg_M1, c_ptr + 3 * rs_c + 1 * SVL ), beta_ );
                            svst1_f32( pg_M1, c_ptr + 3 * rs_c + 1 * SVL, z1 );
                        }
                        if (n > 2 * SVL) {
                            svfloat32_t z2 = svmul_n_f32_x( pg_M2, svget4(zq2, 3), alpha_ );
                            z2 = svmla_n_f32_x( pg_M2, z2, svld1_f32( pg_M2, c_ptr + 3 * rs_c + 2 * SVL ), beta_ );
                            svst1_f32( pg_M2, c_ptr + 3 * rs_c + 2 * SVL, z2 );
                        }
                        if (n > 3 * SVL) {
                            svfloat32_t z3 = svmul_n_f32_x( pg_M3, svget4(zq3, 3), alpha_ );
                            z3 = svmla_n_f32_x( pg_M3, z3, svld1_f32( pg_M3, c_ptr + 3 * rs_c + 3 * SVL ), beta_ );
                            svst1_f32( pg_M3, c_ptr + 3 * rs_c + 3 * SVL, z3 );
                        }
                    }

                    c_ptr += 4 * rs_c;
                }
            }
        }
    }

	// GEMM_UKR_FLUSH_CT( s );

	return;
}

__arm_new( "za" ) __arm_locally_streaming void bli_dgemm_armsme_int_SVLx8SVL
     (
             dim_t      m,
             dim_t      n,
             dim_t      k,
       const void*      alpha,
       const void*      a,
       const void*      b,
       const void*      beta,
             void*      c, inc_t rs_c, inc_t cs_c,
       const auxinfo_t* data,
       const cntx_t*    cntx
     )
{
	uint64_t SVL = svcntsd();

	GEMM_UKR_SETUP_CT_AMBI( d, SVL, 8 * SVL, false );

	double* a_ = (double*)a;
	double* b_ = (double*)b;
	double* c_ = (double*)c;

	svzero_za();

	uint64_t k_;
	uint64_t k_iter = k / 4;
	uint64_t k_left = k % 4;

	for ( k_ = 0; k_ < k_iter; k_++ )
	{
		// Loads
		svfloat64x4_t zL00 = svld1_f64_x4( svptrue_c32(),
			(float64_t*)( &b_[0] ) );
		svfloat64x4_t zL01 = svld1_f64_x4( svptrue_c32(),
			(float64_t*)( &b_[4 * SVL] ) );
		svfloat64_t zR00 = svld1_f64( svptrue_b32(), (float64_t*)( &a_[0] ) );

		svmopa_za64_m( 0, svptrue_b32(), svptrue_b32(), svget4( zL00, 0 ),
			zR00 );
		svmopa_za64_m( 1, svptrue_b32(), svptrue_b32(), svget4( zL00, 1 ),
			zR00 );

		svmopa_za64_m( 2, svptrue_b32(), svptrue_b32(), svget4( zL00, 2 ),
			zR00 );
		svmopa_za64_m( 3, svptrue_b32(), svptrue_b32(), svget4( zL00, 3 ),
			zR00 );

		svmopa_za64_m( 4, svptrue_b32(), svptrue_b32(), svget4( zL01, 0 ),
			zR00 );
		svmopa_za64_m( 5, svptrue_b32(), svptrue_b32(), svget4( zL01, 1 ),
			zR00 );

		svmopa_za64_m( 6, svptrue_b32(), svptrue_b32(), svget4( zL01, 2 ),
			zR00 );
		svmopa_za64_m( 7, svptrue_b32(), svptrue_b32(), svget4( zL01, 3 ),
			zR00 );

		svfloat64x4_t zL02 = svld1_f64_x4( svptrue_c32(),
			(float64_t*)( &b_[8 * SVL] ) );
		svfloat64x4_t zL03 = svld1_f64_x4( svptrue_c32(),
			(float64_t*)( &b_[12 * SVL] ) );
		svfloat64_t zR01 = svld1_f64( svptrue_b32(),
			(float64_t*)( &a_[1 * SVL] ) );

		svmopa_za64_m( 0, svptrue_b32(), svptrue_b32(), svget4( zL02, 0 ),
			zR01 );
		svmopa_za64_m( 1, svptrue_b32(), svptrue_b32(), svget4( zL02, 1 ),
			zR01 );

		svmopa_za64_m( 2, svptrue_b32(), svptrue_b32(), svget4( zL02, 2 ),
			zR01 );
		svmopa_za64_m( 3, svptrue_b32(), svptrue_b32(), svget4( zL02, 3 ),
			zR01 );

		svmopa_za64_m( 4, svptrue_b32(), svptrue_b32(), svget4( zL03, 0 ),
			zR01 );
		svmopa_za64_m( 5, svptrue_b32(), svptrue_b32(), svget4( zL03, 1 ),
			zR01 );

		svmopa_za64_m( 6, svptrue_b32(), svptrue_b32(), svget4( zL03, 2 ),
			zR01 );
		svmopa_za64_m( 7, svptrue_b32(), svptrue_b32(), svget4( zL03, 3 ),
			zR01 );

		svfloat64x4_t zL04 = svld1_f64_x4( svptrue_c32(),
			(float64_t*)( &b_[16 * SVL] ) );
		svfloat64x4_t zL05 = svld1_f64_x4( svptrue_c32(),
			(float64_t*)( &b_[20 * SVL] ) );
		svfloat64_t zR02 = svld1_f64( svptrue_b32(),
			(float64_t*)( &a_[2 * SVL] ) );

		svmopa_za64_m( 0, svptrue_b32(), svptrue_b32(), svget4( zL04, 0 ),
			zR02 );
		svmopa_za64_m( 1, svptrue_b32(), svptrue_b32(), svget4( zL04, 1 ),
			zR02 );

		svmopa_za64_m( 2, svptrue_b32(), svptrue_b32(), svget4( zL04, 2 ),
			zR02 );
		svmopa_za64_m( 3, svptrue_b32(), svptrue_b32(), svget4( zL04, 3 ),
			zR02 );

		svmopa_za64_m( 4, svptrue_b32(), svptrue_b32(), svget4( zL05, 0 ),
			zR02 );
		svmopa_za64_m( 5, svptrue_b32(), svptrue_b32(), svget4( zL05, 1 ),
			zR02 );

		svmopa_za64_m( 6, svptrue_b32(), svptrue_b32(), svget4( zL05, 2 ),
			zR02 );
		svmopa_za64_m( 7, svptrue_b32(), svptrue_b32(), svget4( zL05, 3 ),
			zR02 );

		svfloat64x4_t zL06 = svld1_f64_x4( svptrue_c32(),
			(float64_t*)( &b_[24 * SVL] ) );
		svfloat64x4_t zL07 = svld1_f64_x4( svptrue_c32(),
			(float64_t*)( &b_[28 * SVL] ) );
		svfloat64_t zR03 = svld1_f64( svptrue_b32(),
			(float64_t*)( &a_[3 * SVL] ) );

		svmopa_za64_m( 0, svptrue_b32(), svptrue_b32(), svget4( zL06, 0 ),
			zR03 );
		svmopa_za64_m( 1, svptrue_b32(), svptrue_b32(), svget4( zL06, 1 ),
			zR03 );

		svmopa_za64_m( 2, svptrue_b32(), svptrue_b32(), svget4( zL06, 2 ),
			zR03 );
		svmopa_za64_m( 3, svptrue_b32(), svptrue_b32(), svget4( zL06, 3 ),
			zR03 );

		svmopa_za64_m( 4, svptrue_b32(), svptrue_b32(), svget4( zL07, 0 ),
			zR03 );
		svmopa_za64_m( 5, svptrue_b32(), svptrue_b32(), svget4( zL07, 1 ),
			zR03 );

		svmopa_za64_m( 6, svptrue_b32(), svptrue_b32(), svget4( zL07, 2 ),
			zR03 );
		svmopa_za64_m( 7, svptrue_b32(), svptrue_b32(), svget4( zL07, 3 ),
			zR03 );

		a_ += ( 4 * SVL );
		b_ += ( 4 * 8 * SVL );
	}

	for ( k_ = 0; k_ < k_left; k_ += 1 )
	{
		svfloat64x4_t zL00 = svld1_f64_x4( svptrue_c32(),
			(float64_t*)( &b_[0] ) );
		svfloat64x4_t zL01 = svld1_f64_x4( svptrue_c32(),
			(float64_t*)( &b_[4 * SVL] ) );
		svfloat64_t zR00 = svld1_f64( svptrue_b32(), (float64_t*)( &a_[0] ) );

		svmopa_za64_m( 0, svptrue_b32(), svptrue_b32(), svget4( zL00, 0 ),
			zR00 );
		svmopa_za64_m( 1, svptrue_b32(), svptrue_b32(), svget4( zL00, 1 ),
			zR00 );

		svmopa_za64_m( 2, svptrue_b32(), svptrue_b32(), svget4( zL00, 2 ),
			zR00 );
		svmopa_za64_m( 3, svptrue_b32(), svptrue_b32(), svget4( zL00, 3 ),
			zR00 );

		svmopa_za64_m( 4, svptrue_b32(), svptrue_b32(), svget4( zL01, 0 ),
			zR00 );
		svmopa_za64_m( 5, svptrue_b32(), svptrue_b32(), svget4( zL01, 1 ),
			zR00 );

		svmopa_za64_m( 6, svptrue_b32(), svptrue_b32(), svget4( zL01, 2 ),
			zR00 );
		svmopa_za64_m( 7, svptrue_b32(), svptrue_b32(), svget4( zL01, 3 ),
			zR00 );

		a_ += ( SVL );
		b_ += ( 8 * SVL );
	}

	double beta_ = *(double*)beta;
	double alpha_ = *(double*)alpha;

	const uint64_t result_tile_TL_corner = 0;

	svfloat64_t zbeta = svdup_f64( beta_ );
	svfloat64_t zalpha = svdup_f64( alpha_ );

	if ( cs_c == 1 )
	{
		const uint64_t result_tile_TR_corner = 4 * SVL;

		if ( beta_ == 0 )
		{
			for ( uint64_t tcol = 0; tcol < SVL; tcol += 4 )
			{
				// Read ZA slices into Z regs
				svfloat64_t z0 = svread_ver_za64_m( z0, svptrue_b32(),
					/* tile: */ 0, /* slice: */ tcol + 0 );
				svfloat64_t z1 = svread_ver_za64_m( z1, svptrue_b32(),
					/* tile: */ 1, /* slice: */ tcol + 0 );
				svfloat64_t z2 = svread_ver_za64_m( z2, svptrue_b32(),
					/* tile: */ 2, /* slice: */ tcol + 0 );
				svfloat64_t z3 = svread_ver_za64_m( z3, svptrue_b32(),
					/* tile: */ 3, /* slice: */ tcol + 0 );
				svfloat64_t z4 = svread_ver_za64_m( z3, svptrue_b32(),
					/* tile: */ 4, /* slice: */ tcol + 0 );
				svfloat64_t z5 = svread_ver_za64_m( z3, svptrue_b32(),
					/* tile: */ 5, /* slice: */ tcol + 0 );
				svfloat64_t z6 = svread_ver_za64_m( z3, svptrue_b32(),
					/* tile: */ 6, /* slice: */ tcol + 0 );
				svfloat64_t z7 = svread_ver_za64_m( z3, svptrue_b32(),
					/* tile: */ 7, /* slice: */ tcol + 0 );

				// Scale Z regs by broadcast alpha
				z0 = svmul_f64_z( svptrue_b32(), z0, zalpha );
				z1 = svmul_f64_z( svptrue_b32(), z1, zalpha );
				z2 = svmul_f64_z( svptrue_b32(), z2, zalpha );
				z3 = svmul_f64_z( svptrue_b32(), z3, zalpha );
				z4 = svmul_f64_z( svptrue_b32(), z4, zalpha );
				z5 = svmul_f64_z( svptrue_b32(), z5, zalpha );
				z6 = svmul_f64_z( svptrue_b32(), z6, zalpha );
				z7 = svmul_f64_z( svptrue_b32(), z7, zalpha );

				// Store full result into C
				svfloat64x4_t z400 = svcreate4( z0, z1, z2, z3 );
				svst1_f64_x4( svptrue_c32(),
					&c_[result_tile_TL_corner + ( tcol + 0 ) * rs_c], z400 );

				svfloat64x4_t z600 = svcreate4( z4, z5, z6, z7 );
				svst1_f64_x4( svptrue_c32(),
					&c_[result_tile_TR_corner + ( tcol + 0 ) * rs_c], z600 );

				// tcol + 1
				z0 = svread_ver_za64_m( z0, svptrue_b32(),
					/* tile: */ 0, /* slice: */ tcol + 1 );
				z1 = svread_ver_za64_m( z1, svptrue_b32(),
					/* tile: */ 1, /* slice: */ tcol + 1 );
				z2 = svread_ver_za64_m( z2, svptrue_b32(),
					/* tile: */ 2, /* slice: */ tcol + 1 );
				z3 = svread_ver_za64_m( z3, svptrue_b32(),
					/* tile: */ 3, /* slice: */ tcol + 1 );
				z4 = svread_ver_za64_m( z3, svptrue_b32(),
					/* tile: */ 4, /* slice: */ tcol + 1 );
				z5 = svread_ver_za64_m( z3, svptrue_b32(),
					/* tile: */ 5, /* slice: */ tcol + 1 );
				z6 = svread_ver_za64_m( z3, svptrue_b32(),
					/* tile: */ 6, /* slice: */ tcol + 1 );
				z7 = svread_ver_za64_m( z3, svptrue_b32(),
					/* tile: */ 7, /* slice: */ tcol + 1 );

				z0 = svmul_f64_z( svptrue_b32(), z0, zalpha );
				z1 = svmul_f64_z( svptrue_b32(), z1, zalpha );
				z2 = svmul_f64_z( svptrue_b32(), z2, zalpha );
				z3 = svmul_f64_z( svptrue_b32(), z3, zalpha );
				z4 = svmul_f64_z( svptrue_b32(), z4, zalpha );
				z5 = svmul_f64_z( svptrue_b32(), z5, zalpha );
				z6 = svmul_f64_z( svptrue_b32(), z6, zalpha );
				z7 = svmul_f64_z( svptrue_b32(), z7, zalpha );

				z400 = svcreate4( z0, z1, z2, z3 );
				svst1_f64_x4( svptrue_c32(),
					&c_[result_tile_TL_corner + ( tcol + 1 ) * rs_c], z400 );

				z600 = svcreate4( z4, z5, z6, z7 );
				svst1_f64_x4( svptrue_c32(),
					&c_[result_tile_TR_corner + ( tcol + 1 ) * rs_c], z600 );

				// tcol + 2
				z0 = svread_ver_za64_m( z0, svptrue_b32(),
					/* tile: */ 0, /* slice: */ tcol + 2 );
				z1 = svread_ver_za64_m( z1, svptrue_b32(),
					/* tile: */ 1, /* slice: */ tcol + 2 );
				z2 = svread_ver_za64_m( z2, svptrue_b32(),
					/* tile: */ 2, /* slice: */ tcol + 2 );
				z3 = svread_ver_za64_m( z3, svptrue_b32(),
					/* tile: */ 3, /* slice: */ tcol + 2 );
				z4 = svread_ver_za64_m( z3, svptrue_b32(),
					/* tile: */ 4, /* slice: */ tcol + 2 );
				z5 = svread_ver_za64_m( z3, svptrue_b32(),
					/* tile: */ 5, /* slice: */ tcol + 2 );
				z6 = svread_ver_za64_m( z3, svptrue_b32(),
					/* tile: */ 6, /* slice: */ tcol + 2 );
				z7 = svread_ver_za64_m( z3, svptrue_b32(),
					/* tile: */ 7, /* slice: */ tcol + 2 );

				z0 = svmul_f64_z( svptrue_b32(), z0, zalpha );
				z1 = svmul_f64_z( svptrue_b32(), z1, zalpha );
				z2 = svmul_f64_z( svptrue_b32(), z2, zalpha );
				z3 = svmul_f64_z( svptrue_b32(), z3, zalpha );
				z4 = svmul_f64_z( svptrue_b32(), z4, zalpha );
				z5 = svmul_f64_z( svptrue_b32(), z5, zalpha );
				z6 = svmul_f64_z( svptrue_b32(), z6, zalpha );
				z7 = svmul_f64_z( svptrue_b32(), z7, zalpha );

				z400 = svcreate4( z0, z1, z2, z3 );
				svst1_f64_x4( svptrue_c32(),
					&c_[result_tile_TL_corner + ( tcol + 2 ) * rs_c], z400 );

				z600 = svcreate4( z4, z5, z6, z7 );
				svst1_f64_x4( svptrue_c32(),
					&c_[result_tile_TR_corner + ( tcol + 2 ) * rs_c], z600 );

				// tcol + 3
				z0 = svread_ver_za64_m( z0, svptrue_b32(),
					/* tile: */ 0, /* slice: */ tcol + 3 );
				z1 = svread_ver_za64_m( z1, svptrue_b32(),
					/* tile: */ 1, /* slice: */ tcol + 3 );
				z2 = svread_ver_za64_m( z2, svptrue_b32(),
					/* tile: */ 2, /* slice: */ tcol + 3 );
				z3 = svread_ver_za64_m( z3, svptrue_b32(),
					/* tile: */ 3, /* slice: */ tcol + 3 );
				z4 = svread_ver_za64_m( z3, svptrue_b32(),
					/* tile: */ 4, /* slice: */ tcol + 3 );
				z5 = svread_ver_za64_m( z3, svptrue_b32(),
					/* tile: */ 5, /* slice: */ tcol + 3 );
				z6 = svread_ver_za64_m( z3, svptrue_b32(),
					/* tile: */ 6, /* slice: */ tcol + 3 );
				z7 = svread_ver_za64_m( z3, svptrue_b32(),
					/* tile: */ 7, /* slice: */ tcol + 3 );

				z0 = svmul_f64_z( svptrue_b32(), z0, zalpha );
				z1 = svmul_f64_z( svptrue_b32(), z1, zalpha );
				z2 = svmul_f64_z( svptrue_b32(), z2, zalpha );
				z3 = svmul_f64_z( svptrue_b32(), z3, zalpha );
				z4 = svmul_f64_z( svptrue_b32(), z4, zalpha );
				z5 = svmul_f64_z( svptrue_b32(), z5, zalpha );
				z6 = svmul_f64_z( svptrue_b32(), z6, zalpha );
				z7 = svmul_f64_z( svptrue_b32(), z7, zalpha );

				z400 = svcreate4( z0, z1, z2, z3 );
				svst1_f64_x4( svptrue_c32(),
					&c_[result_tile_TL_corner + ( tcol + 3 ) * rs_c], z400 );

				z600 = svcreate4( z4, z5, z6, z7 );
				svst1_f64_x4( svptrue_c32(),
					&c_[result_tile_TR_corner + ( tcol + 3 ) * rs_c], z600 );
			}
		}
		else
		{
			for ( uint64_t tcol = 0; tcol < SVL; tcol += 4 )
			{
				// Read ZA slices into Z regs
				svfloat64_t z0 = svread_ver_za64_m( z0, svptrue_b32(),
					/* tile: */ 0, /* slice: */ tcol + 0 );
				svfloat64_t z1 = svread_ver_za64_m( z1, svptrue_b32(),
					/* tile: */ 1, /* slice: */ tcol + 0 );
				svfloat64_t z2 = svread_ver_za64_m( z2, svptrue_b32(),
					/* tile: */ 2, /* slice: */ tcol + 0 );
				svfloat64_t z3 = svread_ver_za64_m( z3, svptrue_b32(),
					/* tile: */ 3, /* slice: */ tcol + 0 );
				svfloat64_t z4 = svread_ver_za64_m( z3, svptrue_b32(),
					/* tile: */ 4, /* slice: */ tcol + 0 );
				svfloat64_t z5 = svread_ver_za64_m( z3, svptrue_b32(),
					/* tile: */ 5, /* slice: */ tcol + 0 );
				svfloat64_t z6 = svread_ver_za64_m( z3, svptrue_b32(),
					/* tile: */ 6, /* slice: */ tcol + 0 );
				svfloat64_t z7 = svread_ver_za64_m( z3, svptrue_b32(),
					/* tile: */ 7, /* slice: */ tcol + 0 );

				// Scale Z regs by broadcast alpha
				z0 = svmul_f64_z( svptrue_b32(), z0, zalpha );
				z1 = svmul_f64_z( svptrue_b32(), z1, zalpha );
				z2 = svmul_f64_z( svptrue_b32(), z2, zalpha );
				z3 = svmul_f64_z( svptrue_b32(), z3, zalpha );
				z4 = svmul_f64_z( svptrue_b32(), z4, zalpha );
				z5 = svmul_f64_z( svptrue_b32(), z5, zalpha );
				z6 = svmul_f64_z( svptrue_b32(), z6, zalpha );
				z7 = svmul_f64_z( svptrue_b32(), z7, zalpha );

				// Load C into Z regs

				svfloat64x4_t zq5 = svld1_f64_x4( svptrue_c32(),
					&c_[result_tile_TL_corner + ( ( ( tcol + 0 ) * rs_c ) )] );
				svfloat64x4_t zq6 = svld1_f64_x4( svptrue_c32(),
					&c_[result_tile_TR_corner + ( ( ( tcol + 0 ) * rs_c ) )] );

				// Scale Z regs by broadcast beta
				svfloat64_t z40 = svmla_m( svptrue_b32(), z0, svget4( zq5, 0 ),
					zbeta );
				svfloat64_t z50 = svmla_m( svptrue_b32(), z1, svget4( zq5, 1 ),
					zbeta );
				svfloat64_t z60 = svmla_m( svptrue_b32(), z2, svget4( zq5, 2 ),
					zbeta );
				svfloat64_t z70 = svmla_m( svptrue_b32(), z3, svget4( zq5, 3 ),
					zbeta );
				svfloat64_t z80 = svmla_m( svptrue_b32(), z4, svget4( zq6, 0 ),
					zbeta );
				svfloat64_t z90 = svmla_m( svptrue_b32(), z5, svget4( zq6, 1 ),
					zbeta );
				svfloat64_t za0 = svmla_m( svptrue_b32(), z6, svget4( zq6, 2 ),
					zbeta );
				svfloat64_t zb0 = svmla_m( svptrue_b32(), z7, svget4( zq6, 3 ),
					zbeta );

				// Store full result into C
				svfloat64x4_t z400 = svcreate4( z40, z50, z60, z70 );
				svst1_f64_x4( svptrue_c32(),
					&c_[result_tile_TL_corner + ( tcol + 0 ) * rs_c], z400 );

				svfloat64x4_t z600 = svcreate4( z80, z90, za0, zb0 );
				svst1_f64_x4( svptrue_c32(),
					&c_[result_tile_TR_corner + ( tcol + 0 ) * rs_c], z600 );

				// tcol + 1
				z0 = svread_ver_za64_m( z0, svptrue_b32(),
					/* tile: */ 0, /* slice: */ tcol + 1 );
				z1 = svread_ver_za64_m( z1, svptrue_b32(),
					/* tile: */ 1, /* slice: */ tcol + 1 );
				z2 = svread_ver_za64_m( z2, svptrue_b32(),
					/* tile: */ 2, /* slice: */ tcol + 1 );
				z3 = svread_ver_za64_m( z3, svptrue_b32(),
					/* tile: */ 3, /* slice: */ tcol + 1 );
				z4 = svread_ver_za64_m( z3, svptrue_b32(),
					/* tile: */ 4, /* slice: */ tcol + 1 );
				z5 = svread_ver_za64_m( z3, svptrue_b32(),
					/* tile: */ 5, /* slice: */ tcol + 1 );
				z6 = svread_ver_za64_m( z3, svptrue_b32(),
					/* tile: */ 6, /* slice: */ tcol + 1 );
				z7 = svread_ver_za64_m( z3, svptrue_b32(),
					/* tile: */ 7, /* slice: */ tcol + 1 );

				z0 = svmul_f64_z( svptrue_b32(), z0, zalpha );
				z1 = svmul_f64_z( svptrue_b32(), z1, zalpha );
				z2 = svmul_f64_z( svptrue_b32(), z2, zalpha );
				z3 = svmul_f64_z( svptrue_b32(), z3, zalpha );
				z4 = svmul_f64_z( svptrue_b32(), z4, zalpha );
				z5 = svmul_f64_z( svptrue_b32(), z5, zalpha );
				z6 = svmul_f64_z( svptrue_b32(), z6, zalpha );
				z7 = svmul_f64_z( svptrue_b32(), z7, zalpha );

				zq5 = svld1_f64_x4( svptrue_c32(),
					&c_[result_tile_TL_corner + ( ( ( tcol + 1 ) * rs_c ) )] );
				zq6 = svld1_f64_x4( svptrue_c32(),
					&c_[result_tile_TR_corner + ( ( ( tcol + 1 ) * rs_c ) )] );

				z40 = svmla_m( svptrue_b32(), z0, svget4( zq5, 0 ), zbeta );
				z50 = svmla_m( svptrue_b32(), z1, svget4( zq5, 1 ), zbeta );
				z60 = svmla_m( svptrue_b32(), z2, svget4( zq5, 2 ), zbeta );
				z70 = svmla_m( svptrue_b32(), z3, svget4( zq5, 3 ), zbeta );
				z80 = svmla_m( svptrue_b32(), z4, svget4( zq6, 0 ), zbeta );
				z90 = svmla_m( svptrue_b32(), z5, svget4( zq6, 1 ), zbeta );
				za0 = svmla_m( svptrue_b32(), z6, svget4( zq6, 2 ), zbeta );
				zb0 = svmla_m( svptrue_b32(), z7, svget4( zq6, 3 ), zbeta );

				z400 = svcreate4( z40, z50, z60, z70 );
				svst1_f64_x4( svptrue_c32(),
					&c_[result_tile_TL_corner + ( tcol + 1 ) * rs_c], z400 );

				z600 = svcreate4( z80, z90, za0, zb0 );
				svst1_f64_x4( svptrue_c32(),
					&c_[result_tile_TR_corner + ( tcol + 1 ) * rs_c], z600 );

				// tcol + 2
				z0 = svread_ver_za64_m( z0, svptrue_b32(),
					/* tile: */ 0, /* slice: */ tcol + 2 );
				z1 = svread_ver_za64_m( z1, svptrue_b32(),
					/* tile: */ 1, /* slice: */ tcol + 2 );
				z2 = svread_ver_za64_m( z2, svptrue_b32(),
					/* tile: */ 2, /* slice: */ tcol + 2 );
				z3 = svread_ver_za64_m( z3, svptrue_b32(),
					/* tile: */ 3, /* slice: */ tcol + 2 );
				z4 = svread_ver_za64_m( z3, svptrue_b32(),
					/* tile: */ 4, /* slice: */ tcol + 2 );
				z5 = svread_ver_za64_m( z3, svptrue_b32(),
					/* tile: */ 5, /* slice: */ tcol + 2 );
				z6 = svread_ver_za64_m( z3, svptrue_b32(),
					/* tile: */ 6, /* slice: */ tcol + 2 );
				z7 = svread_ver_za64_m( z3, svptrue_b32(),
					/* tile: */ 7, /* slice: */ tcol + 2 );

				z0 = svmul_f64_z( svptrue_b32(), z0, zalpha );
				z1 = svmul_f64_z( svptrue_b32(), z1, zalpha );
				z2 = svmul_f64_z( svptrue_b32(), z2, zalpha );
				z3 = svmul_f64_z( svptrue_b32(), z3, zalpha );
				z4 = svmul_f64_z( svptrue_b32(), z4, zalpha );
				z5 = svmul_f64_z( svptrue_b32(), z5, zalpha );
				z6 = svmul_f64_z( svptrue_b32(), z6, zalpha );
				z7 = svmul_f64_z( svptrue_b32(), z7, zalpha );

				zq5 = svld1_f64_x4( svptrue_c32(),
					&c_[result_tile_TL_corner + ( ( ( tcol + 2 ) * rs_c ) )] );
				zq6 = svld1_f64_x4( svptrue_c32(),
					&c_[result_tile_TR_corner + ( ( ( tcol + 2 ) * rs_c ) )] );

				z40 = svmla_m( svptrue_b32(), z0, svget4( zq5, 0 ), zbeta );
				z50 = svmla_m( svptrue_b32(), z1, svget4( zq5, 1 ), zbeta );
				z60 = svmla_m( svptrue_b32(), z2, svget4( zq5, 2 ), zbeta );
				z70 = svmla_m( svptrue_b32(), z3, svget4( zq5, 3 ), zbeta );
				z80 = svmla_m( svptrue_b32(), z4, svget4( zq6, 0 ), zbeta );
				z90 = svmla_m( svptrue_b32(), z5, svget4( zq6, 1 ), zbeta );
				za0 = svmla_m( svptrue_b32(), z6, svget4( zq6, 2 ), zbeta );
				zb0 = svmla_m( svptrue_b32(), z7, svget4( zq6, 3 ), zbeta );

				z400 = svcreate4( z40, z50, z60, z70 );
				svst1_f64_x4( svptrue_c32(),
					&c_[result_tile_TL_corner + ( tcol + 2 ) * rs_c], z400 );

				z600 = svcreate4( z80, z90, za0, zb0 );
				svst1_f64_x4( svptrue_c32(),
					&c_[result_tile_TR_corner + ( tcol + 2 ) * rs_c], z600 );

				// tcol + 3
				z0 = svread_ver_za64_m( z0, svptrue_b32(),
					/* tile: */ 0, /* slice: */ tcol + 3 );
				z1 = svread_ver_za64_m( z1, svptrue_b32(),
					/* tile: */ 1, /* slice: */ tcol + 3 );
				z2 = svread_ver_za64_m( z2, svptrue_b32(),
					/* tile: */ 2, /* slice: */ tcol + 3 );
				z3 = svread_ver_za64_m( z3, svptrue_b32(),
					/* tile: */ 3, /* slice: */ tcol + 3 );
				z4 = svread_ver_za64_m( z3, svptrue_b32(),
					/* tile: */ 4, /* slice: */ tcol + 3 );
				z5 = svread_ver_za64_m( z3, svptrue_b32(),
					/* tile: */ 5, /* slice: */ tcol + 3 );
				z6 = svread_ver_za64_m( z3, svptrue_b32(),
					/* tile: */ 6, /* slice: */ tcol + 3 );
				z7 = svread_ver_za64_m( z3, svptrue_b32(),
					/* tile: */ 7, /* slice: */ tcol + 3 );

				z0 = svmul_f64_z( svptrue_b32(), z0, zalpha );
				z1 = svmul_f64_z( svptrue_b32(), z1, zalpha );
				z2 = svmul_f64_z( svptrue_b32(), z2, zalpha );
				z3 = svmul_f64_z( svptrue_b32(), z3, zalpha );
				z4 = svmul_f64_z( svptrue_b32(), z4, zalpha );
				z5 = svmul_f64_z( svptrue_b32(), z5, zalpha );
				z6 = svmul_f64_z( svptrue_b32(), z6, zalpha );
				z7 = svmul_f64_z( svptrue_b32(), z7, zalpha );

				zq5 = svld1_f64_x4( svptrue_c32(),
					&c_[result_tile_TL_corner + ( ( ( tcol + 3 ) * rs_c ) )] );
				zq6 = svld1_f64_x4( svptrue_c32(),
					&c_[result_tile_TR_corner + ( ( ( tcol + 3 ) * rs_c ) )] );

				z40 = svmla_m( svptrue_b32(), z0, svget4( zq5, 0 ), zbeta );
				z50 = svmla_m( svptrue_b32(), z1, svget4( zq5, 1 ), zbeta );
				z60 = svmla_m( svptrue_b32(), z2, svget4( zq5, 2 ), zbeta );
				z70 = svmla_m( svptrue_b32(), z3, svget4( zq5, 3 ), zbeta );
				z80 = svmla_m( svptrue_b32(), z4, svget4( zq6, 0 ), zbeta );
				z90 = svmla_m( svptrue_b32(), z5, svget4( zq6, 1 ), zbeta );
				za0 = svmla_m( svptrue_b32(), z6, svget4( zq6, 2 ), zbeta );
				zb0 = svmla_m( svptrue_b32(), z7, svget4( zq6, 3 ), zbeta );

				z400 = svcreate4( z40, z50, z60, z70 );
				svst1_f64_x4( svptrue_c32(),
					&c_[result_tile_TL_corner + ( tcol + 3 ) * rs_c], z400 );

				z600 = svcreate4( z80, z90, za0, zb0 );
				svst1_f64_x4( svptrue_c32(),
					&c_[result_tile_TR_corner + ( tcol + 3 ) * rs_c], z600 );
			}
		}
	}
	else
	{
		const uint64_t result_tile_1 = SVL * cs_c;
		const uint64_t result_tile_2 = SVL * 2 * cs_c;
		const uint64_t result_tile_3 = SVL * 3 * cs_c;
		const uint64_t result_tile_4 = SVL * 4 * cs_c;
		const uint64_t result_tile_5 = SVL * 5 * cs_c;
		const uint64_t result_tile_6 = SVL * 6 * cs_c;
		const uint64_t result_tile_7 = SVL * 7 * cs_c;

		if ( beta_ == 0 )
		{
			for ( uint64_t tcol = 0; tcol < SVL; tcol += 4 )
			{
				// Read ZA slices into Z regs
				svfloat64_t z0 = svread_hor_za64_m( z0, svptrue_b32(),
					/* tile: */ 0, /* slice: */ tcol + 0 );
				svfloat64_t z1 = svread_hor_za64_m( z1, svptrue_b32(),
					/* tile: */ 1, /* slice: */ tcol + 0 );
				svfloat64_t z2 = svread_hor_za64_m( z2, svptrue_b32(),
					/* tile: */ 2, /* slice: */ tcol + 0 );
				svfloat64_t z3 = svread_hor_za64_m( z3, svptrue_b32(),
					/* tile: */ 3, /* slice: */ tcol + 0 );
				svfloat64_t z4 = svread_hor_za64_m( z4, svptrue_b32(),
					/* tile: */ 4, /* slice: */ tcol + 0 );
				svfloat64_t z5 = svread_hor_za64_m( z5, svptrue_b32(),
					/* tile: */ 5, /* slice: */ tcol + 0 );
				svfloat64_t z6 = svread_hor_za64_m( z6, svptrue_b32(),
					/* tile: */ 6, /* slice: */ tcol + 0 );
				svfloat64_t z7 = svread_hor_za64_m( z7, svptrue_b32(),
					/* tile: */ 7, /* slice: */ tcol + 0 );

				// Scale Z regs by broadcast alpha
				z0 = svmul_f64_z( svptrue_b32(), z0, zalpha );
				z1 = svmul_f64_z( svptrue_b32(), z1, zalpha );
				z2 = svmul_f64_z( svptrue_b32(), z2, zalpha );
				z3 = svmul_f64_z( svptrue_b32(), z3, zalpha );
				z4 = svmul_f64_z( svptrue_b32(), z4, zalpha );
				z5 = svmul_f64_z( svptrue_b32(), z5, zalpha );
				z6 = svmul_f64_z( svptrue_b32(), z6, zalpha );
				z7 = svmul_f64_z( svptrue_b32(), z7, zalpha );

				// Store full result into C
				svst1_f64( svptrue_b32(),
					&c_[result_tile_TL_corner + ( tcol + 0 ) * cs_c], z0 );
				svst1_f64( svptrue_b32(),
					&c_[result_tile_1 + ( tcol + 0 ) * cs_c], z1 );
				svst1_f64( svptrue_b32(),
					&c_[result_tile_2 + ( tcol + 0 ) * cs_c], z2 );
				svst1_f64( svptrue_b32(),
					&c_[result_tile_3 + ( tcol + 0 ) * cs_c], z3 );
				svst1_f64( svptrue_b32(),
					&c_[result_tile_4 + ( tcol + 0 ) * cs_c], z4 );
				svst1_f64( svptrue_b32(),
					&c_[result_tile_5 + ( tcol + 0 ) * cs_c], z5 );
				svst1_f64( svptrue_b32(),
					&c_[result_tile_6 + ( tcol + 0 ) * cs_c], z6 );
				svst1_f64( svptrue_b32(),
					&c_[result_tile_7 + ( tcol + 0 ) * cs_c], z7 );

				// tcol + 1
				z0 = svread_hor_za64_m( z0, svptrue_b32(),
					/* tile: */ 0, /* slice: */ tcol + 1 );
				z1 = svread_hor_za64_m( z1, svptrue_b32(),
					/* tile: */ 1, /* slice: */ tcol + 1 );
				z2 = svread_hor_za64_m( z2, svptrue_b32(),
					/* tile: */ 2, /* slice: */ tcol + 1 );
				z3 = svread_hor_za64_m( z3, svptrue_b32(),
					/* tile: */ 3, /* slice: */ tcol + 1 );
				z4 = svread_hor_za64_m( z4, svptrue_b32(),
					/* tile: */ 4, /* slice: */ tcol + 1 );
				z5 = svread_hor_za64_m( z5, svptrue_b32(),
					/* tile: */ 5, /* slice: */ tcol + 1 );
				z6 = svread_hor_za64_m( z6, svptrue_b32(),
					/* tile: */ 6, /* slice: */ tcol + 1 );
				z7 = svread_hor_za64_m( z7, svptrue_b32(),
					/* tile: */ 7, /* slice: */ tcol + 1 );

				z0 = svmul_f64_z( svptrue_b32(), z0, zalpha );
				z1 = svmul_f64_z( svptrue_b32(), z1, zalpha );
				z2 = svmul_f64_z( svptrue_b32(), z2, zalpha );
				z3 = svmul_f64_z( svptrue_b32(), z3, zalpha );
				z4 = svmul_f64_z( svptrue_b32(), z4, zalpha );
				z5 = svmul_f64_z( svptrue_b32(), z5, zalpha );
				z6 = svmul_f64_z( svptrue_b32(), z6, zalpha );
				z7 = svmul_f64_z( svptrue_b32(), z7, zalpha );

				svst1_f64( svptrue_b32(),
					&c_[result_tile_TL_corner + ( tcol + 1 ) * cs_c], z0 );
				svst1_f64( svptrue_b32(),
					&c_[result_tile_1 + ( tcol + 1 ) * cs_c], z1 );
				svst1_f64( svptrue_b32(),
					&c_[result_tile_2 + ( tcol + 1 ) * cs_c], z2 );
				svst1_f64( svptrue_b32(),
					&c_[result_tile_3 + ( tcol + 1 ) * cs_c], z3 );
				svst1_f64( svptrue_b32(),
					&c_[result_tile_4 + ( tcol + 1 ) * cs_c], z4 );
				svst1_f64( svptrue_b32(),
					&c_[result_tile_5 + ( tcol + 1 ) * cs_c], z5 );
				svst1_f64( svptrue_b32(),
					&c_[result_tile_6 + ( tcol + 1 ) * cs_c], z6 );
				svst1_f64( svptrue_b32(),
					&c_[result_tile_7 + ( tcol + 1 ) * cs_c], z7 );

				// tcol + 2
				z0 = svread_hor_za64_m( z0, svptrue_b32(),
					/* tile: */ 0, /* slice: */ tcol + 2 );
				z1 = svread_hor_za64_m( z1, svptrue_b32(),
					/* tile: */ 1, /* slice: */ tcol + 2 );
				z2 = svread_hor_za64_m( z2, svptrue_b32(),
					/* tile: */ 2, /* slice: */ tcol + 2 );
				z3 = svread_hor_za64_m( z3, svptrue_b32(),
					/* tile: */ 3, /* slice: */ tcol + 2 );
				z4 = svread_hor_za64_m( z4, svptrue_b32(),
					/* tile: */ 4, /* slice: */ tcol + 2 );
				z5 = svread_hor_za64_m( z5, svptrue_b32(),
					/* tile: */ 5, /* slice: */ tcol + 2 );
				z6 = svread_hor_za64_m( z6, svptrue_b32(),
					/* tile: */ 6, /* slice: */ tcol + 2 );
				z7 = svread_hor_za64_m( z7, svptrue_b32(),
					/* tile: */ 7, /* slice: */ tcol + 2 );

				z0 = svmul_f64_z( svptrue_b32(), z0, zalpha );
				z1 = svmul_f64_z( svptrue_b32(), z1, zalpha );
				z2 = svmul_f64_z( svptrue_b32(), z2, zalpha );
				z3 = svmul_f64_z( svptrue_b32(), z3, zalpha );
				z4 = svmul_f64_z( svptrue_b32(), z4, zalpha );
				z5 = svmul_f64_z( svptrue_b32(), z5, zalpha );
				z6 = svmul_f64_z( svptrue_b32(), z6, zalpha );
				z7 = svmul_f64_z( svptrue_b32(), z7, zalpha );

				svst1_f64( svptrue_b32(),
					&c_[result_tile_TL_corner + ( tcol + 2 ) * cs_c], z0 );
				svst1_f64( svptrue_b32(),
					&c_[result_tile_1 + ( tcol + 2 ) * cs_c], z1 );
				svst1_f64( svptrue_b32(),
					&c_[result_tile_2 + ( tcol + 2 ) * cs_c], z2 );
				svst1_f64( svptrue_b32(),
					&c_[result_tile_3 + ( tcol + 2 ) * cs_c], z3 );
				svst1_f64( svptrue_b32(),
					&c_[result_tile_4 + ( tcol + 2 ) * cs_c], z4 );
				svst1_f64( svptrue_b32(),
					&c_[result_tile_5 + ( tcol + 2 ) * cs_c], z5 );
				svst1_f64( svptrue_b32(),
					&c_[result_tile_6 + ( tcol + 2 ) * cs_c], z6 );
				svst1_f64( svptrue_b32(),
					&c_[result_tile_7 + ( tcol + 2 ) * cs_c], z7 );

				// tcol + 3
				z0 = svread_hor_za64_m( z0, svptrue_b32(),
					/* tile: */ 0, /* slice: */ tcol + 3 );
				z1 = svread_hor_za64_m( z1, svptrue_b32(),
					/* tile: */ 1, /* slice: */ tcol + 3 );
				z2 = svread_hor_za64_m( z2, svptrue_b32(),
					/* tile: */ 2, /* slice: */ tcol + 3 );
				z3 = svread_hor_za64_m( z3, svptrue_b32(),
					/* tile: */ 3, /* slice: */ tcol + 3 );
				z4 = svread_hor_za64_m( z4, svptrue_b32(),
					/* tile: */ 4, /* slice: */ tcol + 3 );
				z5 = svread_hor_za64_m( z5, svptrue_b32(),
					/* tile: */ 5, /* slice: */ tcol + 3 );
				z6 = svread_hor_za64_m( z6, svptrue_b32(),
					/* tile: */ 6, /* slice: */ tcol + 3 );
				z7 = svread_hor_za64_m( z7, svptrue_b32(),
					/* tile: */ 7, /* slice: */ tcol + 3 );

				z0 = svmul_f64_z( svptrue_b32(), z0, zalpha );
				z1 = svmul_f64_z( svptrue_b32(), z1, zalpha );
				z2 = svmul_f64_z( svptrue_b32(), z2, zalpha );
				z3 = svmul_f64_z( svptrue_b32(), z3, zalpha );
				z4 = svmul_f64_z( svptrue_b32(), z4, zalpha );
				z5 = svmul_f64_z( svptrue_b32(), z5, zalpha );
				z6 = svmul_f64_z( svptrue_b32(), z6, zalpha );
				z7 = svmul_f64_z( svptrue_b32(), z7, zalpha );

				svst1_f64( svptrue_b32(),
					&c_[result_tile_TL_corner + ( tcol + 3 ) * cs_c], z0 );
				svst1_f64( svptrue_b32(),
					&c_[result_tile_1 + ( tcol + 3 ) * cs_c], z1 );
				svst1_f64( svptrue_b32(),
					&c_[result_tile_2 + ( tcol + 3 ) * cs_c], z2 );
				svst1_f64( svptrue_b32(),
					&c_[result_tile_3 + ( tcol + 3 ) * cs_c], z3 );
				svst1_f64( svptrue_b32(),
					&c_[result_tile_4 + ( tcol + 3 ) * cs_c], z4 );
				svst1_f64( svptrue_b32(),
					&c_[result_tile_5 + ( tcol + 3 ) * cs_c], z5 );
				svst1_f64( svptrue_b32(),
					&c_[result_tile_6 + ( tcol + 3 ) * cs_c], z6 );
				svst1_f64( svptrue_b32(),
					&c_[result_tile_7 + ( tcol + 3 ) * cs_c], z7 );
			}
		}
		else
		{
			for ( uint64_t tcol = 0; tcol < SVL; tcol += 4 )
			{
				// Read ZA slices into Z regs
				svfloat64_t z0 = svread_hor_za64_m( z0, svptrue_b32(),
					/* tile: */ 0, /* slice: */ tcol + 0 );
				svfloat64_t z1 = svread_hor_za64_m( z1, svptrue_b32(),
					/* tile: */ 1, /* slice: */ tcol + 0 );
				svfloat64_t z2 = svread_hor_za64_m( z2, svptrue_b32(),
					/* tile: */ 2, /* slice: */ tcol + 0 );
				svfloat64_t z3 = svread_hor_za64_m( z3, svptrue_b32(),
					/* tile: */ 3, /* slice: */ tcol + 0 );
				svfloat64_t z4 = svread_hor_za64_m( z4, svptrue_b32(),
					/* tile: */ 4, /* slice: */ tcol + 0 );
				svfloat64_t z5 = svread_hor_za64_m( z5, svptrue_b32(),
					/* tile: */ 5, /* slice: */ tcol + 0 );
				svfloat64_t z6 = svread_hor_za64_m( z6, svptrue_b32(),
					/* tile: */ 6, /* slice: */ tcol + 0 );
				svfloat64_t z7 = svread_hor_za64_m( z7, svptrue_b32(),
					/* tile: */ 7, /* slice: */ tcol + 0 );

				// Scale Z regs by broadcast alpha
				z0 = svmul_f64_z( svptrue_b32(), z0, zalpha );
				z1 = svmul_f64_z( svptrue_b32(), z1, zalpha );
				z2 = svmul_f64_z( svptrue_b32(), z2, zalpha );
				z3 = svmul_f64_z( svptrue_b32(), z3, zalpha );
				z4 = svmul_f64_z( svptrue_b32(), z4, zalpha );
				z5 = svmul_f64_z( svptrue_b32(), z5, zalpha );
				z6 = svmul_f64_z( svptrue_b32(), z6, zalpha );
				z7 = svmul_f64_z( svptrue_b32(), z7, zalpha );

				// Load C into Z regs
				svfloat64_t zq0 = svld1_f64( svptrue_b32(),
					&c_[result_tile_TL_corner + ( ( ( tcol + 0 ) * cs_c ) )] );
				svfloat64_t zq1 = svld1_f64( svptrue_b32(),
					&c_[result_tile_1 + ( ( ( tcol + 0 ) * cs_c ) )] );
				svfloat64_t zq2 = svld1_f64( svptrue_b32(),
					&c_[result_tile_2 + ( ( ( tcol + 0 ) * cs_c ) )] );
				svfloat64_t zq3 = svld1_f64( svptrue_b32(),
					&c_[result_tile_3 + ( ( ( tcol + 0 ) * cs_c ) )] );
				svfloat64_t zq4 = svld1_f64( svptrue_b32(),
					&c_[result_tile_4 + ( ( ( tcol + 0 ) * cs_c ) )] );
				svfloat64_t zq5 = svld1_f64( svptrue_b32(),
					&c_[result_tile_5 + ( ( ( tcol + 0 ) * cs_c ) )] );
				svfloat64_t zq6 = svld1_f64( svptrue_b32(),
					&c_[result_tile_6 + ( ( ( tcol + 0 ) * cs_c ) )] );
				svfloat64_t zq7 = svld1_f64( svptrue_b32(),
					&c_[result_tile_7 + ( ( ( tcol + 0 ) * cs_c ) )] );

				// Scale Z regs by broadcast beta
				svfloat64_t z00 = svmla_m( svptrue_b32(), z0, zq0, zbeta );
				svfloat64_t z10 = svmla_m( svptrue_b32(), z1, zq1, zbeta );
				svfloat64_t z20 = svmla_m( svptrue_b32(), z2, zq2, zbeta );
				svfloat64_t z30 = svmla_m( svptrue_b32(), z3, zq3, zbeta );
				svfloat64_t z40 = svmla_m( svptrue_b32(), z4, zq4, zbeta );
				svfloat64_t z50 = svmla_m( svptrue_b32(), z5, zq5, zbeta );
				svfloat64_t z60 = svmla_m( svptrue_b32(), z6, zq6, zbeta );
				svfloat64_t z70 = svmla_m( svptrue_b32(), z7, zq7, zbeta );

				// Store full result into C
				svst1_f64( svptrue_b32(),
					&c_[result_tile_TL_corner + ( tcol + 0 ) * cs_c], z00 );
				svst1_f64( svptrue_b32(),
					&c_[result_tile_1 + ( tcol + 0 ) * cs_c], z10 );
				svst1_f64( svptrue_b32(),
					&c_[result_tile_2 + ( tcol + 0 ) * cs_c], z20 );
				svst1_f64( svptrue_b32(),
					&c_[result_tile_3 + ( tcol + 0 ) * cs_c], z30 );
				svst1_f64( svptrue_b32(),
					&c_[result_tile_4 + ( tcol + 0 ) * cs_c], z40 );
				svst1_f64( svptrue_b32(),
					&c_[result_tile_5 + ( tcol + 0 ) * cs_c], z50 );
				svst1_f64( svptrue_b32(),
					&c_[result_tile_6 + ( tcol + 0 ) * cs_c], z60 );
				svst1_f64( svptrue_b32(),
					&c_[result_tile_7 + ( tcol + 0 ) * cs_c], z70 );

				// tcol + 1
				z0 = svread_hor_za64_m( z0, svptrue_b32(),
					/* tile: */ 0, /* slice: */ tcol + 1 );
				z1 = svread_hor_za64_m( z1, svptrue_b32(),
					/* tile: */ 1, /* slice: */ tcol + 1 );
				z2 = svread_hor_za64_m( z2, svptrue_b32(),
					/* tile: */ 2, /* slice: */ tcol + 1 );
				z3 = svread_hor_za64_m( z3, svptrue_b32(),
					/* tile: */ 3, /* slice: */ tcol + 1 );
				z4 = svread_hor_za64_m( z4, svptrue_b32(),
					/* tile: */ 4, /* slice: */ tcol + 1 );
				z5 = svread_hor_za64_m( z5, svptrue_b32(),
					/* tile: */ 5, /* slice: */ tcol + 1 );
				z6 = svread_hor_za64_m( z6, svptrue_b32(),
					/* tile: */ 6, /* slice: */ tcol + 1 );
				z7 = svread_hor_za64_m( z7, svptrue_b32(),
					/* tile: */ 7, /* slice: */ tcol + 1 );

				z0 = svmul_f64_z( svptrue_b32(), z0, zalpha );
				z1 = svmul_f64_z( svptrue_b32(), z1, zalpha );
				z2 = svmul_f64_z( svptrue_b32(), z2, zalpha );
				z3 = svmul_f64_z( svptrue_b32(), z3, zalpha );
				z4 = svmul_f64_z( svptrue_b32(), z4, zalpha );
				z5 = svmul_f64_z( svptrue_b32(), z5, zalpha );
				z6 = svmul_f64_z( svptrue_b32(), z6, zalpha );
				z7 = svmul_f64_z( svptrue_b32(), z7, zalpha );

				zq0 = svld1_f64( svptrue_b32(),
					&c_[result_tile_TL_corner + ( ( ( tcol + 1 ) * cs_c ) )] );
				zq1 = svld1_f64( svptrue_b32(),
					&c_[result_tile_1 + ( ( ( tcol + 1 ) * cs_c ) )] );
				zq2 = svld1_f64( svptrue_b32(),
					&c_[result_tile_2 + ( ( ( tcol + 1 ) * cs_c ) )] );
				zq3 = svld1_f64( svptrue_b32(),
					&c_[result_tile_3 + ( ( ( tcol + 1 ) * cs_c ) )] );
				zq4 = svld1_f64( svptrue_b32(),
					&c_[result_tile_4 + ( ( ( tcol + 1 ) * cs_c ) )] );
				zq5 = svld1_f64( svptrue_b32(),
					&c_[result_tile_5 + ( ( ( tcol + 1 ) * cs_c ) )] );
				zq6 = svld1_f64( svptrue_b32(),
					&c_[result_tile_6 + ( ( ( tcol + 1 ) * cs_c ) )] );
				zq7 = svld1_f64( svptrue_b32(),
					&c_[result_tile_7 + ( ( ( tcol + 1 ) * cs_c ) )] );

				z00 = svmla_m( svptrue_b32(), z0, zq0, zbeta );
				z10 = svmla_m( svptrue_b32(), z1, zq1, zbeta );
				z20 = svmla_m( svptrue_b32(), z2, zq2, zbeta );
				z30 = svmla_m( svptrue_b32(), z3, zq3, zbeta );
				z40 = svmla_m( svptrue_b32(), z4, zq4, zbeta );
				z50 = svmla_m( svptrue_b32(), z5, zq5, zbeta );
				z60 = svmla_m( svptrue_b32(), z6, zq6, zbeta );
				z70 = svmla_m( svptrue_b32(), z7, zq7, zbeta );

				svst1_f64( svptrue_b32(),
					&c_[result_tile_TL_corner + ( tcol + 1 ) * cs_c], z00 );
				svst1_f64( svptrue_b32(),
					&c_[result_tile_1 + ( tcol + 1 ) * cs_c], z10 );
				svst1_f64( svptrue_b32(),
					&c_[result_tile_2 + ( tcol + 1 ) * cs_c], z20 );
				svst1_f64( svptrue_b32(),
					&c_[result_tile_3 + ( tcol + 1 ) * cs_c], z30 );
				svst1_f64( svptrue_b32(),
					&c_[result_tile_4 + ( tcol + 1 ) * cs_c], z40 );
				svst1_f64( svptrue_b32(),
					&c_[result_tile_5 + ( tcol + 1 ) * cs_c], z50 );
				svst1_f64( svptrue_b32(),
					&c_[result_tile_6 + ( tcol + 1 ) * cs_c], z60 );
				svst1_f64( svptrue_b32(),
					&c_[result_tile_7 + ( tcol + 1 ) * cs_c], z70 );

				// tcol + 2
				z0 = svread_hor_za64_m( z0, svptrue_b32(),
					/* tile: */ 0, /* slice: */ tcol + 2 );
				z1 = svread_hor_za64_m( z1, svptrue_b32(),
					/* tile: */ 1, /* slice: */ tcol + 2 );
				z2 = svread_hor_za64_m( z2, svptrue_b32(),
					/* tile: */ 2, /* slice: */ tcol + 2 );
				z3 = svread_hor_za64_m( z3, svptrue_b32(),
					/* tile: */ 3, /* slice: */ tcol + 2 );
				z4 = svread_hor_za64_m( z4, svptrue_b32(),
					/* tile: */ 4, /* slice: */ tcol + 2 );
				z5 = svread_hor_za64_m( z5, svptrue_b32(),
					/* tile: */ 5, /* slice: */ tcol + 2 );
				z6 = svread_hor_za64_m( z6, svptrue_b32(),
					/* tile: */ 6, /* slice: */ tcol + 2 );
				z7 = svread_hor_za64_m( z7, svptrue_b32(),
					/* tile: */ 7, /* slice: */ tcol + 2 );

				z0 = svmul_f64_z( svptrue_b32(), z0, zalpha );
				z1 = svmul_f64_z( svptrue_b32(), z1, zalpha );
				z2 = svmul_f64_z( svptrue_b32(), z2, zalpha );
				z3 = svmul_f64_z( svptrue_b32(), z3, zalpha );
				z4 = svmul_f64_z( svptrue_b32(), z4, zalpha );
				z5 = svmul_f64_z( svptrue_b32(), z5, zalpha );
				z6 = svmul_f64_z( svptrue_b32(), z6, zalpha );
				z7 = svmul_f64_z( svptrue_b32(), z7, zalpha );

				zq0 = svld1_f64( svptrue_b32(),
					&c_[result_tile_TL_corner + ( ( ( tcol + 2 ) * cs_c ) )] );
				zq1 = svld1_f64( svptrue_b32(),
					&c_[result_tile_1 + ( ( ( tcol + 2 ) * cs_c ) )] );
				zq2 = svld1_f64( svptrue_b32(),
					&c_[result_tile_2 + ( ( ( tcol + 2 ) * cs_c ) )] );
				zq3 = svld1_f64( svptrue_b32(),
					&c_[result_tile_3 + ( ( ( tcol + 2 ) * cs_c ) )] );
				zq4 = svld1_f64( svptrue_b32(),
					&c_[result_tile_4 + ( ( ( tcol + 2 ) * cs_c ) )] );
				zq5 = svld1_f64( svptrue_b32(),
					&c_[result_tile_5 + ( ( ( tcol + 2 ) * cs_c ) )] );
				zq6 = svld1_f64( svptrue_b32(),
					&c_[result_tile_6 + ( ( ( tcol + 2 ) * cs_c ) )] );
				zq7 = svld1_f64( svptrue_b32(),
					&c_[result_tile_7 + ( ( ( tcol + 2 ) * cs_c ) )] );

				z00 = svmla_m( svptrue_b32(), z0, zq0, zbeta );
				z10 = svmla_m( svptrue_b32(), z1, zq1, zbeta );
				z20 = svmla_m( svptrue_b32(), z2, zq2, zbeta );
				z30 = svmla_m( svptrue_b32(), z3, zq3, zbeta );
				z40 = svmla_m( svptrue_b32(), z4, zq4, zbeta );
				z50 = svmla_m( svptrue_b32(), z5, zq5, zbeta );
				z60 = svmla_m( svptrue_b32(), z6, zq6, zbeta );
				z70 = svmla_m( svptrue_b32(), z7, zq7, zbeta );

				svst1_f64( svptrue_b32(),
					&c_[result_tile_TL_corner + ( tcol + 2 ) * cs_c], z00 );
				svst1_f64( svptrue_b32(),
					&c_[result_tile_1 + ( tcol + 2 ) * cs_c], z10 );
				svst1_f64( svptrue_b32(),
					&c_[result_tile_2 + ( tcol + 2 ) * cs_c], z20 );
				svst1_f64( svptrue_b32(),
					&c_[result_tile_3 + ( tcol + 2 ) * cs_c], z30 );
				svst1_f64( svptrue_b32(),
					&c_[result_tile_4 + ( tcol + 2 ) * cs_c], z40 );
				svst1_f64( svptrue_b32(),
					&c_[result_tile_5 + ( tcol + 2 ) * cs_c], z50 );
				svst1_f64( svptrue_b32(),
					&c_[result_tile_6 + ( tcol + 2 ) * cs_c], z60 );
				svst1_f64( svptrue_b32(),
					&c_[result_tile_7 + ( tcol + 2 ) * cs_c], z70 );

				// tcol + 3
				z0 = svread_hor_za64_m( z0, svptrue_b32(),
					/* tile: */ 0, /* slice: */ tcol + 3 );
				z1 = svread_hor_za64_m( z1, svptrue_b32(),
					/* tile: */ 1, /* slice: */ tcol + 3 );
				z2 = svread_hor_za64_m( z2, svptrue_b32(),
					/* tile: */ 2, /* slice: */ tcol + 3 );
				z3 = svread_hor_za64_m( z3, svptrue_b32(),
					/* tile: */ 3, /* slice: */ tcol + 3 );
				z4 = svread_hor_za64_m( z4, svptrue_b32(),
					/* tile: */ 4, /* slice: */ tcol + 3 );
				z5 = svread_hor_za64_m( z5, svptrue_b32(),
					/* tile: */ 5, /* slice: */ tcol + 3 );
				z6 = svread_hor_za64_m( z6, svptrue_b32(),
					/* tile: */ 6, /* slice: */ tcol + 3 );
				z7 = svread_hor_za64_m( z7, svptrue_b32(),
					/* tile: */ 7, /* slice: */ tcol + 3 );

				z0 = svmul_f64_z( svptrue_b32(), z0, zalpha );
				z1 = svmul_f64_z( svptrue_b32(), z1, zalpha );
				z2 = svmul_f64_z( svptrue_b32(), z2, zalpha );
				z3 = svmul_f64_z( svptrue_b32(), z3, zalpha );
				z4 = svmul_f64_z( svptrue_b32(), z4, zalpha );
				z5 = svmul_f64_z( svptrue_b32(), z5, zalpha );
				z6 = svmul_f64_z( svptrue_b32(), z6, zalpha );
				z7 = svmul_f64_z( svptrue_b32(), z7, zalpha );

				zq0 = svld1_f64( svptrue_b32(),
					&c_[result_tile_TL_corner + ( ( ( tcol + 3 ) * cs_c ) )] );
				zq1 = svld1_f64( svptrue_b32(),
					&c_[result_tile_1 + ( ( ( tcol + 3 ) * cs_c ) )] );
				zq2 = svld1_f64( svptrue_b32(),
					&c_[result_tile_2 + ( ( ( tcol + 3 ) * cs_c ) )] );
				zq3 = svld1_f64( svptrue_b32(),
					&c_[result_tile_3 + ( ( ( tcol + 3 ) * cs_c ) )] );
				zq4 = svld1_f64( svptrue_b32(),
					&c_[result_tile_4 + ( ( ( tcol + 3 ) * cs_c ) )] );
				zq5 = svld1_f64( svptrue_b32(),
					&c_[result_tile_5 + ( ( ( tcol + 3 ) * cs_c ) )] );
				zq6 = svld1_f64( svptrue_b32(),
					&c_[result_tile_6 + ( ( ( tcol + 3 ) * cs_c ) )] );
				zq7 = svld1_f64( svptrue_b32(),
					&c_[result_tile_7 + ( ( ( tcol + 3 ) * cs_c ) )] );

				z00 = svmla_m( svptrue_b32(), z0, zq0, zbeta );
				z10 = svmla_m( svptrue_b32(), z1, zq1, zbeta );
				z20 = svmla_m( svptrue_b32(), z2, zq2, zbeta );
				z30 = svmla_m( svptrue_b32(), z3, zq3, zbeta );
				z40 = svmla_m( svptrue_b32(), z4, zq4, zbeta );
				z50 = svmla_m( svptrue_b32(), z5, zq5, zbeta );
				z60 = svmla_m( svptrue_b32(), z6, zq6, zbeta );
				z70 = svmla_m( svptrue_b32(), z7, zq7, zbeta );

				svst1_f64( svptrue_b32(),
					&c_[result_tile_TL_corner + ( tcol + 3 ) * cs_c], z00 );
				svst1_f64( svptrue_b32(),
					&c_[result_tile_1 + ( tcol + 3 ) * cs_c], z10 );
				svst1_f64( svptrue_b32(),
					&c_[result_tile_2 + ( tcol + 3 ) * cs_c], z20 );
				svst1_f64( svptrue_b32(),
					&c_[result_tile_3 + ( tcol + 3 ) * cs_c], z30 );
				svst1_f64( svptrue_b32(),
					&c_[result_tile_4 + ( tcol + 3 ) * cs_c], z40 );
				svst1_f64( svptrue_b32(),
					&c_[result_tile_5 + ( tcol + 3 ) * cs_c], z50 );
				svst1_f64( svptrue_b32(),
					&c_[result_tile_6 + ( tcol + 3 ) * cs_c], z60 );
				svst1_f64( svptrue_b32(),
					&c_[result_tile_7 + ( tcol + 3 ) * cs_c], z70 );
			}
		}
	}

	GEMM_UKR_FLUSH_CT( d );

	return;
}

