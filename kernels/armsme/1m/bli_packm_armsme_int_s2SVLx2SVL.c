/*
 *
 * BLIS An object-based framework for developing high-performance BLAS-like
 * libraries.
 *
 * Copyright (C) 2014, The University of Texas at Austin Copyright (C) 2020,
 * Linaro Limited
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met: - Redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer. -
 * Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution. - Neither the
 * name(s) of the copyright holder(s) nor the names of its contributors may
 * be used to endorse or promote products derived from this software without
 * specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
 * IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 * THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 */

#include <arm_sme.h>
#include <arm_sve.h>

#include "blis.h"

__arm_new( "za" ) __arm_locally_streaming void bli_spackm_armsme_int_2SVLx2SVL
    (
        conj_t conja,
        pack_t schema,
        dim_t cdim_,
        dim_t cdim_max,
        dim_t cdim_bcast,
        dim_t n_,
        dim_t n_max_,
        const void *kappa,
        const void *a, inc_t inca_, inc_t lda_,
        void *p, inc_t ldp_,
        const void *params,
        const cntx_t * cntx
    )
{
    const int64_t cdim = cdim_;
    const int64_t n = n_;
    const int64_t inca = inca_;
    const int64_t lda = lda_;
    const int64_t ldp = ldp_;

    float* restrict a_ = (float*)a;
    float* restrict p_ = (float*)p;

    uint64_t SVL = svcntsw();

    const float* restrict alpha1 = a;
    float* restrict pi1 = p;
    const bool gs = ( inca != 1 && lda != 1 );
    if ( cdim_bcast == 1 && !gs )
    {
        if ( bli_seq1( *( (float*)kappa ) ) )
        {
            if ( inca == 1 )
            // continous memory.packA style
            {
                // Generate a single predicate-as-counter spanning 2 vectors (up to 2 * SVL)
                svcount_t pn = svwhilelt_c32_s64( 0, cdim, 2 );

                // All-true counter for the store due to zero-padding
                svcount_t ptrue = svptrue_c32();

                dim_t k;

                for ( k = n; k >= 4; k -= 4 )
                {
                    svfloat32x2_t tmp_0 = svld1_f32_x2( pn, alpha1 + 0 * lda );
                    svfloat32x2_t tmp_1 = svld1_f32_x2( pn, alpha1 + 1 * lda );
                    svfloat32x2_t tmp_2 = svld1_f32_x2( pn, alpha1 + 2 * lda );
                    svfloat32x2_t tmp_3 = svld1_f32_x2( pn, alpha1 + 3 * lda );

                    svst1_f32_x4( ptrue, pi1 + 0 * ldp, svcreate4( svget2(tmp_0, 0), svget2(tmp_0, 1), svget2(tmp_1, 0), svget2(tmp_1, 1) ) );
                    svst1_f32_x4( ptrue, pi1 + 2 * ldp, svcreate4( svget2(tmp_2, 0), svget2(tmp_2, 1), svget2(tmp_3, 0), svget2(tmp_3, 1) ) );

                    alpha1 += 4 * lda;
                    pi1    += 4 * ldp;
                }

                // Remainder loop
                for ( ; k != 0; --k )
                {
                    svfloat32x2_t tmp = svld1_f32_x2( pn, alpha1 );
                    svst1_f32_x2( ptrue, pi1, tmp );

                    alpha1 += lda;
                    pi1    += ldp;
                }
            }
            else
            {
                for ( uint64_t col = 0; col < n; col += 2 * SVL )
                {
                    int64_t valid_cols = n - col;
                    int64_t valid_rows = ( cdim % ( 2 * SVL ) == 0 ) ? ( 2 * SVL ) : ( cdim % ( 2 * SVL ) );

                    svfloat32_t undef_v = svundef_f32();

                    if ( valid_cols >= 2 * SVL && valid_rows >= 2 * SVL )
                    {
                        // Fast path
                        svcount_t p_all = svptrue_c32();

                        for ( uint64_t trow = 0; trow < SVL; trow += 4 )
                        {
                            const uint64_t tile_UL_corner = trow * inca + col;
                            const uint64_t tile_BL_corner = tile_UL_corner + inca * SVL;

                            svfloat32x2_t zp01 = svld1_f32_x2( p_all, &a_[tile_UL_corner + 0 * inca] );
                            svfloat32x2_t zp23 = svld1_f32_x2( p_all, &a_[tile_UL_corner + 1 * inca] );
                            svfloat32x2_t zp45 = svld1_f32_x2( p_all, &a_[tile_UL_corner + 2 * inca] );
                            svfloat32x2_t zp67 = svld1_f32_x2( p_all, &a_[tile_UL_corner + 3 * inca] );

                            svfloat32x2_t zp89 = svld1_f32_x2( p_all, &a_[tile_BL_corner + 0 * inca] );
                            svfloat32x2_t zpAB = svld1_f32_x2( p_all, &a_[tile_BL_corner + 1 * inca] );
                            svfloat32x2_t zpCD = svld1_f32_x2( p_all, &a_[tile_BL_corner + 2 * inca] );
                            svfloat32x2_t zpEF = svld1_f32_x2( p_all, &a_[tile_BL_corner + 3 * inca] );

                            svfloat32x4_t zq0 = svcreate4( svget2(zp01,0), svget2(zp23,0), svget2(zp45,0), svget2(zp67,0) );
                            svfloat32x4_t zq1 = svcreate4( svget2(zp01,1), svget2(zp23,1), svget2(zp45,1), svget2(zp67,1) );
                            svfloat32x4_t zq2 = svcreate4( svget2(zp89,0), svget2(zpAB,0), svget2(zpCD,0), svget2(zpEF,0) );
                            svfloat32x4_t zq3 = svcreate4( svget2(zp89,1), svget2(zpAB,1), svget2(zpCD,1), svget2(zpEF,1) );

                            svwrite_hor_za32_f32_vg4( 0, trow, zq0 );
                            svwrite_hor_za32_f32_vg4( 1, trow, zq1 );
                            svwrite_hor_za32_f32_vg4( 2, trow, zq2 );
                            svwrite_hor_za32_f32_vg4( 3, trow, zq3 );
                        }
                    }
                    else
                    {
                        // Safe path
                        // Use predicate-as-counter to handle column boundaries
                        svcount_t pn = svwhilelt_c32_s64( 0, valid_cols, 2 );
                        
                        svfloat32x2_t undef_x2 = svundef2_f32();

                        // Only loop as many times as we have valid rows
                        uint64_t top_tile_limit = (valid_rows > SVL) ? SVL : valid_rows;

                        for ( uint64_t trow = 0; trow < top_tile_limit; trow += 4 )
                        {
                            const uint64_t tile_UL_corner = trow * inca + col;
                            const uint64_t tile_BL_corner = tile_UL_corner + inca * SVL;

                            int64_t rows_left_top = valid_rows - trow;
                            int64_t rows_left_bot = valid_rows - ( SVL + trow );

                            svfloat32x2_t zp01 = undef_x2, zp23 = undef_x2, zp45 = undef_x2, zp67 = undef_x2;
                            svfloat32x2_t zp89 = undef_x2, zpAB = undef_x2, zpCD = undef_x2, zpEF = undef_x2;

                            // Top tiles (0 and 1)
                            if ( rows_left_top > 0 ) zp01 = svld1_f32_x2( pn, &a_[tile_UL_corner + 0 * inca] );
                            if ( rows_left_top > 1 ) zp23 = svld1_f32_x2( pn, &a_[tile_UL_corner + 1 * inca] );
                            if ( rows_left_top > 2 ) zp45 = svld1_f32_x2( pn, &a_[tile_UL_corner + 2 * inca] );
                            if ( rows_left_top > 3 ) zp67 = svld1_f32_x2( pn, &a_[tile_UL_corner + 3 * inca] );

                            // Bottom tiles (2 and 3)
                            if ( rows_left_bot > 0 )
                            {
                                if ( rows_left_bot > 0 ) zp89 = svld1_f32_x2( pn, &a_[tile_BL_corner + 0 * inca] );
                                if ( rows_left_bot > 1 ) zpAB = svld1_f32_x2( pn, &a_[tile_BL_corner + 1 * inca] );
                                if ( rows_left_bot > 2 ) zpCD = svld1_f32_x2( pn, &a_[tile_BL_corner + 2 * inca] );
                                if ( rows_left_bot > 3 ) zpEF = svld1_f32_x2( pn, &a_[tile_BL_corner + 3 * inca] );
                            }

                            svfloat32x4_t zq0 = svcreate4( svget2(zp01,0), svget2(zp23,0), svget2(zp45,0), svget2(zp67,0) );
                            svfloat32x4_t zq1 = svcreate4( svget2(zp01,1), svget2(zp23,1), svget2(zp45,1), svget2(zp67,1) );
                            svfloat32x4_t zq2 = svcreate4( svget2(zp89,0), svget2(zpAB,0), svget2(zpCD,0), svget2(zpEF,0) );
                            svfloat32x4_t zq3 = svcreate4( svget2(zp89,1), svget2(zpAB,1), svget2(zpCD,1), svget2(zpEF,1) );

                            svwrite_hor_za32_f32_vg4( 0, trow, zq0 );
                            svwrite_hor_za32_f32_vg4( 1, trow, zq1 );
                            
                            if ( rows_left_bot > 0 ) {
                                svwrite_hor_za32_f32_vg4( 2, trow, zq2 );
                                svwrite_hor_za32_f32_vg4( 3, trow, zq3 );
                            }
                        }
                    }

                    svcount_t p0 = svptrue_c32();

                    // Check if we are at the edge and fewer than 2 * SVL columns remain
                    if ( valid_cols < 2 * SVL )
                    {
                        int rem1 = ( valid_cols > (int)SVL ) ? (int)SVL : valid_cols;
                        int rem2 = ( valid_cols > (int)SVL ) ? ( valid_cols - (int)SVL ) : 0;

                        int tcol = 0;
                        
                        // Process Tiles 0 & 2 (unrolled)
                        for ( ; tcol <= rem1 - 4; tcol += 4 ) {
                            svfloat32x4_t zq0 = svread_ver_za32_f32_vg4( 0, tcol );
                            svfloat32x4_t zq2 = svread_ver_za32_f32_vg4( 2, tcol );                         
                            svst1_f32_x4( p0, &p_[0],       svcreate4( svget4( zq0, 0 ), svget4( zq2, 0 ), svget4( zq0, 1 ), svget4( zq2, 1 ) ) );
                            svst1_f32_x4( p0, &p_[4 * SVL], svcreate4( svget4( zq0, 2 ), svget4( zq2, 2 ), svget4( zq0, 3 ), svget4( zq2, 3 ) ) );
                            p_ += 8 * SVL;
                        }
                        for ( ; tcol <= rem1 - 2; tcol += 2 ) {
                            svfloat32x2_t zq0 = svread_ver_za32_f32_vg2( 0, tcol );
                            svfloat32x2_t zq2 = svread_ver_za32_f32_vg2( 2, tcol );
                            svst1_f32_x4( p0, &p_[0], svcreate4( svget2( zq0, 0 ), svget2( zq2, 0 ), svget2( zq0, 1 ), svget2( zq2, 1 ) ) );
                            p_ += 4 * SVL;
                        }
                        if ( tcol < rem1 ) {
                            svfloat32_t zq0 = svread_ver_za32_m( undef_v, svptrue_b32(), 0, tcol );
                            svfloat32_t zq2 = svread_ver_za32_m( undef_v, svptrue_b32(), 2, tcol );
                            svst1_f32_x2( p0, &p_[0], svcreate2( zq0, zq2 ) );
                            p_ += 2 * SVL;
                        }

                        // Process Tiles 1 & 3 (unrolled)
                        tcol = 0;
                        for ( ; tcol <= rem2 - 4; tcol += 4 ) {
                            svfloat32x4_t zq1 = svread_ver_za32_f32_vg4( 1, tcol );
                            svfloat32x4_t zq3 = svread_ver_za32_f32_vg4( 3, tcol );
                            svst1_f32_x4( p0, &p_[0],       svcreate4( svget4( zq1, 0 ), svget4( zq3, 0 ), svget4( zq1, 1 ), svget4( zq3, 1 ) ) );
                            svst1_f32_x4( p0, &p_[4 * SVL], svcreate4( svget4( zq1, 2 ), svget4( zq3, 2 ), svget4( zq1, 3 ), svget4( zq3, 3 ) ) );
                            p_ += 8 * SVL;
                        }
                        for ( ; tcol <= rem2 - 2; tcol += 2 ) {
                            svfloat32x2_t zq1 = svread_ver_za32_f32_vg2( 1, tcol );
                            svfloat32x2_t zq3 = svread_ver_za32_f32_vg2( 3, tcol );
                            svst1_f32_x4( p0, &p_[0], svcreate4( svget2( zq1, 0 ), svget2( zq3, 0 ), svget2( zq1, 1 ), svget2( zq3, 1 ) ) );
                            p_ += 4 * SVL;
                        }
                        if ( tcol < rem2 ) {
                            svfloat32_t zq1 = svread_ver_za32_m( undef_v, svptrue_b32(), 1, tcol );
                            svfloat32_t zq3 = svread_ver_za32_m( undef_v, svptrue_b32(), 3, tcol );
                            svst1_f32_x2( p0, &p_[0], svcreate2( zq1, zq3 ) );
                            p_ += 2 * SVL;
                        }
                    }
                    else
                    {
                        // Read - as - columns and store (FULL BLOCKS)
                        for ( uint64_t tcol = 0; tcol < SVL; tcol += 4 )
                        {
                            svfloat32x4_t zq0 = svread_ver_za32_f32_vg4( 0, tcol );
                            svfloat32x4_t zq2 = svread_ver_za32_f32_vg4( 2, tcol );
                            svfloat32x4_t zq1 = svread_ver_za32_f32_vg4( 1, tcol );
                            svfloat32x4_t zq3 = svread_ver_za32_f32_vg4( 3, tcol );

                            svst1_f32_x4( p0, &p_[0],       svcreate4( svget4( zq0, 0 ), svget4( zq2, 0 ), svget4( zq0, 1 ), svget4( zq2, 1 ) ) );
                            svst1_f32_x4( p0, &p_[4 * SVL], svcreate4( svget4( zq0, 2 ), svget4( zq2, 2 ), svget4( zq0, 3 ), svget4( zq2, 3 ) ) );

                            svst1_f32_x4( p0, &p_[2 * SVL * SVL + 0],       svcreate4( svget4( zq1, 0 ), svget4( zq3, 0 ), svget4( zq1, 1 ), svget4( zq3, 1 ) ) );
                            svst1_f32_x4( p0, &p_[2 * SVL * SVL + 4 * SVL], svcreate4( svget4( zq1, 2 ), svget4( zq3, 2 ), svget4( zq1, 3 ), svget4( zq3, 3 ) ) );

                            p_ += ( 8 * SVL );
                        }
                        p_ += ( 2 * SVL * SVL );
                    }
                }
            }
        }
        else
        {
            bli_sscal2bbs_mxn
                (
                 conja,
                 cdim_,
                 n_,
                 kappa,
                 a, inca, lda,
                 p_, cdim_bcast, ldp
                );
        }
    }
    else
    {
        bli_sscal2bbs_mxn
            (
             conja,
             cdim_,
             n_,
             kappa,
             a, inca, lda,
             p_, cdim_bcast, ldp
            );
    }

    bli_sset0s_edge
        (
         cdim_ * cdim_bcast, cdim_max * cdim_bcast,
         n_, n_max_,
         p_, ldp
        );
} 