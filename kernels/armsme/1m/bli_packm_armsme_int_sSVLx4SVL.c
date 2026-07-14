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

__arm_new( "za" ) __arm_locally_streaming void bli_spackm_armsme_int_SVLx4SVL
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
    if ( !gs && cdim_bcast )
    {
        if ( bli_seq1( *( (float*)kappa ) ) )
        {
            if ( inca == 1 && ldp == 4 * SVL )
            // continuous memory.packA style
            {
                svcount_t pn = svwhilelt_c32_s64( 0, cdim, 4 );
                svcount_t ptrue = svptrue_c32();

                dim_t k = n;
                for ( ; k >= 4; k -= 4 )
                {
                    svfloat32x4_t tmp_0 = svld1_f32_x4( pn, alpha1 + 0 * lda );
                    svfloat32x4_t tmp_1 = svld1_f32_x4( pn, alpha1 + 1 * lda );
                    svfloat32x4_t tmp_2 = svld1_f32_x4( pn, alpha1 + 2 * lda );
                    svfloat32x4_t tmp_3 = svld1_f32_x4( pn, alpha1 + 3 * lda );

                    svst1_f32_x4( ptrue, pi1 + 0 * ldp, tmp_0 );
                    svst1_f32_x4( ptrue, pi1 + 1 * ldp, tmp_1 );
                    svst1_f32_x4( ptrue, pi1 + 2 * ldp, tmp_2 );
                    svst1_f32_x4( ptrue, pi1 + 3 * ldp, tmp_3 );

                    alpha1 += 4 * lda;
                    pi1 += 4 * ldp;
                }
                for ( ; k != 0; --k )
                {
                    svfloat32x4_t tmp = svld1_f32_x4( pn, alpha1 );
                    svst1_f32_x4( ptrue, pi1, tmp );

                    alpha1 += lda;
                    pi1 += ldp;
                }
            }
            else if ( inca == 1 && ldp == SVL )
            // continuous memory.packA style
            {
                svbool_t p0 = svwhilelt_b32( (int64_t) 0, cdim );
                svbool_t ptrue = svptrue_b32();

                dim_t k = n;
                for ( ; k >= 4; k -= 4 )
                {
                    svfloat32_t t0 = svld1_f32( p0, alpha1 + 0 * lda );
                    svfloat32_t t1 = svld1_f32( p0, alpha1 + 1 * lda );
                    svfloat32_t t2 = svld1_f32( p0, alpha1 + 2 * lda );
                    svfloat32_t t3 = svld1_f32( p0, alpha1 + 3 * lda );

                    svst1_f32( ptrue, pi1 + 0 * ldp, t0 );
                    svst1_f32( ptrue, pi1 + 1 * ldp, t1 );
                    svst1_f32( ptrue, pi1 + 2 * ldp, t2 );
                    svst1_f32( ptrue, pi1 + 3 * ldp, t3 );

                    alpha1 += 4 * lda;
                    pi1 += 4 * ldp;
                }
                for ( ; k != 0; --k )
                {
                    svfloat32_t tmp2 = svld1_f32( p0, alpha1 );
                    svst1_f32( ptrue, pi1, tmp2 );

                    alpha1 += lda;
                    pi1 += ldp;
                }
            }
            else if ( inca != 1 && ldp == SVL )
            {
                svfloat32_t zero_v = svdup_n_f32( 0.0f );
                svfloat32x4_t zero_x4 = svcreate4( zero_v, zero_v, zero_v, zero_v );

                for ( uint64_t col = 0; col < n; col += 4 * SVL )
                {
                    int64_t valid_cols = n - col;
                    int64_t valid_rows = ( cdim % SVL == 0 ) ? SVL : ( cdim % SVL );

                    svcount_t pn = svwhilelt_c32_s64( 0, valid_cols, 4 );

                    if ( valid_cols >= 4 * SVL && valid_rows == SVL )
                    {
                        // Fast path: Perfect SVL x 4*SVL block
                        svcount_t p_all = svptrue_c32();
                        for ( uint64_t trow = 0; trow < SVL; trow += 4 )
                        {
                            const uint64_t tile_UL_corner = trow * inca + col;

                            svfloat32x4_t zp0 = svld1_f32_x4( p_all, &a_[tile_UL_corner + 0 * inca] );
                            svfloat32x4_t zp4 = svld1_f32_x4( p_all, &a_[tile_UL_corner + 1 * inca] );
                            svfloat32x4_t zp8 = svld1_f32_x4( p_all, &a_[tile_UL_corner + 2 * inca] );
                            svfloat32x4_t zp12 = svld1_f32_x4( p_all, &a_[tile_UL_corner + 3 * inca] );

                            svfloat32x4_t zq0 = svcreate4( svget4(zp0,0), svget4(zp4,0), svget4(zp8,0), svget4(zp12,0) );
                            svfloat32x4_t zq1 = svcreate4( svget4(zp0,1), svget4(zp4,1), svget4(zp8,1), svget4(zp12,1) );
                            svfloat32x4_t zq2 = svcreate4( svget4(zp0,2), svget4(zp4,2), svget4(zp8,2), svget4(zp12,2) );
                            svfloat32x4_t zq3 = svcreate4( svget4(zp0,3), svget4(zp4,3), svget4(zp8,3), svget4(zp12,3) );

                            svwrite_hor_za32_f32_vg4( 0, trow, zq0 );
                            svwrite_hor_za32_f32_vg4( 1, trow, zq1 );
                            svwrite_hor_za32_f32_vg4( 2, trow, zq2 );
                            svwrite_hor_za32_f32_vg4( 3, trow, zq3 );
                        }
                    }
                    else
                    {
                        // Safe path: Matrix edge
                        for ( uint64_t trow = 0; trow < valid_rows; trow += 4 )
                        {
                            const uint64_t tile_UL_corner = trow * inca + col;
                            int64_t rows_left = valid_rows - trow;

                            svfloat32x4_t zp0 = zero_x4, zp4 = zero_x4, zp8 = zero_x4, zp12 = zero_x4;

                            if ( rows_left > 0 ) zp0 = svld1_f32_x4( pn, &a_[tile_UL_corner + 0 * inca] );
                            if ( rows_left > 1 ) zp4 = svld1_f32_x4( pn, &a_[tile_UL_corner + 1 * inca] );
                            if ( rows_left > 2 ) zp8 = svld1_f32_x4( pn, &a_[tile_UL_corner + 2 * inca] );
                            if ( rows_left > 3 ) zp12 = svld1_f32_x4( pn, &a_[tile_UL_corner + 3 * inca] );

                            svfloat32x4_t zq0 = svcreate4( svget4(zp0,0), svget4(zp4,0), svget4(zp8,0), svget4(zp12,0) );
                            svfloat32x4_t zq1 = svcreate4( svget4(zp0,1), svget4(zp4,1), svget4(zp8,1), svget4(zp12,1) );
                            svfloat32x4_t zq2 = svcreate4( svget4(zp0,2), svget4(zp4,2), svget4(zp8,2), svget4(zp12,2) );
                            svfloat32x4_t zq3 = svcreate4( svget4(zp0,3), svget4(zp4,3), svget4(zp8,3), svget4(zp12,3) );

                            svwrite_hor_za32_f32_vg4( 0, trow, zq0 );
                            svwrite_hor_za32_f32_vg4( 1, trow, zq1 );
                            svwrite_hor_za32_f32_vg4( 2, trow, zq2 );
                            svwrite_hor_za32_f32_vg4( 3, trow, zq3 );
                        }
                    }

                    svcount_t p0 = svptrue_c32();

                    // Check if we are at the edge and fewer than 4 * SVL columns remain
                    if ( valid_cols < 4 * SVL )
                    {
                        int total_rem = valid_cols;
                        int rem[4];
                        
                        // Partition the remainder among the 4 tiles
                        for ( int i = 0; i < 4; i++ ) {
                            rem[i] = (total_rem > (int)SVL) ? (int)SVL : total_rem;
                            total_rem -= rem[i];
                        }

                        // Macro to force the tile argument to be a compile-time constant
                        #define PROCESS_TILE_REMAINDER( TILE ) \
                        if ( rem[TILE] > 0 ) { \
                            int tcol = 0; \
                            float* p_curr = &p_[TILE * SVL * SVL]; \
                            for ( ; tcol <= rem[TILE] - 4; tcol += 4 ) { \
                                svfloat32x4_t z = svread_ver_za32_f32_vg4( TILE, tcol ); \
                                svst1_f32_x4( p0, p_curr, z ); \
                                p_curr += 4 * SVL; \
                            } \
                            for ( ; tcol <= rem[TILE] - 2; tcol += 2 ) { \
                                svfloat32x2_t z = svread_ver_za32_f32_vg2( TILE, tcol ); \
                                svst1_f32_x2( p0, p_curr, z ); \
                                p_curr += 2 * SVL; \
                            } \
                            if ( tcol < rem[TILE] ) { \
                                svfloat32_t z = svread_ver_za32_m( zero_v, svptrue_b32(), TILE, tcol ); \
                                svst1_f32( svptrue_b32(), p_curr, z ); \
                                p_curr += 1 * SVL; \
                            } \
                        }

                        PROCESS_TILE_REMAINDER( 0 )
                        PROCESS_TILE_REMAINDER( 1 )
                        PROCESS_TILE_REMAINDER( 2 )
                        PROCESS_TILE_REMAINDER( 3 )

                        #undef PROCESS_TILE_REMAINDER

                        // Advance pointer past the buffer
                        p_ += ( 4 * SVL * SVL );
                    }
                    else
                    {
                        // Read - as - columns and store
                        for ( uint64_t tcol = 0; tcol < SVL; tcol += 4 )
                        {
                            svfloat32x4_t zq0 = svread_ver_za32_f32_vg4( 0, tcol );
                            svfloat32x4_t zq2 = svread_ver_za32_f32_vg4( 2, tcol );
                            svfloat32x4_t zq1 = svread_ver_za32_f32_vg4( 1, tcol );
                            svfloat32x4_t zq3 = svread_ver_za32_f32_vg4( 3, tcol );

                            svst1_f32_x4( p0, &p_[0], zq0 );
                            svst1_f32_x4( p0, &p_[SVL * SVL], zq1 );
                            svst1_f32_x4( p0, &p_[2 * SVL * SVL], zq2 );
                            svst1_f32_x4( p0, &p_[3 * SVL * SVL], zq3 );

                            p_ += ( 4 * SVL );
                        }
                        p_ += ( 3 * SVL * SVL );
                    }
                }

                p_ = (float*)p;
            }
            else if ( inca != 1 && ldp == 4 * SVL )
            {
                svfloat32_t zero_v = svdup_n_f32( 0.0f );
                svfloat32x4_t zero_x4 = svcreate4( zero_v, zero_v, zero_v, zero_v );

                for ( uint64_t col = 0; col < n; col += SVL )
                {
                    int64_t valid_cols = n - col;
                    int64_t valid_rows = ( cdim % ( 4 * SVL ) == 0 ) ? ( 4 * SVL ) : ( cdim % ( 4 * SVL ) );

                    svbool_t p_col = svwhilelt_b32( (int64_t)0, valid_cols );

                    if ( valid_cols >= SVL && valid_rows >= 4 * SVL )
                    {
                        // Fast path: Perfect 4*SVL x SVL block
                        svbool_t p_all = svptrue_b32();
                        for ( uint64_t trow = 0; trow < SVL; trow += 4 )
                        {
                            const uint64_t tile_UL_corner = trow * inca + col;
                            const uint64_t tile_BL_corner = tile_UL_corner + inca * SVL;
                            const uint64_t tile_BBL_corner = tile_UL_corner + 2 * inca * SVL;
                            const uint64_t tile_BBBL_corner = tile_UL_corner + 3 * inca * SVL;

                            svfloat32x4_t zq0 = svcreate4( 
                                svld1_f32( p_all, &a_[tile_UL_corner + 0 * inca] ),
                                svld1_f32( p_all, &a_[tile_UL_corner + 1 * inca] ),
                                svld1_f32( p_all, &a_[tile_UL_corner + 2 * inca] ),
                                svld1_f32( p_all, &a_[tile_UL_corner + 3 * inca] ) );

                            svfloat32x4_t zq1 = svcreate4( 
                                svld1_f32( p_all, &a_[tile_BL_corner + 0 * inca] ),
                                svld1_f32( p_all, &a_[tile_BL_corner + 1 * inca] ),
                                svld1_f32( p_all, &a_[tile_BL_corner + 2 * inca] ),
                                svld1_f32( p_all, &a_[tile_BL_corner + 3 * inca] ) );

                            svfloat32x4_t zq2 = svcreate4(
                                svld1_f32( p_all, &a_[tile_BBL_corner + 0 * inca] ),
                                svld1_f32( p_all, &a_[tile_BBL_corner + 1 * inca] ),
                                svld1_f32( p_all, &a_[tile_BBL_corner + 2 * inca] ),
                                svld1_f32( p_all, &a_[tile_BBL_corner + 3 * inca] ) );

                            svfloat32x4_t zq3 = svcreate4(
                                svld1_f32( p_all, &a_[tile_BBBL_corner + 0 * inca] ),
                                svld1_f32( p_all, &a_[tile_BBBL_corner + 1 * inca] ),
                                svld1_f32( p_all, &a_[tile_BBBL_corner + 2 * inca] ),
                                svld1_f32( p_all, &a_[tile_BBBL_corner + 3 * inca] ) );

                            svwrite_hor_za32_f32_vg4( 0, trow, zq0 );
                            svwrite_hor_za32_f32_vg4( 1, trow, zq1 );
                            svwrite_hor_za32_f32_vg4( 2, trow, zq2 );
                            svwrite_hor_za32_f32_vg4( 3, trow, zq3 );
                        }
                    }
                    else
                    {
                        // Safe path: Matrix edge
                        for ( uint64_t trow = 0; trow < SVL; trow += 4 )
                        {
                            svfloat32x4_t zq0 = zero_x4, zq1 = zero_x4, zq2 = zero_x4, zq3 = zero_x4;

                            const uint64_t tile_UL_corner = trow * inca + col;
                            const uint64_t tile_BL_corner = tile_UL_corner + inca * SVL;
                            const uint64_t tile_BBL_corner = tile_UL_corner + 2 * inca * SVL;
                            const uint64_t tile_BBBL_corner = tile_UL_corner + 3 * inca * SVL;

                            int64_t rows_left_t0 = valid_rows - ( 0 * SVL + trow );
                            int64_t rows_left_t1 = valid_rows - ( 1 * SVL + trow );
                            int64_t rows_left_t2 = valid_rows - ( 2 * SVL + trow );
                            int64_t rows_left_t3 = valid_rows - ( 3 * SVL + trow );

                            if ( rows_left_t0 > 0 )
                            {
                                zq0 = svcreate4( 
                                    ( rows_left_t0 > 0 ) ? svld1_f32( p_col, &a_[tile_UL_corner + 0 * inca] ) : zero_v,
                                    ( rows_left_t0 > 1 ) ? svld1_f32( p_col, &a_[tile_UL_corner + 1 * inca] ) : zero_v,
                                    ( rows_left_t0 > 2 ) ? svld1_f32( p_col, &a_[tile_UL_corner + 2 * inca] ) : zero_v,
                                    ( rows_left_t0 > 3 ) ? svld1_f32( p_col, &a_[tile_UL_corner + 3 * inca] ) : zero_v );
                            }

                            if ( rows_left_t1 > 0 )
                            {
                                zq1 = svcreate4( 
                                    ( rows_left_t1 > 0 ) ? svld1_f32( p_col, &a_[tile_BL_corner + 0 * inca] ) : zero_v,
                                    ( rows_left_t1 > 1 ) ? svld1_f32( p_col, &a_[tile_BL_corner + 1 * inca] ) : zero_v,
                                    ( rows_left_t1 > 2 ) ? svld1_f32( p_col, &a_[tile_BL_corner + 2 * inca] ) : zero_v,
                                    ( rows_left_t1 > 3 ) ? svld1_f32( p_col, &a_[tile_BL_corner + 3 * inca] ) : zero_v );
                            }

                            if ( rows_left_t2 > 0 )
                            {
                                zq2 = svcreate4( 
                                    ( rows_left_t2 > 0 ) ? svld1_f32( p_col, &a_[tile_BBL_corner + 0 * inca] ) : zero_v,
                                    ( rows_left_t2 > 1 ) ? svld1_f32( p_col, &a_[tile_BBL_corner + 1 * inca] ) : zero_v,
                                    ( rows_left_t2 > 2 ) ? svld1_f32( p_col, &a_[tile_BBL_corner + 2 * inca] ) : zero_v,
                                    ( rows_left_t2 > 3 ) ? svld1_f32( p_col, &a_[tile_BBL_corner + 3 * inca] ) : zero_v );
                            }

                            if ( rows_left_t3 > 0 )
                            {
                                zq3 = svcreate4( 
                                    ( rows_left_t3 > 0 ) ? svld1_f32( p_col, &a_[tile_BBBL_corner + 0 * inca] ) : zero_v,
                                    ( rows_left_t3 > 1 ) ? svld1_f32( p_col, &a_[tile_BBBL_corner + 1 * inca] ) : zero_v,
                                    ( rows_left_t3 > 2 ) ? svld1_f32( p_col, &a_[tile_BBBL_corner + 2 * inca] ) : zero_v,
                                    ( rows_left_t3 > 3 ) ? svld1_f32( p_col, &a_[tile_BBBL_corner + 3 * inca] ) : zero_v );
                            }

                            svwrite_hor_za32_f32_vg4( 0, trow, zq0 );
                            svwrite_hor_za32_f32_vg4( 1, trow, zq1 );
                            svwrite_hor_za32_f32_vg4( 2, trow, zq2 );
                            svwrite_hor_za32_f32_vg4( 3, trow, zq3 );
                        }
                    }

                    svcount_t p0 = svptrue_c32();

                    // Check if we are at the edge and fewer than SVL columns remain
                    if ( valid_cols < SVL )
                    {
                        int tcol = 0;
                        for ( ; tcol <= valid_cols - 4; tcol += 4 ) {
                            svfloat32x4_t zq0 = svread_ver_za32_f32_vg4( 0, tcol );
                            svfloat32x4_t zq1 = svread_ver_za32_f32_vg4( 1, tcol );
                            svfloat32x4_t zq2 = svread_ver_za32_f32_vg4( 2, tcol );
                            svfloat32x4_t zq3 = svread_ver_za32_f32_vg4( 3, tcol );

                            svst1_f32_x4( p0, p_,             svcreate4( svget4( zq0, 0 ), svget4( zq1, 0 ), svget4( zq2, 0 ), svget4( zq3, 0 ) ) );
                            svst1_f32_x4( p0, p_ + 4 * SVL,   svcreate4( svget4( zq0, 1 ), svget4( zq1, 1 ), svget4( zq2, 1 ), svget4( zq3, 1 ) ) );
                            svst1_f32_x4( p0, p_ + 8 * SVL,   svcreate4( svget4( zq0, 2 ), svget4( zq1, 2 ), svget4( zq2, 2 ), svget4( zq3, 2 ) ) );
                            svst1_f32_x4( p0, p_ + 12 * SVL,  svcreate4( svget4( zq0, 3 ), svget4( zq1, 3 ), svget4( zq2, 3 ), svget4( zq3, 3 ) ) );
                            p_ += 16 * SVL;
                        }
                        for ( ; tcol <= valid_cols - 2; tcol += 2 ) {
                            svfloat32x2_t zq0 = svread_ver_za32_f32_vg2( 0, tcol );
                            svfloat32x2_t zq1 = svread_ver_za32_f32_vg2( 1, tcol );
                            svfloat32x2_t zq2 = svread_ver_za32_f32_vg2( 2, tcol );
                            svfloat32x2_t zq3 = svread_ver_za32_f32_vg2( 3, tcol );

                            svst1_f32_x4( p0, p_,             svcreate4( svget2( zq0, 0 ), svget2( zq1, 0 ), svget2( zq2, 0 ), svget2( zq3, 0 ) ) );
                            svst1_f32_x4( p0, p_ + 4 * SVL,   svcreate4( svget2( zq0, 1 ), svget2( zq1, 1 ), svget2( zq2, 1 ), svget2( zq3, 1 ) ) );
                            p_ += 8 * SVL;
                        }
                        if ( tcol < valid_cols ) {
                            svfloat32_t zq0 = svread_ver_za32_m( zero_v, svptrue_b32(), 0, tcol );
                            svfloat32_t zq1 = svread_ver_za32_m( zero_v, svptrue_b32(), 1, tcol );
                            svfloat32_t zq2 = svread_ver_za32_m( zero_v, svptrue_b32(), 2, tcol );
                            svfloat32_t zq3 = svread_ver_za32_m( zero_v, svptrue_b32(), 3, tcol );

                            svst1_f32_x4( p0, p_, svcreate4( zq0, zq1, zq2, zq3 ) );
                            p_ += 4 * SVL;
                        }
                    }
                    else
                    {
                        // Read - as - columns and store
                        for ( uint64_t tcol = 0; tcol < SVL; tcol += 4 )
                        {
                            svfloat32x4_t zq0 = svread_ver_za32_f32_vg4( 0, tcol );
                            svfloat32x4_t zq1 = svread_ver_za32_f32_vg4( 1, tcol );
                            svfloat32x4_t zq2 = svread_ver_za32_f32_vg4( 2, tcol );
                            svfloat32x4_t zq3 = svread_ver_za32_f32_vg4( 3, tcol );

                            svst1_f32_x4( p0, &p_[0],        svcreate4( svget4( zq0, 0 ), svget4( zq1, 0 ), svget4( zq2, 0 ), svget4( zq3, 0 ) ) );
                            svst1_f32_x4( p0, &p_[4 * SVL],  svcreate4( svget4( zq0, 1 ), svget4( zq1, 1 ), svget4( zq2, 1 ), svget4( zq3, 1 ) ) );
                            svst1_f32_x4( p0, &p_[8 * SVL],  svcreate4( svget4( zq0, 2 ), svget4( zq1, 2 ), svget4( zq2, 2 ), svget4( zq3, 2 ) ) );
                            svst1_f32_x4( p0, &p_[12 * SVL], svcreate4( svget4( zq0, 3 ), svget4( zq1, 3 ), svget4( zq2, 3 ), svget4( zq3, 3 ) ) );

                            p_ += ( 16 * SVL );
                        }
                    }
                }

                p_ = (float*)p;
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