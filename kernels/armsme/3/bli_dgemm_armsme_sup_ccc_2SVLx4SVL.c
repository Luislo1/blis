#include "arm_sme.h"
#include "blis.h"

__arm_new( "za" ) __arm_locally_streaming void bli_dgemm_armsme_sup_ccc_2SVLx4SVL
(
    conj_t           conja,
    conj_t           conjb,
    dim_t            m,
    dim_t            n,
    dim_t            k,
    const void*      alpha,
    const void*      a, inc_t rs_a, inc_t cs_a,
    const void*      b, inc_t rs_b, inc_t cs_b,
    const void*      beta,
    void*            c, inc_t rs_c, inc_t cs_c,
    const auxinfo_t* data,
    const cntx_t*    cntx
)
{
    uint64_t SVL = svcntd();

    double *a_orig = (double *)a;
    double *b_ = (double *)b;
    double *c_ = (double *)c;

    svbool_t  pg   = svptrue_b64();
    svcount_t pg_c = svptrue_c64();

    // Manually aligned stack buffer for packed A (2*SVL rows x k columns)
    double packed_A_raw[2 * SVL * k + 16];
    double *packed_A = (double *)(((uintptr_t)packed_A_raw + 63) & ~63);

    // =========================================================================
    // Phase 1: Transpose A in 2*SVL x 4*SVL blocks using all 8 ZA tiles
    // =========================================================================
    for (uint64_t kk = 0; kk < k; kk += 4 * SVL)
    {
        for (uint64_t trow = 0; trow < SVL; trow += 4)
        {
            double *row_ptr_top = a_orig + trow * rs_a + kk;
            double *row_ptr_bot = a_orig + (trow + SVL) * rs_a + kk;

            // Load 4 contiguous SVL segments across each of the 4 rows (Top half)
            svfloat64x4_t zp0 = svld1_f64_x4( pg_c, row_ptr_top + 0 * rs_a );
            svfloat64x4_t zp1 = svld1_f64_x4( pg_c, row_ptr_top + 1 * rs_a );
            svfloat64x4_t zp2 = svld1_f64_x4( pg_c, row_ptr_top + 2 * rs_a );
            svfloat64x4_t zp3 = svld1_f64_x4( pg_c, row_ptr_top + 3 * rs_a );

            // Load 4 contiguous SVL segments across each of the 4 rows (Bottom half)
            svfloat64x4_t zp4 = svld1_f64_x4( pg_c, row_ptr_bot + 0 * rs_a );
            svfloat64x4_t zp5 = svld1_f64_x4( pg_c, row_ptr_bot + 1 * rs_a );
            svfloat64x4_t zp6 = svld1_f64_x4( pg_c, row_ptr_bot + 2 * rs_a );
            svfloat64x4_t zp7 = svld1_f64_x4( pg_c, row_ptr_bot + 3 * rs_a );

            // Group into 4-row slices per column block
            svfloat64x4_t zq0 = svcreate4( svget4(zp0, 0), svget4(zp1, 0), svget4(zp2, 0), svget4(zp3, 0) );
            svfloat64x4_t zq1 = svcreate4( svget4(zp4, 0), svget4(zp5, 0), svget4(zp6, 0), svget4(zp7, 0) );
            svfloat64x4_t zq2 = svcreate4( svget4(zp0, 1), svget4(zp1, 1), svget4(zp2, 1), svget4(zp3, 1) );
            svfloat64x4_t zq3 = svcreate4( svget4(zp4, 1), svget4(zp5, 1), svget4(zp6, 1), svget4(zp7, 1) );
            svfloat64x4_t zq4 = svcreate4( svget4(zp0, 2), svget4(zp1, 2), svget4(zp2, 2), svget4(zp3, 2) );
            svfloat64x4_t zq5 = svcreate4( svget4(zp4, 2), svget4(zp5, 2), svget4(zp6, 2), svget4(zp7, 2) );
            svfloat64x4_t zq6 = svcreate4( svget4(zp0, 3), svget4(zp1, 3), svget4(zp2, 3), svget4(zp3, 3) );
            svfloat64x4_t zq7 = svcreate4( svget4(zp4, 3), svget4(zp5, 3), svget4(zp6, 3), svget4(zp7, 3) );

            // Write horizontally to ZA tiles
            svwrite_hor_za64_f64_vg4( 0, trow, zq0 );
            svwrite_hor_za64_f64_vg4( 1, trow, zq1 );
            svwrite_hor_za64_f64_vg4( 2, trow, zq2 );
            svwrite_hor_za64_f64_vg4( 3, trow, zq3 );
            svwrite_hor_za64_f64_vg4( 4, trow, zq4 );
            svwrite_hor_za64_f64_vg4( 5, trow, zq5 );
            svwrite_hor_za64_f64_vg4( 6, trow, zq6 );
            svwrite_hor_za64_f64_vg4( 7, trow, zq7 );
        }
        
        for (uint64_t tcol = 0; tcol < SVL; tcol += 4)
        {
            // Read vertical slices (transposing rows into columns)
            svfloat64x4_t zq0 = svread_ver_za64_f64_vg4( 0, tcol );
            svfloat64x4_t zq1 = svread_ver_za64_f64_vg4( 1, tcol );
            svfloat64x4_t zq2 = svread_ver_za64_f64_vg4( 2, tcol );
            svfloat64x4_t zq3 = svread_ver_za64_f64_vg4( 3, tcol );
            svfloat64x4_t zq4 = svread_ver_za64_f64_vg4( 4, tcol );
            svfloat64x4_t zq5 = svread_ver_za64_f64_vg4( 5, tcol );
            svfloat64x4_t zq6 = svread_ver_za64_f64_vg4( 6, tcol );
            svfloat64x4_t zq7 = svread_ver_za64_f64_vg4( 7, tcol );

            // Slice 0 (tcol + 0)
            double *pack_ptr_0 = &packed_A[(kk + tcol + 0 + 0 * SVL) * 2 * SVL];
            double *pack_ptr_1 = &packed_A[(kk + tcol + 0 + 1 * SVL) * 2 * SVL];
            double *pack_ptr_2 = &packed_A[(kk + tcol + 0 + 2 * SVL) * 2 * SVL];
            double *pack_ptr_3 = &packed_A[(kk + tcol + 0 + 3 * SVL) * 2 * SVL];

            svst1_f64_x2( pg_c, pack_ptr_0, svcreate2( svget4(zq0, 0), svget4(zq1, 0) ) );
            svst1_f64_x2( pg_c, pack_ptr_1, svcreate2( svget4(zq2, 0), svget4(zq3, 0) ) );
            svst1_f64_x2( pg_c, pack_ptr_2, svcreate2( svget4(zq4, 0), svget4(zq5, 0) ) );
            svst1_f64_x2( pg_c, pack_ptr_3, svcreate2( svget4(zq6, 0), svget4(zq7, 0) ) );

            // Slice 1 (tcol + 1)
            pack_ptr_0 = &packed_A[(kk + tcol + 1 + 0 * SVL) * 2 * SVL];
            pack_ptr_1 = &packed_A[(kk + tcol + 1 + 1 * SVL) * 2 * SVL];
            pack_ptr_2 = &packed_A[(kk + tcol + 1 + 2 * SVL) * 2 * SVL];
            pack_ptr_3 = &packed_A[(kk + tcol + 1 + 3 * SVL) * 2 * SVL];

            svst1_f64_x2( pg_c, pack_ptr_0, svcreate2( svget4(zq0, 1), svget4(zq1, 1) ) );
            svst1_f64_x2( pg_c, pack_ptr_1, svcreate2( svget4(zq2, 1), svget4(zq3, 1) ) );
            svst1_f64_x2( pg_c, pack_ptr_2, svcreate2( svget4(zq4, 1), svget4(zq5, 1) ) );
            svst1_f64_x2( pg_c, pack_ptr_3, svcreate2( svget4(zq6, 1), svget4(zq7, 1) ) );

            // Slice 2 (tcol + 2)
            pack_ptr_0 = &packed_A[(kk + tcol + 2 + 0 * SVL) * 2 * SVL];
            pack_ptr_1 = &packed_A[(kk + tcol + 2 + 1 * SVL) * 2 * SVL];
            pack_ptr_2 = &packed_A[(kk + tcol + 2 + 2 * SVL) * 2 * SVL];
            pack_ptr_3 = &packed_A[(kk + tcol + 2 + 3 * SVL) * 2 * SVL];

            svst1_f64_x2( pg_c, pack_ptr_0, svcreate2( svget4(zq0, 2), svget4(zq1, 2) ) );
            svst1_f64_x2( pg_c, pack_ptr_1, svcreate2( svget4(zq2, 2), svget4(zq3, 2) ) );
            svst1_f64_x2( pg_c, pack_ptr_2, svcreate2( svget4(zq4, 2), svget4(zq5, 2) ) );
            svst1_f64_x2( pg_c, pack_ptr_3, svcreate2( svget4(zq6, 2), svget4(zq7, 2) ) );

            // Slice 3 (tcol + 3)
            pack_ptr_0 = &packed_A[(kk + tcol + 3 + 0 * SVL) * 2 * SVL];
            pack_ptr_1 = &packed_A[(kk + tcol + 3 + 1 * SVL) * 2 * SVL];
            pack_ptr_2 = &packed_A[(kk + tcol + 3 + 2 * SVL) * 2 * SVL];
            pack_ptr_3 = &packed_A[(kk + tcol + 3 + 3 * SVL) * 2 * SVL];

            svst1_f64_x2( pg_c, pack_ptr_0, svcreate2( svget4(zq0, 3), svget4(zq1, 3) ) );
            svst1_f64_x2( pg_c, pack_ptr_1, svcreate2( svget4(zq2, 3), svget4(zq3, 3) ) );
            svst1_f64_x2( pg_c, pack_ptr_2, svcreate2( svget4(zq4, 3), svget4(zq5, 3) ) );
            svst1_f64_x2( pg_c, pack_ptr_3, svcreate2( svget4(zq6, 3), svget4(zq7, 3) ) );
        }
    }

    svzero_za();

    // =========================================================================
    // Phase 2: GEMM Outer-Product Loop
    // =========================================================================
    double *pack_a_ptr = packed_A;

    for (uint64_t k_ = 0; k_ < k; k_ += 4)
    {
        // Steps 0 and 1 Loads
        svfloat64x4_t zL01 = svld1_f64_x4( pg_c, pack_a_ptr );
        svfloat64x4_t zR0  = svld1_f64_x4( pg_c, b_ + 0 * rs_b );
        svfloat64x4_t zR1  = svld1_f64_x4( pg_c, b_ + 1 * rs_b );

        // Step 0 Outer Products
        svmopa_za64_m( 0, pg, pg, svget4(zL01, 0), svget4(zR0, 0) );
        svmopa_za64_m( 1, pg, pg, svget4(zL01, 1), svget4(zR0, 0) );
        svmopa_za64_m( 2, pg, pg, svget4(zL01, 0), svget4(zR0, 1) );
        svmopa_za64_m( 3, pg, pg, svget4(zL01, 1), svget4(zR0, 1) );
        svmopa_za64_m( 4, pg, pg, svget4(zL01, 0), svget4(zR0, 2) );
        svmopa_za64_m( 5, pg, pg, svget4(zL01, 1), svget4(zR0, 2) );
        svmopa_za64_m( 6, pg, pg, svget4(zL01, 0), svget4(zR0, 3) );
        svmopa_za64_m( 7, pg, pg, svget4(zL01, 1), svget4(zR0, 3) );

        // Step 1 Outer Products
        svmopa_za64_m( 0, pg, pg, svget4(zL01, 2), svget4(zR1, 0) );
        svmopa_za64_m( 1, pg, pg, svget4(zL01, 3), svget4(zR1, 0) );
        svmopa_za64_m( 2, pg, pg, svget4(zL01, 2), svget4(zR1, 1) );
        svmopa_za64_m( 3, pg, pg, svget4(zL01, 3), svget4(zR1, 1) );
        svmopa_za64_m( 4, pg, pg, svget4(zL01, 2), svget4(zR1, 2) );
        svmopa_za64_m( 5, pg, pg, svget4(zL01, 3), svget4(zR1, 2) );
        svmopa_za64_m( 6, pg, pg, svget4(zL01, 2), svget4(zR1, 3) );
        svmopa_za64_m( 7, pg, pg, svget4(zL01, 3), svget4(zR1, 3) );

        // Steps 2 and 3 Loads
        svfloat64x4_t zL23 = svld1_f64_x4( pg_c, pack_a_ptr + 4 * SVL );
        svfloat64x4_t zR2  = svld1_f64_x4( pg_c, b_ + 2 * rs_b );
        svfloat64x4_t zR3  = svld1_f64_x4( pg_c, b_ + 3 * rs_b );

        // Step 2 Outer Products
        svmopa_za64_m( 0, pg, pg, svget4(zL23, 0), svget4(zR2, 0) );
        svmopa_za64_m( 1, pg, pg, svget4(zL23, 1), svget4(zR2, 0) );
        svmopa_za64_m( 2, pg, pg, svget4(zL23, 0), svget4(zR2, 1) );
        svmopa_za64_m( 3, pg, pg, svget4(zL23, 1), svget4(zR2, 1) );
        svmopa_za64_m( 4, pg, pg, svget4(zL23, 0), svget4(zR2, 2) );
        svmopa_za64_m( 5, pg, pg, svget4(zL23, 1), svget4(zR2, 2) );
        svmopa_za64_m( 6, pg, pg, svget4(zL23, 0), svget4(zR2, 3) );
        svmopa_za64_m( 7, pg, pg, svget4(zL23, 1), svget4(zR2, 3) );

        // Step 3 Outer Products
        svmopa_za64_m( 0, pg, pg, svget4(zL23, 2), svget4(zR3, 0) );
        svmopa_za64_m( 1, pg, pg, svget4(zL23, 3), svget4(zR3, 0) );
        svmopa_za64_m( 2, pg, pg, svget4(zL23, 2), svget4(zR3, 1) );
        svmopa_za64_m( 3, pg, pg, svget4(zL23, 3), svget4(zR3, 1) );
        svmopa_za64_m( 4, pg, pg, svget4(zL23, 2), svget4(zR3, 2) );
        svmopa_za64_m( 5, pg, pg, svget4(zL23, 3), svget4(zR3, 2) );
        svmopa_za64_m( 6, pg, pg, svget4(zL23, 2), svget4(zR3, 3) );
        svmopa_za64_m( 7, pg, pg, svget4(zL23, 3), svget4(zR3, 3) );

        pack_a_ptr += 8 * SVL;
        b_         += 4 * rs_b;  
    }

    // =========================================================================
    // Phase 3: Row-Major C Epilogue
    // =========================================================================
    double beta_  = *(const double *)beta;
    double alpha_ = *(const double *)alpha;

    svfloat64_t zbeta  = svdup_f64( beta_ ); 
    svfloat64_t zalpha = svdup_f64( alpha_ );

    const uint64_t result_tile_TL_corner = 0;
    const uint64_t result_tile_BL_corner = SVL * rs_c;

    for ( uint64_t trow = 0; trow < SVL; trow += 1 )
    {
        // Read horizontal rows out of each ZA tile
        svfloat64_t z0 = svread_hor_za64_m( svundef_f64(), pg, 0, trow );
        svfloat64_t z1 = svread_hor_za64_m( svundef_f64(), pg, 1, trow );
        svfloat64_t z2 = svread_hor_za64_m( svundef_f64(), pg, 2, trow );
        svfloat64_t z3 = svread_hor_za64_m( svundef_f64(), pg, 3, trow );
        svfloat64_t z4 = svread_hor_za64_m( svundef_f64(), pg, 4, trow );
        svfloat64_t z5 = svread_hor_za64_m( svundef_f64(), pg, 5, trow );
        svfloat64_t z6 = svread_hor_za64_m( svundef_f64(), pg, 6, trow );
        svfloat64_t z7 = svread_hor_za64_m( svundef_f64(), pg, 7, trow );

        // Scale by alpha
        z0 = svmul_f64_m( pg, z0, zalpha );
        z1 = svmul_f64_m( pg, z1, zalpha );
        z2 = svmul_f64_m( pg, z2, zalpha );
        z3 = svmul_f64_m( pg, z3, zalpha );
        z4 = svmul_f64_m( pg, z4, zalpha );
        z5 = svmul_f64_m( pg, z5, zalpha );
        z6 = svmul_f64_m( pg, z6, zalpha );
        z7 = svmul_f64_m( pg, z7, zalpha );

        // Destination pointers for Top Half (Tiles 0, 2, 4, 6 across 4 column blocks)
        double *c_ptr_0 = &c_[result_tile_TL_corner + trow * rs_c + 0 * SVL];
        double *c_ptr_2 = &c_[result_tile_TL_corner + trow * rs_c + 1 * SVL];
        double *c_ptr_4 = &c_[result_tile_TL_corner + trow * rs_c + 2 * SVL];
        double *c_ptr_6 = &c_[result_tile_TL_corner + trow * rs_c + 3 * SVL];

        // Destination pointers for Bottom Half (Tiles 1, 3, 5, 7 across 4 column blocks)
        double *c_ptr_1 = &c_[result_tile_BL_corner + trow * rs_c + 0 * SVL];
        double *c_ptr_3 = &c_[result_tile_BL_corner + trow * rs_c + 1 * SVL];
        double *c_ptr_5 = &c_[result_tile_BL_corner + trow * rs_c + 2 * SVL];
        double *c_ptr_7 = &c_[result_tile_BL_corner + trow * rs_c + 3 * SVL];

        // Load C, accumulate with beta
        z0 = svmla_m( pg, z0, svld1_f64( pg, c_ptr_0 ), zbeta );
        z2 = svmla_m( pg, z2, svld1_f64( pg, c_ptr_2 ), zbeta );
        z4 = svmla_m( pg, z4, svld1_f64( pg, c_ptr_4 ), zbeta );
        z6 = svmla_m( pg, z6, svld1_f64( pg, c_ptr_6 ), zbeta );

        z1 = svmla_m( pg, z1, svld1_f64( pg, c_ptr_1 ), zbeta );
        z3 = svmla_m( pg, z3, svld1_f64( pg, c_ptr_3 ), zbeta );
        z5 = svmla_m( pg, z5, svld1_f64( pg, c_ptr_5 ), zbeta );
        z7 = svmla_m( pg, z7, svld1_f64( pg, c_ptr_7 ), zbeta );

        // Store C
        svst1_f64( pg, c_ptr_0, z0 );
        svst1_f64( pg, c_ptr_2, z2 );
        svst1_f64( pg, c_ptr_4, z4 );
        svst1_f64( pg, c_ptr_6, z6 );

        svst1_f64( pg, c_ptr_1, z1 );
        svst1_f64( pg, c_ptr_3, z3 );
        svst1_f64( pg, c_ptr_5, z5 );
        svst1_f64( pg, c_ptr_7, z7 );
    }
}