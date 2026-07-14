#include "arm_sme.h"
#include "blis.h"

__arm_new( "za" ) __arm_locally_streaming void bli_sgemm_armsme_sup_test_2SVLx2SVL
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
    uint64_t SVL = svcntsw();
    
    float *a_ = (float *)a;
    float *b_ = (float *)b;
    float *c_ = (float *)c;

    svzero_za();
    svbool_t pg_M0 = svwhilelt_b32((uint64_t)0, (uint64_t)m);
    svbool_t pg_M1 = svwhilelt_b32((uint64_t)SVL, (uint64_t)m);
    
    svbool_t pg_N0 = svwhilelt_b32((uint64_t)0, (uint64_t)n);
    svbool_t pg_N1 = svwhilelt_b32((uint64_t)SVL, (uint64_t)n);
    svbool_t pg = svptrue_b32();
    svcount_t pg_c = svptrue_c32();

    uint64_t k_iter = k / 4;
    uint64_t k_left = k % 4;
    // printf("rs_c:%d, rs_a:%d, rs_b:%d\n", rs_c, rs_a, rs_b);
    for (uint64_t k_ = 0; k_ < k_iter; k_++ )
    {
        // Step 0 Loads
        svfloat32x2_t zL0 = svld1_f32_x2(pg_c, a_ + 0 * cs_a);
        svfloat32x2_t zR0 = svld1_f32_x2(pg_c, b_ + 0 * rs_b);

        // Step 0 Outer Products
        svmopa_za32_m( 0, pg_M0, pg_N0, svget2(zR0, 0), svget2(zL0, 0) );
        svmopa_za32_m( 1, pg_M1, pg_N0, svget2(zR0, 0), svget2(zL0, 1) );
        svmopa_za32_m( 2, pg_M0, pg_N1, svget2(zR0, 1), svget2(zL0, 0) );
        svmopa_za32_m( 3, pg_M1, pg_N1, svget2(zR0, 1), svget2(zL0, 1) );

        // Step 1
        svfloat32x2_t zL1 = svld1_f32_x2(pg_c, a_ + 1 * cs_a);
        svfloat32x2_t zR1 = svld1_f32_x2(pg_c, b_ + 1 * rs_b);

        svmopa_za32_m( 0, pg_M0, pg_N0, svget2(zR1, 0),svget2(zL1, 0) );
        svmopa_za32_m( 1, pg_M1, pg_N0, svget2(zR1, 0),svget2(zL1, 1) );
        svmopa_za32_m( 2, pg_M0, pg_N1, svget2(zR1, 1),svget2(zL1, 0) );
        svmopa_za32_m( 3, pg_M1, pg_N1, svget2(zR1, 1),svget2(zL1, 1) );

        // Step 2
        svfloat32x2_t zL2 = svld1_f32_x2(pg_c, a_ + 2 * cs_a);
        svfloat32x2_t zR2 = svld1_f32_x2(pg_c, b_ + 2 * rs_b);

        svmopa_za32_m( 0, pg_M0, pg_N0, svget2(zR2, 0), svget2(zL2, 0));
        svmopa_za32_m( 1, pg_M1, pg_N0, svget2(zR2, 0), svget2(zL2, 1));
        svmopa_za32_m( 2, pg_M0, pg_N1, svget2(zR2, 1), svget2(zL2, 0));
        svmopa_za32_m( 3, pg_M1, pg_N1, svget2(zR2, 1), svget2(zL2, 1));

        // Step 3
        svfloat32x2_t zL3 = svld1_f32_x2(pg_c, a_ + 3 * cs_a);
        svfloat32x2_t zR3 = svld1_f32_x2(pg_c, b_ + 3 * rs_b);

        svmopa_za32_m( 0, pg_M0, pg_N0, svget2(zR3, 0), svget2(zL3, 0));
        svmopa_za32_m( 1, pg_M1, pg_N0, svget2(zR3, 0), svget2(zL3, 1));
        svmopa_za32_m( 2, pg_M0, pg_N1, svget2(zR3, 1), svget2(zL3, 0));
        svmopa_za32_m( 3, pg_M1, pg_N1, svget2(zR3, 1), svget2(zL3, 1));

        a_ += 4 * cs_a;
        b_ += 4 * rs_b;  
    }

    // Remainder Loop
    for (uint64_t k_ = 0; k_ < k_left; k_ += 1 )
    {
        svfloat32x2_t zL = svld1_f32_x2(pg_c, a_);
        svfloat32x2_t zR = svld1_f32_x2(pg_c, b_);

        svmopa_za32_m( 0, pg_M0, pg_N0, svget2(zR, 0), svget2(zL, 0) );
        svmopa_za32_m( 1, pg_M1, pg_N0, svget2(zR, 0), svget2(zL, 1) );
        svmopa_za32_m( 2, pg_M0, pg_N1, svget2(zR, 1), svget2(zL, 0) );
        svmopa_za32_m( 3, pg_M1, pg_N1, svget2(zR, 1), svget2(zL, 1) );

        a_ += cs_a;
        b_ += rs_b;  
    }

    float beta_ = *(float *)beta;
    float alpha_ = *(float *)alpha;

    svfloat32_t zbeta = svdup_f32( beta_ ); 
    svfloat32_t zalpha = svdup_f32( alpha_ );

    const uint64_t result_tile_TL_corner = 0;
    const uint64_t result_tile_BL_corner = SVL * rs_c;

    if (m == 2 * SVL && n == 2 * SVL) 
    {
        // Fast Path: Row-Major C
        for ( uint64_t trow = 0; trow < SVL; trow += 4 )
        {
            // Read 4 rows at once out of each ZA tile
            svfloat32x4_t zq0 = svread_ver_za32_f32_vg4( 0, trow );
            svfloat32x4_t zq1 = svread_ver_za32_f32_vg4( 1, trow );
            svfloat32x4_t zq2 = svread_ver_za32_f32_vg4( 2, trow );
            svfloat32x4_t zq3 = svread_ver_za32_f32_vg4( 3, trow );

            // Row 0 (trow + 0)
            {
                uint64_t r_trow = trow + 0;

                svfloat32_t z0 = svget4(zq0, 0);
                svfloat32_t z1 = svget4(zq1, 0);
                svfloat32_t z2 = svget4(zq2, 0);
                svfloat32_t z3 = svget4(zq3, 0);

                // Scale by alpha
                z0 = svmul_f32_m( pg, z0, zalpha );
                z1 = svmul_f32_m( pg, z1, zalpha );
                z2 = svmul_f32_m( pg, z2, zalpha );
                z3 = svmul_f32_m( pg, z3, zalpha );

                float *c_ptr_0 = &c_[result_tile_TL_corner + r_trow * rs_c];
                float *c_ptr_1 = &c_[result_tile_BL_corner + r_trow * rs_c];

                // Load C, scale by beta, and store back
                svfloat32x2_t zq_c02 = svld1_f32_x2( pg_c, c_ptr_0 );
                z0 = svmla_m( pg, z0, svget2(zq_c02, 0), zbeta );
                z1 = svmla_m( pg, z1, svget2(zq_c02, 1), zbeta );
                svst1_f32_x2( pg_c, c_ptr_0, svcreate2( z0, z1 ) );

                svfloat32x2_t zq_c13 = svld1_f32_x2( pg_c, c_ptr_1 );
                z2 = svmla_m( pg, z2, svget2(zq_c13, 0), zbeta );
                z3 = svmla_m( pg, z3, svget2(zq_c13, 1), zbeta );
                svst1_f32_x2( pg_c, c_ptr_1, svcreate2( z2, z3 ) );
            }

            // Row 1 (trow + 1)
            {
                uint64_t r_trow = trow + 1;

                svfloat32_t z0 = svget4(zq0, 1);
                svfloat32_t z1 = svget4(zq1, 1);
                svfloat32_t z2 = svget4(zq2, 1);
                svfloat32_t z3 = svget4(zq3, 1);

                z0 = svmul_f32_m( pg, z0, zalpha );
                z1 = svmul_f32_m( pg, z1, zalpha );
                z2 = svmul_f32_m( pg, z2, zalpha );
                z3 = svmul_f32_m( pg, z3, zalpha );

                float *c_ptr_0 = &c_[result_tile_TL_corner + r_trow * rs_c];
                float *c_ptr_1 = &c_[result_tile_BL_corner + r_trow * rs_c];

                svfloat32x2_t zq_c02 = svld1_f32_x2( pg_c, c_ptr_0 );
                z0 = svmla_m( pg, z0, svget2(zq_c02, 0), zbeta );
                z1 = svmla_m( pg, z1, svget2(zq_c02, 1), zbeta );
                svst1_f32_x2( pg_c, c_ptr_0, svcreate2( z0, z1 ) );

                svfloat32x2_t zq_c13 = svld1_f32_x2( pg_c, c_ptr_1 );
                z2 = svmla_m( pg, z2, svget2(zq_c13, 0), zbeta );
                z3 = svmla_m( pg, z3, svget2(zq_c13, 1), zbeta );
                svst1_f32_x2( pg_c, c_ptr_1, svcreate2( z2, z3 ) );
            }

            // Row 2 (trow + 2)
            {
                uint64_t r_trow = trow + 2;

                svfloat32_t z0 = svget4(zq0, 2);
                svfloat32_t z1 = svget4(zq1, 2);
                svfloat32_t z2 = svget4(zq2, 2);
                svfloat32_t z3 = svget4(zq3, 2);

                z0 = svmul_f32_m( pg, z0, zalpha );
                z1 = svmul_f32_m( pg, z1, zalpha );
                z2 = svmul_f32_m( pg, z2, zalpha );
                z3 = svmul_f32_m( pg, z3, zalpha );

                float *c_ptr_0 = &c_[result_tile_TL_corner + r_trow * rs_c];
                float *c_ptr_1 = &c_[result_tile_BL_corner + r_trow * rs_c];

                svfloat32x2_t zq_c02 = svld1_f32_x2( pg_c, c_ptr_0 );
                z1 = svmla_m( pg, z1, svget2(zq_c02, 0), zbeta );
                z2 = svmla_m( pg, z2, svget2(zq_c02, 1), zbeta );
                svst1_f32_x2( pg_c, c_ptr_0, svcreate2( z1, z2 ) );

                svfloat32x2_t zq_c13 = svld1_f32_x2( pg_c, c_ptr_1 );
                z2 = svmla_m( pg, z2, svget2(zq_c13, 0), zbeta );
                z3 = svmla_m( pg, z3, svget2(zq_c13, 1), zbeta );
                svst1_f32_x2( pg_c, c_ptr_1, svcreate2( z2, z3 ) );
            }

            // Row 3 (trow + 3)
            {
                uint64_t r_trow = trow + 3;

                svfloat32_t z0 = svget4(zq0, 3);
                svfloat32_t z1 = svget4(zq1, 3);
                svfloat32_t z2 = svget4(zq2, 3);
                svfloat32_t z3 = svget4(zq3, 3);

                z0 = svmul_f32_m( pg, z0, zalpha );
                z1 = svmul_f32_m( pg, z1, zalpha );
                z2 = svmul_f32_m( pg, z2, zalpha );
                z3 = svmul_f32_m( pg, z3, zalpha );

                float *c_ptr_0 = &c_[result_tile_TL_corner + r_trow * rs_c];
                float *c_ptr_1 = &c_[result_tile_BL_corner + r_trow * rs_c];

                svfloat32x2_t zq_c02 = svld1_f32_x2( pg_c, c_ptr_0 );
                z0 = svmla_m( pg, z0, svget2(zq_c02, 0), zbeta );
                z1 = svmla_m( pg, z1, svget2(zq_c02, 1), zbeta );
                svst1_f32_x2( pg_c, c_ptr_0, svcreate2( z0, z1 ) );

                svfloat32x2_t zq_c13 = svld1_f32_x2( pg_c, c_ptr_1 );
                z2 = svmla_m( pg, z2, svget2(zq_c13, 0), zbeta );
                z3 = svmla_m( pg, z3, svget2(zq_c13, 1), zbeta );
                svst1_f32_x2( pg_c, c_ptr_1, svcreate2( z2, z3 ) );
            }
        }
    }
    else 
    {
        // Edge Path: Row-Major C
        for ( uint64_t trow = 0; trow < SVL; trow += 1 )
        {
            bool valid_row_0 = (trow < m);
            bool valid_row_1 = (trow + SVL < m);

            // Top Half (Tiles 0 & 2)
            if (valid_row_0) 
            {
                svfloat32_t z0 = svread_hor_za32_m( svundef_f32(), pg_N0, 0, trow );
                svfloat32_t z2 = svread_hor_za32_m( svundef_f32(), pg_N1, 2, trow );

                z0 = svmul_f32_m( pg_N0, z0, zalpha );
                z2 = svmul_f32_m( pg_N1, z2, zalpha );

                float *c_ptr_0 = &c_[result_tile_TL_corner + trow * rs_c];
                float *c_ptr_2 = &c_[result_tile_TL_corner + SVL + trow * rs_c];

                z0 = svmla_m( pg_N0, z0, svld1_f32(pg_N0, c_ptr_0), zbeta );
                z2 = svmla_m( pg_N1, z2, svld1_f32(pg_N1, c_ptr_2), zbeta );

                svst1_f32( pg_N0, c_ptr_0, z0 );
                svst1_f32( pg_N1, c_ptr_2, z2 );
            }

            // Bottom Half (Tiles 1 & 3)
            if (valid_row_1) 
            {
                svfloat32_t z1 = svread_hor_za32_m( svundef_f32(), pg_N0, 1, trow );
                svfloat32_t z3 = svread_hor_za32_m( svundef_f32(), pg_N1, 3, trow );

                z1 = svmul_f32_m( pg_N0, z1, zalpha );
                z3 = svmul_f32_m( pg_N1, z3, zalpha );

                float *c_ptr_1 = &c_[result_tile_BL_corner + trow * rs_c];
                float *c_ptr_3 = &c_[result_tile_BL_corner + SVL + trow * rs_c];

                z1 = svmla_m( pg_N0, z1, svld1_f32(pg_N0, c_ptr_1), zbeta );
                z3 = svmla_m( pg_N1, z3, svld1_f32(pg_N1, c_ptr_3), zbeta );

                svst1_f32( pg_N0, c_ptr_1, z1 );
                svst1_f32( pg_N1, c_ptr_3, z3 );
            }
        }
    }
}