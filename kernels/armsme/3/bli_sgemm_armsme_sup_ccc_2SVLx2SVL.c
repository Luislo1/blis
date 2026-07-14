#include "arm_sme.h"
#include "blis.h"

__arm_new( "za" ) __arm_locally_streaming void bli_sgemm_armsme_sup_ccc_2SVLx2SVL
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

    float *a_orig = (float *)a;
    float *b_ = (float *)b;
    float *c_ = (float *)c;

    svbool_t pg = svptrue_b32();
    svcount_t pg_c = svptrue_c32();

    // Manually aligned stack buffer
    float packed_A_raw[2 * SVL * k + 16];
    float *packed_A = (float *)(((uintptr_t)packed_A_raw + 63) & ~63);

    // printf("rs_c:%d, rs_a:%d, rs_b:%d\n", rs_c, rs_a, rs_b);
    for (uint64_t kk = 0; kk < k; kk += 2 * SVL)
    {
        for (uint64_t trow = 0; trow < SVL; trow += 4)
        {
            float *row_ptr_top = a_orig + trow * rs_a + kk;
            float *row_ptr_bot = a_orig + (trow + SVL) * rs_a + kk;

            svfloat32x2_t zp01 = svld1_f32_x2( pg_c, row_ptr_top + 0 * rs_a );
            svfloat32x2_t zp23 = svld1_f32_x2( pg_c, row_ptr_top + 1 * rs_a );
            svfloat32x2_t zp45 = svld1_f32_x2( pg_c, row_ptr_top + 2 * rs_a );
            svfloat32x2_t zp67 = svld1_f32_x2( pg_c, row_ptr_top + 3 * rs_a );

            svfloat32x2_t zp89 = svld1_f32_x2( pg_c, row_ptr_bot + 0 * rs_a );
            svfloat32x2_t zpAB = svld1_f32_x2( pg_c, row_ptr_bot + 1 * rs_a );
            svfloat32x2_t zpCD = svld1_f32_x2( pg_c, row_ptr_bot + 2 * rs_a );
            svfloat32x2_t zpEF = svld1_f32_x2( pg_c, row_ptr_bot + 3 * rs_a );

            svfloat32x4_t zq0 = svcreate4( svget2(zp01,0), svget2(zp23,0), svget2(zp45,0), svget2(zp67,0) );
            svfloat32x4_t zq1 = svcreate4( svget2(zp01,1), svget2(zp23,1), svget2(zp45,1), svget2(zp67,1) );
            svfloat32x4_t zq2 = svcreate4( svget2(zp89,0), svget2(zpAB,0), svget2(zpCD,0), svget2(zpEF,0) );
            svfloat32x4_t zq3 = svcreate4( svget2(zp89,1), svget2(zpAB,1), svget2(zpCD,1), svget2(zpEF,1) );

            svwrite_hor_za32_f32_vg4( 0, trow, zq0 );
            svwrite_hor_za32_f32_vg4( 1, trow, zq1 );
            svwrite_hor_za32_f32_vg4( 2, trow, zq2 );
            svwrite_hor_za32_f32_vg4( 3, trow, zq3 );
        }
        
        for (uint64_t tcol = 0; tcol < SVL; tcol += 4)
        {
            svfloat32x4_t zq0 = svread_ver_za32_f32_vg4( 0, tcol );
            svfloat32x4_t zq2 = svread_ver_za32_f32_vg4( 2, tcol );
            svfloat32x4_t zq1 = svread_ver_za32_f32_vg4( 1, tcol );
            svfloat32x4_t zq3 = svread_ver_za32_f32_vg4( 3, tcol );

            float *pack_ptr_L = &packed_A[(kk + tcol + 0) * 2 * SVL];
            float *pack_ptr_R = &packed_A[(kk + tcol + 0 + SVL) * 2 * SVL];
            
            svst1_f32_x2(pg_c, pack_ptr_L, svcreate2(svget4(zq0, 0), svget4(zq2, 0)));
            svst1_f32_x2(pg_c, pack_ptr_R, svcreate2(svget4(zq1, 0), svget4(zq3, 0)));

            pack_ptr_L = &packed_A[(kk + tcol + 1) * 2 * SVL];
            pack_ptr_R = &packed_A[(kk + tcol + 1 + SVL) * 2 * SVL];
            
            svst1_f32_x2(pg_c, pack_ptr_L, svcreate2(svget4(zq0, 1), svget4(zq2, 1)));
            svst1_f32_x2(pg_c, pack_ptr_R, svcreate2(svget4(zq1, 1), svget4(zq3, 1)));

            pack_ptr_L = &packed_A[(kk + tcol + 2) * 2 * SVL];
            pack_ptr_R = &packed_A[(kk + tcol + 2 + SVL) * 2 * SVL];
            
            svst1_f32_x2(pg_c, pack_ptr_L, svcreate2(svget4(zq0, 2), svget4(zq2, 2)));
            svst1_f32_x2(pg_c, pack_ptr_R, svcreate2(svget4(zq1, 2), svget4(zq3, 2)));

            pack_ptr_L = &packed_A[(kk + tcol + 3) * 2 * SVL];
            pack_ptr_R = &packed_A[(kk + tcol + 3 + SVL) * 2 * SVL];
            
            svst1_f32_x2(pg_c, pack_ptr_L, svcreate2(svget4(zq0, 3), svget4(zq2, 3)));
            svst1_f32_x2(pg_c, pack_ptr_R, svcreate2(svget4(zq1, 3), svget4(zq3, 3)));
        }
    }

    svzero_za();

    float *pack_a_ptr = packed_A;

    for (uint64_t k_ = 0; k_ < k; k_ += 4)
    {
        // Steps 0 and 1 Loads
        svfloat32x4_t zL01 = svld1_f32_x4(pg_c, pack_a_ptr);
        svfloat32x2_t zR0 = svld1_f32_x2(pg_c, b_ + 0 * rs_b);
        svfloat32x2_t zR1 = svld1_f32_x2(pg_c, b_ + 1 * rs_b);

        // Step 0 Outer Products
        svmopa_za32_m( 0, pg, pg, svget4(zL01, 0), svget2(zR0, 0));
        svmopa_za32_m( 1, pg, pg, svget4(zL01, 1), svget2(zR0, 0));
        svmopa_za32_m( 2, pg, pg, svget4(zL01, 0), svget2(zR0, 1));
        svmopa_za32_m( 3, pg, pg, svget4(zL01, 1), svget2(zR0, 1));

        // Step 1 Outer Products
        svmopa_za32_m( 0, pg, pg, svget4(zL01, 2), svget2(zR1, 0));
        svmopa_za32_m( 1, pg, pg, svget4(zL01, 3), svget2(zR1, 0));
        svmopa_za32_m( 2, pg, pg, svget4(zL01, 2), svget2(zR1, 1));
        svmopa_za32_m( 3, pg, pg, svget4(zL01, 3), svget2(zR1, 1));

        // Steps 2 and 3
        svfloat32x4_t zL23 = svld1_f32_x4(pg_c, pack_a_ptr + 4 * SVL);
        svfloat32x2_t zR2 = svld1_f32_x2(pg_c, b_ + 2 * rs_b);
        svfloat32x2_t zR3 = svld1_f32_x2(pg_c, b_ + 3 * rs_b);

        svmopa_za32_m( 0, pg, pg, svget4(zL23, 0), svget2(zR2, 0));
        svmopa_za32_m( 1, pg, pg, svget4(zL23, 1), svget2(zR2, 0));
        svmopa_za32_m( 2, pg, pg, svget4(zL23, 0), svget2(zR2, 1));
        svmopa_za32_m( 3, pg, pg, svget4(zL23, 1), svget2(zR2, 1));

        svmopa_za32_m( 0, pg, pg, svget4(zL23, 2), svget2(zR3, 0));
        svmopa_za32_m( 1, pg, pg, svget4(zL23, 3), svget2(zR3, 0));
        svmopa_za32_m( 2, pg, pg, svget4(zL23, 2), svget2(zR3, 1));
        svmopa_za32_m( 3, pg, pg, svget4(zL23, 3), svget2(zR3, 1));

        pack_a_ptr += 8 * SVL;
        b_         += 4 * rs_b;  
    }
    //     for (uint64_t k_ = 0; k_ < k; k_ += 4)
    // {
    //     svfloat32x4_t zL01 = svld1_f32_x4(pg_c, pack_a_ptr);
    //     svfloat32x4_t zL23 = svld1_f32_x4(pg_c, pack_a_ptr + 4 * SVL);

    //     svfloat32x2_t zR0 = svld1_f32_x2(pg_c, b_ + 0 * rs_b);
    //     svfloat32x2_t zR1 = svld1_f32_x2(pg_c, b_ + 1 * rs_b);
    //     svfloat32x2_t zR2 = svld1_f32_x2(pg_c, b_ + 2 * rs_b);
    //     svfloat32x2_t zR3 = svld1_f32_x2(pg_c, b_ + 3 * rs_b);

    //     svmopa_za32_m( 0, pg, pg,  svget2(zR0, 0), svget4(zL01, 0));
    //     svmopa_za32_m( 1, pg, pg,  svget2(zR0, 0), svget4(zL01, 1));
    //     svmopa_za32_m( 2, pg, pg,  svget2(zR0, 1), svget4(zL01, 0));
    //     svmopa_za32_m( 3, pg, pg,  svget2(zR0, 1), svget4(zL01, 1));

    //     svmopa_za32_m( 0, pg, pg, svget2(zR1, 0), svget4(zL01, 2) );
    //     svmopa_za32_m( 1, pg, pg, svget2(zR1, 0), svget4(zL01, 3) );
    //     svmopa_za32_m( 2, pg, pg, svget2(zR1, 1), svget4(zL01, 2) );
    //     svmopa_za32_m( 3, pg, pg, svget2(zR1, 1), svget4(zL01, 3) );

    //     svmopa_za32_m( 0, pg, pg, svget2(zR2, 0), svget4(zL23, 0) );
    //     svmopa_za32_m( 1, pg, pg, svget2(zR2, 0), svget4(zL23, 1) );
    //     svmopa_za32_m( 2, pg, pg, svget2(zR2, 1), svget4(zL23, 0) );
    //     svmopa_za32_m( 3, pg, pg, svget2(zR2, 1), svget4(zL23, 1) );

    //     svmopa_za32_m( 0, pg, pg, svget2(zR3, 0), svget4(zL23, 2));
    //     svmopa_za32_m( 1, pg, pg, svget2(zR3, 0), svget4(zL23, 3) );
    //     svmopa_za32_m( 2, pg, pg, svget2(zR3, 1), svget4(zL23, 2));
    //     svmopa_za32_m( 3, pg, pg, svget2(zR3, 1), svget4(zL23, 3) );

    //     pack_a_ptr += 8 * SVL;
    //     b_         += 4 * rs_b;  
    // }

    
    float beta_ = *(float *)beta;
    float alpha_ = *(float *)alpha;

    svfloat32_t zbeta = svdup_f32( beta_ ); 
    svfloat32_t zalpha = svdup_f32( alpha_ );

    // Row-Major C epilogue
    const uint64_t result_tile_TL_corner = 0;
    const uint64_t result_tile_BL_corner = SVL * rs_c;

    for ( uint64_t trow = 0; trow < SVL; trow += 1 )
    {
        // Read 4 rows at once out of each ZA tile
        svfloat32_t z0 = svread_hor_za32_m( svundef_f32(), pg, 0, trow );
        svfloat32_t z1 = svread_hor_za32_m( svundef_f32(), pg, 1, trow );
        svfloat32_t z2 = svread_hor_za32_m( svundef_f32(), pg, 2, trow );
        svfloat32_t z3 = svread_hor_za32_m( svundef_f32(), pg, 3, trow );

        // Scale by alpha
        z0 = svmul_f32_m( pg, z0, zalpha );
        z1 = svmul_f32_m( pg, z1, zalpha );
        z2 = svmul_f32_m( pg, z2, zalpha );
        z3 = svmul_f32_m( pg, z3, zalpha );

        float *c_ptr_0 = &c_[result_tile_TL_corner + trow * rs_c];
        float *c_ptr_2 = &c_[result_tile_TL_corner + SVL + trow * rs_c];
        float *c_ptr_1 = &c_[result_tile_BL_corner + trow * rs_c];
        float *c_ptr_3 = &c_[result_tile_BL_corner + SVL + trow * rs_c];

        // Load C, scale by beta
        z0 = svmla_m( pg, z0, svld1_f32(pg, c_ptr_0), zbeta );
        z2 = svmla_m( pg, z2, svld1_f32(pg, c_ptr_2), zbeta );
        z1 = svmla_m( pg, z1, svld1_f32(pg, c_ptr_1), zbeta );
        z3 = svmla_m( pg, z3, svld1_f32(pg, c_ptr_3), zbeta );

        // Store C
        svst1_f32( pg, c_ptr_0, z0 );
        svst1_f32( pg, c_ptr_2, z2 );
        svst1_f32( pg, c_ptr_1, z1 );
        svst1_f32( pg, c_ptr_3, z3 );
    }
}