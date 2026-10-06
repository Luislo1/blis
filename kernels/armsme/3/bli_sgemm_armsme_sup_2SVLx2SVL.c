#include "arm_sme.h"
#include "blis.h"

__arm_new( "za" ) __arm_locally_streaming void bli_sgemm_armsme_sup_2SVLx2SVL
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
    svzero_za();

    // Define all loop-invariant predicates ONCE
    svbool_t pg_M0 = svwhilelt_b32((uint64_t)0, (uint64_t)m);
    svbool_t pg_M1 = svwhilelt_b32((uint64_t)SVL, (uint64_t)m);
    svbool_t pg_N0 = svwhilelt_b32((uint64_t)0, (uint64_t)n);
    svbool_t pg_N1 = svwhilelt_b32((uint64_t)SVL, (uint64_t)n);
    svbool_t pg = svptrue_b32();
    svcount_t pg_c = svptrue_c32();

    uint64_t k_iter = k / 4;
    uint64_t k_left = k % 4;

    const float *a_ptr = (const float *)a;
    const float *b_ptr = (const float *)b;

    // K-LOOP 
    for (uint64_t k_ = 0; k_ < k_iter; k_++ )
    {
        // PRE-LOAD STEP 0 & 1
        svfloat32x2_t zL0 = svld1_f32_x2(pg_c, a_ptr + 0 * cs_a);
        svfloat32x2_t zR0 = svld1_f32_x2(pg_c, b_ptr + 0 * rs_b);
        
        svfloat32x2_t zL1 = svld1_f32_x2(pg_c, a_ptr + 1 * cs_a);
        svfloat32x2_t zR1 = svld1_f32_x2(pg_c, b_ptr + 1 * rs_b);

        // MATH STEP 0
        svmopa_za32_m( 0, pg_M0, pg_N0, svget2(zL0, 0), svget2(zR0, 0));
        svmopa_za32_m( 1, pg_M1, pg_N0, svget2(zL0, 1), svget2(zR0, 0));
        svmopa_za32_m( 2, pg_M0, pg_N1, svget2(zL0, 0), svget2(zR0, 1));
        svmopa_za32_m( 3, pg_M1, pg_N1, svget2(zL0, 1), svget2(zR0, 1));

        // LOAD STEP 2
        svfloat32x2_t zL2 = svld1_f32_x2(pg_c, a_ptr + 2 * cs_a);
        svfloat32x2_t zR2 = svld1_f32_x2(pg_c, b_ptr + 2 * rs_b);

        // MATH STEP 1
        svmopa_za32_m( 0, pg_M0, pg_N0, svget2(zL1, 0), svget2(zR1, 0));
        svmopa_za32_m( 1, pg_M1, pg_N0, svget2(zL1, 1), svget2(zR1, 0));
        svmopa_za32_m( 2, pg_M0, pg_N1, svget2(zL1, 0), svget2(zR1, 1));
        svmopa_za32_m( 3, pg_M1, pg_N1, svget2(zL1, 1), svget2(zR1, 1));

        // LOAD STEP 3
        svfloat32x2_t zL3 = svld1_f32_x2(pg_c, a_ptr + 3 * cs_a);
        svfloat32x2_t zR3 = svld1_f32_x2(pg_c, b_ptr + 3 * rs_b);

        // MATH STEP 2
        svmopa_za32_m( 0, pg_M0, pg_N0, svget2(zL2, 0), svget2(zR2, 0));
        svmopa_za32_m( 1, pg_M1, pg_N0, svget2(zL2, 1), svget2(zR2, 0));
        svmopa_za32_m( 2, pg_M0, pg_N1, svget2(zL2, 0), svget2(zR2, 1));
        svmopa_za32_m( 3, pg_M1, pg_N1, svget2(zL2, 1), svget2(zR2, 1));

        // MATH STEP 3
        svmopa_za32_m( 0, pg_M0, pg_N0, svget2(zL3, 0), svget2(zR3, 0));
        svmopa_za32_m( 1, pg_M1, pg_N0, svget2(zL3, 1), svget2(zR3, 0));
        svmopa_za32_m( 2, pg_M0, pg_N1, svget2(zL3, 0), svget2(zR3, 1));
        svmopa_za32_m( 3, pg_M1, pg_N1, svget2(zL3, 1), svget2(zR3, 1));

        a_ptr += 4 * cs_a;
        b_ptr += 4 * rs_b;  
    }

    // REMAINDER LOOP
    for (uint64_t k_ = 0; k_ < k_left; k_ += 1 )
    {
        svfloat32x2_t zL = svld1_f32_x2(pg_c, a_ptr);
        svfloat32x2_t zR = svld1_f32_x2(pg_c, b_ptr);

        svmopa_za32_m( 0, pg_M0, pg_N0, svget2(zL, 0), svget2(zR, 0));
        svmopa_za32_m( 1, pg_M1, pg_N0, svget2(zL, 1), svget2(zR, 0));
        svmopa_za32_m( 2, pg_M0, pg_N1, svget2(zL, 0), svget2(zR, 1));
        svmopa_za32_m( 3, pg_M1, pg_N1, svget2(zL, 1), svget2(zR, 1));

        a_ptr += cs_a;
        b_ptr += rs_b;  
    }

    // EPILOGUE
    float beta_ = *(float *)beta;
    float alpha_ = *(float *)alpha;
    float *c_ = (float *)c;

    if (m == 2 * SVL && n == 2 * SVL) 
    {
        // FAST PATH
        float *c_ptr_0 = c_;
        float *c_ptr_1 = c_ + SVL * rs_c;

        if (beta_ == 0 )
        {
            for ( uint64_t trow = 0; trow < SVL; trow += 4 )
            {
                // Read 4 rows at once out of each ZA tile
                svfloat32x4_t zq0 = svread_hor_za32_f32_vg4( 0, trow );
                svfloat32x4_t zq1 = svread_hor_za32_f32_vg4( 1, trow );
                svfloat32x4_t zq2 = svread_hor_za32_f32_vg4( 2, trow );
                svfloat32x4_t zq3 = svread_hor_za32_f32_vg4( 3, trow );

                // Row 0 (trow + 0)
                {
                    svfloat32_t z0 = svmul_n_f32_x( pg, svget4(zq0, 0), alpha_ );
                    svfloat32_t z1 = svmul_n_f32_x( pg, svget4(zq1, 0), alpha_ );
                    svfloat32_t z2 = svmul_n_f32_x( pg, svget4(zq2, 0), alpha_ );
                    svfloat32_t z3 = svmul_n_f32_x( pg, svget4(zq3, 0), alpha_ );

                    svst1_f32_x2( pg_c, c_ptr_0 + 0 * rs_c, svcreate2( z0, z2 ) );
                    svst1_f32_x2( pg_c, c_ptr_1 + 0 * rs_c, svcreate2( z1, z3 ) );
                }

                // Row 1 (trow + 1)
                {
                    svfloat32_t z0 = svmul_n_f32_x( pg, svget4(zq0, 1), alpha_ );
                    svfloat32_t z1 = svmul_n_f32_x( pg, svget4(zq1, 1), alpha_ );
                    svfloat32_t z2 = svmul_n_f32_x( pg, svget4(zq2, 1), alpha_ );
                    svfloat32_t z3 = svmul_n_f32_x( pg, svget4(zq3, 1), alpha_ );

                    svst1_f32_x2( pg_c, c_ptr_0 + 1 * rs_c, svcreate2( z0, z2 ) );
                    svst1_f32_x2( pg_c, c_ptr_1 + 1 * rs_c, svcreate2( z1, z3 ) );
                }

                // Row 2 (trow + 2)
                {
                    svfloat32_t z0 = svmul_n_f32_x( pg, svget4(zq0, 2), alpha_ );
                    svfloat32_t z1 = svmul_n_f32_x( pg, svget4(zq1, 2), alpha_ );
                    svfloat32_t z2 = svmul_n_f32_x( pg, svget4(zq2, 2), alpha_ );
                    svfloat32_t z3 = svmul_n_f32_x( pg, svget4(zq3, 2), alpha_ );

                    svst1_f32_x2( pg_c, c_ptr_0 + 2 * rs_c, svcreate2( z0, z2 ) );
                    svst1_f32_x2( pg_c, c_ptr_1 + 2 * rs_c, svcreate2( z1, z3 ) );
                }

                // Row 3 (trow + 3)
                {
                    svfloat32_t z0 = svmul_n_f32_x( pg, svget4(zq0, 3), alpha_ );
                    svfloat32_t z1 = svmul_n_f32_x( pg, svget4(zq1, 3), alpha_ );
                    svfloat32_t z2 = svmul_n_f32_x( pg, svget4(zq2, 3), alpha_ );
                    svfloat32_t z3 = svmul_n_f32_x( pg, svget4(zq3, 3), alpha_ );

                    svst1_f32_x2( pg_c, c_ptr_0 + 3 * rs_c, svcreate2( z0, z2 ) );
                    svst1_f32_x2( pg_c, c_ptr_1 + 3 * rs_c, svcreate2( z1, z3 ) );
                }
                
                // Step pointers for next 4 rows
                c_ptr_0 += 4 * rs_c;
                c_ptr_1 += 4 * rs_c;
            }
        }
        else {
            for ( uint64_t trow = 0; trow < SVL; trow += 4 )
            {
                // Read 4 rows at once out of each ZA tile
                svfloat32x4_t zq0 = svread_hor_za32_f32_vg4( 0, trow );
                svfloat32x4_t zq1 = svread_hor_za32_f32_vg4( 1, trow );
                svfloat32x4_t zq2 = svread_hor_za32_f32_vg4( 2, trow );
                svfloat32x4_t zq3 = svread_hor_za32_f32_vg4( 3, trow );

                // Row 0 (trow + 0)
                {
                    svfloat32_t z0 = svmul_n_f32_x( pg, svget4(zq0, 0), alpha_ );
                    svfloat32_t z1 = svmul_n_f32_x( pg, svget4(zq1, 0), alpha_ );
                    svfloat32_t z2 = svmul_n_f32_x( pg, svget4(zq2, 0), alpha_ );
                    svfloat32_t z3 = svmul_n_f32_x( pg, svget4(zq3, 0), alpha_ );

                    svfloat32x2_t zq_c02 = svld1_f32_x2( pg_c, c_ptr_0 + 0 * rs_c );
                    z0 = svmla_n_f32_x( pg, z0, svget2(zq_c02, 0), beta_ );
                    z2 = svmla_n_f32_x( pg, z2, svget2(zq_c02, 1), beta_ );
                    svst1_f32_x2( pg_c, c_ptr_0 + 0 * rs_c, svcreate2( z0, z2 ) );

                    svfloat32x2_t zq_c13 = svld1_f32_x2( pg_c, c_ptr_1 + 0 * rs_c );
                    z1 = svmla_n_f32_x( pg, z1, svget2(zq_c13, 0), beta_ );
                    z3 = svmla_n_f32_x( pg, z3, svget2(zq_c13, 1), beta_ );
                    svst1_f32_x2( pg_c, c_ptr_1 + 0 * rs_c, svcreate2( z1, z3 ) );
                }

                // Row 1 (trow + 1)
                {
                    svfloat32_t z0 = svmul_n_f32_x( pg, svget4(zq0, 1), alpha_ );
                    svfloat32_t z1 = svmul_n_f32_x( pg, svget4(zq1, 1), alpha_ );
                    svfloat32_t z2 = svmul_n_f32_x( pg, svget4(zq2, 1), alpha_ );
                    svfloat32_t z3 = svmul_n_f32_x( pg, svget4(zq3, 1), alpha_ );

                    svfloat32x2_t zq_c02 = svld1_f32_x2( pg_c, c_ptr_0 + 1 * rs_c );
                    z0 = svmla_n_f32_x( pg, z0, svget2(zq_c02, 0), beta_ );
                    z2 = svmla_n_f32_x( pg, z2, svget2(zq_c02, 1), beta_ );
                    svst1_f32_x2( pg_c, c_ptr_0 + 1 * rs_c, svcreate2( z0, z2 ) );

                    svfloat32x2_t zq_c13 = svld1_f32_x2( pg_c, c_ptr_1 + 1 * rs_c );
                    z1 = svmla_n_f32_x( pg, z1, svget2(zq_c13, 0), beta_ );
                    z3 = svmla_n_f32_x( pg, z3, svget2(zq_c13, 1), beta_ );
                    svst1_f32_x2( pg_c, c_ptr_1 + 1 * rs_c, svcreate2( z1, z3 ) );
                }

                // Row 2 (trow + 2)
                {
                    svfloat32_t z0 = svmul_n_f32_x( pg, svget4(zq0, 2), alpha_ );
                    svfloat32_t z1 = svmul_n_f32_x( pg, svget4(zq1, 2), alpha_ );
                    svfloat32_t z2 = svmul_n_f32_x( pg, svget4(zq2, 2), alpha_ );
                    svfloat32_t z3 = svmul_n_f32_x( pg, svget4(zq3, 2), alpha_ );

                    svfloat32x2_t zq_c02 = svld1_f32_x2( pg_c, c_ptr_0 + 2 * rs_c );
                    z0 = svmla_n_f32_x( pg, z0, svget2(zq_c02, 0), beta_ );
                    z2 = svmla_n_f32_x( pg, z2, svget2(zq_c02, 1), beta_ );
                    svst1_f32_x2( pg_c, c_ptr_0 + 2 * rs_c, svcreate2( z0, z2 ) );

                    svfloat32x2_t zq_c13 = svld1_f32_x2( pg_c, c_ptr_1 + 2 * rs_c );
                    z1 = svmla_n_f32_x( pg, z1, svget2(zq_c13, 0), beta_ );
                    z3 = svmla_n_f32_x( pg, z3, svget2(zq_c13, 1), beta_ );
                    svst1_f32_x2( pg_c, c_ptr_1 + 2 * rs_c, svcreate2( z1, z3 ) );
                }

                // Row 3 (trow + 3)
                {
                    svfloat32_t z0 = svmul_n_f32_x( pg, svget4(zq0, 3), alpha_ );
                    svfloat32_t z1 = svmul_n_f32_x( pg, svget4(zq1, 3), alpha_ );
                    svfloat32_t z2 = svmul_n_f32_x( pg, svget4(zq2, 3), alpha_ );
                    svfloat32_t z3 = svmul_n_f32_x( pg, svget4(zq3, 3), alpha_ );

                    svfloat32x2_t zq_c02 = svld1_f32_x2( pg_c, c_ptr_0 + 3 * rs_c );
                    z0 = svmla_n_f32_x( pg, z0, svget2(zq_c02, 0), beta_ );
                    z2 = svmla_n_f32_x( pg, z2, svget2(zq_c02, 1), beta_ );
                    svst1_f32_x2( pg_c, c_ptr_0 + 3 * rs_c, svcreate2( z0, z2 ) );

                    svfloat32x2_t zq_c13 = svld1_f32_x2( pg_c, c_ptr_1 + 3 * rs_c );
                    z1 = svmla_n_f32_x( pg, z1, svget2(zq_c13, 0), beta_ );
                    z3 = svmla_n_f32_x( pg, z3, svget2(zq_c13, 1), beta_ );
                    svst1_f32_x2( pg_c, c_ptr_1 + 3 * rs_c, svcreate2( z1, z3 ) );
                }
                
                // Step pointers for next 4 rows
                c_ptr_0 += 4 * rs_c;
                c_ptr_1 += 4 * rs_c;
            }
        }
    }
    else 
    {
        // EDGE PATH
        if (beta_ == 0) {
            for ( uint64_t trow = 0; trow < SVL; trow += 1 )
            {
                bool valid_row_0 = (trow < m);
                bool valid_row_1 = (trow + SVL < m);

                // Top Half (Tiles 0 & 2)
                if (valid_row_0) 
                {
                    svfloat32_t z0 = svread_hor_za32_m( svundef_f32(), pg_N0, 0, trow );
                    svfloat32_t z2 = svread_hor_za32_m( svundef_f32(), pg_N1, 2, trow );

                    z0 = svmul_n_f32_z( pg_N0, z0, alpha_ );
                    z2 = svmul_n_f32_z( pg_N1, z2, alpha_ );

                    float *c_ptr_0 = c_ + trow * rs_c;
                    float *c_ptr_2 = c_ + SVL + trow * rs_c;

                    svst1_f32( pg_N0, c_ptr_0, z0 );
                    svst1_f32( pg_N1, c_ptr_2, z2 );
                }

                // Bottom Half (Tiles 1 & 3)
                if (valid_row_1) 
                {
                    svfloat32_t z1 = svread_hor_za32_m( svundef_f32(), pg_N0, 1, trow );
                    svfloat32_t z3 = svread_hor_za32_m( svundef_f32(), pg_N1, 3, trow );

                    z1 = svmul_n_f32_z( pg_N0, z1, alpha_ );
                    z3 = svmul_n_f32_z( pg_N1, z3, alpha_ );

                    float *c_ptr_1 = c_ + SVL * rs_c + trow * rs_c;
                    float *c_ptr_3 = c_ + SVL * rs_c + SVL + trow * rs_c;

                    svst1_f32( pg_N0, c_ptr_1, z1 );
                    svst1_f32( pg_N1, c_ptr_3, z3 );
                }
            }
        }
        else {
            for ( uint64_t trow = 0; trow < SVL; trow += 1 )
            {
                bool valid_row_0 = (trow < m);
                bool valid_row_1 = (trow + SVL < m);

                // Top Half (Tiles 0 & 2)
                if (valid_row_0) 
                {
                    svfloat32_t z0 = svread_hor_za32_m( svundef_f32(), pg_N0, 0, trow );
                    svfloat32_t z2 = svread_hor_za32_m( svundef_f32(), pg_N1, 2, trow );

                    z0 = svmul_n_f32_z( pg_N0, z0, alpha_ );
                    z2 = svmul_n_f32_z( pg_N1, z2, alpha_ );

                    float *c_ptr_0 = c_ + trow * rs_c;
                    float *c_ptr_2 = c_ + SVL + trow * rs_c;

                    z0 = svmla_n_f32_m( pg_N0, z0, svld1_f32(pg_N0, c_ptr_0), beta_ );
                    z2 = svmla_n_f32_m( pg_N1, z2, svld1_f32(pg_N1, c_ptr_2), beta_ );

                    svst1_f32( pg_N0, c_ptr_0, z0 );
                    svst1_f32( pg_N1, c_ptr_2, z2 );
                }

                // Bottom Half (Tiles 1 & 3)
                if (valid_row_1) 
                {
                    svfloat32_t z1 = svread_hor_za32_m( svundef_f32(), pg_N0, 1, trow );
                    svfloat32_t z3 = svread_hor_za32_m( svundef_f32(), pg_N1, 3, trow );

                    z1 = svmul_n_f32_z( pg_N0, z1, alpha_ );
                    z3 = svmul_n_f32_z( pg_N1, z3, alpha_ );

                    float *c_ptr_1 = c_ + SVL * rs_c + trow * rs_c;
                    float *c_ptr_3 = c_ + SVL * rs_c + SVL + trow * rs_c;

                    z1 = svmla_n_f32_m( pg_N0, z1, svld1_f32(pg_N0, c_ptr_1), beta_ );
                    z3 = svmla_n_f32_m( pg_N1, z3, svld1_f32(pg_N1, c_ptr_3), beta_ );

                    svst1_f32( pg_N0, c_ptr_1, z1 );
                    svst1_f32( pg_N1, c_ptr_3, z3 );
                }
            }
        }

    }
}