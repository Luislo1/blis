#include "arm_sme.h"
#include "blis.h"

__arm_new( "za" ) __arm_locally_streaming void bli_dgemm_armsme_sup_2SVLx4SVL
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
    
    svzero_za();
    // Define all loop-invariant predicates ONCE
    svbool_t pg_M0 = svwhilelt_b64((uint64_t)0, (uint64_t)m);
    svbool_t pg_M1 = svwhilelt_b64((uint64_t)SVL, (uint64_t)m);
    
    svbool_t pg_N0 = svwhilelt_b64((uint64_t)0, (uint64_t)n);
    svbool_t pg_N1 = svwhilelt_b64((uint64_t)SVL, (uint64_t)n);
    svbool_t pg_N2 = svwhilelt_b64((uint64_t)(2 * SVL), (uint64_t)n);
    svbool_t pg_N3 = svwhilelt_b64((uint64_t)(3 * SVL), (uint64_t)n);

    svbool_t  pg   = svptrue_b64();
    svcount_t pg_c = svptrue_c64();

    uint64_t k_iter = k / 4;
    uint64_t k_left = k % 4;

    const double *a_ptr = (const double *)a;
    const double *b_ptr = (const double *)b;

    // K-LOOP 
    for (uint64_t k_ = 0; k_ < k_iter; k_++ )
    {
        // PRE-LOAD STEP 0 & 1 (A: 2 vectors, B: 4 vectors)
        svfloat64x2_t zL0 = svld1_f64_x2(pg_c, a_ptr + 0 * cs_a);
        svfloat64x4_t zR0 = svld1_f64_x4(pg_c, b_ptr + 0 * rs_b);
        
        svfloat64x2_t zL1 = svld1_f64_x2(pg_c, a_ptr + 1 * cs_a);
        svfloat64x4_t zR1 = svld1_f64_x4(pg_c, b_ptr + 1 * rs_b);

        // MATH STEP 0
        svmopa_za64_m( 0, pg_M0, pg_N0, svget2(zL0, 0), svget4(zR0, 0));
        svmopa_za64_m( 1, pg_M1, pg_N0, svget2(zL0, 1), svget4(zR0, 0));
        svmopa_za64_m( 2, pg_M0, pg_N1, svget2(zL0, 0), svget4(zR0, 1));
        svmopa_za64_m( 3, pg_M1, pg_N1, svget2(zL0, 1), svget4(zR0, 1));
        svmopa_za64_m( 4, pg_M0, pg_N2, svget2(zL0, 0), svget4(zR0, 2));
        svmopa_za64_m( 5, pg_M1, pg_N2, svget2(zL0, 1), svget4(zR0, 2));
        svmopa_za64_m( 6, pg_M0, pg_N3, svget2(zL0, 0), svget4(zR0, 3));
        svmopa_za64_m( 7, pg_M1, pg_N3, svget2(zL0, 1), svget4(zR0, 3));

        // LOAD STEP 2
        svfloat64x2_t zL2 = svld1_f64_x2(pg_c, a_ptr + 2 * cs_a);
        svfloat64x4_t zR2 = svld1_f64_x4(pg_c, b_ptr + 2 * rs_b);

        // MATH STEP 1
        svmopa_za64_m( 0, pg_M0, pg_N0, svget2(zL1, 0), svget4(zR1, 0));
        svmopa_za64_m( 1, pg_M1, pg_N0, svget2(zL1, 1), svget4(zR1, 0));
        svmopa_za64_m( 2, pg_M0, pg_N1, svget2(zL1, 0), svget4(zR1, 1));
        svmopa_za64_m( 3, pg_M1, pg_N1, svget2(zL1, 1), svget4(zR1, 1));
        svmopa_za64_m( 4, pg_M0, pg_N2, svget2(zL1, 0), svget4(zR1, 2));
        svmopa_za64_m( 5, pg_M1, pg_N2, svget2(zL1, 1), svget4(zR1, 2));
        svmopa_za64_m( 6, pg_M0, pg_N3, svget2(zL1, 0), svget4(zR1, 3));
        svmopa_za64_m( 7, pg_M1, pg_N3, svget2(zL1, 1), svget4(zR1, 3));

        // LOAD STEP 3
        svfloat64x2_t zL3 = svld1_f64_x2(pg_c, a_ptr + 3 * cs_a);
        svfloat64x4_t zR3 = svld1_f64_x4(pg_c, b_ptr + 3 * rs_b);

        // MATH STEP 2
        svmopa_za64_m( 0, pg_M0, pg_N0, svget2(zL2, 0), svget4(zR2, 0));
        svmopa_za64_m( 1, pg_M1, pg_N0, svget2(zL2, 1), svget4(zR2, 0));
        svmopa_za64_m( 2, pg_M0, pg_N1, svget2(zL2, 0), svget4(zR2, 1));
        svmopa_za64_m( 3, pg_M1, pg_N1, svget2(zL2, 1), svget4(zR2, 1));
        svmopa_za64_m( 4, pg_M0, pg_N2, svget2(zL2, 0), svget4(zR2, 2));
        svmopa_za64_m( 5, pg_M1, pg_N2, svget2(zL2, 1), svget4(zR2, 2));
        svmopa_za64_m( 6, pg_M0, pg_N3, svget2(zL2, 0), svget4(zR2, 3));
        svmopa_za64_m( 7, pg_M1, pg_N3, svget2(zL2, 1), svget4(zR2, 3));

        // MATH STEP 3
        svmopa_za64_m( 0, pg_M0, pg_N0, svget2(zL3, 0), svget4(zR3, 0));
        svmopa_za64_m( 1, pg_M1, pg_N0, svget2(zL3, 1), svget4(zR3, 0));
        svmopa_za64_m( 2, pg_M0, pg_N1, svget2(zL3, 0), svget4(zR3, 1));
        svmopa_za64_m( 3, pg_M1, pg_N1, svget2(zL3, 1), svget4(zR3, 1));
        svmopa_za64_m( 4, pg_M0, pg_N2, svget2(zL3, 0), svget4(zR3, 2));
        svmopa_za64_m( 5, pg_M1, pg_N2, svget2(zL3, 1), svget4(zR3, 2));
        svmopa_za64_m( 6, pg_M0, pg_N3, svget2(zL3, 0), svget4(zR3, 3));
        svmopa_za64_m( 7, pg_M1, pg_N3, svget2(zL3, 1), svget4(zR3, 3));

        a_ptr += 4 * cs_a;
        b_ptr += 4 * rs_b;  
    }

    // REMAINDER LOOP
    for (uint64_t k_ = 0; k_ < k_left; k_ += 1 )
    {
        svfloat64x2_t zL = svld1_f64_x2(pg_c, a_ptr);
        svfloat64x4_t zR = svld1_f64_x4(pg_c, b_ptr);

        svmopa_za64_m( 0, pg_M0, pg_N0, svget2(zL, 0), svget4(zR, 0));
        svmopa_za64_m( 1, pg_M1, pg_N0, svget2(zL, 1), svget4(zR, 0));
        svmopa_za64_m( 2, pg_M0, pg_N1, svget2(zL, 0), svget4(zR, 1));
        svmopa_za64_m( 3, pg_M1, pg_N1, svget2(zL, 1), svget4(zR, 1));
        svmopa_za64_m( 4, pg_M0, pg_N2, svget2(zL, 0), svget4(zR, 2));
        svmopa_za64_m( 5, pg_M1, pg_N2, svget2(zL, 1), svget4(zR, 2));
        svmopa_za64_m( 6, pg_M0, pg_N3, svget2(zL, 0), svget4(zR, 3));
        svmopa_za64_m( 7, pg_M1, pg_N3, svget2(zL, 1), svget4(zR, 3));

        a_ptr += cs_a;
        b_ptr += rs_b;  
    }

    // EPILOGUE
    double beta_  = *(const double *)beta;
    double alpha_ = *(const double *)alpha;
    double *c_    = (double *)c;

    if (m == 2 * SVL && n == 4 * SVL) 
    {
        // FAST PATH
        double *c_ptr_0 = c_;
        double *c_ptr_1 = c_ + SVL * rs_c;

for ( uint64_t trow = 0; trow < SVL; trow += 4 )
        {
            // Read 4 rows at once out of each of the 8 ZA tiles
            svfloat64x4_t zq0 = svread_hor_za64_f64_vg4( 0, trow );
            svfloat64x4_t zq1 = svread_hor_za64_f64_vg4( 1, trow );
            svfloat64x4_t zq2 = svread_hor_za64_f64_vg4( 2, trow );
            svfloat64x4_t zq3 = svread_hor_za64_f64_vg4( 3, trow );
            svfloat64x4_t zq4 = svread_hor_za64_f64_vg4( 4, trow );
            svfloat64x4_t zq5 = svread_hor_za64_f64_vg4( 5, trow );
            svfloat64x4_t zq6 = svread_hor_za64_f64_vg4( 6, trow );
            svfloat64x4_t zq7 = svread_hor_za64_f64_vg4( 7, trow );

            // Row 0 (trow + 0)
            {
                // Top Half (Tiles 0, 2, 4, 6)
                svfloat64_t z0 = svmul_n_f64_x( pg, svget4(zq0, 0), alpha_ );
                svfloat64_t z2 = svmul_n_f64_x( pg, svget4(zq2, 0), alpha_ );
                svfloat64_t z4 = svmul_n_f64_x( pg, svget4(zq4, 0), alpha_ );
                svfloat64_t z6 = svmul_n_f64_x( pg, svget4(zq6, 0), alpha_ );

                svfloat64x4_t zq_c0 = svld1_f64_x4( pg_c, c_ptr_0 + 0 * rs_c );
                z0 = svmla_n_f64_x( pg, z0, svget4(zq_c0, 0), beta_ );
                z2 = svmla_n_f64_x( pg, z2, svget4(zq_c0, 1), beta_ );
                z4 = svmla_n_f64_x( pg, z4, svget4(zq_c0, 2), beta_ );
                z6 = svmla_n_f64_x( pg, z6, svget4(zq_c0, 3), beta_ );
                svst1_f64_x4( pg_c, c_ptr_0 + 0 * rs_c, svcreate4( z0, z2, z4, z6 ) );

                // Bottom Half (Tiles 1, 3, 5, 7)
                svfloat64_t z1 = svmul_n_f64_x( pg, svget4(zq1, 0), alpha_ );
                svfloat64_t z3 = svmul_n_f64_x( pg, svget4(zq3, 0), alpha_ );
                svfloat64_t z5 = svmul_n_f64_x( pg, svget4(zq5, 0), alpha_ );
                svfloat64_t z7 = svmul_n_f64_x( pg, svget4(zq7, 0), alpha_ );

                svfloat64x4_t zq_c1 = svld1_f64_x4( pg_c, c_ptr_1 + 0 * rs_c );
                z1 = svmla_n_f64_x( pg, z1, svget4(zq_c1, 0), beta_ );
                z3 = svmla_n_f64_x( pg, z3, svget4(zq_c1, 1), beta_ );
                z5 = svmla_n_f64_x( pg, z5, svget4(zq_c1, 2), beta_ );
                z7 = svmla_n_f64_x( pg, z7, svget4(zq_c1, 3), beta_ );
                svst1_f64_x4( pg_c, c_ptr_1 + 0 * rs_c, svcreate4( z1, z3, z5, z7 ) );
            }

            // Row 1 (trow + 1)
            {
                // Top Half (Tiles 0, 2, 4, 6)
                svfloat64_t z0 = svmul_n_f64_x( pg, svget4(zq0, 1), alpha_ );
                svfloat64_t z2 = svmul_n_f64_x( pg, svget4(zq2, 1), alpha_ );
                svfloat64_t z4 = svmul_n_f64_x( pg, svget4(zq4, 1), alpha_ );
                svfloat64_t z6 = svmul_n_f64_x( pg, svget4(zq6, 1), alpha_ );

                svfloat64x4_t zq_c0 = svld1_f64_x4( pg_c, c_ptr_0 + 1 * rs_c );
                z0 = svmla_n_f64_x( pg, z0, svget4(zq_c0, 0), beta_ );
                z2 = svmla_n_f64_x( pg, z2, svget4(zq_c0, 1), beta_ );
                z4 = svmla_n_f64_x( pg, z4, svget4(zq_c0, 2), beta_ );
                z6 = svmla_n_f64_x( pg, z6, svget4(zq_c0, 3), beta_ );
                svst1_f64_x4( pg_c, c_ptr_0 + 1 * rs_c, svcreate4( z0, z2, z4, z6 ) );

                // Bottom Half (Tiles 1, 3, 5, 7)
                svfloat64_t z1 = svmul_n_f64_x( pg, svget4(zq1, 1), alpha_ );
                svfloat64_t z3 = svmul_n_f64_x( pg, svget4(zq3, 1), alpha_ );
                svfloat64_t z5 = svmul_n_f64_x( pg, svget4(zq5, 1), alpha_ );
                svfloat64_t z7 = svmul_n_f64_x( pg, svget4(zq7, 1), alpha_ );

                svfloat64x4_t zq_c1 = svld1_f64_x4( pg_c, c_ptr_1 + 1 * rs_c );
                z1 = svmla_n_f64_x( pg, z1, svget4(zq_c1, 0), beta_ );
                z3 = svmla_n_f64_x( pg, z3, svget4(zq_c1, 1), beta_ );
                z5 = svmla_n_f64_x( pg, z5, svget4(zq_c1, 2), beta_ );
                z7 = svmla_n_f64_x( pg, z7, svget4(zq_c1, 3), beta_ );
                svst1_f64_x4( pg_c, c_ptr_1 + 1 * rs_c, svcreate4( z1, z3, z5, z7 ) );
            }

            // Row 2 (trow + 2)
            {
                // Top Half (Tiles 0, 2, 4, 6)
                svfloat64_t z0 = svmul_n_f64_x( pg, svget4(zq0, 2), alpha_ );
                svfloat64_t z2 = svmul_n_f64_x( pg, svget4(zq2, 2), alpha_ );
                svfloat64_t z4 = svmul_n_f64_x( pg, svget4(zq4, 2), alpha_ );
                svfloat64_t z6 = svmul_n_f64_x( pg, svget4(zq6, 2), alpha_ );

                svfloat64x4_t zq_c0 = svld1_f64_x4( pg_c, c_ptr_0 + 2 * rs_c );
                z0 = svmla_n_f64_x( pg, z0, svget4(zq_c0, 0), beta_ );
                z2 = svmla_n_f64_x( pg, z2, svget4(zq_c0, 1), beta_ );
                z4 = svmla_n_f64_x( pg, z4, svget4(zq_c0, 2), beta_ );
                z6 = svmla_n_f64_x( pg, z6, svget4(zq_c0, 3), beta_ );
                svst1_f64_x4( pg_c, c_ptr_0 + 2 * rs_c, svcreate4( z0, z2, z4, z6 ) );

                // Bottom Half (Tiles 1, 3, 5, 7)
                svfloat64_t z1 = svmul_n_f64_x( pg, svget4(zq1, 2), alpha_ );
                svfloat64_t z3 = svmul_n_f64_x( pg, svget4(zq3, 2), alpha_ );
                svfloat64_t z5 = svmul_n_f64_x( pg, svget4(zq5, 2), alpha_ );
                svfloat64_t z7 = svmul_n_f64_x( pg, svget4(zq7, 2), alpha_ );

                svfloat64x4_t zq_c1 = svld1_f64_x4( pg_c, c_ptr_1 + 2 * rs_c );
                z1 = svmla_n_f64_x( pg, z1, svget4(zq_c1, 0), beta_ );
                z3 = svmla_n_f64_x( pg, z3, svget4(zq_c1, 1), beta_ );
                z5 = svmla_n_f64_x( pg, z5, svget4(zq_c1, 2), beta_ );
                z7 = svmla_n_f64_x( pg, z7, svget4(zq_c1, 3), beta_ );
                svst1_f64_x4( pg_c, c_ptr_1 + 2 * rs_c, svcreate4( z1, z3, z5, z7 ) );
            }

            // Row 3 (trow + 3)
            {
                // Top Half (Tiles 0, 2, 4, 6)
                svfloat64_t z0 = svmul_n_f64_x( pg, svget4(zq0, 3), alpha_ );
                svfloat64_t z2 = svmul_n_f64_x( pg, svget4(zq2, 3), alpha_ );
                svfloat64_t z4 = svmul_n_f64_x( pg, svget4(zq4, 3), alpha_ );
                svfloat64_t z6 = svmul_n_f64_x( pg, svget4(zq6, 3), alpha_ );

                svfloat64x4_t zq_c0 = svld1_f64_x4( pg_c, c_ptr_0 + 3 * rs_c );
                z0 = svmla_n_f64_x( pg, z0, svget4(zq_c0, 0), beta_ );
                z2 = svmla_n_f64_x( pg, z2, svget4(zq_c0, 1), beta_ );
                z4 = svmla_n_f64_x( pg, z4, svget4(zq_c0, 2), beta_ );
                z6 = svmla_n_f64_x( pg, z6, svget4(zq_c0, 3), beta_ );
                svst1_f64_x4( pg_c, c_ptr_0 + 3 * rs_c, svcreate4( z0, z2, z4, z6 ) );

                // Bottom Half (Tiles 1, 3, 5, 7)
                svfloat64_t z1 = svmul_n_f64_x( pg, svget4(zq1, 3), alpha_ );
                svfloat64_t z3 = svmul_n_f64_x( pg, svget4(zq3, 3), alpha_ );
                svfloat64_t z5 = svmul_n_f64_x( pg, svget4(zq5, 3), alpha_ );
                svfloat64_t z7 = svmul_n_f64_x( pg, svget4(zq7, 3), alpha_ );

                svfloat64x4_t zq_c1 = svld1_f64_x4( pg_c, c_ptr_1 + 3 * rs_c );
                z1 = svmla_n_f64_x( pg, z1, svget4(zq_c1, 0), beta_ );
                z3 = svmla_n_f64_x( pg, z3, svget4(zq_c1, 1), beta_ );
                z5 = svmla_n_f64_x( pg, z5, svget4(zq_c1, 2), beta_ );
                z7 = svmla_n_f64_x( pg, z7, svget4(zq_c1, 3), beta_ );
                svst1_f64_x4( pg_c, c_ptr_1 + 3 * rs_c, svcreate4( z1, z3, z5, z7 ) );
            }
            
            // Step pointers for next 4 rows
            c_ptr_0 += 4 * rs_c;
            c_ptr_1 += 4 * rs_c;
        }
    }
    else 
    {
        // EDGE PATH
        for ( uint64_t trow = 0; trow < SVL; trow += 1 )
        {
            bool valid_row_0 = (trow < m);
            bool valid_row_1 = (trow + SVL < m);

            // Top Half (Tiles 0, 2, 4, 6)
            if (valid_row_0) 
            {
                svfloat64_t z0 = svread_hor_za64_m( svundef_f64(), pg_N0, 0, trow );
                svfloat64_t z2 = svread_hor_za64_m( svundef_f64(), pg_N1, 2, trow );
                svfloat64_t z4 = svread_hor_za64_m( svundef_f64(), pg_N2, 4, trow );
                svfloat64_t z6 = svread_hor_za64_m( svundef_f64(), pg_N3, 6, trow );

                z0 = svmul_n_f64_z( pg_N0, z0, alpha_ );
                z2 = svmul_n_f64_z( pg_N1, z2, alpha_ );
                z4 = svmul_n_f64_z( pg_N2, z4, alpha_ );
                z6 = svmul_n_f64_z( pg_N3, z6, alpha_ );

                double *c_ptr_0 = c_ + trow * rs_c;
                double *c_ptr_2 = c_ + 1 * SVL + trow * rs_c;
                double *c_ptr_4 = c_ + 2 * SVL + trow * rs_c;
                double *c_ptr_6 = c_ + 3 * SVL + trow * rs_c;

                z0 = svmla_n_f64_m( pg_N0, z0, svld1_f64(pg_N0, c_ptr_0), beta_ );
                z2 = svmla_n_f64_m( pg_N1, z2, svld1_f64(pg_N1, c_ptr_2), beta_ );
                z4 = svmla_n_f64_m( pg_N2, z4, svld1_f64(pg_N2, c_ptr_4), beta_ );
                z6 = svmla_n_f64_m( pg_N3, z6, svld1_f64(pg_N3, c_ptr_6), beta_ );

                svst1_f64( pg_N0, c_ptr_0, z0 );
                svst1_f64( pg_N1, c_ptr_2, z2 );
                svst1_f64( pg_N2, c_ptr_4, z4 );
                svst1_f64( pg_N3, c_ptr_6, z6 );
            }

            // Bottom Half (Tiles 1, 3, 5, 7)
            if (valid_row_1) 
            {
                svfloat64_t z1 = svread_hor_za64_m( svundef_f64(), pg_N0, 1, trow );
                svfloat64_t z3 = svread_hor_za64_m( svundef_f64(), pg_N1, 3, trow );
                svfloat64_t z5 = svread_hor_za64_m( svundef_f64(), pg_N2, 5, trow );
                svfloat64_t z7 = svread_hor_za64_m( svundef_f64(), pg_N3, 7, trow );

                z1 = svmul_n_f64_z( pg_N0, z1, alpha_ );
                z3 = svmul_n_f64_z( pg_N1, z3, alpha_ );
                z5 = svmul_n_f64_z( pg_N2, z5, alpha_ );
                z7 = svmul_n_f64_z( pg_N3, z7, alpha_ );

                double *c_ptr_1 = c_ + SVL * rs_c + trow * rs_c;
                double *c_ptr_3 = c_ + SVL * rs_c + 1 * SVL + trow * rs_c;
                double *c_ptr_5 = c_ + SVL * rs_c + 2 * SVL + trow * rs_c;
                double *c_ptr_7 = c_ + SVL * rs_c + 3 * SVL + trow * rs_c;

                z1 = svmla_n_f64_m( pg_N0, z1, svld1_f64(pg_N0, c_ptr_1), beta_ );
                z3 = svmla_n_f64_m( pg_N1, z3, svld1_f64(pg_N1, c_ptr_3), beta_ );
                z5 = svmla_n_f64_m( pg_N2, z5, svld1_f64(pg_N2, c_ptr_5), beta_ );
                z7 = svmla_n_f64_m( pg_N3, z7, svld1_f64(pg_N3, c_ptr_7), beta_ );

                svst1_f64( pg_N0, c_ptr_1, z1 );
                svst1_f64( pg_N1, c_ptr_3, z3 );
                svst1_f64( pg_N2, c_ptr_5, z5 );
                svst1_f64( pg_N3, c_ptr_7, z7 );
            }
        }
    }
}