#include <arm_acle.h>
#include <arm_sme.h>
#include "blis.h"

__arm_new( "za" ) __arm_locally_streaming void bli_dgemmtrsm_l_armsme_int_2SVLx4SVL
	(
			  dim_t      m,
			  dim_t      n,
			  dim_t      k,
		const double*    alpha,
		const double*    a10, const double* a11,
		const double*    b01, const double* b11,
			  double*    c11, inc_t rs_c, inc_t cs_c,
		const auxinfo_t* data,
		const cntx_t*    cntx
	) 
{
	uint64_t SVL = svcntd();
	GEMMTRSM_UKR_SETUP_CT_AMBI( d, 2 * SVL, 4 * SVL, false );

	// =========================================================================
	// Phase 1: GEMM Update (ZA = A10 * B01)
	// =========================================================================

	double *a_ = (double *)a10;
	double *b_ = (double *)b01;

	double *a_next = (double *)bli_auxinfo_next_a( data );
	double *b_next = (double *)bli_auxinfo_next_b( data );

	double *c_ = (double *)c11;

	const uint64_t c0 = 0;
	const uint64_t c1 = c0 + SVL;
	const uint64_t c2 = c1 + SVL;
	const uint64_t c3 = c2 + SVL;

	if ( cs_c != 1 )
	{
		for ( int i = 0; i < 8; i++ )
		{
			__pldx( 1, 1, 0, (double *)&c_[c0 + i * cs_c] );
			__pldx( 1, 1, 0, (double *)&c_[c1 + i * cs_c] );
			__pldx( 1, 1, 0, (double *)&c_[c2 + i * cs_c] );
			__pldx( 1, 1, 0, (double *)&c_[c3 + i * cs_c] );
		}
	}
	else
	{
		for ( int i = 0; i < 8; i++ )
		{
			__pldx( 1, 1, 0, (double *)&c_[c0 + i * rs_c] );
			__pldx( 1, 1, 0, (double *)&c_[c1 + i * rs_c] );
			__pldx( 1, 1, 0, (double *)&c_[c2 + i * rs_c] );
			__pldx( 1, 1, 0, (double *)&c_[c3 + i * rs_c] );
		}
	}

	svzero_za();
	
	uint64_t k_;
	uint64_t k_iter = k / 4;
	uint64_t k_left = k % 4;

	for ( k_ = 0; k_ < k_iter; k_++ )
	{
		// --- k step 0 ---
		svfloat64x2_t zL00 = svld1_f64_x2( svptrue_c64(), (float64_t *)( &a_[0] ) );
		svfloat64x4_t zR00 = svld1_f64_x4( svptrue_c64(), (float64_t *)( &b_[0] ) );

		__pldx( 0, 1, 1, (double *)&a_next[0] );
		__pldx( 0, 1, 1, (double *)&b_next[0] );

		svmopa_za64_m( 0, svptrue_b64(), svptrue_b64(), svget2( zL00, 0 ), svget4( zR00, 0 ) );
		svmopa_za64_m( 1, svptrue_b64(), svptrue_b64(), svget2( zL00, 1 ), svget4( zR00, 0 ) );
		svmopa_za64_m( 2, svptrue_b64(), svptrue_b64(), svget2( zL00, 0 ), svget4( zR00, 1 ) );
		svmopa_za64_m( 3, svptrue_b64(), svptrue_b64(), svget2( zL00, 1 ), svget4( zR00, 1 ) );
		svmopa_za64_m( 4, svptrue_b64(), svptrue_b64(), svget2( zL00, 0 ), svget4( zR00, 2 ) );
		svmopa_za64_m( 5, svptrue_b64(), svptrue_b64(), svget2( zL00, 1 ), svget4( zR00, 2 ) );
		svmopa_za64_m( 6, svptrue_b64(), svptrue_b64(), svget2( zL00, 0 ), svget4( zR00, 3 ) );
		svmopa_za64_m( 7, svptrue_b64(), svptrue_b64(), svget2( zL00, 1 ), svget4( zR00, 3 ) );

		// --- k step 1 ---
		svfloat64x2_t zL01 = svld1_f64_x2( svptrue_c64(), (float64_t *)( &a_[2 * SVL] ) );
		svfloat64x4_t zR01 = svld1_f64_x4( svptrue_c64(), (float64_t *)( &b_[4 * SVL] ) );

		svmopa_za64_m( 0, svptrue_b64(), svptrue_b64(), svget2( zL01, 0 ), svget4( zR01, 0 ) );
		svmopa_za64_m( 1, svptrue_b64(), svptrue_b64(), svget2( zL01, 1 ), svget4( zR01, 0 ) );
		svmopa_za64_m( 2, svptrue_b64(), svptrue_b64(), svget2( zL01, 0 ), svget4( zR01, 1 ) );
		svmopa_za64_m( 3, svptrue_b64(), svptrue_b64(), svget2( zL01, 1 ), svget4( zR01, 1 ) );
		svmopa_za64_m( 4, svptrue_b64(), svptrue_b64(), svget2( zL01, 0 ), svget4( zR01, 2 ) );
		svmopa_za64_m( 5, svptrue_b64(), svptrue_b64(), svget2( zL01, 1 ), svget4( zR01, 2 ) );
		svmopa_za64_m( 6, svptrue_b64(), svptrue_b64(), svget2( zL01, 0 ), svget4( zR01, 3 ) );
		svmopa_za64_m( 7, svptrue_b64(), svptrue_b64(), svget2( zL01, 1 ), svget4( zR01, 3 ) );

		// --- k step 2 ---
		svfloat64x2_t zL02 = svld1_f64_x2( svptrue_c64(), (float64_t *)( &a_[4 * SVL] ) );
		svfloat64x4_t zR02 = svld1_f64_x4( svptrue_c64(), (float64_t *)( &b_[8 * SVL] ) );

		__pldx( 0, 1, 1, (double *)&a_next[4 * SVL] );
		__pldx( 0, 1, 1, (double *)&b_next[8 * SVL] );

		svmopa_za64_m( 0, svptrue_b64(), svptrue_b64(), svget2( zL02, 0 ), svget4( zR02, 0 ) );
		svmopa_za64_m( 1, svptrue_b64(), svptrue_b64(), svget2( zL02, 1 ), svget4( zR02, 0 ) );
		svmopa_za64_m( 2, svptrue_b64(), svptrue_b64(), svget2( zL02, 0 ), svget4( zR02, 1 ) );
		svmopa_za64_m( 3, svptrue_b64(), svptrue_b64(), svget2( zL02, 1 ), svget4( zR02, 1 ) );
		svmopa_za64_m( 4, svptrue_b64(), svptrue_b64(), svget2( zL02, 0 ), svget4( zR02, 2 ) );
		svmopa_za64_m( 5, svptrue_b64(), svptrue_b64(), svget2( zL02, 1 ), svget4( zR02, 2 ) );
		svmopa_za64_m( 6, svptrue_b64(), svptrue_b64(), svget2( zL02, 0 ), svget4( zR02, 3 ) );
		svmopa_za64_m( 7, svptrue_b64(), svptrue_b64(), svget2( zL02, 1 ), svget4( zR02, 3 ) );

		// --- k step 3 ---
		svfloat64x2_t zL03 = svld1_f64_x2( svptrue_c64(), (float64_t *)( &a_[6 * SVL] ) );
		svfloat64x4_t zR03 = svld1_f64_x4( svptrue_c64(), (float64_t *)( &b_[12 * SVL] ) );

		svmopa_za64_m( 0, svptrue_b64(), svptrue_b64(), svget2( zL03, 0 ), svget4( zR03, 0 ) );
		svmopa_za64_m( 1, svptrue_b64(), svptrue_b64(), svget2( zL03, 1 ), svget4( zR03, 0 ) );
		svmopa_za64_m( 2, svptrue_b64(), svptrue_b64(), svget2( zL03, 0 ), svget4( zR03, 1 ) );
		svmopa_za64_m( 3, svptrue_b64(), svptrue_b64(), svget2( zL03, 1 ), svget4( zR03, 1 ) );
		svmopa_za64_m( 4, svptrue_b64(), svptrue_b64(), svget2( zL03, 0 ), svget4( zR03, 2 ) );
		svmopa_za64_m( 5, svptrue_b64(), svptrue_b64(), svget2( zL03, 1 ), svget4( zR03, 2 ) );
		svmopa_za64_m( 6, svptrue_b64(), svptrue_b64(), svget2( zL03, 0 ), svget4( zR03, 3 ) );
		svmopa_za64_m( 7, svptrue_b64(), svptrue_b64(), svget2( zL03, 1 ), svget4( zR03, 3 ) );

		a_ += ( 2 * 4 * SVL );
		b_ += ( 4 * 4 * SVL );

		a_next += ( 2 * 4 * SVL );
		b_next += ( 4 * 4 * SVL );
	}

	for ( k_ = 0; k_ < k_left; k_++ )
	{
		svfloat64x2_t zL00 = svld1_f64_x2( svptrue_c64(), (float64_t *)( &a_[0] ) );
		svfloat64x4_t zR00 = svld1_f64_x4( svptrue_c64(), (float64_t *)( &b_[0] ) );

		svmopa_za64_m( 0, svptrue_b64(), svptrue_b64(), svget2( zL00, 0 ), svget4( zR00, 0 ) );
		svmopa_za64_m( 1, svptrue_b64(), svptrue_b64(), svget2( zL00, 1 ), svget4( zR00, 0 ) );
		svmopa_za64_m( 2, svptrue_b64(), svptrue_b64(), svget2( zL00, 0 ), svget4( zR00, 1 ) );
		svmopa_za64_m( 3, svptrue_b64(), svptrue_b64(), svget2( zL00, 1 ), svget4( zR00, 1 ) );
		svmopa_za64_m( 4, svptrue_b64(), svptrue_b64(), svget2( zL00, 0 ), svget4( zR00, 2 ) );
		svmopa_za64_m( 5, svptrue_b64(), svptrue_b64(), svget2( zL00, 1 ), svget4( zR00, 2 ) );
		svmopa_za64_m( 6, svptrue_b64(), svptrue_b64(), svget2( zL00, 0 ), svget4( zR00, 3 ) );
		svmopa_za64_m( 7, svptrue_b64(), svptrue_b64(), svget2( zL00, 1 ), svget4( zR00, 3 ) );

		a_ += ( 2 * SVL );
		b_ += ( 4 * SVL );
	}

	// =========================================================================
	// Phase 2: TRSM (Left-Lower Solve)
	// Solve A11 * X = alpha * B11 - ZA
	// =========================================================================

	double alpha_ = *alpha;
	double *b11_ = (double *)b11;
	const double *a11_ = (const double *)a11;
	
	svuint64_t z_indices = svindex_u64( 0, 1 );

	// Predicates for writing to the final C11 output
	svbool_t p_n_0 = svwhilelt_b64_u64( 0 * SVL, n );
	svbool_t p_n_1 = svwhilelt_b64_u64( 1 * SVL, n );
	svbool_t p_n_2 = svwhilelt_b64_u64( 2 * SVL, n );
	svbool_t p_n_3 = svwhilelt_b64_u64( 3 * SVL, n );

	for ( dim_t i = 0; i < m; i++ )
	{
		svfloat64_t z_za_0, z_za_1, z_za_2, z_za_3;

		// 1. Read row i of the accumulator from ZA
		if ( i < SVL )
		{
			z_za_0 = svread_hor_za64_m( svundef_f64(), svptrue_b64(), 0, i );
			z_za_1 = svread_hor_za64_m( svundef_f64(), svptrue_b64(), 2, i );
			z_za_2 = svread_hor_za64_m( svundef_f64(), svptrue_b64(), 4, i );
			z_za_3 = svread_hor_za64_m( svundef_f64(), svptrue_b64(), 6, i );
		}
		else
		{
			z_za_0 = svread_hor_za64_m( svundef_f64(), svptrue_b64(), 1, i - SVL );
			z_za_1 = svread_hor_za64_m( svundef_f64(), svptrue_b64(), 3, i - SVL );
			z_za_2 = svread_hor_za64_m( svundef_f64(), svptrue_b64(), 5, i - SVL );
			z_za_3 = svread_hor_za64_m( svundef_f64(), svptrue_b64(), 7, i - SVL );
		}

		// 2. Load input RHS strictly from packed B11
		svfloat64_t z_c_0 = svld1_f64( svptrue_b64(), &b11_[i * (4 * SVL) + 0 * SVL] );
		svfloat64_t z_c_1 = svld1_f64( svptrue_b64(), &b11_[i * (4 * SVL) + 1 * SVL] );
		svfloat64_t z_c_2 = svld1_f64( svptrue_b64(), &b11_[i * (4 * SVL) + 2 * SVL] );
		svfloat64_t z_c_3 = svld1_f64( svptrue_b64(), &b11_[i * (4 * SVL) + 3 * SVL] );

		// Scale by alpha
		z_c_0 = svmul_n_f64_z( svptrue_b64(), z_c_0, alpha_ );
		z_c_1 = svmul_n_f64_z( svptrue_b64(), z_c_1, alpha_ );
		z_c_2 = svmul_n_f64_z( svptrue_b64(), z_c_2, alpha_ );
		z_c_3 = svmul_n_f64_z( svptrue_b64(), z_c_3, alpha_ );

		// 3. Compute RHS: alpha * B11 - ZA
		svfloat64_t z_x_0 = svsub_f64_z( svptrue_b64(), z_c_0, z_za_0 );
		svfloat64_t z_x_1 = svsub_f64_z( svptrue_b64(), z_c_1, z_za_1 );
		svfloat64_t z_x_2 = svsub_f64_z( svptrue_b64(), z_c_2, z_za_2 );
		svfloat64_t z_x_3 = svsub_f64_z( svptrue_b64(), z_c_3, z_za_3 );

		// 4. Multiply by inverse of diagonal: X = RHS * A11[i, i]
		double diag_inv = a11_[i * (2 * SVL) + i];
		z_x_0 = svmul_n_f64_z( svptrue_b64(), z_x_0, diag_inv );
		z_x_1 = svmul_n_f64_z( svptrue_b64(), z_x_1, diag_inv );
		z_x_2 = svmul_n_f64_z( svptrue_b64(), z_x_2, diag_inv );
		z_x_3 = svmul_n_f64_z( svptrue_b64(), z_x_3, diag_inv );

		// 5. Store X back to packed buffer B11
		svst1_f64( svptrue_b64(), &b11_[i * (4 * SVL) + 0 * SVL], z_x_0 );
		svst1_f64( svptrue_b64(), &b11_[i * (4 * SVL) + 1 * SVL], z_x_1 );
		svst1_f64( svptrue_b64(), &b11_[i * (4 * SVL) + 2 * SVL], z_x_2 );
		svst1_f64( svptrue_b64(), &b11_[i * (4 * SVL) + 3 * SVL], z_x_3 );

		// 6. Direct contiguous store if row-major (cs_c == 1)
		if ( cs_c == 1 )
		{
			double *c_row = &c11[i * rs_c];
			svst1_f64( p_n_0, c_row + 0 * SVL, z_x_0 );
			svst1_f64( p_n_1, c_row + 1 * SVL, z_x_1 );
			svst1_f64( p_n_2, c_row + 2 * SVL, z_x_2 );
			svst1_f64( p_n_3, c_row + 3 * SVL, z_x_3 );
		}

		// 7. Rank-1 Update of ZA for the remaining rows strictly BELOW i
		if ( i < m - 1 )
		{
			svfloat64_t z_a11_top = svld1_f64( svptrue_b64(), &a11_[i * (2 * SVL) + 0] );
			svfloat64_t z_a11_bot = svld1_f64( svptrue_b64(), &a11_[i * (2 * SVL) + SVL] );

			svbool_t p_m_top = svcmpgt_n_u64( svptrue_b64(), z_indices, i );
			svbool_t p_m_bot = ( i < SVL ) ? svptrue_b64() : svcmpgt_n_u64( svptrue_b64(), z_indices, i - SVL );
			
			if ( i < SVL )
			{
				svmopa_za64_m( 0, p_m_top, svptrue_b64(), z_a11_top, z_x_0 );
				svmopa_za64_m( 2, p_m_top, svptrue_b64(), z_a11_top, z_x_1 );
				svmopa_za64_m( 4, p_m_top, svptrue_b64(), z_a11_top, z_x_2 );
				svmopa_za64_m( 6, p_m_top, svptrue_b64(), z_a11_top, z_x_3 );
			}
			svmopa_za64_m( 1, p_m_bot, svptrue_b64(), z_a11_bot, z_x_0 );
			svmopa_za64_m( 3, p_m_bot, svptrue_b64(), z_a11_bot, z_x_1 );
			svmopa_za64_m( 5, p_m_bot, svptrue_b64(), z_a11_bot, z_x_2 );
			svmopa_za64_m( 7, p_m_bot, svptrue_b64(), z_a11_bot, z_x_3 );
		}
	}

	// =========================================================================
	// Deferred non-contiguous store into C11
	// =========================================================================
	if ( cs_c != 1 )
	{
		// Column-Major: Transpose using ZA
		// 1. Write the computed rows horizontally into ZA tiles
		for ( int64_t r = 0; r < m; r++ )
		{
			svfloat64_t row_0 = svld1_f64( svptrue_b64(), &b11_[r * (4 * SVL) + 0 * SVL] );
			svfloat64_t row_1 = svld1_f64( svptrue_b64(), &b11_[r * (4 * SVL) + 1 * SVL] );
			svfloat64_t row_2 = svld1_f64( svptrue_b64(), &b11_[r * (4 * SVL) + 2 * SVL] );
			svfloat64_t row_3 = svld1_f64( svptrue_b64(), &b11_[r * (4 * SVL) + 3 * SVL] );

			if ( r < SVL )
			{
				svwrite_hor_za64_m( 0, r, svptrue_b64(), row_0 );
				svwrite_hor_za64_m( 2, r, svptrue_b64(), row_1 );
				svwrite_hor_za64_m( 4, r, svptrue_b64(), row_2 );
				svwrite_hor_za64_m( 6, r, svptrue_b64(), row_3 );
			}
			else
			{
				svwrite_hor_za64_m( 1, r - SVL, svptrue_b64(), row_0 );
				svwrite_hor_za64_m( 3, r - SVL, svptrue_b64(), row_1 );
				svwrite_hor_za64_m( 5, r - SVL, svptrue_b64(), row_2 );
				svwrite_hor_za64_m( 7, r - SVL, svptrue_b64(), row_3 );
			}
		}

		svbool_t p_m_top = svwhilelt_b64_u64( 0, m );
		svbool_t p_m_bot = svwhilelt_b64_u64( SVL, m );

		// 2. Read vertically and write contiguous column vectors into C11
		for ( dim_t col = 0; col < n; col++ )
		{
			double *c_col = &c11[col * cs_c];

			if ( col < SVL )
			{
				svfloat64_t col_top = svread_ver_za64_m( svundef_f64(), svptrue_b64(), 0, col );
				svst1_f64( p_m_top, c_col, col_top );
				if ( m > SVL )
				{
					svfloat64_t col_bot = svread_ver_za64_m( svundef_f64(), svptrue_b64(), 1, col );
					svst1_f64( p_m_bot, c_col + SVL, col_bot );
				}
			}
			else if ( col < 2 * SVL )
			{
				svfloat64_t col_top = svread_ver_za64_m( svundef_f64(), svptrue_b64(), 2, col - SVL );
				svst1_f64( p_m_top, c_col, col_top );
				if ( m > SVL )
				{
					svfloat64_t col_bot = svread_ver_za64_m( svundef_f64(), svptrue_b64(), 3, col - SVL );
					svst1_f64( p_m_bot, c_col + SVL, col_bot );
				}
			}
			else if ( col < 3 * SVL )
			{
				svfloat64_t col_top = svread_ver_za64_m( svundef_f64(), svptrue_b64(), 4, col - 2 * SVL );
				svst1_f64( p_m_top, c_col, col_top );
				if ( m > SVL )
				{
					svfloat64_t col_bot = svread_ver_za64_m( svundef_f64(), svptrue_b64(), 5, col - 2 * SVL );
					svst1_f64( p_m_bot, c_col + SVL, col_bot );
				}
			}
			else
			{
				svfloat64_t col_top = svread_ver_za64_m( svundef_f64(), svptrue_b64(), 6, col - 3 * SVL );
				svst1_f64( p_m_top, c_col, col_top );
				if ( m > SVL )
				{
					svfloat64_t col_bot = svread_ver_za64_m( svundef_f64(), svptrue_b64(), 7, col - 3 * SVL );
					svst1_f64( p_m_bot, c_col + SVL, col_bot );
				}
			}
		}
	}

	// =========================================================================
	// Phase 3: Zero-pad the unwritten rows of B11 for the macro-kernel
	// =========================================================================
	for ( dim_t pad_i = m; pad_i < 2 * SVL; pad_i++ )
	{
		svst1_f64( svptrue_b64(), &b11_[pad_i * (4 * SVL) + 0 * SVL], svdup_n_f64( 0.0 ) );
		svst1_f64( svptrue_b64(), &b11_[pad_i * (4 * SVL) + 1 * SVL], svdup_n_f64( 0.0 ) );
		svst1_f64( svptrue_b64(), &b11_[pad_i * (4 * SVL) + 2 * SVL], svdup_n_f64( 0.0 ) );
		svst1_f64( svptrue_b64(), &b11_[pad_i * (4 * SVL) + 3 * SVL], svdup_n_f64( 0.0 ) );
	}

	GEMMTRSM_UKR_FLUSH_CT( d );
	return;
}