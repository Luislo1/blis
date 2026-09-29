#include <arm_acle.h>
#include <arm_sme.h>
#include "blis.h"

__arm_new( "za" ) __arm_locally_streaming void bli_sgemmtrsm_l_armsme_int_2SVLx2SVL
	(
			  dim_t      m,
			  dim_t      n,
			  dim_t      k,
		const float*     alpha,
		const float*     a10, const float* a11,
		const float*     b01, const float* b11,
			  float*     c11, inc_t rs_c, inc_t cs_c,
		const auxinfo_t* data,
		const cntx_t*    cntx
	) 
{
	uint64_t SVL = svcntsw();
	GEMMTRSM_UKR_SETUP_CT_AMBI( s, 2 * SVL, 2 * SVL, false );
	//  GEMMTRSM_UKR_SETUP_CT_ANY( s, 2 * SVL, 2 * SVL, false );
	// GEMMTRSM_UKR_SETUP_CT_ALIGNED( s, 2 * SVL, 2 * SVL, false, 64 );

	// =========================================================================
	// Phase 1: GEMM Update (ZA = A10 * B01)
	// =========================================================================

	float *a_ = (float *)a10;
	float *b_ = (float *)b01;

	float *a_next = (float *)bli_auxinfo_next_a( data );
	float *b_next = (float *)bli_auxinfo_next_b( data );

	float *c_ = (float *)c11;

	const uint64_t result_tile_TL_corner_ = 0;
	const uint64_t result_tile_TR_corner_ = result_tile_TL_corner_ + SVL;

	if ( cs_c != 1 )
	{
		__pldx( 1, 1, 0, (float *)&c_[result_tile_TL_corner_ + ( ( ( 0 + 0 ) * cs_c ) )] );
		__pldx( 1, 1, 0, (float *)&c_[result_tile_TL_corner_ + ( ( ( 1 + 0 ) * cs_c ) )] );
		__pldx( 1, 1, 0, (float *)&c_[result_tile_TL_corner_ + ( ( ( 2 + 0 ) * cs_c ) )] );
		__pldx( 1, 1, 0, (float *)&c_[result_tile_TL_corner_ + ( ( ( 3 + 0 ) * cs_c ) )] );
		__pldx( 1, 1, 0, (float *)&c_[result_tile_TL_corner_ + ( ( ( 4 + 0 ) * cs_c ) )] );
		__pldx( 1, 1, 0, (float *)&c_[result_tile_TL_corner_ + ( ( ( 5 + 0 ) * cs_c ) )] );
		__pldx( 1, 1, 0, (float *)&c_[result_tile_TL_corner_ + ( ( ( 6 + 0 ) * cs_c ) )] );
		__pldx( 1, 1, 0, (float *)&c_[result_tile_TL_corner_ + ( ( ( 7 + 0 ) * cs_c ) )] );

		__pldx( 1, 1, 0, (float *)&c_[result_tile_TR_corner_ + ( ( ( 0 + 0 ) * cs_c ) )] );
		__pldx( 1, 1, 0, (float *)&c_[result_tile_TR_corner_ + ( ( ( 1 + 0 ) * cs_c ) )] );
		__pldx( 1, 1, 0, (float *)&c_[result_tile_TR_corner_ + ( ( ( 2 + 0 ) * cs_c ) )] );
		__pldx( 1, 1, 0, (float *)&c_[result_tile_TR_corner_ + ( ( ( 3 + 0 ) * cs_c ) )] );
		__pldx( 1, 1, 0, (float *)&c_[result_tile_TR_corner_ + ( ( ( 4 + 0 ) * cs_c ) )] );
		__pldx( 1, 1, 0, (float *)&c_[result_tile_TR_corner_ + ( ( ( 5 + 0 ) * cs_c ) )] );
		__pldx( 1, 1, 0, (float *)&c_[result_tile_TR_corner_ + ( ( ( 6 + 0 ) * cs_c ) )] );
		__pldx( 1, 1, 0, (float *)&c_[result_tile_TR_corner_ + ( ( ( 7 + 0 ) * cs_c ) )] );
	}
	else
	{
		__pldx( 1, 1, 0, (float *)&c_[result_tile_TL_corner_ + ( ( ( 0 + 0 ) * rs_c ) )] );
		__pldx( 1, 1, 0, (float *)&c_[result_tile_TL_corner_ + ( ( ( 1 + 0 ) * rs_c ) )] );
		__pldx( 1, 1, 0, (float *)&c_[result_tile_TL_corner_ + ( ( ( 2 + 0 ) * rs_c ) )] );
		__pldx( 1, 1, 0, (float *)&c_[result_tile_TL_corner_ + ( ( ( 3 + 0 ) * rs_c ) )] );
		__pldx( 1, 1, 0, (float *)&c_[result_tile_TL_corner_ + ( ( ( 4 + 0 ) * rs_c ) )] );
		__pldx( 1, 1, 0, (float *)&c_[result_tile_TL_corner_ + ( ( ( 5 + 0 ) * rs_c ) )] );
		__pldx( 1, 1, 0, (float *)&c_[result_tile_TL_corner_ + ( ( ( 6 + 0 ) * rs_c ) )] );
		__pldx( 1, 1, 0, (float *)&c_[result_tile_TL_corner_ + ( ( ( 7 + 0 ) * rs_c ) )] );

		__pldx( 1, 1, 0, (float *)&c_[result_tile_TR_corner_ + ( ( ( 0 + 0 ) * rs_c ) )] );
		__pldx( 1, 1, 0, (float *)&c_[result_tile_TR_corner_ + ( ( ( 1 + 0 ) * rs_c ) )] );
		__pldx( 1, 1, 0, (float *)&c_[result_tile_TR_corner_ + ( ( ( 2 + 0 ) * rs_c ) )] );
		__pldx( 1, 1, 0, (float *)&c_[result_tile_TR_corner_ + ( ( ( 3 + 0 ) * rs_c ) )] );
		__pldx( 1, 1, 0, (float *)&c_[result_tile_TR_corner_ + ( ( ( 4 + 0 ) * rs_c ) )] );
		__pldx( 1, 1, 0, (float *)&c_[result_tile_TR_corner_ + ( ( ( 5 + 0 ) * rs_c ) )] );
		__pldx( 1, 1, 0, (float *)&c_[result_tile_TR_corner_ + ( ( ( 6 + 0 ) * rs_c ) )] );
		__pldx( 1, 1, 0, (float *)&c_[result_tile_TR_corner_ + ( ( ( 7 + 0 ) * rs_c ) )] );
	}

	svzero_za();
	
	uint64_t k_;
	uint64_t k_iter = k / 8;
	uint64_t k_left = k % 8;

	for ( k_ = 0; k_ < k_iter; k_++ )
	{
		svfloat32x4_t zL00 = svld1_f32_x4( svptrue_c32(), (float32_t *)( &a_[0] ) );
		svfloat32x4_t zR00 = svld1_f32_x4( svptrue_c32(), (float32_t *)( &b_[0] ) );

		svmopa_za32_m( 0, svptrue_b32(), svptrue_b32(), svget4( zL00, 0 ), svget4( zR00, 0 ) );
		svmopa_za32_m( 1, svptrue_b32(), svptrue_b32(), svget4( zL00, 1 ), svget4( zR00, 0 ) );

		__pldx( 0, 1, 1, (float *)&a_next[0] );

		svmopa_za32_m( 2, svptrue_b32(), svptrue_b32(), svget4( zL00, 0 ), svget4( zR00, 1 ) );
		svmopa_za32_m( 3, svptrue_b32(), svptrue_b32(), svget4( zL00, 1 ), svget4( zR00, 1 ) );

		svmopa_za32_m( 0, svptrue_b32(), svptrue_b32(), svget4( zL00, 2 ), svget4( zR00, 2 ) );
		svmopa_za32_m( 1, svptrue_b32(), svptrue_b32(), svget4( zL00, 3 ), svget4( zR00, 2 ) );

		svmopa_za32_m( 2, svptrue_b32(), svptrue_b32(), svget4( zL00, 2 ), svget4( zR00, 3 ) );
		svmopa_za32_m( 3, svptrue_b32(), svptrue_b32(), svget4( zL00, 3 ), svget4( zR00, 3 ) );

		svfloat32x4_t zL02 = svld1_f32_x4( svptrue_c32(), (float32_t *)( &a_[( 4 * SVL )] ) );
		svfloat32x4_t zR02 = svld1_f32_x4( svptrue_c32(), (float32_t *)( &b_[( 4 * SVL )] ) );

		svmopa_za32_m( 0, svptrue_b32(), svptrue_b32(), svget4( zL02, 0 ), svget4( zR02, 0 ) );
		svmopa_za32_m( 1, svptrue_b32(), svptrue_b32(), svget4( zL02, 1 ), svget4( zR02, 0 ) );

		__pldx( 0, 1, 1, (float *)&a_next[4 * SVL] );

		svmopa_za32_m( 2, svptrue_b32(), svptrue_b32(), svget4( zL02, 0 ), svget4( zR02, 1 ) );
		svmopa_za32_m( 3, svptrue_b32(), svptrue_b32(), svget4( zL02, 1 ), svget4( zR02, 1 ) );

		svmopa_za32_m( 0, svptrue_b32(), svptrue_b32(), svget4( zL02, 2 ), svget4( zR02, 2 ) );
		svmopa_za32_m( 1, svptrue_b32(), svptrue_b32(), svget4( zL02, 3 ), svget4( zR02, 2 ) );
		svmopa_za32_m( 2, svptrue_b32(), svptrue_b32(), svget4( zL02, 2 ), svget4( zR02, 3 ) );
		svmopa_za32_m( 3, svptrue_b32(), svptrue_b32(), svget4( zL02, 3 ), svget4( zR02, 3 ) );

		svfloat32x4_t zL04 = svld1_f32_x4( svptrue_c32(), (float32_t *)( &a_[8 * SVL] ) );
		svfloat32x4_t zR04 = svld1_f32_x4( svptrue_c32(), (float32_t *)( &b_[8 * SVL] ) );

		svmopa_za32_m( 0, svptrue_b32(), svptrue_b32(), svget4( zL04, 0 ), svget4( zR04, 0 ) );
		svmopa_za32_m( 1, svptrue_b32(), svptrue_b32(), svget4( zL04, 1 ), svget4( zR04, 0 ) );

		__pldx( 0, 1, 1, (float *)&a_next[8 * SVL] );

		svmopa_za32_m( 2, svptrue_b32(), svptrue_b32(), svget4( zL04, 0 ), svget4( zR04, 1 ) );
		svmopa_za32_m( 3, svptrue_b32(), svptrue_b32(), svget4( zL04, 1 ), svget4( zR04, 1 ) );

		svmopa_za32_m( 0, svptrue_b32(), svptrue_b32(), svget4( zL04, 2 ), svget4( zR04, 2 ) );
		svmopa_za32_m( 1, svptrue_b32(), svptrue_b32(), svget4( zL04, 3 ), svget4( zR04, 2 ) );

		svmopa_za32_m( 2, svptrue_b32(), svptrue_b32(), svget4( zL04, 2 ), svget4( zR04, 3 ) );
		svmopa_za32_m( 3, svptrue_b32(), svptrue_b32(), svget4( zL04, 3 ), svget4( zR04, 3 ) );

		svfloat32x4_t zL06 = svld1_f32_x4( svptrue_c32(), (float32_t *)( &a_[( 12 * SVL )] ) );
		svfloat32x4_t zR06 = svld1_f32_x4( svptrue_c32(), (float32_t *)( &b_[( 12 * SVL )] ) );

		svmopa_za32_m( 0, svptrue_b32(), svptrue_b32(), svget4( zL06, 0 ), svget4( zR06, 0 ) );
		svmopa_za32_m( 1, svptrue_b32(), svptrue_b32(), svget4( zL06, 1 ), svget4( zR06, 0 ) );

		__pldx( 0, 1, 1, (float *)&a_next[12 * SVL] );

		svmopa_za32_m( 2, svptrue_b32(), svptrue_b32(), svget4( zL06, 0 ), svget4( zR06, 1 ) );
		svmopa_za32_m( 3, svptrue_b32(), svptrue_b32(), svget4( zL06, 1 ), svget4( zR06, 1 ) );

		svmopa_za32_m( 0, svptrue_b32(), svptrue_b32(), svget4( zL06, 2 ), svget4( zR06, 2 ) );
		svmopa_za32_m( 1, svptrue_b32(), svptrue_b32(), svget4( zL06, 3 ), svget4( zR06, 2 ) );

		svmopa_za32_m( 2, svptrue_b32(), svptrue_b32(), svget4( zL06, 2 ), svget4( zR06, 3 ) );
		svmopa_za32_m( 3, svptrue_b32(), svptrue_b32(), svget4( zL06, 3 ), svget4( zR06, 3 ) );

		a_ += ( 2 * 8 * SVL );
		b_ += ( 2 * 8 * SVL );

		a_next += ( 2 * 8 * SVL );
		b_next += ( 2 * 8 * SVL );
	}

	for ( k_ = 0; k_ < k_left; k_ += 1 )
	{
		svfloat32x2_t zL00 = svld1_f32_x2( svptrue_c32(), (float32_t *)( &a_[0] ) );
		svfloat32x2_t zR00 = svld1_f32_x2( svptrue_c32(), (float32_t *)( &b_[0] ) );

		svmopa_za32_m( 0, svptrue_b32(), svptrue_b32(), svget2( zL00, 0 ), svget2( zR00, 0 ) );
		svmopa_za32_m( 1, svptrue_b32(), svptrue_b32(), svget2( zL00, 1 ), svget2( zR00, 0 ) );

		svmopa_za32_m( 2, svptrue_b32(), svptrue_b32(), svget2( zL00, 0 ), svget2( zR00, 1 ) );
		svmopa_za32_m( 3, svptrue_b32(), svptrue_b32(), svget2( zL00, 1 ), svget2( zR00, 1 ) );

		a_ += ( 2 * SVL );
		b_ += ( 2 * SVL );
	}


	// =========================================================================
	// Phase 2: TRSM (Left-Lower Solve)
	// Solve A11 * X = alpha * B11 - ZA
	// =========================================================================

	float alpha_ = *alpha;
	float *b11_ = (float *)b11;
	const float *a11_ = (const float *)a11;
	
	svuint32_t z_indices = svindex_u32(0, 1);

	// Predicates for writing to the final C11 output
	svbool_t p_n_L = svwhilelt_b32_u64( 0, n );
	svbool_t p_n_R = svwhilelt_b32_u64( SVL, n );

	// =========================================================================
	// Phase 2: TRSM (Left-Lower Solve)
	// Iterate FORWARD row-by-row from 0 up to 'm - 1'
	// =========================================================================
	for ( dim_t i = 0; i < m; i++ )
	{
		svfloat32_t z_za_L, z_za_R;

		// 1. Read row i of the accumulator from ZA
		if ( i < SVL )
		{
			z_za_L = svread_hor_za32_m( svundef_f32(), svptrue_b32(), 0, i );
			z_za_R = svread_hor_za32_m( svundef_f32(), svptrue_b32(), 2, i );
		}
		else
		{
			z_za_L = svread_hor_za32_m( svundef_f32(), svptrue_b32(), 1, i - SVL );
			z_za_R = svread_hor_za32_m( svundef_f32(), svptrue_b32(), 3, i - SVL );
		}

		// 2. Load input RHS strictly from packed B11
		svfloat32_t z_c_L = svld1_f32( svptrue_b32(), &b11_[i * (2 * SVL) + 0] );
		svfloat32_t z_c_R = svld1_f32( svptrue_b32(), &b11_[i * (2 * SVL) + SVL] );

		// Scale by alpha
		z_c_L = svmul_n_f32_z( svptrue_b32(), z_c_L, alpha_ );
		z_c_R = svmul_n_f32_z( svptrue_b32(), z_c_R, alpha_ );

		// 3. Compute RHS: alpha * B11 - ZA
		svfloat32_t z_x_L = svsub_f32_z( svptrue_b32(), z_c_L, z_za_L );
		svfloat32_t z_x_R = svsub_f32_z( svptrue_b32(), z_c_R, z_za_R );

		// 4. Multiply by inverse of diagonal: X = RHS * A11[i, i]
		float diag_inv = a11_[i * (2 * SVL) + i];
		z_x_L = svmul_n_f32_z( svptrue_b32(), z_x_L, diag_inv );
		z_x_R = svmul_n_f32_z( svptrue_b32(), z_x_R, diag_inv );

		// 5. Store X back to packed buffer B11
		svst1_f32( svptrue_b32(), &b11_[i * (2 * SVL) + 0], z_x_L );
		svst1_f32( svptrue_b32(), &b11_[i * (2 * SVL) + SVL], z_x_R );

		// 6. Direct contiguous store if row-major (cs_c == 1)
		// If non-contiguous, writing to C11 is deferred to avoid TRSM pipeline stalls
		if ( cs_c == 1 )
		{
			float *c_row = &c11[i * rs_c];
			svst1_f32( p_n_L, c_row, z_x_L );
			svst1_f32( p_n_R, c_row + SVL, z_x_R );
		}
		// 7. Rank-1 Update of ZA for the remaining rows strictly BELOW i
		if ( i < m - 1 )
		{
			svfloat32_t z_a11_top = svld1_f32( svptrue_b32(), &a11_[i * (2 * SVL) + 0] );
			svfloat32_t z_a11_bot = svld1_f32( svptrue_b32(), &a11_[i * (2 * SVL) + SVL] );

			svbool_t p_m_top = svcmpgt_n_u32( svptrue_b32(), z_indices, i );
			svbool_t p_m_bot = ( i < SVL ) ? svptrue_b32() : svcmpgt_n_u32( svptrue_b32(), z_indices, i - SVL );
			
			// Accumulate: ZA += A11 * X
			if ( i < SVL )
			{
				svmopa_za32_m( 0, p_m_top, svptrue_b32(), z_a11_top, z_x_L );
				svmopa_za32_m( 2, p_m_top, svptrue_b32(), z_a11_top, z_x_R );
			}
			svmopa_za32_m( 1, p_m_bot, svptrue_b32(), z_a11_bot, z_x_L );
			svmopa_za32_m( 3, p_m_bot, svptrue_b32(), z_a11_bot, z_x_R );
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
			svfloat32_t row_L = svld1_f32( svptrue_b32(), &b11_[r * (2 * SVL) + 0] );
			svfloat32_t row_R = svld1_f32( svptrue_b32(), &b11_[r * (2 * SVL) + SVL] );

			if ( r < SVL )
			{
				svwrite_hor_za32_m( 0, r, svptrue_b32(), row_L );
				svwrite_hor_za32_m( 2, r, svptrue_b32(), row_R );
			}
			else
			{
				svwrite_hor_za32_m( 1, r - SVL, svptrue_b32(), row_L );
				svwrite_hor_za32_m( 3, r - SVL, svptrue_b32(), row_R );
			}
		}

		svbool_t p_m_top = svwhilelt_b32_u64( 0, m );
		svbool_t p_m_bot = svwhilelt_b32_u64( SVL, m );

		// 2. Read vertically and write contiguous column vectors into C11
		for ( dim_t col = 0; col < n; col++ )
		{
			float *c_col = &c11[col * cs_c];

			if ( col < SVL )
			{
				svfloat32_t col_top = svread_ver_za32_m( svundef_f32(), svptrue_b32(), 0, col );
				svst1_f32( p_m_top, c_col, col_top );
				if ( m > SVL )
				{
					svfloat32_t col_bot = svread_ver_za32_m( svundef_f32(), svptrue_b32(), 1, col );
					svst1_f32( p_m_bot, c_col + SVL, col_bot );
				}
			}
			else
			{
				svfloat32_t col_top = svread_ver_za32_m( svundef_f32(), svptrue_b32(), 2, col - SVL );
				svst1_f32( p_m_top, c_col, col_top );
				if ( m > SVL )
				{
					svfloat32_t col_bot = svread_ver_za32_m( svundef_f32(), svptrue_b32(), 3, col - SVL );
					svst1_f32( p_m_bot, c_col + SVL, col_bot );
				}
			}
		}
	}

	// =========================================================================
	// Phase 3: Zero-pad the unwritten rows of B11 for the macro-kernel
	// =========================================================================
	for ( dim_t pad_i = m; pad_i < 2 * SVL; pad_i++ )
	{
		svst1_f32( svptrue_b32(), &b11_[pad_i * (2 * SVL) + 0], svdup_n_f32(0.0f) );
		svst1_f32( svptrue_b32(), &b11_[pad_i * (2 * SVL) + SVL], svdup_n_f32(0.0f) );
	}

	GEMMTRSM_UKR_FLUSH_CT( s );
	return;
}