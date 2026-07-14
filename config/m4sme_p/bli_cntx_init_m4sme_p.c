/*

   BLIS
   An object-based framework for developing high-performance BLAS-like
   libraries.

   Copyright (C) 2014, The University of Texas at Austin

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

#include "blis.h"

void bli_cntx_init_m4sme_p( cntx_t* cntx )
{
	blksz_t blkszs[ BLIS_NUM_BLKSZS ];

	// Set default kernel blocksizes and functions.
	bli_cntx_init_m4sme_p_ref( cntx );

	// -------------------------------------------------------------------------
#if 1 // CONFIG_1
	// Update the context with optimized native gemm micro-kernels.
	bli_cntx_set_ukrs
	(
	  cntx,

	  // level-3
	  BLIS_GEMM_UKR, BLIS_FLOAT,  bli_sgemm_armsme_int_2SVLx2SVL,
	  BLIS_GEMM_UKR, BLIS_DOUBLE, bli_dgemm_armsme_int_4SVLx2SVL,
	  BLIS_PACKM_KER, BLIS_FLOAT, bli_spackm_armsme_int_2SVLx2SVL,
	  BLIS_PACKM_KER, BLIS_DOUBLE, bli_dpackm_armsme_int_4SVLx2SVL,

	//   BLIS_GEMMSUP_RRR_UKR, BLIS_FLOAT, bli_sgemm_armsme_sup_ccc_2SVLx2SVL,
	  BLIS_GEMMSUP_RRC_UKR, BLIS_FLOAT, bli_sgemm_armsme_sup_double_pack_2SVLx2SVL,
	  BLIS_GEMMSUP_RCC_UKR, BLIS_FLOAT, bli_sgemm_armsme_sup_crr_2SVLx2SVL,

	  BLIS_GEMMSUP_CRR_UKR, BLIS_FLOAT, bli_sgemm_armsme_sup_crr_2SVLx2SVL,
	  BLIS_GEMMSUP_CRC_UKR, BLIS_FLOAT, bli_sgemm_armsme_sup_double_pack_2SVLx2SVL,
	//   BLIS_GEMMSUP_CCC_UKR, BLIS_FLOAT, bli_sgemm_armsme_sup_ccc_2SVLx2SVL,
	  BLIS_GEMMSUP_RCR_UKR, BLIS_FLOAT, bli_sgemm_armsme_sup_2SVLx2SVL,
	  BLIS_GEMMSUP_CCR_UKR, BLIS_FLOAT, bli_sgemm_armsme_sup_2SVLx2SVL,
	  BLIS_GEMMSUP_RRR_UKR, BLIS_FLOAT, bli_sgemm_armsme_sup_ccc_2SVLx2SVL,
	//   BLIS_GEMMSUP_RRC_UKR, BLIS_FLOAT, bli_sgemm_armsme_sup_fallback,
	//   BLIS_GEMMSUP_RCC_UKR, BLIS_FLOAT, bli_sgemm_armsme_sup_fallback,
	//   BLIS_GEMMSUP_CRR_UKR, BLIS_FLOAT, bli_sgemm_armsme_sup_fallback,
	//   BLIS_GEMMSUP_CRC_UKR, BLIS_FLOAT, bli_sgemm_armsme_sup_fallback,
	  BLIS_GEMMSUP_CCC_UKR, BLIS_FLOAT, bli_sgemm_armsme_sup_ccc_2SVLx2SVL,

	  BLIS_VA_END
	);

	// Update the context with storage preferences.
	bli_cntx_set_ukr_prefs
	(
	  cntx,

	  // level-3
	  BLIS_GEMM_UKR_ROW_PREF, BLIS_FLOAT,  FALSE,
	  BLIS_GEMM_UKR_ROW_PREF, BLIS_DOUBLE, FALSE,

	  BLIS_VA_END
	);

	// Initialize level-3 blocksize objects with architecture-specific values.
	//                                           s      d      c      z
	bli_blksz_init_easy( &blkszs[ BLIS_MR ],    32,    32,    -1,    -1 );
	bli_blksz_init_easy( &blkszs[ BLIS_NR ],    32,    16,    -1,    -1 );
	bli_blksz_init_easy( &blkszs[ BLIS_MC ],    768,   608,   -1,    -1 );
	bli_blksz_init_easy( &blkszs[ BLIS_KC ],    3648,  2240,  -1,    -1 );
	bli_blksz_init_easy( &blkszs[ BLIS_NC ],    10240, 3072,  -1,    -1 );


	bli_blksz_init_easy( &blkszs[ BLIS_MT ],    1504,   -1,   -1,    -1 );
	bli_blksz_init_easy( &blkszs[ BLIS_NT ],    1504,   -1,   -1,    -1 );
	bli_blksz_init_easy( &blkszs[ BLIS_KT ],    1,   -1,   -1,    -1 );

	bli_blksz_init_easy( &blkszs[ BLIS_MR_SUP ],  32,    32,    -1,    -1 );
	bli_blksz_init_easy( &blkszs[ BLIS_NR_SUP ],  32,    16,    -1,    -1 );
	bli_blksz_init_easy( &blkszs[ BLIS_MC_SUP ],  32,    32,    -1,    -1 );
	bli_blksz_init_easy( &blkszs[ BLIS_KC_SUP ],  2080,  1504,  -1,    -1 );
	bli_blksz_init_easy( &blkszs[ BLIS_NC_SUP ],  32,    16,    -1,    -1 );
#endif

#if 0 // CONFIG_2
// Update the context with optimized native gemm micro-kernels.
	bli_cntx_set_ukrs
	(
	  cntx,

	  // level-3
	  BLIS_GEMM_UKR, BLIS_FLOAT,  bli_sgemm_armsme_int_SVLx4SVL,
	  BLIS_GEMM_UKR, BLIS_DOUBLE, bli_dgemm_armsme_int_SVLx8SVL,
	  BLIS_PACKM_KER, BLIS_FLOAT, bli_spackm_armsme_int_SVLx4SVL,
	  BLIS_PACKM_KER, BLIS_DOUBLE, bli_dpackm_armsme_int_SVLx8SVL,
	  BLIS_VA_END
	);

	// Update the context with storage preferences.
	bli_cntx_set_ukr_prefs
	(
	  cntx,

	  // level-3
	  BLIS_GEMM_UKR_ROW_PREF, BLIS_FLOAT,  TRUE,
	  BLIS_GEMM_UKR_ROW_PREF, BLIS_DOUBLE, TRUE,

	  BLIS_VA_END
	);

	// Initialize level-3 blocksize objects with architecture-specific values.
	//                                           s      d      c      z
	bli_blksz_init_easy( &blkszs[ BLIS_MR ],    16,    8,     -1,    -1 );
	bli_blksz_init_easy( &blkszs[ BLIS_NR ],    64,    64,    -1,    -1 );
	bli_blksz_init_easy( &blkszs[ BLIS_MC ],    512,   512,   -1,    -1 );
	bli_blksz_init_easy( &blkszs[ BLIS_KC ],    4032,  2016,  -1,    -1 );
	bli_blksz_init_easy( &blkszs[ BLIS_NC ], 	10240, 3072,  -1,    -1 );
#endif

#if 0 // CONFIG_3
	// Update the context with optimized native gemm micro-kernels.
	bli_cntx_set_ukrs
	(
	  cntx,

	  // level-3
	  BLIS_GEMM_UKR, BLIS_FLOAT,  bli_sgemm_armsme_int_4SVLxSVL,
	  BLIS_GEMM_UKR, BLIS_DOUBLE, bli_dgemm_armsme_int_8SVLxSVL,
	  BLIS_PACKM_KER, BLIS_FLOAT, bli_spackm_armsme_int_SVLx4SVL,
	  BLIS_PACKM_KER, BLIS_DOUBLE, bli_dpackm_armsme_int_SVLx8SVL,
	  BLIS_VA_END
	);

	// Update the context with storage preferences.
	 bli_cntx_set_ukr_prefs
	(
	  cntx,

	  // level-3
	  BLIS_GEMM_UKR_ROW_PREF, BLIS_FLOAT,  FALSE,
	  BLIS_GEMM_UKR_ROW_PREF, BLIS_DOUBLE, FALSE,

	  BLIS_VA_END
	);

	// Initialize level-3 blocksize objects with architecture-specific values.
	//                                           s      d      c      z
	bli_blksz_init_easy( &blkszs[ BLIS_MR ],    64,    64,    -1,    -1 );
	bli_blksz_init_easy( &blkszs[ BLIS_NR ],    16,    8,     -1,    -1 );
	bli_blksz_init_easy( &blkszs[ BLIS_MC ],    512,   512,   -1,    -1 );
	bli_blksz_init_easy( &blkszs[ BLIS_KC ],    4032,  2016,  -1,    -1 );
	bli_blksz_init_easy( &blkszs[ BLIS_NC ], 	10240, 3072,  -1,    -1 );
#endif
#if 0 // CONFIG_4
	// Update the context with optimized native gemm micro-kernels.
	bli_cntx_set_ukrs
	(
	  cntx,

	  // level-3
	  BLIS_GEMM_UKR, BLIS_FLOAT,  bli_sgemm_armsme_int_2SVLx2SVL,
	  BLIS_GEMM_UKR, BLIS_DOUBLE, bli_dgemm_armsme_int_2SVLx4SVL,
	  BLIS_PACKM_KER, BLIS_FLOAT, bli_spackm_armsme_int_2SVLx2SVL,
	  BLIS_PACKM_KER, BLIS_DOUBLE, bli_dpackm_armsme_int_4SVLx2SVL,
	  BLIS_VA_END
	);

	// Update the context with storage preferences.
	bli_cntx_set_ukr_prefs
	(
	  cntx,

	  // level-3
	  BLIS_GEMM_UKR_ROW_PREF, BLIS_FLOAT,  TRUE,
	  BLIS_GEMM_UKR_ROW_PREF, BLIS_DOUBLE, TRUE,

	  BLIS_VA_END
	);

	// Initialize level-3 blocksize objects with architecture-specific values.
	//                                           s      d      c      z
	bli_blksz_init_easy( &blkszs[ BLIS_MR ],    32,    16,    -1,    -1 );
	bli_blksz_init_easy( &blkszs[ BLIS_NR ],    32,    32,    -1,    -1 );
	bli_blksz_init_easy( &blkszs[ BLIS_MC ],    768,   608,   -1,    -1 );
	bli_blksz_init_easy( &blkszs[ BLIS_KC ],    3648,  2240,  -1,    -1 );
	bli_blksz_init_easy( &blkszs[ BLIS_NC ],    10240, 3072,  -1,    -1 );
#endif

	// Update the context with the current architecture's register and cache
	// blocksizes (and multiples) for native execution.
	bli_cntx_set_blkszs
	(
	  cntx,

	  // level-3
	  BLIS_NC, &blkszs[ BLIS_NC ], BLIS_NR,
	  BLIS_KC, &blkszs[ BLIS_KC ], BLIS_KR,
	  BLIS_MC, &blkszs[ BLIS_MC ], BLIS_MR,
	  BLIS_NR, &blkszs[ BLIS_NR ], BLIS_NR,
	  BLIS_MR, &blkszs[ BLIS_MR ], BLIS_MR,

	  BLIS_MT, &blkszs[ BLIS_MT ], BLIS_MT,
	  BLIS_NT, &blkszs[ BLIS_NT ], BLIS_NT,
	  BLIS_KT, &blkszs[ BLIS_KT ], BLIS_KT,

	  BLIS_MR_SUP, &blkszs[ BLIS_MR_SUP ], BLIS_MR_SUP,
	  BLIS_NR_SUP, &blkszs[ BLIS_NR_SUP ], BLIS_NR_SUP,
	  BLIS_MC_SUP, &blkszs[ BLIS_MC_SUP ], BLIS_MC_SUP,
	  BLIS_KC_SUP, &blkszs[ BLIS_KC_SUP ], BLIS_KC_SUP,
	  BLIS_NC_SUP, &blkszs[ BLIS_NC_SUP ], BLIS_NC_SUP,
	  	  
	  BLIS_VA_END
	);
}
