#include "blis.h"

// A unified fallback SUP kernel for unsupported matrix layouts
void bli_sgemm_armsme_sup_fallback(
    conj_t conja, conj_t conjb, dim_t m, dim_t n, dim_t k,
    void* alpha, void* a, inc_t rs_a, inc_t cs_a,
    void* b, inc_t rs_b, inc_t cs_b,
    void* beta, void* c, inc_t rs_c, inc_t cs_c,
    auxinfo_t* data, cntx_t* cntx
)
{
    // 1. Create a local copy of the BLIS context
    cntx_t local_cntx = *cntx;
    
    // 2. Disable SUP in the local context by setting thresholds to 0
    blksz_t bsz_disable;
    bli_blksz_init_easy( &bsz_disable, 0, 0, 0, 0 );
    
    // Overwrite the primary SUP decision thresholds
    bli_cntx_set_blkszs(
        &local_cntx,
        BLIS_MT, &bsz_disable, BLIS_MT,
        BLIS_NT, &bsz_disable, BLIS_NT,
        BLIS_KT, &bsz_disable, BLIS_KT,
        BLIS_VA_END
    );

    // 3. Create a single-threaded runtime environment for this sub-problem
    rntm_t local_rntm = BLIS_RNTM_INITIALIZER;
    bli_rntm_set_num_threads( 1, &local_rntm );

    // 4. Translate conjugation flags
    trans_t transa = ( conja == BLIS_CONJUGATE ? BLIS_CONJ_NO_TRANSPOSE : BLIS_NO_TRANSPOSE );
    trans_t transb = ( conjb == BLIS_CONJUGATE ? BLIS_CONJ_NO_TRANSPOSE : BLIS_NO_TRANSPOSE );

    // 5. Recursively invoke the BLIS expert GEMM interface
    bli_sgemm_ex(
        transa, transb,
        m, n, k,
        (float*)alpha,
        (float*)a, rs_a, cs_a,
        (float*)b, rs_b, cs_b,
        (float*)beta,
        (float*)c, rs_c, cs_c,
        &local_cntx,
        &local_rntm
    );
}