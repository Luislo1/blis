#
#
#  BLIS
#  An object-based framework for developing high-performance BLAS-like
#  libraries.
#
#  Copyright (C) 2014, The University of Texas at Austin
#  Copyright (C) 2017 - 2020, Advanced Micro Devices, Inc. All rights reserved.
#
#  ... [Standard Copyright Header] ...
#

#
# --- Makefile PHONY target definitions ----------------------------------------
#

# 1. ADDED: 'accelerate' to PHONY targets
.PHONY: all \
        blis openblas atlas mkl accelerate \
        check-env check-env-mk check-lib \
        clean cleanx



#
# --- Determine makefile fragment location -------------------------------------
#

ifneq ($(strip $(BLIS_INSTALL_PATH)),)
LIB_PATH   := $(BLIS_INSTALL_PATH)/lib
INC_PATH   := $(BLIS_INSTALL_PATH)/include/blis
SHARE_PATH := $(BLIS_INSTALL_PATH)/share/blis
else
DIST_PATH  := ..
LIB_PATH    = ../lib/$(CONFIG_NAME)
INC_PATH    = ../include/$(CONFIG_NAME)
SHARE_PATH := ..
endif



#
# --- Include common makefile definitions --------------------------------------
#

-include $(SHARE_PATH)/common.mk



#
# --- BLAS and LAPACK implementations ------------------------------------------
#

BLAS_LIB_PATH  := $(HOME)/flame/lib
MKL_LIB_PATH   := $(HOME)/intel/mkl/lib/intel64

# OpenBLAS
OPENBLAS_LIB   := $(BLAS_LIB_PATH)/libopenblas.a

# MKL
MKL_LIB        := -L$(MKL_LIB_PATH) \
                  -lmkl_intel_lp64 \
                  -lmkl_core \
                  -lmkl_sequential \
                  -lpthread -lm -ldl

# 2. ADDED: Accelerate & OpenMP Apple Silicon Flags
ACCELERATE_CFLAGS  := -I/opt/homebrew/opt/libomp/include -Xpreprocessor -fopenmp -flax-vector-conversions
ACCELERATE_LDFLAGS := -L/opt/homebrew/opt/libomp/lib -lomp -framework Accelerate -flax-vector-conversions


#
# --- General build definitions ------------------------------------------------
#

TEST_SRC_PATH  := .
TEST_OBJ_PATH  := .
TEST_OBJS      := $(patsubst $(TEST_SRC_PATH)/%.c, \
                             $(TEST_OBJ_PATH)/%.o, \
                             $(wildcard $(TEST_SRC_PATH)/*.c))

CINCFLAGS      := -I$(INC_PATH)
CFLAGS         := $(call get-user-cflags-for,$(CONFIG_NAME))
CFLAGS         += -I$(TEST_SRC_PATH)



#
# --- Targets/rules ------------------------------------------------------------
#

# TEST_OPS := gemm  gemm_qos gemm_classic gemm_xnu gemm_xnu_p 
# TEST_OPS := gemm_classic gemm_classic_d gemm_classic_sm gemm_classic_sm_d gemm_rcr gemm_rcr_sm gemm_barrido gemm_rcr_skinny gemm_skinny gemm_config
# TEST_OPS := hemm herk her2k trmm trsm gemm_rcr gemm_rcr_d
# TEST_OPS := gemm_xnu gemm_xnu_p gemm_rcr gemm_rcr_xnu gemm gemm_qos gemm_classic 
 TEST_OPS := gemm_rcr gemm_qos gemm_scan
ifeq ($(BUILD_GEMMT),yes)
TEST_OPS := $(TEST_OPS) gemmt
endif

test-bins = $(foreach op, $(TEST_OPS), test_$(op)_$(1).x)

TEST_BINS_BLIS       := $(call test-bins,blis)
TEST_BINS_OPENBLAS   := $(call test-bins,openblas)
TEST_BINS_MKL        := $(call test-bins,mkl)
# 3. ADDED: Accelerate test bin array
TEST_BINS_ACCELERATE := $(call test-bins,accelerate)


# 4. ADDED: accelerate to "all" and created its specific target rule
all: blis openblas mkl accelerate

blis: check-env $(TEST_BINS_BLIS)

openblas: check-env $(TEST_BINS_OPENBLAS)

mkl: check-env $(TEST_BINS_MKL)

accelerate: check-env $(TEST_BINS_ACCELERATE)


# --Object file rules --

$(TEST_OBJ_PATH)/%.o: $(TEST_SRC_PATH)/%.c
	$(CC) $(CFLAGS) -c $< -o $@


test_%_openblas.o: test_%.c
	$(CC) $(CFLAGS) -DBLAS=\"openblas\" -c $< -o $@

test_%_mkl.o: test_%.c
	$(CC) $(CFLAGS) -DBLAS=\"mkl\" -c $< -o $@

test_%_blis.o: test_%.c
	$(CC) $(CFLAGS) -DBLIS -c $< -o $@

# 5. ADDED: Object compilation rule for Accelerate
test_%_accelerate.o: test_%.c
	$(CC) $(CFLAGS) $(ACCELERATE_CFLAGS) -DACCELERATE -DBLAS=\"ACCELERATE\" -c $< -o $@


# -- Executable file rules --

test_%_openblas.x: test_%_openblas.o $(LIBBLIS_LINK)
	$(LINKER) $<             $(OPENBLAS_LIB) $(LIBBLIS_LINK) $(LDFLAGS) -o $@

test_%_mkl.x: test_%_mkl.o $(LIBBLIS_LINK)
	$(LINKER) $<             $(MKL_LIB)      $(LIBBLIS_LINK) $(LDFLAGS) -o $@

test_%_blis.x: test_%_blis.o $(LIBBLIS_LINK)
	$(LINKER) $<                             $(LIBBLIS_LINK) $(LDFLAGS) -o $@

# 6. ADDED: Executable linking rule for Accelerate
test_%_accelerate.x: test_%_accelerate.o $(LIBBLIS_LINK)
	$(LINKER) $<             $(ACCELERATE_LDFLAGS) $(LIBBLIS_LINK) $(LDFLAGS) -o $@


# -- Environment check rules --

check-env: check-lib

check-env-mk:
ifeq ($(CONFIG_MK_PRESENT),no)
	$(error Cannot proceed: config.mk not detected! Run configure first)
endif

check-lib: check-env-mk
ifeq ($(wildcard $(LIBBLIS_LINK)),)
	$(error Cannot proceed: BLIS library not yet built! Run make first)
endif


# -- Clean rules --

clean: cleanx

cleanx:
	- $(RM_F) *.o *.x
