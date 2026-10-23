// SPDX-License-Identifier: Apache-2.0
// Fixed six-row sequence wrapper. C1/masked arithmetic remains in the core
// pair kernel; this unit only selects its transaction-partner behavior.
#define DSV41_COMPRESSOR_BATCH 1
#define DSV41_COMPRESSOR_SEQUENCE_MASKED 1
#include "deepseek_v41_compressor_pair_bf16_gaudi2.c"
