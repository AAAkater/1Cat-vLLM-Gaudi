// SPDX-License-Identifier: Apache-2.0
// Fixed-six masked sequence schema. Reuse the qualified pair registration and
// numerical path; only the new GUID/kernel selects padding-safe semantics.
#define DSV41_COMPRESSOR_PAIR_ROWS 6
#define DSV41_COMPRESSOR_PAIR_MIN_ROWS 6
#define DSV41_COMPRESSOR_PAIR_SCHEMA \
    "custom_deepseek_v41_compressor_sequence_masked_bf16_gaudi2"
#define DSV41_COMPRESSOR_PAIR_ORDERED \
    "custom_deepseek_v41_compressor_sequence_masked_ordered_bf16_gaudi2"
#define DSV41_COMPRESSOR_PAIR_GUID \
    "custom_deepseek_v41_compressor_sequence_masked_bf16_gaudi2"
#include "hpu_dsv41_compressor_pair_pt2.cpp"
