// SPDX-License-Identifier: Apache-2.0
#include "deepseek_v41_compressor_sequence_masked_gaudi2.hpp"
#include <cstring>

extern unsigned char
    _binary___deepseek_v41_compressor_sequence_masked_bf16_gaudi2_o_start;
extern unsigned char
    _binary___deepseek_v41_compressor_sequence_masked_bf16_gaudi2_o_end;

namespace {
using namespace tpc_lib_api;

bool tensor(const Tensor& value, unsigned type, unsigned rows) {
    return value.geometry.dataType == type && value.geometry.dims == 2 &&
           value.geometry.maxSizes[0] == 512 &&
           value.geometry.maxSizes[1] == rows;
}

void feature_rows(TensorAccessPattern& access, unsigned rows) {
    access.allRequired = false;
    access.mapping[0] = {0, 128, 0, 127};
    access.mapping[1] = {1, 0, 0, static_cast<int>(rows - 1)};
}
}

tpc_lib_api::GlueCodeReturn
DeepseekV41CompressorSequenceMaskedGaudi2::GetGcDefinitions(
    tpc_lib_api::HabanaKernelParams* in,
    tpc_lib_api::HabanaKernelInstantiation* out) {
    if (!in || !out) return GLUE_FAILED;
    if (in->inputTensorNr != 5) return GLUE_INCOMPATIBLE_INPUT_COUNT;
    if (in->outputTensorNr != 1) return GLUE_INCOMPATIBLE_OUTPUT_COUNT;
    if (in->inputTensors[0].geometry.dims != 2 ||
        in->inputTensors[2].geometry.dims != 2)
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    const auto rows = in->inputTensors[2].geometry.maxSizes[1];
    if (rows != 6 ||
        !tensor(in->inputTensors[0], DATA_F32, 8) ||
        !tensor(in->inputTensors[1], DATA_F32, 8) ||
        !tensor(in->inputTensors[2], DATA_F32, rows) ||
        !tensor(in->inputTensors[3], DATA_F32, rows))
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    const auto& position = in->inputTensors[4].geometry;
    if (position.dataType != DATA_I32 || position.dims != 1 ||
        position.maxSizes[0] != rows)
        return GLUE_INCOMPATIBLE_INPUT_SIZE;
    if (!tensor(in->outputTensors[0], DATA_BF16, rows))
        return GLUE_INCOMPATIBLE_OUTPUT_SIZE;

    feature_rows(out->inputTensorAccessPattern[0], 8);
    feature_rows(out->inputTensorAccessPattern[1], 8);
    feature_rows(out->inputTensorAccessPattern[2], rows);
    feature_rows(out->inputTensorAccessPattern[3], rows);
    out->inputTensorAccessPattern[4].allRequired = true;
    feature_rows(out->outputTensorAccessPattern[0], rows);
    out->indexSpaceRank = 2;
    out->indexSpaceGeometry[0] = 4;
    out->indexSpaceGeometry[1] = rows;
    out->kernel.paramsNr = 0;

    auto* begin =
        &_binary___deepseek_v41_compressor_sequence_masked_bf16_gaudi2_o_start;
    auto* end =
        &_binary___deepseek_v41_compressor_sequence_masked_bf16_gaudi2_o_end;
    const auto capacity = out->kernel.elfSize;
    out->kernel.elfSize = end - begin;
    if (capacity < out->kernel.elfSize) return GLUE_INSUFFICIENT_ELF_BUFFER;
    std::memcpy(out->kernel.kernelElf, begin, out->kernel.elfSize);
    return GLUE_SUCCESS;
}
