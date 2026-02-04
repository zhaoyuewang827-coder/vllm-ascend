/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file sum_lstm_tiling.cpp
 * \brief
 */
#include "sum_lstm_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include <algorithm>

namespace optiling {

// 计算对齐值 (对齐到 32 字节边界)
static inline uint32_t AlignUp32B(uint32_t size, uint32_t elemSize)
{
    uint32_t alignElems = 32 / elemSize;  // 32B 对应的元素数
    return (size + alignElems - 1) / alignElems * alignElems;
}

static ge::graphStatus TilingFunc4SumLstm(gert::TilingContext* context)
{
    if (context == nullptr) {
        return ge::GRAPH_FAILED;
    }

    SumLstmTilingData tiling;

    // 获取输入 tensor 信息
    const gert::StorageShape* states4dShape = context->GetInputShape(0);  // states_4d
    const gert::StorageShape* prevCellShape = context->GetInputShape(2);  // prev_cell

    if (states4dShape == nullptr || prevCellShape == nullptr) {
        return ge::GRAPH_FAILED;
    }

    // 获取数据类型
    auto dtype = context->GetInputDesc(0)->GetDataType();
    uint32_t dataTypeSize = (dtype == ge::DT_FLOAT16) ? 2 : 4;
    tiling.set_dataTypeSize(dataTypeSize);

    // 解析形状参数
    const gert::Shape& statesShape = states4dShape->GetStorageShape();
    int32_t dimCount = statesShape.GetDimNum();

    // 最后一维是 4D (gatedDim)
    uint32_t gatedDim = static_cast<uint32_t>(statesShape.GetDim(dimCount - 1));
    uint32_t hiddenDim = gatedDim / 4;

    // 计算总样本数 (除最后一维外的所有维度乘积)
    uint32_t totalSamples = 1;
    for (int32_t i = 0; i < dimCount - 1; ++i) {
        totalSamples *= static_cast<uint32_t>(statesShape.GetDim(i));
    }

    tiling.set_totalSamples(totalSamples);
    tiling.set_hiddenDim(hiddenDim);
    tiling.set_gatedDim(gatedDim);

    // 计算对齐后的维度
    uint32_t hiddenDimAligned = AlignUp32B(hiddenDim, dataTypeSize);
    uint32_t gatedDimAligned = AlignUp32B(gatedDim, dataTypeSize);
    // float = 4 bytes, 且至少 1024 个元素 (4096 字节, 与 D=1024 匹配)
    // 小于此值的 D 在 VEC 操作中会产生错误结果
    uint32_t floatHiddenDimAligned = std::max(AlignUp32B(hiddenDim, 4), (uint32_t)1024);
    tiling.set_hiddenDimAligned(hiddenDimAligned);
    tiling.set_gatedDimAligned(gatedDimAligned);
    tiling.set_floatHiddenDimAligned(floatHiddenDimAligned);

    // 检查可选输入 (在 UB 计算之前，用于确定预加载策略)
    uint32_t hasWCell = (context->GetInputTensor(3) != nullptr) ? 1 : 0;
    uint32_t hasBCell = (context->GetInputTensor(4) != nullptr) ? 1 : 0;
    uint32_t hasWState = (context->GetInputTensor(5) != nullptr) ? 1 : 0;
    uint32_t hasBState = (context->GetInputTensor(6) != nullptr) ? 1 : 0;
    uint32_t numWeights = hasWCell + hasBCell + hasWState + hasBState;

    // 获取硬件平台信息
    auto platformInfo = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t coreNum = platformInfo.GetCoreNumAic();
    if (coreNum == 0) {
        coreNum = 1;
    }

    // 获取 UB 大小
    uint64_t ubSize = 0;
    platformInfo.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);

    // 计算 UB 内存
    // 固定内存 (不随 tileSamples 缩放):
    //   5 个 float buffer: 5 * floatHiddenDimAligned * 4
    //   optBuffer (half): hiddenDimAligned * dataTypeSize (保守估计，总是计入)
    uint32_t fixedBytes = 5 * floatHiddenDimAligned * 4 + hiddenDimAligned * dataTypeSize;

    // 每样本 I/O 字节 (不含缓冲倍数):
    //   输入: states_4d(4D_aligned), z4_4d(4D_aligned), prev_cell(D_aligned) — half
    //   输出: out_state(D_aligned), out_cell(D_aligned) — half
    uint32_t perSampleBytesBase = (2 * gatedDimAligned + 3 * hiddenDimAligned) * dataTypeSize;

    // 优先尝试双缓冲，不够则降为单缓冲
    uint64_t availableUb = ubSize * 8 / 10;  // 使用 80% 的 UB
    uint32_t bufferCount = 2;
    uint32_t perSampleBytes = perSampleBytesBase * bufferCount;
    uint64_t availableForTiles = (availableUb > fixedBytes) ? (availableUb - fixedBytes) : 0;
    uint32_t maxSamplesPerTile = (perSampleBytes > 0) ? static_cast<uint32_t>(availableForTiles / perSampleBytes) : 0;

    if (maxSamplesPerTile == 0) {
        // 双缓冲放不下，降为单缓冲
        bufferCount = 1;
        perSampleBytes = perSampleBytesBase;
        availableForTiles = (availableUb > fixedBytes) ? (availableUb - fixedBytes) : 0;
        maxSamplesPerTile = (perSampleBytes > 0) ? static_cast<uint32_t>(availableForTiles / perSampleBytes) : 0;
    }
    if (maxSamplesPerTile == 0) {
        maxSamplesPerTile = 1;
    }

    // 分核策略: 均匀分配样本到各核心
    uint32_t usedCoreNum = std::min(coreNum, totalSamples);
    uint32_t samplesPerCore = totalSamples / usedCoreNum;
    uint32_t remainSamples = totalSamples % usedCoreNum;

    tiling.set_coreNum(usedCoreNum);
    tiling.set_samplesPerCore(samplesPerCore);
    tiling.set_remainSamples(remainSamples);

    // 计算每个核心的 tile 数量
    uint32_t tileSamples = std::min(maxSamplesPerTile, samplesPerCore + 1);
    uint32_t tileNumPerCore = (samplesPerCore + tileSamples - 1) / tileSamples;
    if (tileNumPerCore == 0) {
        tileNumPerCore = 1;
    }
    uint32_t lastTileSamples = samplesPerCore - (tileNumPerCore - 1) * tileSamples;
    if (lastTileSamples == 0) {
        lastTileSamples = tileSamples;
    }

    tiling.set_tileNumPerCore(tileNumPerCore);
    tiling.set_tileSamples(tileSamples);
    tiling.set_lastTileSamples(lastTileSamples);
    tiling.set_ubBufferSize(tileSamples * hiddenDimAligned * dataTypeSize);
    tiling.set_bufferCount(bufferCount);

    // 判断是否可以预加载权重
    uint32_t preloadWeights = 0;
    if (numWeights > 0) {
        // 预加载需要额外 3 个权重槽位 (共 4 个, 已有 1 个在 fixedBytes 中)
        uint32_t extraPreloadBytes = 3 * hiddenDimAligned * dataTypeSize;
        uint64_t usedBytes = (uint64_t)fixedBytes + (uint64_t)tileSamples * perSampleBytesBase * bufferCount;
        if (usedBytes + extraPreloadBytes <= availableUb) {
            preloadWeights = 1;
        }
    }
    tiling.set_preloadWeights(preloadWeights);

    // 获取算子属性
    const auto* attrs = context->GetAttrs();
    float alpha = 1.0f;
    float epsCell = 1e-6f;
    float epsState = 1e-6f;
    bool useFastGelu = true;

    if (attrs != nullptr) {
        const float* alphaPtr = attrs->GetAttrPointer<float>(0);
        const float* epsCellPtr = attrs->GetAttrPointer<float>(1);
        const float* epsStatePtr = attrs->GetAttrPointer<float>(2);
        const bool* useFastGeluPtr = attrs->GetAttrPointer<bool>(3);

        if (alphaPtr) alpha = *alphaPtr;
        if (epsCellPtr) epsCell = *epsCellPtr;
        if (epsStatePtr) epsState = *epsStatePtr;
        if (useFastGeluPtr) useFastGelu = *useFastGeluPtr;
    }

    tiling.set_alpha(alpha);
    tiling.set_epsCell(epsCell);
    tiling.set_epsState(epsState);
    tiling.set_useFastGelu(useFastGelu ? 1 : 0);

    tiling.set_hasWCell(hasWCell);
    tiling.set_hasBCell(hasBCell);
    tiling.set_hasWState(hasWState);
    tiling.set_hasBState(hasBState);

    // 设置 Tiling 数据
    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

    // 设置 Block 维度 (核心数)
    context->SetBlockDim(usedCoreNum);

    // 设置 Tiling Key (0: 标准路径)
    context->SetTilingKey(0);

    return ge::GRAPH_SUCCESS;
}

}  // namespace optiling

namespace ops {
IMPL_OP_OPTILING(SumLstm)
    .Tiling(optiling::TilingFunc4SumLstm);
}  // namespace ops
