#ifndef SUM_LSTM_TILING_H
#define SUM_LSTM_TILING_H

#include "register/tilingdata_base.h"

namespace optiling {

BEGIN_TILING_DATA_DEF(SumLstmTilingData)
    // 基础参数
    TILING_DATA_FIELD_DEF(uint32_t, totalSamples);      // 总样本数 (batch * ...)
    TILING_DATA_FIELD_DEF(uint32_t, hiddenDim);         // 隐藏维度 D
    TILING_DATA_FIELD_DEF(uint32_t, gatedDim);          // 4D (4 * hiddenDim)

    // 分核参数
    TILING_DATA_FIELD_DEF(uint32_t, coreNum);           // 使用的核心数
    TILING_DATA_FIELD_DEF(uint32_t, samplesPerCore);    // 每个核心处理的样本数
    TILING_DATA_FIELD_DEF(uint32_t, remainSamples);     // 余数样本数

    // Tile 参数
    TILING_DATA_FIELD_DEF(uint32_t, tileNumPerCore);    // 每个核心的 tile 数量
    TILING_DATA_FIELD_DEF(uint32_t, tileSamples);       // 每个 tile 处理的样本数
    TILING_DATA_FIELD_DEF(uint32_t, lastTileSamples);   // 最后一个 tile 的样本数

    // 对齐参数
    TILING_DATA_FIELD_DEF(uint32_t, hiddenDimAligned);  // D 对齐到 32B 后的元素数 (half)
    TILING_DATA_FIELD_DEF(uint32_t, gatedDimAligned);   // 4D 对齐到 32B 后的元素数 (half)
    TILING_DATA_FIELD_DEF(uint32_t, floatHiddenDimAligned); // D 对齐到 32B 后的元素数 (float)

    // 算子属性
    TILING_DATA_FIELD_DEF(float, alpha);                // z 张量缩放系数
    TILING_DATA_FIELD_DEF(float, epsCell);              // Cell RMSNorm epsilon
    TILING_DATA_FIELD_DEF(float, epsState);             // State RMSNorm epsilon
    TILING_DATA_FIELD_DEF(uint32_t, useFastGelu);       // 是否使用快速 GELU

    // 可选输入标志
    TILING_DATA_FIELD_DEF(uint32_t, hasWCell);          // 是否有 w_cell
    TILING_DATA_FIELD_DEF(uint32_t, hasBCell);          // 是否有 b_cell
    TILING_DATA_FIELD_DEF(uint32_t, hasWState);         // 是否有 w_state
    TILING_DATA_FIELD_DEF(uint32_t, hasBState);         // 是否有 b_state

    // 数据类型信息
    TILING_DATA_FIELD_DEF(uint32_t, dataTypeSize);      // 数据类型字节数 (2 for fp16/bf16)

    // UB 内存参数
    TILING_DATA_FIELD_DEF(uint32_t, ubBufferSize);      // 单个 buffer 大小
    TILING_DATA_FIELD_DEF(uint32_t, bufferCount);       // I/O 队列缓冲深度 (1=单缓冲, 2=双缓冲)
    TILING_DATA_FIELD_DEF(uint32_t, preloadWeights);    // 是否预加载权重 (0=每样本加载, 1=预加载)

END_TILING_DATA_DEF;

// 注册 Tiling 数据结构
REGISTER_TILING_DATA_CLASS(SumLstm, SumLstmTilingData)

}  // namespace optiling

#endif  // SUM_LSTM_TILING_H
