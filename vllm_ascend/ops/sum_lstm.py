# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from typing import Optional, Tuple

import torch

from vllm_ascend.utils import enable_custom_op


def npu_sum_lstm(
    states_4d: torch.Tensor,
    z4_4d: torch.Tensor,
    prev_cell: torch.Tensor,
    w_cell: Optional[torch.Tensor] = None,
    b_cell: Optional[torch.Tensor] = None,
    w_state: Optional[torch.Tensor] = None,
    b_state: Optional[torch.Tensor] = None,
    alpha: float = 1.0,
    eps_cell: float = 1e-6,
    eps_state: float = 1e-6,
    use_fast_gelu: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    SumLstm custom operator for Ascend NPU.

    This operator implements a modified LSTM cell with RMSNorm and GELU activations.

    Args:
        states_4d: Input states tensor of shape (..., 4*D), containing gate inputs
                   [s0(forget), s1(input), s2(output), s3(cell)]
        z4_4d: Additional input tensor of shape (..., 4*D), containing
               [z0(forget), z1(input), z2(output), z3(cell)]
        prev_cell: Previous cell state of shape (..., D)
        w_cell: Optional weight for cell normalization, shape (D,)
        b_cell: Optional bias for cell normalization, shape (D,)
        w_state: Optional weight for state normalization, shape (D,)
        b_state: Optional bias for state normalization, shape (D,)
        alpha: Scaling factor for z4_4d inputs (default: 1.0)
        eps_cell: Epsilon for cell RMSNorm (default: 1e-6)
        eps_state: Epsilon for state RMSNorm (default: 1e-6)
        use_fast_gelu: Whether to use fast GELU approximation (default: True)

    Returns:
        Tuple of (out_state, out_cell), both of shape (..., D)

    Computation:
        1. Compute gate pre-activations:
           - pre_f = s0 + alpha * z0
           - pre_i = s1 + alpha * z1
           - pre_o = s2 + alpha * z2
           - cpre = s3 + alpha * z3
        2. Apply sigmoid to gates: f, i, o = sigmoid(pre_f, pre_i, pre_o)
        3. Cell activation: cact = GELU(RMSNorm(cpre) * w_cell + b_cell)
        4. New cell: out_cell = prev_cell * f + cact * i
        5. State activation: sact = GELU(RMSNorm(out_cell) * w_state + b_state)
        6. Output state: out_state = sact * o
    """
    if not enable_custom_op():
        raise RuntimeError(
            "npu_sum_lstm requires custom op to be enabled. "
            "Please call enable_custom_op() first."
        )

    return torch.ops._C_ascend.npu_sum_lstm(
        states_4d,
        z4_4d,
        prev_cell,
        w_cell,
        b_cell,
        w_state,
        b_state,
        alpha,
        eps_cell,
        eps_state,
        use_fast_gelu,
    )


class SumLstmCell(torch.nn.Module):
    """
    SumLstm cell module for Ascend NPU.

    This module wraps the npu_sum_lstm operator for use in PyTorch models.

    Args:
        hidden_dim: Hidden dimension size
        alpha: Scaling factor for z4_4d inputs (default: 1.0)
        eps_cell: Epsilon for cell RMSNorm (default: 1e-6)
        eps_state: Epsilon for state RMSNorm (default: 1e-6)
        use_fast_gelu: Whether to use fast GELU approximation (default: True)
        use_cell_weights: Whether to use learnable cell weights (default: False)
        use_state_weights: Whether to use learnable state weights (default: False)
    """

    def __init__(
        self,
        hidden_dim: int,
        alpha: float = 1.0,
        eps_cell: float = 1e-6,
        eps_state: float = 1e-6,
        use_fast_gelu: bool = True,
        use_cell_weights: bool = False,
        use_state_weights: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.alpha = alpha
        self.eps_cell = eps_cell
        self.eps_state = eps_state
        self.use_fast_gelu = use_fast_gelu

        # Optional learnable weights
        if use_cell_weights:
            self.w_cell = torch.nn.Parameter(torch.ones(hidden_dim))
            self.b_cell = torch.nn.Parameter(torch.zeros(hidden_dim))
        else:
            self.register_parameter('w_cell', None)
            self.register_parameter('b_cell', None)

        if use_state_weights:
            self.w_state = torch.nn.Parameter(torch.ones(hidden_dim))
            self.b_state = torch.nn.Parameter(torch.zeros(hidden_dim))
        else:
            self.register_parameter('w_state', None)
            self.register_parameter('b_state', None)

    def forward(
        self,
        states_4d: torch.Tensor,
        z4_4d: torch.Tensor,
        prev_cell: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of SumLstm cell.

        Args:
            states_4d: Input states tensor of shape (..., 4*hidden_dim)
            z4_4d: Additional input tensor of shape (..., 4*hidden_dim)
            prev_cell: Previous cell state of shape (..., hidden_dim)

        Returns:
            Tuple of (out_state, out_cell), both of shape (..., hidden_dim)
        """
        return npu_sum_lstm(
            states_4d,
            z4_4d,
            prev_cell,
            self.w_cell,
            self.b_cell,
            self.w_state,
            self.b_state,
            self.alpha,
            self.eps_cell,
            self.eps_state,
            self.use_fast_gelu,
        )
