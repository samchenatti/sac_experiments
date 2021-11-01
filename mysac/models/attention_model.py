# pylint: disable=no-member

import torch
from mysac.models.mlp import PolicyModel as MLPPolicyModel
from mysac.models.mlp import QModel as MLPQModel
from mysac.utils import get_device
from torch import nn

# Numerical stability
LOG_SIG_MAX = 2
LOG_SIG_MIN = -20

# Use the values as observed in Rlkit
B_INIT_VALUE = 0.1
W_INIT_VALUE = 3e-3


class AttentionBase(nn.Module):
    def __init__(self, num_inputs: int, num_outputs: int):
        super(AttentionBase, self).__init__()

        num_inputs = num_inputs + 1

        self.query = nn.Linear(
            in_features=num_inputs,
            out_features=num_outputs,
            bias=False
        )

        self.key = nn.Linear(
            in_features=num_inputs,
            out_features=num_outputs,
            bias=False
        )

        self.value = nn.Linear(
            in_features=num_inputs,
            out_features=num_outputs,
            bias=False
        )

        self.post_linear = nn.Linear(
            in_features=num_inputs,
            out_features=num_outputs
        )

        self.multi_head_attention = nn.MultiheadAttention(
            embed_dim=num_outputs,
            num_heads=1,
            batch_first=True
        )

        self.recurrent_layer = nn.LSTM(
            num_inputs,
            32,
            batch_first=True
        )

        self.norm_1 = nn.LayerNorm(normalized_shape=num_outputs)
        self.norm_2 = nn.LayerNorm(normalized_shape=num_outputs)

    def forward(self, state: torch.tensor) -> torch.tensor:
        if True:
            batch_size = state.shape[0]
            num_frames = state.shape[1]

            position_enconding = torch.arange(num_frames).to(get_device())
            position_enconding *= num_frames

            position_enconding = position_enconding.repeat(batch_size)
            position_enconding = position_enconding.reshape(batch_size, 20, 1)

            state = torch.cat([state, position_enconding], dim=-1)

        Q = self.query(state)
        K = self.key(state)
        V = self.value(state)

        context, _ = self.multi_head_attention(query=Q, key=K, value=V)

        context = self.norm_1(context + state)

        linear_context = self.post_linear(context)

        _, (embedded, _) = self.recurrent_layer(
            self.norm_2(context + linear_context)
        )

        return embedded.squeeze(0)


class QModel(nn.Module):
    def __init__(
        self, hidden_size: int, post_attention: bool = False, **kwargs
    ):
        super(QModel, self).__init__()

        self.attention_base = AttentionBase(num_inputs=10, num_outputs=11)

        del kwargs['num_inputs']

        self.mlp_q = MLPQModel(num_inputs=32, hidden_sizes=32, **kwargs)

        print('Q Model:', self)

    def forward(self, state: torch.tensor, action: torch.tensor):
        state = self.attention_base.forward(state)

        return self.mlp_q.forward(observations=state, actions=action)


class PolicyModel(nn.Module):
    def __init__(
        self,
        *args,
        hidden_size: int,
        num_inputs: int,
        post_attention: bool = False,
        **kwargs
    ):
        super(PolicyModel, self).__init__()

        self.attention_base = AttentionBase(num_inputs=10, num_outputs=11)

        self.mlp_policy = MLPPolicyModel(
            *args, num_inputs=32, hidden_sizes=32, **kwargs)

    def forward(self, state: torch.tensor):
        if len(state.shape) == 2:
            state = state.unsqueeze(0)

        state = self.attention_base.forward(state)

        mean, std = self.mlp_policy.forward(observations=state)

        if mean.shape[0] == 1:
            mean = mean.squeeze(0)
            std = std.squeeze(0)

        return mean, std
