import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models._blocks.mlp import DeepSeekExpert, LinearDO, Linear, SimpleMLP
from models._blocks.moe import MoEPredictor, MoEWithSharedExp, TokenMoE


class MultiHeadSelfAttention(nn.Module):
    """
    多头自注意力机制实现，支持batch_first和src_key_padding_mask
    """
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1, gated_sdpa: bool = False):
        super(MultiHeadSelfAttention, self).__init__()
        assert embed_dim % num_heads == 0, "embed_dim必须能被num_heads整除"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # 线性变换层
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # gated attention
        self.gated_sdpa = gated_sdpa
        if gated_sdpa:
            self.gate_proj = nn.Linear(embed_dim, self.num_heads * self.head_dim, bias=False)
            self.gated_sdpa_start_idx = 0 if gated_sdpa == -1 else gated_sdpa
        
        self.dropout = nn.Dropout(dropout)
        self._reset_parameters()
        self.attn_weights = None
    
    def _reset_parameters(self):
        # Xavier初始化
        for module in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.)
                
    def extra_repr(self) -> str:
        return f"embed_dim={self.embed_dim}, num_heads={self.num_heads}, dropout={self.dropout.p}, gated_sdpa={self.gated_sdpa}"
    
    def forward(self, x, kv: torch.Tensor = None, src_key_padding_mask: torch.Tensor = None, softmax_partition: list[int] = None):
        """
        前向传播
        Args:
            x: 输入张量，形状为 (batch_size, seq_len, embed_dim)
            kv: K-V张量，可选，形状为 (batch_size, seq2_len, embed_dim)
            src_key_padding_mask: 填充掩码，形状为 (batch_size, kv_len)
                                True表示对应位置是填充，需要被mask
            softmax_partition:  softmax分区，L1+L2+L3+...=L
        Returns:
            输出张量，形状为 (batch_size, seq_len, embed_dim)
        """
        batch_size, seq_len, embed_dim = x.size()
        
        # 计算Q, K, V
        Q = self.q_proj(x)  # (batch_size, seq_len, embed_dim)
        kv = kv if kv is not None else x
        kv_len = kv.size(1)
        K = self.k_proj(kv)  # (batch_size, seq2_len, embed_dim)
        V = self.v_proj(kv)  # (batch_size, seq2_len, embed_dim)
        
        # 重塑为多头格式
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        K = K.view(batch_size, kv_len, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, kv_len, head_dim)
        V = V.view(batch_size, kv_len, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, kv_len, head_dim)
        
        # 计算注意力分数
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (batch_size, num_heads, seq_len, kv_len)
        
        # 应用填充掩码
        if src_key_padding_mask is not None:
            # 将掩码扩展到多头
            # src_key_padding_mask: (batch_size, kv_len)
            # 需要扩展为: (batch_size, 1, 1, kv_len)
            mask = src_key_padding_mask.unsqueeze(1).unsqueeze(2)
            # 将True位置设置为非常大的负数，这样softmax后这些位置的权重会接近0
            attn_scores = attn_scores.masked_fill(mask, float('-inf'))
        
        # 计算注意力权重
        if softmax_partition is not None:
            # 对每个分区进行softmax
            start, attn_weights = 0, torch.zeros_like(attn_scores)
            for (partition_len, partition_weight) in softmax_partition:
                attn_weights[:, :, :, start:start+partition_len] = F.softmax(attn_scores[:, :, :, start:start+partition_len], dim=-1) * partition_weight
                start += partition_len
            attn_weights /= len(softmax_partition)
        else:
            attn_weights = F.softmax(attn_scores, dim=-1)
        self.attn_weights = attn_weights.detach().mean(1)
        attn_weights = self.dropout(attn_weights)
        
        # 应用注意力权重到V上
        output = torch.matmul(attn_weights, V)  # (batch_size, num_heads, seq_len, head_dim)
        
        # Gated Attention for Large Language Models
        if self.gated_sdpa and x.shape[1] >= self.gated_sdpa_start_idx:
            # score: [batch_size, seq_len, num_heads * head_dim]
            gate_scores = self.gate_proj(x[:, self.gated_sdpa_start_idx:]).sigmoid()
            # -> [batch_size, num_heads, seq_len, head_dim]
            gate_scores = gate_scores.view(batch_size, seq_len-self.gated_sdpa_start_idx, self.num_heads, self.head_dim).transpose(1, 2)
            output[:, :, self.gated_sdpa_start_idx:, :].mul_(gate_scores)
        
        # 合并多头
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        
        # 输出投影
        output = self.out_proj(output)
        
        return output


def get_normal_ffn(embed_dim: int, dim_feedforward: int = 2048, dropout: float = 0.1, activation: nn.Module = nn.ReLU()):
    return nn.Sequential(
        nn.Linear(embed_dim, dim_feedforward),
        activation,
        nn.Dropout(dropout),
        nn.Linear(dim_feedforward, embed_dim)
    )


class TransformerEncoderLayer(nn.Module):
    """
    Transformer编码器层实现，batch_first，支持src_key_padding_mask
    """
    def __init__(self, embed_dim: int, num_heads: int = 8, dim_feedforward: int = 2048,
                 dropout: float = 0.1, activation: str = "silu", norm_first: bool = False,
                 moe_ffn: bool = False, gated_sdpa: bool = False, use_flash_attn: bool = False):
        super(TransformerEncoderLayer, self).__init__()
        self.norm_first = norm_first
        self.moe_ffn = moe_ffn
        self.use_flash_attn = use_flash_attn
        
        # 激活函数
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "silu":
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"不支持的激活函数: {activation}")
        
        # 多头自注意力 - 支持flash-attn2
        if use_flash_attn:
            from models._blocks.flash_attention import FlashAttentionMHSA
            self.self_attn = FlashAttentionMHSA(embed_dim, num_heads, dropout, gated_sdpa)
        else:
            self.self_attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout, gated_sdpa)
        self.code_hack_atten_weights = None
        
        # 前馈网络
        if moe_ffn:
            # exps = nn.ModuleList([get_normal_ffn(embed_dim, dim_feedforward, dropout, self.activation) for _ in range(48)])
            exps = nn.ModuleList([Linear(embed_dim, embed_dim, dropout) for _ in range(12)])
            # exps = nn.ModuleList([Linear(embed_dim, embed_dim, dropout) for _ in range(12)] + [nn.Identity()]*12)
            # exps = nn.ModuleList([SimpleMLP(embed_dim, embed_dim, dropout, embed_dim//2) for _ in range(128)])
            ffn = MoEPredictor(exps, 1, 1, 'sum', True, hid_dim=embed_dim, dropout=dropout)
            shared_ffn = get_normal_ffn(embed_dim, dim_feedforward, dropout, self.activation)
            # shared_ffn = DeepSeekExpert(embed_dim, dim_feedforward)
            self.ffn = MoEWithSharedExp(TokenMoE(ffn), shared_ffn)
            # self.ffn = TokenMoE(ffn)
        else:
            self.ffn = get_normal_ffn(embed_dim, dim_feedforward, dropout, self.activation)
        
        # 层归一化
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def extra_repr(self) -> str:
        return f"norm_first={self.norm_first}, MoE FFN={self.moe_ffn}, use_flash_attn={self.use_flash_attn}"
        
    @property
    def balance_loss(self):
        if self.moe_ffn:
            return self.ffn.balance_loss
        return 0
    
    def forward(self, src, src_key_padding_mask=None, softmax_partition: list[int] = None,
                only_return_first_token: int = 0, **kwargs):
        """
        前向传播
        Args:
            src: 输入张量，形状为 (batch_size, seq_len, embed_dim)
            src_key_padding_mask: 填充掩码，形状为 (batch_size, seq_len)
                                True表示对应位置是填充，需要被mask
            softmax_partition:  softmax分区，L1+L2+L3+...=L
        
        Returns:
            输出张量，形状为 (batch_size, seq_len, embed_dim)
        """
        # 多头自注意力 + 残差连接 + 层归一化
        if self.norm_first:
            src = self.norm1(src)
        # if only_return_first_token > 0 and not self.training:
        #     # fast inference
        #     src2 = self.self_attn(src[:, :only_return_first_token, :], kv=src, src_key_padding_mask=src_key_padding_mask,
        #                           softmax_partition=softmax_partition)
        # else:
        src2 = self.self_attn(src, src_key_padding_mask=src_key_padding_mask)
        self.code_hack_atten_weights = self.self_attn.attn_weights
        
        if only_return_first_token > 0:
            src, src2 = src[:, :only_return_first_token, :], src2[:, :only_return_first_token, :]
        
        src = src + self.dropout1(src2)
        if not self.norm_first:
            src = self.norm1(src)
        
        # 前馈网络 + 残差连接 + 层归一化
        if self.norm_first:
            src = self.norm2(src)
        src2 = self.ffn(src)
        src = src + self.dropout2(src2)
        if not self.norm_first:
            src = self.norm2(src)
        
        return src


class TransformerEncoder(nn.Module):
    """
    Transformer编码器实现，支持batch_first和src_key_padding_mask
    """
    def __init__(self, encoder_layer, num_layers, norm=None, batch_first: bool = False):
        super(TransformerEncoder, self).__init__()
        if isinstance(encoder_layer, (list, tuple)):
            self.layers = nn.ModuleList(encoder_layer)
        else:
            self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm
        self.batch_first = batch_first
        
    @property
    def balance_loss(self):
        if getattr(self.layers[0], 'moe_ffn', None):
            return sum([m.balance_loss if m.moe_ffn else 0 for m in self.layers])
        else:
            return 0
    
    def forward(self, src, src_key_padding_mask=None, softmax_partition: list[int] = None,
                only_return_first_token: int = 0, **kwargs):
        """
        前向传播
        Args:
            src: 输入张量，形状为 (batch_size, seq_len, embed_dim)
            src_key_padding_mask: 填充掩码，形状为 (batch_size, seq_len)
                                True表示对应位置是填充，需要被mask
            softmax_partition:  softmax分区，L1+L2+L3+...=L
            only_return_first_token: 是否只返回第一个token的输出, 节省计算, 0表示不开启, 其他值表示只返回前N个token的输出
        
        Returns:
            输出张量，形状为 (batch_size, seq_len, embed_dim)
        """
        if not self.batch_first:
            src = src.transpose(0, 1)
        output = src
        
        for i, mod in enumerate(self.layers):
            output = mod(output, src_key_padding_mask=src_key_padding_mask,
                         softmax_partition=softmax_partition,
                         only_return_first_token=only_return_first_token if (i == self.num_layers - 1) else False,
                         **kwargs)
        
        if self.norm is not None:
            output = self.norm(output)
        
        if not self.batch_first:
            output = output.transpose(0, 1)
        return output
    
    
if __name__ == '__main__':
    # 测试TransformerEncoderLayer
    layer = TransformerEncoderLayer(embed_dim=512, num_heads=8, dim_feedforward=2048, dropout=0.1, moe_ffn=True, gated_sdpa=True)
    layer.eval()
    src = torch.randn(1, 10, 512)  # (batch_size, seq_len, embed_dim)
    src[:, 0] *= 0.5
    src[:, 1:6] *= 1.5
    src[:, 6:] *= 10    
    mask = torch.zeros(1, 10, dtype=torch.bool)  # 全False表示无填充
    partition = [(1, 0.5), (5, 0.3), (4, 1)]
    output1 = layer(src, src_key_padding_mask=mask)
    output2 = layer(src, src_key_padding_mask=mask, only_return_first_token=1)
    torch.testing.assert_close(output1[:, 0, :], output2[:, 0, :])
    
    # # test MHSA
    # attn = layer.self_attn
    # attn.eval()
    # output3 = attn(src, src_key_padding_mask=mask)
    # output4 = attn(src[:, 0:1, :], src, src_key_padding_mask=mask)
    # torch.testing.assert_close(output3[0, 0], output4[0, 0])
    
    # access balance loss
    encoder = TransformerEncoder(layer, num_layers=1, batch_first=True)
    encoder.train()
    output = encoder(src, src_key_padding_mask=mask)
    print(output.shape)
    print(encoder.balance_loss)
