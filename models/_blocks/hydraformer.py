import torch
import torch.nn as nn
import torch.nn.functional as F

from models._blocks.mlp import DeepSeekExpert, Linear, LinearDO, SimpleMLP
from models._blocks.moe import MoEPredictor, MoEWithSharedExp, TokenMoE
from models._blocks.transformer import (MultiHeadSelfAttention,
                                        TransformerEncoder, get_normal_ffn)


class TransformerEncoderLayer(nn.Module):
    """
    Transformer编码器层实现，支持batch_first和src_key_padding_mask
    """
    def __init__(self, embed_dim: int, num_heads: int = 8, dim_feedforward: int = 2048,
                 dropout: float = 0.1, activation: str = "silu", norm_first: bool = False,
                 n_modal: int = 3, gated_sdpa: bool = False, use_flash_attn: bool = False):
        super(TransformerEncoderLayer, self).__init__()
        self.norm_first = norm_first
        self.n_modal = n_modal
        self.gated_sdpa = gated_sdpa
        self.use_flash_attn = use_flash_attn
        self.moe_ffn = True
        self.moe_loss_scale = 0#1#e-7
        
        # 激活函数
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU(0.1)
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "silu":
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"不支持的激活函数: {activation}")
        
        # 多头自注意力
        if use_flash_attn:
            from models._blocks.flash_attention import FlashAttentionMHSA
            self.self_attn = FlashAttentionMHSA(embed_dim, num_heads, dropout, gated_sdpa)
        else:
            self.self_attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout, gated_sdpa=gated_sdpa)
        self.code_hack_atten_weights = None
        
        # 前馈网络
        base_ffn, ffns = DeepSeekExpert(embed_dim, dim_feedforward), []
        # base_ffn, ffns = TokenMoE(MoEPredictor(nn.ModuleList([DeepSeekExpert(embed_dim, dim_feedforward), nn.Identity()]),
        #                                        1, 1, 'sum', True, hid_dim=embed_dim, dropout=dropout)), []
        for _ in range(n_modal):
            # exps = nn.ModuleList([Linear(embed_dim, embed_dim, dropout) for _ in range(6)] + [nn.Identity()]*6)
            exps = nn.ModuleList([Linear(embed_dim, embed_dim, dropout) for _ in range(12)])
            ffn = MoEPredictor(exps, 2, 1, 'weighted', True, hid_dim=embed_dim, dropout=dropout)
            ffn = MoEWithSharedExp(TokenMoE(ffn), base_ffn)
            # ffn = TokenMoE(ffn)
            ffns.append(ffn)
        self.ffns = nn.ModuleList(ffns)
        
        # 层归一化
        self.norm1 = nn.ModuleList([nn.LayerNorm(embed_dim, eps=1e-6) for _ in range(n_modal)])
        self.norm2 = nn.ModuleList([nn.LayerNorm(embed_dim, eps=1e-6) for _ in range(n_modal)])
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    # @property
    # def balance_loss(self):
    #     return sum([m.balance_loss for m in self.ffns]) * self.moe_loss_scale
        
    def extra_repr(self) -> str:
        return f"norm_first={self.norm_first}, n_modal={self.n_modal}, moe_loss_scale={self.moe_loss_scale}"
    
    def _same_shape_modal_forward(self, model: nn.ModuleList, x: torch.tensor, modal_idx: torch.Tensor):
        """
        模态特定前向传播
        Args:
            model: ModuleList
            x: 输入张量，形状为 (batch_size, seq_len, embed_dim)
            modal_idx: 模态索引，形状为 (seq_len)
        
        Returns:
            输出张量，形状为 (batch_size, seq_len, embed_dim)
        """
        # 前馈网络
        out = torch.zeros_like(x)
        for idx in range(self.n_modal):
            if (modal_idx == idx).any():
                out[:, modal_idx == idx, :] = model[idx](x[:, modal_idx == idx, :])
        return out
    
    def forward(self, src, src_key_padding_mask, modal_idx: torch.Tensor,
                only_return_first_token: int = 0, **kwargs):
        """
        前向传播
        Args:
            src: 输入张量，形状为 (batch_size, seq_len, embed_dim)
            src_key_padding_mask: 填充掩码，形状为 (batch_size, seq_len)
                                True表示对应位置是填充，需要被mask
            modal_idx: 模态索引，形状为 (seq_len)
        
        Returns:
            输出张量，形状为 (batch_size, seq_len, embed_dim)
        """
        # 多头自注意力 + 残差连接 + 层归一化
        if self.norm_first:
            src = self._same_shape_modal_forward(self.norm1, src, modal_idx)
        src2 = self.self_attn(src, src_key_padding_mask=src_key_padding_mask)
        # if only_return_first_token > 0:
        #     src2 = self.self_attn(src[:, :only_return_first_token, :], kv=src, src_key_padding_mask=src_key_padding_mask)
        # else:
        #     src2 = self.self_attn(src, src_key_padding_mask=src_key_padding_mask)
        self.code_hack_atten_weights = self.self_attn.attn_weights
        
        if only_return_first_token > 0:
            modal_idx = modal_idx[:only_return_first_token]
            src, src2 = src[:, :only_return_first_token, :], src2[:, :only_return_first_token, :]
            
        # 残差连接 + Dropout
        src = src + self.dropout1(src2)
        if not self.norm_first:
            src = self._same_shape_modal_forward(self.norm1, src, modal_idx)
        
        # 前馈网络 + 残差连接 + 层归一化
        if self.norm_first:
            src = self._same_shape_modal_forward(self.norm2, src, modal_idx)
        src2 = self._same_shape_modal_forward(self.ffns, src, modal_idx)
        src = src + self.dropout2(src2)
        if not self.norm_first:
            src = self._same_shape_modal_forward(self.norm2, src, modal_idx)
        
        return src
    
    
if __name__ == "__main__":
    encoder_layer = TransformerEncoderLayer(embed_dim=512, num_heads=8, dim_feedforward=2048, dropout=0.1, n_modal=3)
    encoder = TransformerEncoder(encoder_layer, num_layers=2, norm=nn.LayerNorm(512), batch_first=True)
    src = torch.randn(2, 10, 512)
    src_key_padding_mask = torch.zeros(2, 10, dtype=torch.bool)
    modal_idx = torch.randint(0, 3, (10,))
    output = encoder(src, src_key_padding_mask, modal_idx=modal_idx, only_return_first_token=1)
    print(output.shape)
    print(encoder.balance_loss)