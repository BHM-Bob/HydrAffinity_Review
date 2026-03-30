
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from mbapy.dl_torch.bb import EncoderLayer, PositionwiseFeedforwardLayer, RoPE


@torch.compile
def add_rope(x: torch.Tensor):
    """
    Add Rotary Position Embedding (RoPE) to the input tensor.
    Args:
        x: Input tensor of shape [N, L, D].
    Returns:
        Tensor with RoPE applied, shape [N, L, D].
    """
    _, seq_len, dim = x.shape
    half_dim = dim // 2
    freq_seq = torch.arange(half_dim, device=x.device, dtype=torch.float32)
    inv_freq = 1.0 / (10000 ** (freq_seq / half_dim))
    pos_seq = torch.arange(seq_len, device=x.device, dtype=torch.float32)
    sinusoid_inp = torch.einsum("i,j->ij", pos_seq, inv_freq)
    sin, cos = sinusoid_inp.sin(), sinusoid_inp.cos()
    x1, x2 = x[..., :half_dim], x[..., half_dim:]
    x = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return x


@torch.compile
def add_rope1(x: torch.Tensor):
    _, seq_len, dim = x.shape
    half_dim = dim // 2
    device = x.device
    
    # 更稳定的频率计算（使用对数避免大指数）
    inv_freq = torch.exp(-torch.arange(0, half_dim, device=device) * (math.log(10000.0) / half_dim))
    
    # 使用双精度计算位置编码
    pos_seq = torch.arange(seq_len, device=device, dtype=torch.float64)
    sinusoid_inp = torch.einsum("i,j->ij", pos_seq, inv_freq.double())
    
    # 更精确的三角函数计算
    sin = torch.sin(sinusoid_inp).to(x.dtype)
    cos = torch.cos(sinusoid_inp).to(x.dtype)
    
    x1, x2 = x[..., :half_dim], x[..., half_dim:]
    
    # 添加数值保护
    with torch.autocast(device_type='cuda', enabled=False):  # 禁用混合精度
        rotated_x1 = x1 * cos - x2 * sin
        rotated_x2 = x1 * sin + x2 * cos
    
    # 数值保护
    # rotated_x1 = torch.clamp(rotated_x1, min=-1e4, max=1e4)
    # rotated_x2 = torch.clamp(rotated_x2, min=-1e4, max=1e4)
    
    out = torch.cat([rotated_x1, rotated_x2], dim=-1)
    print(f'add_rope input and output range: INPUT {x.min().item():9.0f} {x.max().item():9.0f}, OUTPUT {out.min().item():9.0f} {out.max().item():9.0f}')
    return out


# remain for compatibility
class AttnBase(nn.Module):
    """TransformerEncoder with pred-token, input is [N, L, feat_dim], output is [N, 1, feat_dim].
    Args:
        feat_dim (int): Feature dimension of the input.
        num_layers (int): Number of transformer encoder layers.
        nhead (int): Number of attention heads.
        dropout (float): Dropout rate for the transformer encoder.
        
    forward method:
        feat (torch.Tensor): Input tensor of shape [N, L, feat_dim].
        mask (torch.Tensor): Optional attention mask of shape [N, L], False means padding.
        Returns:
            torch.Tensor: Output tensor of shape [N, 1, feat_dim].
    """
    def __init__(self, feat_dim: int = 384, num_layers: int = 2, nhead: int = 8, dropout: float = 0.4):
        super().__init__()
        # Learnable prediction token
        self.pred_token = nn.Parameter(torch.zeros(1, 1, feat_dim))  # [1, 1, 384]
        nn.init.xavier_uniform_(self.pred_token)  # Initialize with Xavier uniform
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=feat_dim, nhead=nhead,
                                                   dim_feedforward=4*feat_dim,
                                                   dropout=dropout, activation='relu')
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    def forward(self, feat: torch.Tensor, mask: torch.Tensor):
        batch_size, _, _ = feat.shape  # [N, L, 384]
        
        # Add learnable prediction token
        pred_token = self.pred_token.expand(batch_size, -1, -1)  # [N, 1, 384]
        smile_feat = torch.cat([pred_token, feat], dim=1)  # [N, L+1, 384]
        
        # Add RoPE (Rotary Position Embedding)
        smile_feat = add_rope(smile_feat)  # [N, L+1, 384]
        
        # Create attention mask for Transformer
        if mask is not None:
            extended_mask = torch.cat([torch.ones(batch_size, 1, device=mask.device), mask], dim=1)  # [N, L+1]
            transformer_mask = ~extended_mask.bool()  # [N, L+1]
        else:
            transformer_mask = None
        
        # Pass through Transformer
        transformer_out = self.transformer(smile_feat.transpose(0, 1), src_key_padding_mask=transformer_mask)  # [L+1, N, 384]
        transformer_out = transformer_out.transpose(0, 1)  # [N, L+1, 384]
        
        # Extract prediction token output
        return transformer_out[:, 0:1, :]  # [N, 1, 384]


class MiniMHSA(nn.Module):
    """random init weight and do not need grad, reduce learnable params"""
    def __init__(self, feat_dim: int = 384, n_head: int = 8, dropout: float = 0.4):
        super().__init__()
        self.feat_dim = feat_dim
        self.n_head = n_head
        self.dropout = dropout
        self.head_dim = feat_dim // n_head
        self.qkv_proj = nn.Linear(feat_dim, 3 * feat_dim)
        self.out_proj = nn.Linear(feat_dim, feat_dim)
        self.dropout = nn.Dropout(dropout)
        # random init weight
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        # disable grad
        self.qkv_proj.requires_grad_(False)
        self.out_proj.requires_grad_(False)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """x: [N, L, feat_dim], mask: [N, L], False means padding"""
        qkv = self.qkv_proj(x)  # [N, L, 3*feat_dim]
        q, k, v = torch.split(qkv, self.feat_dim, dim=-1)  # [N, L, feat_dim]
        q = q.view(q.shape[0], q.shape[1], self.n_head, self.head_dim).transpose(1, 2)  # [N, n_head, L, head_dim]
        k = k.view(k.shape[0], k.shape[1], self.n_head, self.head_dim).transpose(1, 2)  # [N, n_head, L, head_dim]
        v = v.view(v.shape[0], v.shape[1], self.n_head, self.head_dim).transpose(1, 2)  # [N, n_head, L, head_dim]
        # compute attention score
        attn_score = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [N, n_head, L, L]
        # apply mask
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(2)  # [N, 1, 1, L]
            attn_score = attn_score.masked_fill(mask, -1e9)
        # softmax
        attn_score = torch.softmax(attn_score, dim=-1)  # [N, n_head, L, L]
        # dropout
        attn_score = self.dropout(attn_score)  # [N, n_head, L, L]
        # compute output
        out = torch.matmul(attn_score, v)  # [N, n_head, L, head_dim]
        out = out.transpose(1, 2).contiguous().view(out.shape[0], out.shape[2], self.feat_dim)  # [N, L, feat_dim]
        # linear projection
        out = self.out_proj(out)  # [N, L, feat_dim]
        return out


class MiniTransformerEncodeLayer(nn.Module):
    """use MiniMHSA, only has layernorm and MHSA, no feedforward"""
    def __init__(self, feat_dim: int = 384, n_head: int = 8, dropout: float = 0.4):
        super().__init__()
        self.attn = MiniMHSA(feat_dim, n_head, dropout)
        self.norm = nn.LayerNorm(feat_dim, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """x: [N, L, feat_dim], mask: [N, L], False means padding"""
        # MHSA
        attn_out = self.attn(x, mask)  # [N, L, feat_dim]
        # dropout
        attn_out = self.dropout(attn_out)  # [N, L, feat_dim]
        # residual connection and layernorm
        x = self.norm(x + attn_out)  # [N, L, feat_dim]
        return x


class PredTokenAttn(nn.Module):
    def __init__(self, n_token: int = 1, feat_dim: int = 384, n_layer: int = 2, 
                 n_head: int = 8, dropout: float = 0.4, RMSNorm: bool = False,
                 use_rope: bool = True, norm_first: bool = False, softmax_partition: bool = False,
                 moe_ffn: list[int] = False, hydraformer: int = 0, use_method_id: str = None,
                 gated_sdpa: bool = False, low_mem_transformer: bool = False):
        """TransformerEncoder with pred-token, input is [N, L, feat_dim], output is [N, n_token, feat_dim].
        Args:
            n_token (int): Number of prediction tokens.
            feat_dim (int): Feature dimension of the input.
            n_layer (int): Number of transformer encoder layers.
            n_head (int): Number of attention heads.
            dropout (float): Dropout rate for the transformer encoder.
            RMSNorm (bool): Whether to use RMSNorm instead of LayerNorm.
            use_rope (bool): Whether to use RoPE (Rotary Position Embedding).
            norm_first (bool): Whether to use norm-first architecture.
            softmax_partition (bool): Whether to use softmax partition. It will replace pytorch transformer with User Defined TransformerEncoder.
            moe_ffn (list[int]): Whether to use MoE feedforward network in each layer. If not None, it should be a list of bool with length n_layer.
            hydraformer (int): Number of modal in hydraformer. If not 0, it will use hydraformer.
            use_method_id (str): Whether to use mid token. If not None, it should be 'replace' or 'external'.
                - replace: use mid token to replace pred token.
                - external: use mid token as external token added into the seq after pred token.
            gated_sdpa (bool): Whether to use gated SDPA.
            low_mem_transformer (bool): Whether to use low memory transformer.
            
        forward method:
            feat (torch.Tensor): Input tensor of shape [N, L, feat_dim].
            mask (torch.Tensor): Optional attention mask of shape [N, L], False means padding.
            modal_idx (torch.Tensor): Optional modal index of shape [L], used in hydraformer.
            Returns:
                torch.Tensor: Output tensor of shape [N, n_token, feat_dim].
        """
        super().__init__()
        # Learnable prediction token
        self.n_token = self.mask_n_token = n_token
        self.n_head = n_head
        self.use_rope = use_rope
        self.norm_first = norm_first        
        self.RMSNorm = RMSNorm
        self.softmax_partition = softmax_partition
        self.moe_ffn = moe_ffn
        self.hydraformer = hydraformer
        self.use_mid_token = use_method_id
        self.gated_sdpa = gated_sdpa
        self.low_mem_transformer = low_mem_transformer
        
        # pred token
        if not self.use_mid_token:
            self.pred_token = nn.Parameter(torch.zeros(1, n_token, feat_dim))  # [1, n_token, 384]
        elif 'replace' in self.use_mid_token:
            self.mid_token = nn.Embedding(3, feat_dim)
        elif 'external' in self.use_mid_token:
            self.pred_token = nn.Parameter(torch.zeros(1, n_token, feat_dim))  # [1, n_token, 384]
            nn.init.xavier_uniform_(self.pred_token)  # Initialize with Xavier uniform
            self.mid_token = nn.Embedding(3, feat_dim)
            self.mask_n_token += 1
        else:
            raise ValueError(f"use_mid_token should be [], 'replace' or 'external', but got {self.use_mid_token}")
        nn.init.xavier_uniform_(self.pred_token if not self.use_mid_token else self.mid_token.weight)  # Initialize with Xavier uniform
        
        # Transformer Encoder
        self.std_transformer = False
        if (not softmax_partition) and (not moe_ffn) and (hydraformer == 0) and (not gated_sdpa) and (not low_mem_transformer):
            self.std_transformer = True
            encoder_layer = nn.TransformerEncoderLayer(d_model=feat_dim, nhead=n_head,
                                                    dim_feedforward=4*feat_dim,
                                                    dropout=dropout, activation='relu', norm_first=norm_first)
            if RMSNorm:
                encoder_layer.norm1 = nn.RMSNorm(feat_dim, eps=1e-6)
                encoder_layer.norm2 = nn.RMSNorm(feat_dim, eps=1e-6)
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)
        elif hydraformer > 0:
            from models._blocks.hydraformer import TransformerEncoderLayer, TransformerEncoder
            layer = TransformerEncoderLayer(feat_dim, n_head, 4*feat_dim, dropout, norm_first=norm_first,
                                            n_modal=hydraformer, gated_sdpa=gated_sdpa, use_flash_attn=low_mem_transformer)
            self.transformer = TransformerEncoder(layer, n_layer, batch_first=False)
        else:
            from models._blocks.transformer import TransformerEncoderLayer, TransformerEncoder
            layers = []
            for _, is_moe in enumerate(moe_ffn or [False]*n_layer):
                layers.append(TransformerEncoderLayer(feat_dim, n_head, dropout=dropout, norm_first=norm_first, moe_ffn=is_moe,
                                                      gated_sdpa=gated_sdpa, use_flash_attn=low_mem_transformer))
            self.transformer = TransformerEncoder(layers, n_layer, batch_first=False)
        
    def extra_repr(self) -> str:
        return f"n_token={self.n_token}, mask_n_token={self.mask_n_token}, use_mid_token={self.use_mid_token}, n_head={self.n_head}, RMSNorm={self.RMSNorm},"\
                    f"use_rope={self.use_rope}, norm_first={self.norm_first}, softmax_partition={self.softmax_partition}, moe_ffn={self.moe_ffn},"\
                    f"hydraformer={self.hydraformer}, gated_sdpa={self.gated_sdpa}, low_mem_transformer={self.low_mem_transformer}"
    
    @property
    def balance_loss(self):
        if hasattr(self.transformer, 'balance_loss'):
            return self.transformer.balance_loss
        raise AttributeError("TransformerEncoder does not have balance_loss attribute")
        
    def forward(self, feat: torch.Tensor, mask: torch.Tensor, pta_return_src: bool = False,
                only_return_first_token: int = 0, mid: torch.Tensor = None, **kwargs):
        only_return_first_token = only_return_first_token if not pta_return_src else 0
        batch_size, _, _ = feat.shape  # [N, L, D]
        if not self.std_transformer:
            kwargs['only_return_first_token'] = only_return_first_token
        
        # Add learnable prediction token
        if not self.use_mid_token:
            pred_token = self.pred_token.expand(batch_size, -1, -1)  # [N, n_token, D]
        elif 'replace' in self.use_mid_token:
            pred_token = self.mid_token(mid)  # [N, n_token, D]
        elif 'external' in self.use_mid_token:
            _pred_token = self.pred_token.expand(batch_size, -1, -1)  # [N, n_token, D]
            mid_token = self.mid_token(mid)  # [N, 1, D]
            pred_token = torch.cat([_pred_token, mid_token], dim=1)  # [N, n_token+1, D]
        
        smile_feat = torch.cat([pred_token, feat], dim=1)  # [N, n_token+L, D]
        
        # Add RoPE (Rotary Position Embedding) if enabled
        if self.use_rope:
            smile_feat = add_rope(smile_feat)  # [N, n_token+L, D]
        
        # Create attention mask for Transformer
        if mask is not None:
            extended_mask = torch.cat([torch.ones(batch_size, self.mask_n_token, device=mask.device), mask], dim=1)  # [N, n_token+L]
            transformer_mask = ~extended_mask.bool()  # [N, n_token+L]
        else:
            transformer_mask = None
        
        # Pass through Transformer
        transformer_out = self.transformer(smile_feat.transpose(0, 1),
                                           src_key_padding_mask=transformer_mask,
                                           **kwargs)  # [n_token+L, N, 384]
        transformer_out = transformer_out.transpose(0, 1)  # [N, n_token+L, 384]
        
        if pta_return_src:
            return transformer_out[:, :self.n_token, :], transformer_out[:, self.n_token:, :]
        # Extract prediction token output
        return transformer_out[:, :self.n_token, :]  # [N, n_token, 384]


class CrossAttn(nn.Module):
    """交叉注意力对齐模块"""
    def __init__(self, feat_dim: int = 384, n_layer: int = 2, n_head: int = 8, dropout: float = 0.4):
        super().__init__()
        self.rope = RoPE(feat_dim, 1000)
        # 交叉注意力层
        self.cross_attn = nn.MultiheadAttention(embed_dim=feat_dim, num_heads=n_head, dropout=dropout)
        # 共享的Transformer编码层
        decoder_layer = nn.TransformerDecoderLayer(d_model=feat_dim, nhead=n_head,
                                                 dim_feedforward=4*feat_dim,
                                                 dropout=dropout, activation='relu')
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=n_layer)
        
    def get_attn(self, q: torch.Tensor, k: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # 应用RoPE位置编码: [N, L, D] -> [N, L, D]
        q, k = add_rope(q), add_rope(k)
        
        # 交叉注意力计算
        _, attn_output_weights = self.cross_attn(
            query=q.transpose(0, 1), 
            key=k.transpose(0, 1), 
            value=k.transpose(0, 1)
        )  # [Lq, N, D]
        
        return attn_output_weights

    def forward(self, q: torch.Tensor, k: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # 应用RoPE位置编码: [N, L, D] -> [N, L, D]
        q, k = add_rope(q), add_rope(k)
        
        # 交叉注意力计算
        attn_output, _ = self.cross_attn(
            query=q.transpose(0, 1), 
            key=k.transpose(0, 1), 
            value=k.transpose(0, 1)
        )  # [Lq, N, D]
        
        # 通过Transformer解码层
        decoder_q = self.transformer(
            tgt=attn_output,
            memory=q.transpose(0, 1),
            tgt_key_padding_mask=mask
        ).transpose(0, 1)  # [N, Lq, D]
        decoder_k = self.transformer(
            tgt=attn_output,
            memory=k.transpose(0, 1),
            tgt_key_padding_mask=mask
        ).transpose(0, 1)  # [N, Lq, D]
        
        return decoder_q, decoder_k


class MLDecoderLite(nn.Module):
    def __init__(self, num_classes: int, input_dim: int, n_head: int = 8, dropout: float = 0.3):
        """MLDecoderLite
        Args:
            num_classes (int): Number of classes.
            input_dim (int): Input dimension.
            n_heads (int, optional): Number of attention heads. Defaults to 8.
            
        forward:
            x (torch.Tensor): Input tensor of shape [b, L, C].
            return (torch.Tensor): Output tensor of shape [b, NC].
        """
        super(MLDecoderLite, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.head_dim = input_dim // n_head
        self.n_head = n_head
        # ML_Decoder : CrossAttention
        self.gq = nn.Parameter(torch.zeros(1, num_classes, input_dim))
        torch.nn.init.xavier_uniform_(self.gq)
        self.cross_attn = nn.MultiheadAttention(input_dim, n_head, batch_first=True)
        # ML_Decoder : FFN
        self.self_attn_layer_norm = nn.LayerNorm(input_dim)
        self.ff_layer_norm = nn.LayerNorm(input_dim)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(input_dim, 4*input_dim, dropout)
        self.dropout_FFN = nn.Dropout(dropout)
        # ML_Decoder : groupFC
        self.groupFC = nn.Linear(input_dim, 1)  

    def forward(self, x:torch.Tensor)->torch.Tensor:
        if len(x.shape) == 2:
            x.unsqueeze_(1)
        # x: [b, L, C]
        batch_size = x.shape[0]
        # gq: [1, NC, C] => [b, NC, C]
        gq = self.gq.repeat(batch_size, 1, 1)
        # _x: [b, NC, C]
        _x = self.cross_attn(gq, x, x)[0]
        # x: [b, NC, C]
        x = self.self_attn_layer_norm(gq + self.dropout_FFN(_x))
        # positionwise feedforward
        _x = self.positionwise_feedforward(x)
        # dropout, residual and layer norm
        x = self.ff_layer_norm(x + self.dropout_FFN(_x))
        # x: [b, NC, C] => [b, NC]
        x = self.groupFC(x).reshape(batch_size, 1, -1)
        return x


class BANLayer(nn.Module):
    def __init__(self, v_dim, q_dim, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # 特征投影
        self.v_proj = nn.Linear(v_dim, num_heads * self.head_dim)
        self.q_proj = nn.Linear(q_dim, num_heads * self.head_dim)
        
        # 双线性注意力权重
        self.att_weight = nn.Parameter(torch.Tensor(1, num_heads, self.head_dim, self.head_dim))
        nn.init.xavier_uniform_(self.att_weight)
        
        # 输出层
        self.output = nn.Sequential(
            nn.Linear(2 * num_heads * self.head_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.register_buffer('scale', torch.ones(1) * self.head_dim ** -0.5)

    def forward(self, v, q):
        b, v_num, _ = v.shape
        q_num = q.size(1)
        
        # 特征投影
        v_proj = self.v_proj(v)  # [b, v_num, h*head]
        q_proj = self.q_proj(q)  # [b, q_num, h*head]
        
        # 重塑为多头格式
        v_proj = v_proj.view(b, v_num, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        q_proj = q_proj.view(b, q_num, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # 双线性注意力计算
        att_logits = torch.einsum('bhvi,hij,bhqj->bhvq', 
                                 v_proj,
                                 self.att_weight[0],  # [head, h, h]
                                 q_proj) * self.scale
        
        # 注意力权重
        att_weights = F.softmax(att_logits, dim=-1)  # [b, head, v_num, q_num]
        
        # 特征聚合
        attended_v = torch.einsum('bhvq,bhqj->bhvj', att_weights, q_proj)  # [b, head, v_num, h]
        attended_q = torch.einsum('bhvq,bhvi->bhqi', att_weights, v_proj)  # [b, head, q_num, h]
        
        # 对称池化
        pooled_v = attended_v.mean(dim=2)  # [b, head, h]
        pooled_q = attended_q.mean(dim=2)  # [b, head, h]
        
        # 特征融合
        fused = torch.cat([pooled_v, pooled_q], dim=-1)  # [b, head, 2*h]
        fused_flat = fused.reshape(b, -1)  # [b, head*2*h]
        
        return self.output(fused_flat)
    

class CoAttentionLayer(nn.Module):
    def __init__(self, hid_dim: int, n_head: int = 8, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.v_self_attn = nn.MultiheadAttention(hid_dim, n_head, dropout, batch_first=True)
        self.v_dropout = nn.Dropout(dropout)
        self.v_norm = nn.LayerNorm(hid_dim)
        
        self.q_self_attn = nn.MultiheadAttention(hid_dim, n_head, dropout, batch_first=True)
        self.q_dropout = nn.Dropout(dropout)
        self.q_norm = nn.LayerNorm(hid_dim)
        
        self.vq_attn = nn.MultiheadAttention(hid_dim, n_head, dropout, batch_first=True)
        self.vq_dropout = nn.Dropout(dropout)
        self.vq_norm = nn.LayerNorm(hid_dim)
        
        self.qv_attn = nn.MultiheadAttention(hid_dim, n_head, dropout, batch_first=True)
        self.qv_dropout = nn.Dropout(dropout)
        self.qv_norm = nn.LayerNorm(hid_dim)
        
        self.v_ffn = PositionwiseFeedforwardLayer(hid_dim, 4*hid_dim, dropout)
        self.v_ffn_dropout = nn.Dropout(dropout)
        self.v_ffn_norm = nn.LayerNorm(hid_dim)
        
        self.q_ffn = PositionwiseFeedforwardLayer(hid_dim, 4*hid_dim, dropout)
        self.q_ffn_dropout = nn.Dropout(dropout)
        self.q_ffn_norm = nn.LayerNorm(hid_dim)
        
    def forward(self, v: torch.Tensor, v_mask: torch.Tensor,
                q: torch.Tensor, q_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        v: [b, v_num, hid_dim]
        v_mask: [b, v_num]
        q: [b, q_num, hid_dim]
        q_mask: [b, q_num]
        """
        # self-attention
        v_attn = self.v_norm(v+self.v_dropout(self.v_self_attn(v, v, v, v_mask)[0]))
        q_attn = self.q_norm(q+self.q_dropout(self.q_self_attn(q, q, q, q_mask)[0]))
        # cross-attention, will cause nan, don't kown why
        # vq_mask = (v_mask.unsqueeze(-1) | q_mask.unsqueeze(1)).repeat_interleave(self.n_head, dim=0)  # [b*nhead, v_num, q_num]
        # qv_mask = (q_mask.unsqueeze(-1) | v_mask.unsqueeze(1)).repeat_interleave(self.n_head, dim=0)  # [b*nhead, q_num, v_num]
        
        v_attn = self.vq_attn(
            query=v_attn,
            key=q_attn,
            value=q_attn,
            key_padding_mask=q_mask,
            # attn_mask=vq_mask
        )[0]
        
        q_attn = self.qv_attn(
            query=q_attn,
            key=v_attn,
            value=v_attn,
            key_padding_mask=v_mask,
            # attn_mask=qv_mask
        )[0]
        
        v = self.vq_norm(v+self.vq_dropout(v_attn))
        q = self.qv_norm(q+self.qv_dropout(q_attn))
        
        # output
        v = self.v_ffn_norm(v+self.v_ffn_dropout(self.v_ffn(v)))
        q = self.q_ffn_norm(q+self.q_ffn_dropout(self.q_ffn(q)))
        return v, q


class CoAttention(nn.Module):
    """modified from https://github.com/chengeng17/HitScreen/blob/main/coattention.py"""
    def __init__(self, hid_dim: int, n_head: int, n_layer: int, dropout=0.1):
        super().__init__()
        self.backbone = nn.ModuleList([CoAttentionLayer(hid_dim, n_head, dropout=dropout) for _ in range(n_layer)])
        self.att_net = nn.Linear(hid_dim, 1)

    def attention_pooling(self, v, q, att_map):
        # att_map = att_map.squeeze(-1)
        # # einsum('bvk,bvq,bqk->bk', v, att_map, q) 的等价基础张量运算：
        # # 1. 先计算 att_map 与 q 的批量矩阵乘：att_map [b,v,q] × q [b,q,k] → [b,v,k]
        # #    对应 einsum 中的 'bvq,bqk->bvk'
        # tmp = torch.bmm(att_map.float(), q.float())          # [b, v, k]
        # # 2. 再计算 v 与上一步结果的批量矩阵乘：v [b,v,k] 与 tmp [b,v,k] 做逐元素乘后沿 v 维求和
        # #    对应 einsum 中的 'bvk,bvk->bk'（先逐元素乘，再对维度 v 求和）
        # fusion_logits = (v.float() * tmp).sum(dim=1)         # [b, k]
        fusion_logits = torch.einsum('bvk,bvq,bqk->bk', (v.float(), att_map.float(), q.float()))
        # add scale to fusion_logits
        v_num, q_num = v.size(1), q.size(1)
        seq_scale = 1.0 / torch.sqrt(torch.tensor(v_num * q_num, dtype=torch.float32, device=v.device))
        return fusion_logits * seq_scale

    def forward(self, v: torch.Tensor, v_mask: torch.Tensor,
                q: torch.Tensor, q_mask: torch.Tensor, use_pool: bool = True):
        """
        v: [b, v_num, hid_dim]
        v_mask: [b, v_num]
        q: [b, q_num, hid_dim]
        q_mask: [b, q_num]
        """
        v_mask, q_mask = ~v_mask.bool(), ~q_mask.bool()
        # v: [b, v_num, hid_dim], q: [b, q_num, hid_dim]
        for layer in self.backbone:
            v, q = layer(v, v_mask, q, q_mask)
        # [b, v_num, 1, hid_dim] * [b, 1, q_num, hid_dim] -> [b, v_num, q_num, hid_dim]
        att_scores = v.unsqueeze(2) * q.unsqueeze(1)
        att_scores = self.att_net(att_scores).squeeze(-1) # [b, v_num, q_num]
        # [b, v_num, q_num, 1]
        corss_mask = (v_mask.unsqueeze(-1) | q_mask.unsqueeze(1))  # [b, v_num, q_num]
        att_scores = att_scores.masked_fill(corss_mask, -1e9)
        att_maps = torch.softmax(att_scores, dim=-1) # [b, v_num, q_num]
        if use_pool:
            return self.attention_pooling(v, q, att_maps)
        return v, q, att_maps
    

if __name__ == '__main__':
    # test code
    from mbapy.dl_torch.utils import init_model_parameter
    # test add_rope
    x = torch.randn(128, 257, 512)
    x1 = add_rope1(x)
    x2 = add_rope(x)
    print(add_rope(x).shape)
    ## test BANLayer
    ban = BANLayer(384, 1536, 256, num_heads=8)
    v = torch.randn(2, 256, 384)
    q = torch.randn(2, 1, 1536)
    out = ban(v, q)
    print(out.shape)
    # test CoAttention
    coattn = init_model_parameter(CoAttention(256, 8, 2), {})
    v = torch.randn(2, 256, 256)
    v_mask = torch.randint(0, 2, (2, 256), dtype=torch.bool)
    q = torch.randn(2, 16, 256)
    q_mask = torch.randint(0, 2, (2, 16), dtype=torch.bool)
    out = coattn(v, v_mask, q, q_mask)
    print(out.shape)