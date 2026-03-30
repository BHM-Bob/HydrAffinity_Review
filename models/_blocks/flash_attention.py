import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import pad_input, unpad_input
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    print("Warning: flash-attn not available, falling back to standard attention")


class FlashAttentionMHSA(nn.Module):
    """
    使用flash-attn2实现的多头自注意力机制
    保持与MultiHeadSelfAttention相同的接口
    """
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1, gated_sdpa: bool = False):
        super(FlashAttentionMHSA, self).__init__()
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
        
        # gated attention (flash-attn不支持，需要fallback)
        self.gated_sdpa = gated_sdpa
        if gated_sdpa:
            self.gate_proj = nn.Linear(embed_dim, self.num_heads * self.head_dim, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        self._reset_parameters()
        self.attn_weights = None
        
        # 检查flash-attn可用性
        self.use_flash_attn = FLASH_ATTN_AVAILABLE
    
    def _reset_parameters(self):
        # Xavier初始化
        for module in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.)
                
    def extra_repr(self) -> str:
        return f"embed_dim={self.embed_dim}, num_heads={self.num_heads}, dropout={self.dropout.p}, gated_sdpa={self.gated_sdpa}"
    
    def _flash_attention(self, Q, K, V, src_key_padding_mask=None):
        """使用flash-attn2实现注意力"""
        batch_size, seq_len, embed_dim = Q.size()
        kv_len = K.size(1)
        
        # 转换为半精度（flash-attn要求）
        original_dtype = Q.dtype
        if original_dtype == torch.float32:
            Q = Q.half()
            K = K.half()
            V = V.half()
        
        # 重塑为flash-attn需要的格式: (batch_size, seq_len, num_heads, head_dim)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = K.view(batch_size, kv_len, self.num_heads, self.head_dim)
        V = V.view(batch_size, kv_len, self.num_heads, self.head_dim)
        
        # 处理填充掩码
        if src_key_padding_mask is not None:
            # 使用varlen函数处理填充
            Q_unpad, indices_q, cu_seqlens_q, max_seqlen_q, _ = unpad_input(Q, src_key_padding_mask)
            K_unpad, _, cu_seqlens_k, max_seqlen_k, _ = unpad_input(K, src_key_padding_mask)
            V_unpad, _, _, _, _ = unpad_input(V, src_key_padding_mask)
            
            output_unpad = flash_attn_varlen_func(
                Q_unpad, K_unpad, V_unpad,
                cu_seqlens_q=cu_seqlens_q, max_seqlen_q=max_seqlen_q,
                cu_seqlens_k=cu_seqlens_k, max_seqlen_k=max_seqlen_k,
                dropout_p=self.dropout.p if self.training else 0.0,
                softmax_scale=self.scale,
                causal=False
            )
            
            # 重新填充输出
            output = pad_input(output_unpad, indices_q, batch_size, seq_len)
        else:
            # 无填充掩码，使用标准flash-attn
            output = flash_attn_func(
                Q, K, V,
                dropout_p=self.dropout.p if self.training else 0.0,
                softmax_scale=self.scale,
                causal=False
            )
        
        # 重塑回原始格式
        output = output.contiguous().view(batch_size, seq_len, embed_dim)
        
        # 转换回原始精度
        if original_dtype == torch.float32:
            output = output.float()
        
        # flash-attn不返回注意力权重，设置为None
        self.attn_weights = None
        
        return output
    
    def forward(self, x, kv: torch.Tensor = None, src_key_padding_mask: torch.Tensor = None, **kwargs):
        """
        前向传播
        保持与MultiHeadSelfAttention相同的接口
        """
        batch_size, seq_len, embed_dim = x.size()
        
        # 计算Q, K, V
        Q = self.q_proj(x)
        kv = kv if kv is not None else x
        K = self.k_proj(kv)
        V = self.v_proj(kv)
        
        output = self._flash_attention(Q, K, V, src_key_padding_mask)
        
        # Gated Attention (支持与flash-attn结合使用)
        if self.gated_sdpa:
            gate_scores = self.gate_proj(x).sigmoid()
            gate_scores = gate_scores.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            # 需要重新reshape output以应用门控
            output_reshaped = output.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            output_reshaped = output_reshaped * gate_scores
            output = output_reshaped.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        
        # 输出投影
        output = self.out_proj(output)
        
        return output
