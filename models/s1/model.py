from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models._blocks.attn import (AttnBase, BANLayer, CoAttention, CrossAttn,
                                 MiniTransformerEncodeLayer, MLDecoderLite,
                                 PredTokenAttn, add_rope)
from models._blocks.mlp import (FC, BSplineLayer, BSplineLayer2, HighwayBase,
                                HighwayMLP1, HighwayMLP2, Linear, LinearDO,
                                MeanPool, SimpleMLP, SimpleMLPN)
from models._blocks.moe import MoEPredictor, MoEWithSharedExp, TokenMoE


class AdaptiveInputScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        return x * torch.sigmoid(self.scale) + self.bias


class BVIPredictor(nn.Module):
    def __init__(self, hid_dim: int = 256, dropout: float = 0.4):
        super().__init__()
        self.mu_fc = nn.Sequential(nn.Linear(hid_dim, hid_dim),
                                    nn.LeakyReLU(),
                                    nn.Dropout(dropout))
        self.logvar_fc = nn.Sequential(nn.Linear(hid_dim, hid_dim),
                                       nn.LeakyReLU(),
                                       nn.Dropout(dropout))
        self.out_fc = nn.Sequential(nn.Linear(hid_dim, 1),
                                    nn.Dropout(dropout))
        
    def forward(self, x):
        mu = self.mu_fc(x)  # [N, 1, hid_dim]
        logvar = self.logvar_fc(x)  # [N, 1, hid_dim]
        sigma = torch.exp(0.5 * logvar)
        z = torch.randn_like(mu) * sigma + mu  # Reparameterization trick
        return self.out_fc(z)


class BVIPredictorV2(nn.Module):
    def __init__(self, hid_dim: int = 256, dropout: float = 0.4, latent_dim: int = 8):
        super().__init__()
        # 潜在空间维度扩展
        self.mu_fc = nn.Sequential(
            nn.Linear(hid_dim, latent_dim),
            nn.LayerNorm(latent_dim)
        )
        self.logvar_fc = nn.Sequential(
            nn.Linear(hid_dim, latent_dim),
            nn.Tanh()  # 约束logvar范围
            # or lambda x: torch.where(x > 0, torch.sigmoid(x), x)
        )
        # 解码器部分
        self.out_decoder = SimpleMLP(latent_dim, 1, dropout, hid_dim=hid_dim)

    def forward(self, x):
        mu = self.mu_fc(x)  # [N, latent_dim]
        logvar = self.logvar_fc(x) * 0.5  # 缩放logvar
        
        sigma = torch.exp(logvar)
        z = mu + sigma * torch.randn_like(mu)
        return self.out_decoder(z).sum(dim=-1)  # 潜在空间聚合


class MLP_BVIPredictor(BVIPredictor):
    def __init__(self, hid_dim: int = 256, dropout: float = 0.4):
        super().__init__()
        self.mu_fc = SimpleMLP(hid_dim, hid_dim, dropout)
        self.logvar_fc = SimpleMLP(hid_dim, hid_dim, dropout)
        self.out_fc = SimpleMLP(hid_dim, 1, dropout)
        

class MLD_BVIPredictor(BVIPredictor):
    def __init__(self, hid_dim: int = 256, dropout: float = 0.4):
        super().__init__()
        self.mu_fc = SimpleMLP(hid_dim, hid_dim, dropout)
        self.logvar_fc = SimpleMLP(hid_dim, hid_dim, dropout)
        self.out_fc = MLDecoderLite(1, hid_dim, 8, dropout)
        
        
class BiLevelPredictor(nn.Module):
    def __init__(self, hid_dim: int = 256, dropout: float = 0.4):
        super().__init__()
        self.marco_mlp = LinearDO(hid_dim, 24, dropout) # predict interger number: 0~23
        self.microm_mlp = LinearDO(hid_dim+24, 1, dropout) # predict float number: -1~1
        self.marco_probs = None  # 存储marco部分的概率分布
    
    def forward(self, x):
        # x: [N, L, D] or [N, D] -> [N, 24+1]
        marco_pred: torch.Tensor = self.marco_mlp(x)
        microm_pred: torch.Tensor = self.microm_mlp(torch.cat([x, marco_pred], dim=-1)).sigmoid()
        
        # 使用softmax代替argmax，使其可微
        self.marco_probs = marco_pred.detach().softmax(dim=-1)  # 存储marco部分的概率分布
        
        if x.dim() == 3:
            return torch.cat([marco_pred, microm_pred], dim=-1).squeeze()
        return torch.cat([marco_pred.squeeze(-1), microm_pred], dim=-1)


def _get_predictor(name: str, gatter: str, router_noise: bool, hid_dim: int,
                   n_head: int, dropout: float, out_dim: int = 1, router_act: str = 'lambda', moe_n_head: int = 1):
    SUPPORTED_PREDICTORS = {'Linear': Linear, 'LinearDO': LinearDO, 'MLP': SimpleMLP, 'MLPN': SimpleMLPN, 'MLDecoder': MLDecoderLite,
                            'BVI': BVIPredictor, 'BVI_V2': BVIPredictorV2, 'FC': FC, 'MeanPool': MeanPool,
                            'MLP_BVI': MLP_BVIPredictor, 'MLD_BVI': MLD_BVIPredictor,
                            'BSL': BSplineLayer, 'BSL2': BSplineLayer2, 'BiLevel': BiLevelPredictor,
                            'HighwayBase': HighwayBase, 'HighwayMLP1': HighwayMLP1, 'HighwayMLP2': HighwayMLP2}
    if isinstance(name, list):
        # MoE: [pred_type: str, n_exp: str, topk: str], such as ['MLP', '4', '2']
        if len(name) == 3 and name[0] in SUPPORTED_PREDICTORS and name[1].isdigit() and name[2].isdigit():
            n_exp, topk = int(name[1]), int(name[2])
            predictors = nn.ModuleList([_get_predictor(name[0], gatter, router_noise, hid_dim, n_head, dropout, out_dim//moe_n_head, router_act, moe_n_head) for _ in range(n_exp)])
            return MoEPredictor(predictors, topk, moe_n_head, gatter, router_noise, router_act, hid_dim, dropout)
        # MoE: [pred_type1: str, pred_type2: str, ..., topk: str], such as ['MLP', 'BVI', '1']
        elif len(name) >= 3 and all(i in SUPPORTED_PREDICTORS for i in name[:-1]) and name[-1].isdigit():
            predictors = nn.ModuleList([_get_predictor(i, gatter, router_noise, hid_dim, n_head, dropout, out_dim//moe_n_head, router_act, moe_n_head) for i in name[:-1]])
            return MoEPredictor(predictors, int(name[-1]), moe_n_head, gatter, router_noise, router_act, hid_dim, dropout)
        else:
            raise ValueError(f'Unknown predictor name: {name} while it is a list for MoE')
    elif name == 'FC':
        return FC(hid_dim, 200, out_dim, dropout, 2)
    elif name in {'Linear', 'LinearDO', 'MLP', 'MLPN', 'MeanPool', 'BSL', 'BSL2', 'HighwayBase', 'HighwayMLP1', 'HighwayMLP2'}:
        return SUPPORTED_PREDICTORS[name](hid_dim, out_dim, dropout)
    elif name == 'MLDecoder':
        return MLDecoderLite(out_dim, hid_dim, n_head, dropout)
    elif name in {'BVI', 'BVI_V2', 'MLP_BVI', 'MLD_BVI', 'BiLevel'}:
        return SUPPORTED_PREDICTORS[name](hid_dim, dropout)
    else:
        raise ValueError(f'Unknown predictor name: {name}')
    
    
class BasedArch(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def calcu_moe_loss(self, *args, **kwargs):
        self.MoE_balance_loss = 0
        for module in self.MoE_modules:
            balance_loss = getattr(self, module).balance_loss
            if isinstance(balance_loss, torch.Tensor):
                self.MoE_balance_loss += balance_loss
        if isinstance(self.MoE_balance_loss, int):
            self.MoE_balance_loss = torch.zeros(1)
        return self.MoE_balance_loss


class Arch1(BasedArch):
    def __init__(self, lig_dim: int = 384, prot_dim: int = 1536, hid_dim: int = 256, RMSNorm: bool = False, use_rope: bool = False,
                 lig_pred: str = 'LinearDO', prot_pred: str = 'LinearDO', lig_moe_n_head: int = 1, prot_moe_n_head: int = 1,
                 pred_n: int = 1, n_layer: int = 2, modal_token: bool = False, softmax_partition: bool = False,
                 pred_method: str|list[str, int] = 'MLP', shared_exp: str|list[str, int] = None,
                 gatter: str = 'sum', router_noise: bool = False, router_act: str = 'lambda', prot_scale: bool = False,
                 n_head: int = 8, dropout: float = 0.4, lig_emb_dim: int = 384, use_method_id: str = None,
                 gated_sdpa: bool = False, token_moe_mask: bool = False, **kwargs):
        super().__init__()
        self.lig_dim = lig_dim
        self.prot_dim = prot_dim
        self.hid_dim = hid_dim
        self.pred_n = pred_n
        self.softmax_partition = softmax_partition
        self.hydraformer = 3 if kwargs.get('hydraformer', False) else 0 # include pred token, there are 3 modals in total
        self.use_method_id = use_method_id
        if use_method_id and 'separate_hydraformer' in use_method_id:
            self.hydraformer += 1 # add method token as separate in hydraformer
        self.modal_token = modal_token        
        self.pred_method = pred_method
        self.shared_exp = shared_exp
        self.prot_scale = prot_scale
        self.dropout = dropout
        self.token_moe_mask = token_moe_mask
        self.low_mem_transformer = kwargs.get('low_mem_transformer', False)
        
        # lig type is token, use embedding layer
        if lig_dim == 1:
            lig_dim = self.lig_dim = lig_emb_dim
            self.lig_emb = nn.Embedding(512, lig_dim, padding_idx=0)
        # lig feature is from pretrained model
        else:
            self.lig_emb = None
        
        self.lig_fc = _get_predictor(lig_pred, gatter, True, lig_dim, n_head, dropout, hid_dim, router_act, lig_moe_n_head)
        if kwargs.get('lig_token_moe', False) and isinstance(self.lig_fc, MoEPredictor):
            self.lig_fc = TokenMoE(self.lig_fc)
        # if isinstance(self.lig_fc, (MoEPredictor, TokenMoE)):
        #     self.lig_fc = MoEWithSharedExp(self.lig_fc, Linear(lig_dim, hid_dim, dropout))
        self.prot_fc = _get_predictor(prot_pred, gatter, True, prot_dim, n_head, dropout, hid_dim, router_act, prot_moe_n_head)
        # if isinstance(self.prot_fc, MoEPredictor):
        #     self.prot_fc = MoEWithSharedExp(self.prot_fc, Linear(prot_dim, hid_dim, dropout))
        if self.prot_scale:
            self.prot_scaler = AdaptiveInputScaler()
        
        self.attn = PredTokenAttn(pred_n, hid_dim, n_layer=n_layer, n_head=n_head, dropout=dropout,
                                  RMSNorm=RMSNorm, use_rope=use_rope, softmax_partition=softmax_partition,
                                  moe_ffn=kwargs.get('moe_ffn', None), norm_first=kwargs.get('norm_first', False),
                                  hydraformer=self.hydraformer, use_method_id=use_method_id, gated_sdpa=gated_sdpa,
                                  low_mem_transformer=self.low_mem_transformer)
        if modal_token:
            self.modal_embedding = nn.Embedding(2, hid_dim)
        if pred_n > 1:
            self.gate = SimpleMLP(hid_dim+1, 1, dropout)

        self.predictor = _get_predictor(pred_method, gatter, router_noise, hid_dim, n_head, dropout, 1, router_act)
        
        # MoE settings
        ## record MoE
        self.MoE_modules = list(filter(lambda x: hasattr(getattr(self, x), 'balance_loss'),
                                       ['lig_fc', 'prot_fc', 'attn', 'predictor']))
        self.is_MoE = bool(self.MoE_modules)
        self.pred_is_MoE = isinstance(self.predictor, MoEPredictor)
        self.MoE_balance_loss = 0
        if isinstance(self.predictor, MoEPredictor):
            self.predictor.reset_expert_activation()
            if self.shared_exp:
                self.shared_MoE = _get_predictor(shared_exp, gatter, router_noise, hid_dim, n_head, dropout, 1, router_act)
        else:
            assert not router_noise, 'router noise only enable when model is MoE'
            
    def extra_repr(self) -> str:
        return f'MoE={self.MoE_modules}, prot_scale={self.prot_scale}'
            
    def feat_encode(self, smile_feat: torch.Tensor, mask: torch.Tensor, prot_feat: torch.Tensor,
                    noise_rate: float = 0.0, **kwargs):
        """encode original input feature into hidden dim"""
        # smile_feat: [N, L, 384], mask: [N, L], prot_feat: [N, 1536]
        if self.lig_emb is not None:
            smile_feat = self.lig_emb(smile_feat.long())  # [N, L, 384]
            smile_feat += torch.randn_like(smile_feat) * noise_rate
        if self.token_moe_mask:
            lig_feat = self.lig_fc(smile_feat, mask)
        else:
            lig_feat = self.lig_fc(smile_feat)  # [N, L, 256]
        prot_feat = self.prot_fc(prot_feat)  # [N, 1, 256]
        if self.prot_scale:
            prot_feat = self.prot_scaler(prot_feat)
        if kwargs.get('attn_each_feat', False):
            lig_feat = self.attn.transformer.layers[0](lig_feat.transpose(0, 1), src_key_padding_mask=~mask.bool()).transpose(0, 1)
            # since Arch1 prot feat has only length of 1, skip attn for prot_feat
        return lig_feat, prot_feat.unsqueeze(1) if len(prot_feat.shape) == 2 else prot_feat
        
    def feat_forward(self, mid: torch.Tensor, lig_feat: torch.Tensor, mask: torch.Tensor, prot_feat: torch.Tensor,
                     cat_mask: torch.Tensor = None):
        """merge ligand hidden state and protein hidden state into model hidden state"""
        # mid: [N], lig_feat: [N, L, 256], mask: [N, L], prot_feat: [N, 1, 256]
        if self.modal_token:
            prot_feat += self.modal_embedding(torch.zeros(1, device=lig_feat.device).long())
            lig_feat[mask.bool()] += self.modal_embedding(torch.ones(1, device=lig_feat.device).long())
        x = torch.cat([prot_feat, lig_feat], dim=1)  # [N, L+1, 256]
        mask = cat_mask if cat_mask is not None else torch.cat([torch.ones_like(mask[:, :prot_feat.size(1)]), mask], dim=1)  # [N, 1+L]
        if self.softmax_partition:
            partition = [(1, 1), (prot_feat.size(1), 1), (lig_feat.size(1), 1)]
            x = self.attn(x, mask, softmax_partition=partition, only_return_first_token=self.pred_n)  # [N, 1, 256]
        elif self.hydraformer:
            if not self.use_method_id or 'replace' in self.use_method_id: # no method token OR replace mid, 3 types token in total
                modal_idx = torch.zeros(x.size(1)+self.pred_n, device=x.device).long()
                modal_idx[1], modal_idx[2:] = 1, 2
            elif 'shared_hydraformer' in self.use_method_id: # shared hydraformer, 3 types token in total
                modal_idx = torch.zeros(x.size(1)+1+self.pred_n, device=x.device).long()
                modal_idx[2], modal_idx[3:] = 1, 2
            else: # separate hydraformer, 4 types token in total
                modal_idx = torch.zeros(x.size(1)+1+self.pred_n, device=x.device).long()
                modal_idx[1], modal_idx[2], modal_idx[3:] = 1, 2, 3
            x = self.attn(x, mask, mid=mid, modal_idx=modal_idx, only_return_first_token=self.pred_n)  # [N, 1, 256]
        else:
            x = self.attn(x, mask, mid=mid, only_return_first_token=self.pred_n)  # [N, 1, 256]
        return x
    
    def predict_from_feat(self, feat: torch.Tensor):
        """predict from model hidden state"""
        # feat: [N, 1, D]
        out = self.predictor(feat).reshape(-1, 1)  # [N, 1]
        if self.shared_exp is not None:
            out += self.shared_MoE(feat).view(-1, 1)  # [N, 1]
        return out.reshape(-1, 1)
        
    def forward(self, mid: torch.Tensor, smile_feat: torch.Tensor, mask: torch.Tensor, prot_feat: torch.Tensor, **kwargs):
        """forward entire model"""
        # smile_feat: [N, L, 384], mask: [N, L], prot_feat: [N, 1536]
        lig_feat, prot_feat = self.feat_encode(smile_feat, mask, prot_feat, **kwargs)
        x = self.feat_forward(mid, lig_feat, mask, prot_feat)  # [N, pred_n, D]
        if self.pred_n > 1:
            batch_size = x.size(0)
            logits = self.predict_from_feat(x.reshape(batch_size*self.pred_n, self.hid_dim)).reshape(batch_size, self.pred_n, 1)  # [N, pred_n, 1]
            gate = self.gate(torch.cat([x, logits], dim=-1)).squeeze(-1).softmax(dim=-1)  # [N, pred_n, D+1] -> [N, pred_n, 1] -> [N, pred_n]
            return (logits.reshape(batch_size, self.pred_n) * gate).sum(dim=-1, keepdim=True)  # [N, 1]
        else:
            return self.predict_from_feat(x)
    
    
class Arch11(Arch1):
    """Only works when both L-dim of ligand and protein are 1"""
    def __init__(self, lig_dim: int = 384, prot_dim: int = 1536, hid_dim: int = 256, RMSNorm: bool = False, use_rope: bool = False,
                 lig_pred: str = 'LinearDO', prot_pred: str = 'LinearDO', n_layer: int = 2,
                 pred_method: str | list[str, int] = 'MLP', shared_exp: str | list[str, int] = None,
                 gatter: str = 'sum', router_noise: bool = False, router_act: str = 'lambda',
                 prot_scale: bool = False, n_head: int = 8, dropout: float = 0.4, lig_emb_dim: int = 384):
        super().__init__(lig_dim, prot_dim, hid_dim, RMSNorm, use_rope, lig_pred, prot_pred, n_layer, pred_method,
                         shared_exp, gatter, router_noise, router_act, prot_scale, n_head, dropout,
                         lig_emb_dim)
        self.attn = None
        self.lig_gate_fc = nn.Sequential(
            nn.Linear(hid_dim * 2, hid_dim),
            nn.Sigmoid()
        )
        self.prot_gate_fc = nn.Sequential(
            nn.Linear(hid_dim * 2, hid_dim),
            nn.Sigmoid()
        )
        self.fusion_fc = SimpleMLP(hid_dim * 2, hid_dim, dropout)
            
    def feat_encode(self, smile_feat: torch.Tensor, mask: torch.Tensor, prot_feat: torch.Tensor,
                    noise_rate: float = 0.0, **kwargs):
        """encode original input feature into hidden dim"""
        # smile_feat: [N, L, 384], mask: [N, L], prot_feat: [N, 1536]
        if self.lig_emb is not None:
            smile_feat = self.lig_emb(smile_feat.long())  # [N, L, 384]
            smile_feat += torch.randn_like(smile_feat) * noise_rate
        if len(smile_feat.shape) == 3:
            smile_feat = smile_feat.mean(dim=1)  # [N, 384]
        lig_feat = self.lig_fc(smile_feat)  # [N, L, 256]
        prot_feat = self.prot_fc(prot_feat)  # [N, 1, 256]
        if self.prot_scale:
            prot_feat = self.prot_scaler(prot_feat)
        return lig_feat, prot_feat.unsqueeze(1) if len(prot_feat.shape) == 2 else prot_feat
        
    def feat_forward(self, mid: torch.Tensor, lig_feat: torch.Tensor, mask: torch.Tensor, prot_feat: torch.Tensor,
                     cat_mask: torch.Tensor = None, return_gate: bool = False):
        # lig_feat: [N, 1, D], prot_feat: [N, 1, D]
        cat_feat = torch.cat([lig_feat, prot_feat], dim=-1)  # [N, 1, 2*D]
        lig_gate = self.lig_gate_fc(cat_feat)  # [N, 1, D]
        prot_gate = self.prot_gate_fc(cat_feat)  # [N, 1, D]
        fused_feat = self.fusion_fc(cat_feat)  # [N, 1, D]
        x = lig_feat * lig_gate + prot_feat * prot_gate + (2 - lig_gate - prot_gate) * fused_feat
        if return_gate:
            return x / 2, lig_gate, prot_gate
        return x / 2

class Arch2(BasedArch):
    def __init__(self, lig_dim: int = 384, prot_dim: int = 1536, prot_n: int = 8, hid_dim: int = 256,
                 RMSNorm: bool = False, use_rope: bool = False, feat_mini_mhsa: bool = False, norm_first: bool = False,
                 lig_pred: str = 'LinearDO', prot_pred: str = 'LinearDO', lig_moe_n_head: int = 1, prot_moe_n_head: int = 1,
                 n_layer: int = 2, pred_method: str|list[str, int] = 'MLP', shared_exp: str|list[str, int] = None,
                 gatter: str = 'sum', router_noise: bool = False, router_act: str = 'lambda', prot_scale: bool = False,
                 n_head: int = 8, dropout: float = 0.4, use_method_id: str = None, **kwargs):
        super().__init__(hid_dim, dropout)
        self.lig_dim = lig_dim
        self.prot_dim = prot_dim
        self.prot_n = prot_n
        self.pred_n = 1
        self.hid_dim = hid_dim
        self.pred_method = pred_method
        self.shared_exp = shared_exp
        self.prot_scale = prot_scale
        self.dropout = dropout
        self.hydraformer = 3 if kwargs.get('hydraformer', False) else 0 # include pred token, there are 3 modals in total
        self.use_method_id = use_method_id
        if use_method_id and 'separate_hydraformer' in use_method_id:
            self.hydraformer += 1 # add method token as separate in hydraformer
        self.low_mem_transformer = kwargs.get('low_mem_transformer', False)
        
        self.lig_fc = _get_predictor(lig_pred, gatter, True, lig_dim, n_head, dropout, hid_dim, router_act, lig_moe_n_head)
        if kwargs.get('lig_token_moe', False) and isinstance(self.lig_fc, MoEPredictor):
            self.lig_fc = TokenMoE(self.lig_fc)
        self.prot_multi_moe = False
        if prot_pred == 'LinearDO':
            self.prot_fc = nn.ModuleList([nn.Sequential(nn.Linear(prot_dim, hid_dim),
                                        nn.LeakyReLU(),
                                        nn.Dropout(dropout)) for _ in range(prot_n)])
        elif isinstance(prot_pred, list) and prot_pred[0] == 'multi_moe':
            self.prot_multi_moe = True
            self.prot_fc = nn.ModuleList([_get_predictor(prot_pred[1:], gatter, True, prot_dim, n_head, dropout, hid_dim, router_act, prot_moe_n_head) for _ in range(prot_n)])
        else:
            self.prot_fc = _get_predictor(prot_pred, 'cat' if prot_n > 1 else gatter, True, prot_dim, n_head, dropout, hid_dim, router_act, prot_moe_n_head)
        if self.prot_scale:
            self.prot_scaler = AdaptiveInputScaler()
        
        # feat MiniTransformer
        self.feat_mini_mhsa = feat_mini_mhsa
        if self.feat_mini_mhsa:
            self.feat_transformer = MiniTransformerEncodeLayer(hid_dim, n_head, dropout)
        
        self.attn = PredTokenAttn(1, hid_dim, n_layer=n_layer, n_head=n_head, dropout=dropout,
                                  RMSNorm=RMSNorm, use_rope=use_rope, norm_first=norm_first, use_method_id=use_method_id,
                                  low_mem_transformer=self.low_mem_transformer, hydraformer=self.hydraformer)

        self.predictor = _get_predictor(pred_method, gatter, router_noise, hid_dim, n_head, dropout, 1, router_act)
        self.MoE_modules = list(filter(lambda x: hasattr(getattr(self, x), 'balance_loss'),
                                       ['lig_fc', 'prot_fc', 'attn', 'predictor']))
        self.is_MoE = bool(self.MoE_modules)
        self.pred_is_MoE = isinstance(self.predictor, MoEPredictor)
        if isinstance(self.predictor, MoEPredictor):
            self.predictor.reset_expert_activation()
            if self.shared_exp:
                self.shared_MoE = _get_predictor(shared_exp, gatter, router_noise, hid_dim, n_head, dropout, 1, router_act)
            
    def feat_encode(self, smile_feat: torch.Tensor, mask: torch.Tensor, prot_feat: torch.Tensor, **kwargs):
        # smile_feat: [N, L, 384], mask: [N, L], prot_feat: [N, 1536]
        lig_feat = self.lig_fc(smile_feat)  # [N, L, 256]
        if isinstance(self.prot_fc, nn.ModuleList) and not self.prot_multi_moe:
            prot_feat = torch.cat([fc(prot_feat).unsqueeze(1) for fc in self.prot_fc], dim=1)  # [N, prot_n, 256]
        elif self.prot_multi_moe:
            prot_feat = torch.cat([fc(prot_feat) for fc in self.prot_fc], dim=1)  # [N, prot_n, 256]
        else: # MoEPredictor
            prot_feat = self.prot_fc(prot_feat)  # [N, prot_n, 256]
        if self.prot_scale:
            prot_feat = self.prot_scaler(prot_feat)
        if self.feat_mini_mhsa:
            lig_feat = self.feat_transformer(lig_feat, mask=~mask.bool())
            tmp_prot_mask = torch.ones_like(mask[:, :self.prot_n], dtype=torch.bool)
            prot_feat = self.feat_transformer(prot_feat, mask=~tmp_prot_mask)
        return lig_feat, prot_feat
        
    def feat_forward(self, mid: torch.Tensor, lig_feat: torch.Tensor, mask: torch.Tensor, prot_feat: torch.Tensor,
                     cat_mask: torch.Tensor = None):
        # smile_feat: [N, L, 384], mask: [N, L], prot_feat: [N, 1536]
        x = torch.cat([prot_feat, lig_feat], dim=1)  # [N, L+1, 256]
        mask = cat_mask if cat_mask is not None else torch.cat([torch.ones(mask.shape[0], self.prot_n, device=mask.device), mask], dim=1)  # [N, 1+L]
        if self.hydraformer:
            if not self.use_method_id: # no method token, 3 types token in total
                modal_idx = torch.zeros(x.size(1)+self.pred_n, device=x.device).long()
                modal_idx[1:prot_feat.size(1)+1], modal_idx[prot_feat.size(1)+1:] = 1, 2
            x = self.attn(x, mask, mid=mid, modal_idx=modal_idx, only_return_first_token=self.pred_n)  # [N, 1, 256]
        else:
            x = self.attn(x, mask, mid=mid)  # [N, 1, 256]
        return x
    
    def predict_from_feat(self, feat: torch.Tensor):
        # feat: [N, 1, D]
        out = self.predictor(feat)  # [N, 1]
        if self.shared_exp is not None:
            out += self.shared_MoE(feat).view(-1, 1)  # [N, 1]
        return out.reshape(-1, 1)        
    
    def forward(self, mid: torch.Tensor, smile_feat: torch.Tensor, mask: torch.Tensor, prot_feat: torch.Tensor, **kwargs):
        # smile_feat: [N, L, 384], mask: [N, L], prot_feat: [N, 1536]
        lig_feat, prot_feat = self.feat_encode(smile_feat, mask, prot_feat, **kwargs)
        x = self.feat_forward(mid, lig_feat, mask, prot_feat)  # [N, 1, 256]
        return self.predict_from_feat(x)
    
    
class Arch21(Arch2):
    """extend prot_n to lig_n, so this arch has both lig_n and prot_n"""
    def __init__(self, lig_dim: int = 384, prot_dim: int = 1536, lig_n: int = 4, prot_n: int = 8, hid_dim: int = 256,
                 RMSNorm: bool = False, use_rope: bool = False,
                 lig_pred: str = 'LinearDO', prot_pred: str = 'LinearDO',
                 n_layer: int = 2, pred_method: str|list[str, int] = 'MLP', shared_exp: str|list[str, int] = None,
                 gatter: str = 'sum', router_noise: bool = False, router_act: str = 'lambda', prot_scale: bool = False,
                 n_head: int = 8, dropout: float = 0.4):
        super().__init__(lig_dim, prot_dim, prot_n, hid_dim, RMSNorm, use_rope, False, False, lig_pred, prot_pred, 1, 1,
                         n_layer, pred_method, shared_exp, gatter, router_noise, router_act, prot_scale,
                         n_head, dropout)
        self.lig_n = lig_n
        self.lig_fc = nn.ModuleList([nn.Sequential(nn.Linear(lig_dim, hid_dim),
                                    nn.LeakyReLU(),
                                    nn.Dropout(dropout)) for _ in range(lig_n)])
            
    def feat_encode(self, smile_feat: torch.Tensor, mask: torch.Tensor, prot_feat: torch.Tensor, **kwargs):
        # smile_feat: [N, L, 384], mask: [N, L], prot_feat: [N, 1536]
        lig_feat = torch.cat([fc(smile_feat) for fc in self.lig_fc], dim=1)  # [N, L*lig_n, 256]
        prot_feat = torch.cat([fc(prot_feat).unsqueeze(1) for fc in self.prot_fc], dim=1)  # [N, prot_n, 256]
        if self.prot_scale:
            prot_feat = self.prot_scaler(prot_feat)
        if kwargs.get('attn_each_feat', False):
            lig_feat = self.attn.transformer.layers[0](lig_feat.transpose(0, 1), src_key_padding_mask=~mask.bool()).transpose(0, 1)
            tmp_prot_mask = torch.ones_like(mask[:, 0], dtype=torch.bool)
            prot_feat = self.attn.transformer.layers[0](prot_feat.transpose(0, 1), src_key_padding_mask=~tmp_prot_mask).transpose(0, 1)
        return lig_feat, prot_feat
        
    def feat_forward(self, mid: torch.Tensor, lig_feat: torch.Tensor, mask: torch.Tensor, prot_feat: torch.Tensor):
        # smile_feat: [N, L, 384], mask: [N, L], prot_feat: [N, 1536]
        x = torch.cat([prot_feat, lig_feat], dim=1)  # [N, L+1, 256]
        mask = torch.cat([torch.ones(mask.shape[0], self.prot_n, device=mask.device), mask.repeat(1, self.lig_n)], dim=1)  # [N, prot_n+L]
        x = self.attn(x, mask)  # [N, 1, 256]
        return x
    

class Arch22(Arch21):
    """extend a n_pred_token, as the param to PredTokenAttn"""
    def __init__(self, lig_dim: int = 384, prot_dim: int = 1536, lig_n: int = 4, prot_n: int = 8, pred_n: int = 8,
                 hid_dim: int = 256, RMSNorm: bool = False, use_rope: bool = False,
                 lig_pred: str = 'LinearDO', prot_pred: str = 'LinearDO',
                 n_layer: int = 2, pred_method: str|list[str, int] = 'MLP', shared_exp: str|list[str, int] = None,
                 gatter: str = 'sum', router_noise: bool = False, router_act: str = 'lambda', prot_scale: bool = False,
                 n_head: int = 8, dropout: float = 0.4):
        super().__init__(lig_dim, prot_dim, lig_n, prot_n, hid_dim, RMSNorm, use_rope, lig_pred, prot_pred,
                         n_layer, pred_method, shared_exp, gatter, router_noise, router_act, prot_scale,
                         n_head, dropout)
        self.pred_n = pred_n
        self.attn = PredTokenAttn(pred_n, hid_dim, n_layer=n_layer, n_head=n_head, dropout=dropout, RMSNorm=RMSNorm, use_rope=use_rope)
        self.gate = SimpleMLP(hid_dim+1, 1, dropout)
    
    def forward(self, mid: torch.Tensor, smile_feat: torch.Tensor, mask: torch.Tensor, prot_feat: torch.Tensor, **kwargs):
        batch_size = smile_feat.shape[0]
        # smile_feat: [N, L, 384], mask: [N, L], prot_feat: [N, 1536]
        lig_feat, prot_feat = self.feat_encode(smile_feat, mask, prot_feat, **kwargs)
        x = self.feat_forward(mid, lig_feat, mask, prot_feat)  # [N, pred_n, D]
        logits = self.predict_from_feat(x.reshape(batch_size*self.pred_n, self.hid_dim)).reshape(batch_size, self.pred_n, 1)  # [N, pred_n, 1]
        gate = self.gate(torch.cat([x, logits], dim=-1)).squeeze(-1).softmax(dim=-1)  # [N, pred_n, D+1] -> [N, pred_n, 1] -> [N, pred_n]
        return (logits.reshape(batch_size, self.pred_n) * gate).sum(dim=-1, keepdim=True)  # [N, 1]


class Arch23(Arch21):
    """extend a n_pred_token, as the param to PredTokenAttn,
    apply average opts to pred tokens before prediction"""
    def __init__(self, lig_dim: int = 384, prot_dim: int = 1536, lig_n: int = 4, prot_n: int = 8, pred_n: int = 8,
                 hid_dim: int = 256, RMSNorm: bool = False, use_rope: bool = False,
                 lig_pred: str = 'LinearDO', prot_pred: str = 'LinearDO',
                 n_layer: int = 2, pred_method: str|list[str, int] = 'MLP', shared_exp: str|list[str, int] = None,
                 gatter: str = 'sum', router_noise: bool = False, router_act: str = 'lambda', prot_scale: bool = False,
                 n_head: int = 8, dropout: float = 0.4):
        super().__init__(lig_dim, prot_dim, lig_n, prot_n, hid_dim, RMSNorm, use_rope, lig_pred, prot_pred,
                         n_layer, pred_method, shared_exp, gatter, router_noise, router_act, prot_scale,
                         n_head, dropout)
        self.pred_n = pred_n
        self.attn = PredTokenAttn(pred_n, hid_dim, n_layer=n_layer, n_head=n_head, dropout=dropout, RMSNorm=RMSNorm, use_rope=use_rope)
    
    def forward(self, mid: torch.Tensor, smile_feat: torch.Tensor, mask: torch.Tensor, prot_feat: torch.Tensor, **kwargs):
        # smile_feat: [N, L, 384], mask: [N, L], prot_feat: [N, 1536]
        lig_feat, prot_feat = self.feat_encode(smile_feat, mask, prot_feat, **kwargs)
        x = self.feat_forward(mid, lig_feat, mask, prot_feat)  # [N, pred_n, D]
        return self.predict_from_feat(x.mean(dim=1))  # [N, 1]


class Arch24(Arch21):
    """extend a n_pred_token, as the param to PredTokenAttn,
    apply average opts to pred logits after prediction"""
    def __init__(self, lig_dim: int = 384, prot_dim: int = 1536, lig_n: int = 4, prot_n: int = 8, pred_n: int = 8,
                 hid_dim: int = 256, RMSNorm: bool = False, use_rope: bool = False,
                 lig_pred: str = 'LinearDO', prot_pred: str = 'LinearDO',
                 n_layer: int = 2, pred_method: str|list[str, int] = 'MLP', shared_exp: str|list[str, int] = None,
                 gatter: str = 'sum', router_noise: bool = False, router_act: str = 'lambda', prot_scale: bool = False,
                 n_head: int = 8, dropout: float = 0.4):
        super().__init__(lig_dim, prot_dim, lig_n, prot_n, hid_dim, RMSNorm, use_rope, lig_pred, prot_pred,
                         n_layer, pred_method, shared_exp, gatter, router_noise, router_act, prot_scale,
                         n_head, dropout)
        self.pred_n = pred_n
        self.attn = PredTokenAttn(pred_n, hid_dim, n_layer=n_layer, n_head=n_head, dropout=dropout, RMSNorm=RMSNorm, use_rope=use_rope)
    
    def forward(self, mid: torch.Tensor, smile_feat: torch.Tensor, mask: torch.Tensor, prot_feat: torch.Tensor, **kwargs):
        batch_size = smile_feat.shape[0]
        # smile_feat: [N, L, 384], mask: [N, L], prot_feat: [N, 1536]
        lig_feat, prot_feat = self.feat_encode(smile_feat, mask, prot_feat, **kwargs)
        x = self.feat_forward(mid, lig_feat, mask, prot_feat)  # [N, pred_n, D]
        logits = self.predict_from_feat(x.reshape(batch_size*self.pred_n, self.hid_dim))  # [N*pred_n, 1]
        return logits.reshape(batch_size, self.pred_n).mean(dim=-1, keepdim=True)  # [N, 1]


class Arch3(BasedArch):
    """HierarchicalCrossAttn层次化交叉注意力网络"""
    def __init__(self, lig_dim: int = 384, prot_dim: int = 1536, hid_dim: int = 256, RMSNorm: bool = False, use_rope: bool = False,
                 lig_pred: str = 'LinearDO', prot_pred: str = 'LinearDO',
                 n_layer: int = 2, pred_method: str|list[str, int] = 'MLP', shared_exp: str|list[str, int] = None,
                 gatter: str = 'sum', router_noise: bool = False, router_act: str = 'lambda', prot_scale: bool = False,
                 n_head: int = 8, dropout: float = 0.4, lig_emb_dim: int = 384):
        super().__init__()
        self.lig_dim = lig_dim
        self.prot_dim = prot_dim
        self.hid_dim = hid_dim
        self.pred_method = pred_method
        self.shared_exp = shared_exp
        self.dropout = dropout
        # 特征投影
        self.lig_proj = nn.Linear(lig_dim, hid_dim)
        self.prot_proj = nn.Linear(prot_dim, hid_dim)
        
        # 层次化注意力层
        self.attn_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(hid_dim, n_head, 4*hid_dim, dropout,
                                      activation='gelu', batch_first=True)
            for _ in range(n_layer)
        ])
        self.lig_norms = nn.ModuleList([nn.LayerNorm(hid_dim) for _ in range(n_layer-1)])
        self.prot_norms = nn.ModuleList([nn.LayerNorm(hid_dim) for _ in range(n_layer-1)])
        
        self.register_buffer('scale', 1 / torch.sqrt(torch.tensor(hid_dim)))
        
        self.predictor = _get_predictor(pred_method, gatter, router_noise, hid_dim, n_head, dropout, 1, router_act)
        self.MoE_modules = list(filter(lambda x: hasattr(getattr(self, x), 'balance_loss'),
                                       ['lig_proj', 'prot_proj', 'predictor']))
        self.is_MoE = bool(self.MoE_modules)
        self.pred_is_MoE = isinstance(self.predictor, MoEPredictor)
        if isinstance(self.predictor, MoEPredictor):
            self.predictor.reset_expert_activation()
            if self.shared_exp:
                self.shared_MoE = _get_predictor(shared_exp, gatter, router_noise, hid_dim, n_head, dropout, 1, router_act)

    def _cross_attn(self, q_feat, padding_mask, k_feat):
        # [N, q_L, D] @ [N, D, k_L] -> [N, q_L, k_L]
        attn_mask = torch.matmul(q_feat, k_feat.transpose(1,2)) * self.scale
        attn_scores = attn_mask.masked_fill(~padding_mask, -1e9)
        
        attn_weights = F.softmax(attn_scores, dim=-1)  # [N, q_L, k_L]
        return torch.matmul(attn_weights, k_feat) # [N, q_L, D]
        # return torch.einsum('nld,nl->nld',
        #                 F.softmax(attn_scores, dim=-1) @ k_feat,
        #                 torch.sigmoid(attn_scores.mean(-1)))
        
    def feat_encode(self, lig_feat: torch.Tensor, mask: torch.Tensor, prot_feat: torch.Tensor):
        # lig_feat: [N, L, 384], mask: [N, L], prot_feat: [N, 1536]
        lig_feat = self.lig_proj(lig_feat)  # [N, L, 256]
        prot_feat = self.prot_proj(prot_feat).unsqueeze(1)  # [N, 1, 256]
        return lig_feat, prot_feat

    def feat_forward(self, lig_feat: torch.Tensor, mask: torch.Tensor, prot_feat: torch.Tensor):
        # 特征投影
        lig_feat = add_rope(lig_feat)
        prot_feat = add_rope(prot_feat)
        # padding mask [N, prot_L(1), lig_L]
        padding_mask = mask.unsqueeze(1).bool()

        # 层次化处理
        for i, attn_layer in enumerate(self.attn_layers):
            # 交叉注意力
            ctx_feat = self._cross_attn(prot_feat, padding_mask, lig_feat)
            # 残差连接+层归一化
            prot_feat = attn_layer(prot_feat + ctx_feat)
            # 特征精炼
            if i < len(self.attn_layers)-1:
                ctx_lig = self._cross_attn(lig_feat, padding_mask.transpose(1,2), prot_feat)
                # 残差连接+层归一化
                lig_feat = lig_feat + ctx_lig
                lig_feat = self.lig_norms[i](lig_feat)
                prot_feat = self.prot_norms[i](prot_feat)
                
        return prot_feat
    
    def predict_from_feat(self, feat: torch.Tensor):
        # feat: [N, 1, D]
        out = self.predictor(feat)  # [N, 1]
        if self.shared_exp is not None:
            out += self.shared_MoE(feat).view(-1, 1)  # [N, 1]
        return out.reshape(-1, 1)
    
    def forward(self, mid: torch.Tensor, lig_feat: torch.Tensor, mask: torch.Tensor, prot_feat: torch.Tensor, **kwargs):
        # smile_feat: [N, L, 384], mask: [N, L], prot_feat: [N, 1536]
        lig_feat, prot_feat = self.feat_encode(lig_feat, mask, prot_feat)
        x = self.feat_forward(lig_feat, mask, prot_feat)  # [N, 1, 256]
        return self.predict_from_feat(x)


class Arch31(Arch3):
    """优化后的层次化交叉注意力网络（蛋白质作为全局上下文）"""
    def __init__(self, lig_dim: int = 384, prot_dim: int = 1536, hid_dim: int = 256, RMSNorm: bool = False, use_rope: bool = False,
                 lig_pred: str = 'LinearDO', prot_pred: str = 'LinearDO',
                 n_layer: int = 2, pred_method: str|list[str, int] = 'MLP', shared_exp: str|list[str, int] = None,
                 gatter: str = 'sum', router_noise: bool = False, router_act: str = 'lambda', prot_scale: bool = False,
                 n_head: int = 8, dropout: float = 0.4, lig_emb_dim: int = 384):
        super().__init__()
        self.n_layer = n_layer
        self.lig_dim = lig_dim
        self.prot_dim = prot_dim
        self.hid_dim = hid_dim
        self.pred_method = pred_method
        self.shared_exp = shared_exp
        self.dropout = dropout
        
        # 特征投影
        self.lig_proj = nn.Linear(lig_dim, hid_dim)
        self.prot_proj = nn.Linear(prot_dim, hid_dim)
        
        # 蛋白质作为全局上下文的注意力机制
        self.global_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(hid_dim, n_head, dropout=dropout, batch_first=True)
            for _ in range(n_layer)
        ])
        
        # 配体自注意力层
        self.lig_self_attn_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(hid_dim, n_head, 4*hid_dim, dropout,
                                      activation='gelu', batch_first=True)
            for _ in range(n_layer)
        ])
        
        # 层归一化
        self.lig_norms = nn.ModuleList([nn.LayerNorm(hid_dim) for _ in range(n_layer)])
        self.prot_norm = nn.LayerNorm(hid_dim)
        
        # predictor
        self.predictor = _get_predictor(pred_method, gatter, router_noise, hid_dim, n_head, dropout)
        self.MoE_modules = list(filter(lambda x: hasattr(getattr(self, x), 'balance_loss'),
                                       ['lig_proj', 'prot_proj', 'predictor']))
        self.is_MoE = bool(self.MoE_modules)
        self.pred_is_MoE = isinstance(self.predictor, MoEPredictor)
        if isinstance(self.predictor, MoEPredictor):
            self.predictor.reset_expert_activation()
            if self.shared_exp:
                self.shared_MoE = _get_predictor(shared_exp, gatter, router_noise, hid_dim, n_head, dropout, 1, router_act)
            
    def feat_encode(self, lig_feat: torch.Tensor, mask: torch.Tensor, prot_feat: torch.Tensor):
        # lig_feat: [N, L, 384], mask: [N, L], prot_feat: [N, 1536]
        lig_feat = self.lig_proj(lig_feat)  # [N, L, 256]
        prot_feat = self.prot_proj(prot_feat).unsqueeze(1)  # [N, 1, 256]
        return lig_feat, prot_feat

    def feat_forward(self, lig_feat: torch.Tensor, mask: torch.Tensor, prot_feat: torch.Tensor):
        # 特征投影
        lig_feat = add_rope(lig_feat)  # [N, L, D]
        
        # 创建配体掩码 (对于MultiheadAttention，key_padding_mask: True表示需要mask的位置)
        padding_mask = ~mask.bool()  # 注意：这里我们取反，因为PyTorch要求True表示需要mask的位置
        
        # 层次化处理
        for i in range(self.n_layer):
            # 使用配体特征作为query，蛋白质特征作为key和value
            # 这样蛋白质特征作为全局上下文指导配体特征更新
            ctx_lig, _ = self.global_attn_layers[i](
                query=lig_feat,
                key=prot_feat,
                value=prot_feat,
                key_padding_mask=None  # 蛋白质特征只有1个token，不需要掩码
            )
            
            # 残差连接 + 层归一化
            lig_feat = lig_feat + ctx_lig
            lig_feat = self.lig_norms[i](lig_feat)
            
            # 配体自注意力精炼
            lig_feat = self.lig_self_attn_layers[i](
                lig_feat, 
                src_key_padding_mask=padding_mask
            )
        
        # 最终使用蛋白质特征作为全局上下文进行预测
        # 首先将蛋白质特征与精炼后的配体特征融合
        # 使用注意力机制聚合配体特征到蛋白质特征
        aggregated, _ = self.global_attn_layers[-1](
            query=prot_feat,
            key=lig_feat,
            value=lig_feat,
            key_padding_mask=padding_mask
        )
        
        # 残差连接 + 层归一化
        prot_feat = self.prot_norm(prot_feat + aggregated)
        return prot_feat  # [N, 1, D]
    
    def predict_from_feat(self, feat: torch.Tensor):
        # feat: [N, 1, D]
        out = self.predictor(feat)  # [N, 1]
        if self.shared_exp is not None:
            out += self.shared_MoE(feat).view(-1, 1)  # [N, 1]
        return out.reshape(-1, 1)
    
    
class Arch4(Arch1):
    def __init__(self, lig_dim: int = 384, prot_dim: int = 1536, hid_dim: int = 256, RMSNorm: bool = False, use_rope: bool = False,
                 lig_pred: str = 'LinearDO', prot_pred: str = 'LinearDO',
                 n_layer: int = 2, pred_method: str|list[str, int] = 'MLP', shared_exp: str|list[str, int] = None,
                 gatter: str = 'sum', router_noise: bool = False, router_act: str = 'lambda',
                 prot_scale: bool = False, n_head: int = 8, dropout: float = 0.4, lig_emb_dim: int = 384):
        super().__init__(lig_dim, prot_dim, hid_dim, RMSNorm, use_rope, lig_pred, prot_pred,
                         1, 1, 1, n_layer, False, False, 
                         pred_method, shared_exp, gatter, router_noise, router_act,
                         prot_scale, n_head, dropout, lig_emb_dim)
        self.cross_attn = CrossAttn(hid_dim, 1, n_head, dropout)
        
    def feat_forward(self, smile_feat: torch.Tensor, mask: torch.Tensor, prot_feat: torch.Tensor):
        # smile_feat: [N, L, D], mask: [N, L], prot_feat: [N, 1, D]
        lig_feat = self.attn(smile_feat, mask)  # [N, 1, D]
        # cross-attention for pose and smiles: -> [N, 1, D]
        aligned_lig_feat, aligned_prot_feat = self.cross_attn(lig_feat, prot_feat)
        return aligned_lig_feat, aligned_prot_feat
    
    def predict_from_feat(self, feat: tuple[torch.Tensor, torch.Tensor]):
        """predict from model hidden state"""
        aligned_lig_feat, aligned_prot_feat = feat
        lig_out = self.predictor(aligned_lig_feat).reshape(-1, 1)  # [N, 1]
        prot_out = self.predictor(aligned_prot_feat).reshape(-1, 1)  # [N, 1]
        if self.shared_exp is not None:
            lig_out += self.shared_MoE(aligned_lig_feat).view(-1, 1)  # [N, 1]
            prot_out += self.shared_MoE(aligned_prot_feat).view(-1, 1)  # [N, 1]
        return (lig_out + prot_out) / 2
        
    def forward(self, mid: torch.Tensor, smile_feat: torch.Tensor, mask: torch.Tensor, prot_feat: torch.Tensor, **kwargs):
        # smile_feat: [N, L, 384], mask: [N, L], prot_feat: [N, 1536]
        lig_feat, prot_feat = self.feat_encode(smile_feat, mask, prot_feat)
        aligned_lig_feat, aligned_prot_feat = self.feat_forward(lig_feat, mask, prot_feat)  # [N, 1, 256]
        return self.predict_from_feat((aligned_lig_feat, aligned_prot_feat))
    
    
class Arch41(Arch4):
    """https://github.com/Bigrock-dd/AlphaPPIMI/blob/master/src/cross_attention_ppimi.py#L9"""
    def __init__(self, lig_dim: int = 384, prot_dim: int = 1536, hid_dim: int = 256, RMSNorm: bool = False, use_rope: bool = False,
                 lig_pred: str = 'LinearDO', prot_pred: str = 'LinearDO',
                 n_layer: int = 2, pred_method: str|list[str, int] = 'MLP', shared_exp: str|list[str, int] = None,
                 gatter: str = 'sum', router_noise: bool = False, router_act: str = 'lambda',
                 prot_scale: bool = False, n_head: int = 8, dropout: float = 0.4, lig_emb_dim: int = 384):
        super().__init__(lig_dim, prot_dim, hid_dim, RMSNorm, use_rope, lig_pred, prot_pred, n_layer,
                         pred_method, shared_exp, gatter, router_noise, router_act,
                         prot_scale, n_head, dropout, lig_emb_dim)
        self.gate_fc = nn.Sequential(
            nn.Linear(hid_dim * 2, hid_dim),
            nn.Sigmoid()
        )
        self.fusion_fc = nn.Sequential(
            nn.Linear(hid_dim * 2, hid_dim),
            nn.LayerNorm(hid_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout)
        )
        
    def fuse(self, lig_feat: torch.Tensor, prot_feat: torch.Tensor):
        # lig_feat: [N, 1, D], prot_feat: [N, 1, D]
        cat_feat = torch.cat([lig_feat, prot_feat], dim=-1)  # [N, 1, 2*D]
        gate = self.gate_fc(cat_feat)  # [N, 1, D]
        fused_feat = self.fusion_fc(cat_feat)  # [N, 1, D]
        return lig_feat * gate + (1 - gate) * fused_feat
        
    def forward(self, mid: torch.Tensor, smile_feat: torch.Tensor, mask: torch.Tensor, prot_feat: torch.Tensor, **kwargs):
        # smile_feat: [N, L, 384], mask: [N, L], prot_feat: [N, 1536]
        lig_feat, prot_feat = self.feat_encode(smile_feat, mask, prot_feat)
        aligned_lig_feat, aligned_prot_feat = self.feat_forward(lig_feat, mask, prot_feat)  # [N, 1, 256]
        fused_feat = self.fuse(aligned_lig_feat, aligned_prot_feat)  # [N, 1, 256]
        out = self.predictor(fused_feat).reshape(-1, 1)  # [N, 1]
        if self.shared_exp is not None:
            out += self.shared_MoE(fused_feat).view(-1, 1)  # [N, 1]
        return out
    
    
class Arch42(Arch41):
    def __init__(self, lig_dim: int = 384, prot_dim: int = 1536, hid_dim: int = 256, RMSNorm: bool = False, use_rope: bool = False,
                 lig_pred: str = 'LinearDO', prot_pred: str = 'LinearDO',
                 n_layer: int = 2, pred_method: str|list[str, int] = 'MLP', shared_exp: str|list[str, int] = None,
                 gatter: str = 'sum', router_noise: bool = False, router_act: str = 'lambda',
                 prot_scale: bool = False, n_head: int = 8, dropout: float = 0.4, lig_emb_dim: int = 384):
        super().__init__(lig_dim, prot_dim, hid_dim, RMSNorm, use_rope, lig_pred, prot_pred, n_layer,
                         pred_method, shared_exp, gatter, router_noise, router_act,
                         prot_scale, n_head, dropout, lig_emb_dim)
        self.fusion_fc = None
        
    def fuse(self, lig_feat: torch.Tensor, prot_feat: torch.Tensor):
        # lig_feat: [N, 1, D], prot_feat: [N, 1, D]
        cat_feat = torch.cat([lig_feat, prot_feat], dim=-1)  # [N, 1, 2*D]
        gate = self.gate_fc(cat_feat)  # [N, 1, D]
        return lig_feat * gate + (1 - gate) * prot_feat
    
    
class Arch43(Arch1):
    def __init__(self, lig_dim: int = 384, prot_dim: int = 1536, hid_dim: int = 256, RMSNorm: bool = False, use_rope: bool = False,
                 lig_pred: str = 'LinearDO', prot_pred: str = 'LinearDO',
                 n_layer: int = 2, pred_method: str|list[str, int] = 'MLP', shared_exp: str|list[str, int] = None,
                 gatter: str = 'sum', router_noise: bool = False, router_act: str = 'lambda',
                 prot_scale: bool = False, n_head: int = 8, dropout: float = 0.4, lig_emb_dim: int = 384):
        super().__init__(lig_dim, prot_dim, hid_dim, RMSNorm, use_rope, lig_pred, prot_pred,
                         1, 1, 1, n_layer, False, False,
                         pred_method, shared_exp, gatter, router_noise, router_act,
                         prot_scale, n_head, dropout, lig_emb_dim)
        self.attn = CrossAttn(hid_dim, n_layer, n_head, dropout)
        self.ban_layer = BANLayer(hid_dim, hid_dim, hid_dim, n_head, dropout)
        self.MoE_modules = list(filter(lambda x: hasattr(getattr(self, x), 'balance_loss'),
                                       ['lig_fc', 'prot_fc', 'attn', 'predictor']))
        
    def feat_forward(self, mid: torch.Tensor, smile_feat: torch.Tensor, mask: torch.Tensor, prot_feat: torch.Tensor):
        lig_feat, prot_feat = self.attn(smile_feat, prot_feat)
        return self.ban_layer(lig_feat, prot_feat).unsqueeze(1)  # [N, 1, D]
        
        
class Arch44(Arch43):
    def __init__(self, lig_dim: int = 384, prot_dim: int = 1536, hid_dim: int = 256, RMSNorm: bool = False, use_rope: bool = False,
                 lig_pred: str = 'LinearDO', prot_pred: str = 'LinearDO',
                 n_layer: int = 2, pred_method: str|list[str, int] = 'MLP', shared_exp: str|list[str, int] = None,
                 gatter: str = 'sum', router_noise: bool = False, router_act: str = 'lambda',
                 prot_scale: bool = False, n_head: int = 8, dropout: float = 0.4, lig_emb_dim: int = 384):
        super().__init__(lig_dim, prot_dim, hid_dim, RMSNorm, use_rope, lig_pred, prot_pred, n_layer,
                         pred_method, shared_exp, gatter, router_noise, router_act,
                         prot_scale, n_head, dropout, lig_emb_dim)
        self.attn = CoAttention(hid_dim, n_head, n_layer, dropout)
        
    def feat_forward(self, mid: torch.Tensor, smile_feat: torch.Tensor, mask: torch.Tensor, prot_feat: torch.Tensor):
        prot_mask = torch.ones_like(prot_feat[:, :, 0], dtype=torch.float32).bool()
        # in ligand mask, one indicates the exists of token
        return self.attn(smile_feat, mask.bool(), prot_feat, prot_mask).unsqueeze(1)



if __name__ == '__main__':
    import os
    
    import pandas as pd
    import seaborn as sns
    from mbapy.dl_torch.utils import init_model_parameter, set_random_seed
    from mbapy.plot import save_show
    from tqdm import tqdm

    from models.s1.data_loader import get_data_loader
    set_random_seed(0)
    
    df = pd.read_csv(f'./data/train.csv')
    ds = get_data_loader(torch.load(os.path.expanduser(f'path-to-your-data-folder/SMILES_PepDoRA.pt'), weights_only=False),
                         torch.load(os.path.expanduser(f'path-to-your-data-folder/protein_esm2-3B_mean_each_mean.pt'), weights_only=False),
                         df, None, None, 0, None, None, 'cpu', 128, True, 0, None, False)
    
    model = Arch1(384, 2560, lig_pred=['LinearDO', '12', '6'], hid_dim=512)
    model = init_model_parameter(model, {})
    print(model)
    model.train()
    for idx, mid, lig_feat, mask, prot_feat, pKa in tqdm(ds):
        out = model(mid, lig_feat, mask, prot_feat)
        break
    print(out.shape)
    lig_feat, prot_feat = model.feat_encode(lig_feat, mask, prot_feat)
    print(model.feat_forward(mid, lig_feat, mask, prot_feat).shape)
    
    print(model.calcu_moe_loss(nn.CrossEntropyLoss()))
    F.mse_loss(out.view(-1), pKa.view(-1)).backward()
