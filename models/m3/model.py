import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from mbapy.dl_torch.utils import init_model_parameter

from models._blocks.attn import MiniTransformerEncodeLayer, PredTokenAttn
from models._blocks.mlp import Linear, LinearDO, SimpleMLP
from models._blocks.moe import MoMixture
from models.m3.data_loader import GraphDataset
from models.s1.model import Arch1 as S1Arch1
from models.s1.model import (TokenMoE, MoEPredictor, _get_predictor)


def get_data_shape_from_dataset(dataset: GraphDataset):
    return {k: v[0].shape for k, v in dataset[0].items() if k not in {'pKa', 'idx', 'mid'}}


class Arch1(S1Arch1):
    def __init__(self, data_shapes: dict[str, torch.Size], hid_dim: int,
                 RMSNorm: bool = False, use_rope: bool = True, norm_first: bool = False, feat_mini_mhsa: bool = False, softmax_partition: bool = False,
                 n_layer: int = 2, pred_method: str|list[str, int] = 'MLP', shared_exp: str|list[str, int] = None,
                 gatter: str = 'sum', router_noise: bool = False, router_act: str = 'lambda', prot_scale: bool = False,
                 n_head: int = 8, dropout: float = 0.4,
                 hydraformer: bool = False, **kwargs):
        super().__init__()
        self.data_shapes = data_shapes
        self.lig_fc = self.prot_fc = None
        self.lig_emb_dim = kwargs.get('lig_emb_dim', 384)
        self.hid_dim = hid_dim
        self.pred_method = pred_method
        self.shared_exp = shared_exp
        self.prot_scale = prot_scale
        self.dropout = dropout
        self.feat_mini_mhsa = feat_mini_mhsa
        self.softmax_partition = softmax_partition
        self.pred_n = 1
        self.use_method_id = kwargs.get('use_method_id', [])
        self.hydraformer = len(data_shapes) + 1 if hydraformer else 0
        if self.use_method_id and 'separate_hydraformer' in self.use_method_id:
            self.hydraformer += 1 # add method token as separate in hydraformer
        self.low_mem_transformer = kwargs.get('low_mem_transformer', False)
        self.moe_ffn = kwargs.get('moe_ffn', False)
        self.gated_sdpa = kwargs.get('gated_sdpa', False)
        self.modal2idx = {}
        # init feature encoders
        self.feat_encoders = nn.ModuleDict()
        for name, shape in data_shapes.items():
            self.modal2idx[name] = len(self.modal2idx) + 1 # zero is for pred token, Arch1 is 1 pred token
            modal_class = 'prot' if self.check_is_prot_feat(name) else 'lig'
            modal_enc = _get_predictor(kwargs.get(f'{modal_class}_pred', 'LinearDO'), gatter, router_noise,
                                       shape[-1] if name not in {'token', 'PepDoRA-token'} else self.lig_emb_dim,
                                       n_head, dropout, hid_dim, router_act)
            if kwargs.get('lig_token_moe', False) and isinstance(modal_enc, MoEPredictor) and shape[0] > 1:
                modal_enc = TokenMoE(modal_enc)
            self.feat_encoders[name] = modal_enc
            if name in {'token', 'PepDoRA-token'}:
                self.feat_encoders[name] = nn.Sequential(
                    nn.Embedding(512, self.lig_emb_dim, padding_idx=0),
                    modal_enc
                )
        if feat_mini_mhsa:
            self.feat_transformer = MiniTransformerEncodeLayer(hid_dim, n_head, dropout)
        
        self.attn = PredTokenAttn(1, hid_dim, n_layer=n_layer, n_head=n_head, dropout=dropout,
                                  RMSNorm=RMSNorm, use_rope=use_rope, norm_first=norm_first,
                                  softmax_partition=softmax_partition, hydraformer=self.hydraformer,
                                  use_method_id=self.use_method_id, moe_ffn=self.moe_ffn,
                                  low_mem_transformer=self.low_mem_transformer, gated_sdpa=self.gated_sdpa)
        if self.hydraformer:
            self._register_modal_idx()
            print(f'modal_idx: {len(self.modal_idx)}', self.modal_idx)
        self.predictor = _get_predictor(pred_method, gatter, router_noise, hid_dim, n_head, dropout, 1, router_act)
        
        self.MoE_modules = list(filter(lambda x: hasattr(getattr(self, x), 'balance_loss'),
                                       ['lig_fc', 'prot_fc', 'attn', 'predictor']))
        self.MoE_balance_loss = 0
        self.is_MoE = bool(self.MoE_modules)
        self.pred_is_MoE = isinstance(self.predictor, MoEPredictor)
        if isinstance(self.predictor, MoEPredictor):
            self.predictor.reset_expert_activation()
            if self.shared_exp:
                self.shared_MoE = _get_predictor(shared_exp, gatter, router_noise, hid_dim, n_head, dropout, 1, router_act)
        else:
            assert not router_noise, 'router noise only enable when model is MoE'

    def check_is_prot_feat(self, name: str):
        if name.startswith('esm') or name in {'SaProt', 'ProSST-2048', 'UniMol-v3-p'}:
            return True
        return False

    def _register_modal_idx(self):
        def _assign_modal_idx(modal_idx: torch.Tensor, lens: list[int]):
            stop = 0 # start from 1, 0 is pred token
            for i, len_i in enumerate(lens):
                modal_idx[stop:stop+len_i] = i
                stop += len_i
            return modal_idx
        # make modal idx for hydraformer
        lens = list(map(lambda x: x[0], self.data_shapes.values()))
        if not self.use_method_id: # no method token, 1+P+L types token in total
            self.register_buffer('modal_idx', torch.zeros(self.pred_n+sum(lens), dtype=torch.long))
            self.modal_idx = _assign_modal_idx(self.modal_idx, [1]+lens)
        elif 'shared_hydraformer' in self.use_method_id: # shared hydraformer, 1+P+L types token in total
            self.register_buffer('modal_idx', torch.zeros(self.pred_n+1+sum(lens), dtype=torch.long))
            self.modal_idx = _assign_modal_idx(self.modal_idx, [2]+lens)
        else: # separate hydraformer, 1+1+P+L types token in total
            self.register_buffer('modal_idx', torch.zeros(self.pred_n+1+sum(lens), dtype=torch.long))
            self.modal_idx = _assign_modal_idx(self.modal_idx, [1, 1]+lens)
            
    def extra_repr(self) -> str:
        return f'MoE={self.MoE_modules}, prot_scale={self.prot_scale}, data_shapes={self.data_shapes}'
            
    def calcu_moe_loss(self, *args, **kwargs):
        self.MoE_balance_loss = super().calcu_moe_loss(*args, **kwargs)
        for module in self.feat_encoders.values():
            balance_loss = getattr(module, 'balance_loss', None)
            if isinstance(balance_loss, torch.Tensor):
                self.MoE_balance_loss += balance_loss.to(self.MoE_balance_loss.device)
        return self.MoE_balance_loss
            
    def feat_encode(self, feats: dict[str: list[torch.Tensor, torch.Tensor]], noise_rate: float = 0.0, **kwargs):
        """encode original input feature into hidden dim"""
        feats_hid = {}
        for name, (feat, mask) in feats.items():
            if name not in {'token', 'PepDoRA-token'}:
                feat = self.feat_encoders[name](feat)
            else:
                feat = self.feat_encoders[name][0](feat)
                feat += torch.randn_like(feat) * noise_rate
                feat = self.feat_encoders[name][1](feat)
            if len(feat.shape) == 2:
                feat = feat.unsqueeze(1)
            if self.feat_mini_mhsa:
                feat = self.feat_transformer(feat, mask=~mask.bool())
            feats_hid[name] = [feat, mask]
        return feats_hid
        
    def feat_forward(self, feats_hid: dict[str: list[torch.Tensor, torch.Tensor]], shuffle: bool = False, **kwargs):
        """merge hidden state into model hidden state"""
        # feats_hid: dict of feature tensors, keys are 'PepDoRA', values are batch of feature and mask
        # shuffle: whether to shuffle the order of features for each sample independently
        if not 'pta_return_src' in kwargs:
            kwargs['only_return_first_token'] = self.pred_n
        
        if not shuffle:
            # Original implementation: concatenate features in fixed order
            x = torch.cat([feats_hid[name][0] for name in feats_hid], dim=1)  # [N, L, D]
            lens = {self.modal2idx[name]: feats_hid[name][0].size(1) for name in feats_hid}
            mask = torch.cat([feats_hid[name][1] for name in feats_hid], dim=1)  # [N, L]
        else:
            # Shuffle implementation: each sample gets a random feature order
            batch_size = next(iter(feats_hid.values()))[0].size(0)
            num_features = len(feats_hid)
            feature_names = list(feats_hid.keys())
            device = next(iter(feats_hid.values()))[0].device
            
            # Calculate the length of each feature for all samples
            feature_lengths = []
            for name in feature_names:
                feat_tensor = feats_hid[name][0]  # [N, Li, D]
                feature_lengths.append(feat_tensor.size(1))
            
            # Generate random permutation for each sample
            # This creates a [N, num_features] tensor where each row is a permutation of [0, 1, ..., num_features-1]
            permutations = torch.stack([torch.randperm(num_features, device=device) for _ in range(batch_size)])
            
            # First, concatenate all features in the original order
            x = torch.cat([feats_hid[name][0] for name in feature_names], dim=1)  # [N, L, D]
            mask = torch.cat([feats_hid[name][1] for name in feature_names], dim=1)  # [N, L]
            
            # Create index mapping for shuffling
            # For each sample, we need to map from original positions to shuffled positions
            total_length = sum(feature_lengths)
            
            # Create a tensor of shape [N, total_length] where each row contains the indices of features
            # in the order they should appear after shuffling
            indices = torch.zeros(batch_size, total_length, dtype=torch.long, device=device)
            
            # For each sample, fill in the indices according to the permutation
            for i in range(batch_size):
                current_pos = 0
                for j in range(num_features):
                    feat_idx = permutations[i, j].item()
                    _ = feature_names[feat_idx]
                    feat_length = feature_lengths[feat_idx]
                    
                    # Calculate the start position of this feature in the original concatenated tensor
                    orig_start = sum(feature_lengths[:feat_idx])
                    orig_end = orig_start + feat_length
                    
                    # Set the indices for this feature in the shuffled order
                    indices[i, current_pos:current_pos+feat_length] = torch.arange(orig_start, orig_end, device=device)
                    current_pos += feat_length
            
            # Use advanced indexing to reorder the features according to the indices
            # We need to expand indices to work with the 3D tensor
            batch_indices = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, total_length)
            x = x[batch_indices, indices]
            mask = mask[batch_indices, indices]
        
        if self.softmax_partition:
            softmax_partition = [(i, 1) for i in [1]+lens]
            x = self.attn(x, mask, softmax_partition=softmax_partition, **kwargs)  # [N, 1, D]
        elif self.hydraformer:
            x = self.attn(x, mask, modal_idx=self.modal_idx, **kwargs)  # [N, 1, D]
        else:
            x = self.attn(x, mask, **kwargs)  # [N, 1, D]
        return x
        
    def forward(self, feats: dict[str: list[torch.Tensor, torch.Tensor]], noise_rate: float = 0.0, **kwargs):
        """feats: dict of feature tensors, keys are 'PepDoRA', values are batch of feature and mask"""
        # normal forward
        feats_hid = self.feat_encode(feats, noise_rate=noise_rate, **kwargs)
        x = self.feat_forward(feats_hid, shuffle=kwargs.get('shuffle', False), **kwargs)  # [N, 1, D]
        return self.predict_from_feat(x)

        
class Arch14(Arch1):
    """apply Hydraformer to the ligand and the protein, not each modals"""
    def __init__(self, data_shapes: dict[str, torch.Size], hid_dim: int,
                 RMSNorm: bool = False, use_rope: bool = True, norm_first: bool = False, feat_mini_mhsa: bool = False, softmax_partition: bool = False,
                 n_layer: int = 2, pred_method: str|list[str, int] = 'MLP', shared_exp: str|list[str, int] = None,
                 gatter: str = 'sum', router_noise: bool = False, router_act: str = 'lambda', prot_scale: bool = False,
                 n_head: int = 8, dropout: float = 0.4, **kwargs):
        super().__init__(data_shapes, hid_dim, RMSNorm, use_rope, False, False, False, n_layer, pred_method, shared_exp, gatter,
                        router_noise, router_act, prot_scale, n_head, dropout, **kwargs)
        self.attn = PredTokenAttn(1, hid_dim, n_layer=n_layer, n_head=n_head, dropout=dropout,
                                  RMSNorm=RMSNorm, use_rope=use_rope, norm_first=False,
                                  softmax_partition=False, hydraformer=4 if self.use_method_id and 'separate_hydraformer' in self.use_method_id else 3,
                                  use_method_id=self.use_method_id, gated_sdpa=self.gated_sdpa, low_mem_transformer=self.low_mem_transformer)
        # get ligand modal total len and protein modal total len
        self.lig_modal_seq_len = sum([data_shapes[name][-2] for name in data_shapes if not self.check_is_prot_feat(name)])
        self.prot_modal_seq_len = sum([data_shapes[name][-2] for name in data_shapes if self.check_is_prot_feat(name)])
        # make modal idx for hydraformer
        if not self.use_method_id:
            self.modal_idx[self.pred_n:self.pred_n+self.prot_modal_seq_len] = 1 # prot modals first
            self.modal_idx[self.pred_n+self.prot_modal_seq_len:] = 2
        elif 'separate_hydraformer' in self.use_method_id:
            self.modal_idx[self.pred_n+1:self.pred_n+1+self.prot_modal_seq_len] = 2 # prot modals first
            self.modal_idx[self.pred_n+1+self.prot_modal_seq_len:] = 3
        else: # shared hydraformer
            self.modal_idx[self.pred_n+1:self.pred_n+1+self.prot_modal_seq_len] = 1 # prot modals first
            self.modal_idx[self.pred_n+1+self.prot_modal_seq_len:] = 2
        print(f'modal_idx: {len(self.modal_idx)}', self.modal_idx)
        
    def feat_forward(self, feats_hid: dict[str: list[torch.Tensor, torch.Tensor]], shuffle: bool = False, **kwargs):
        """merge hidden state into model hidden state"""
        # feats_hid: dict of feature tensors, keys are 'PepDoRA', values are batch of feature and mask
        # shuffle: whether to shuffle the order of features for each sample independently
        if not 'pta_return_src' in kwargs:
            kwargs['only_return_first_token'] = self.pred_n
        
        x = torch.cat([feats_hid[name][0] for name in feats_hid], dim=1)  # [N, L, D]
        mask = torch.cat([feats_hid[name][1] for name in feats_hid], dim=1)  # [N, L]
        x = self.attn(x, mask, modal_idx=self.modal_idx, **kwargs)  # [N, 1, D]
        return x        


if __name__ == '__main__':
    import os
    import pandas as pd
    from mbapy.dl_torch.utils import init_model_parameter, set_random_seed

    from models.m3.data_loader import get_data_loader
    set_random_seed(0)
    
    df = pd.read_csv(f'./data/valid.csv')
    lig_data = torch.load(os.path.expanduser(f'path-to-your-data-folder/SMILES_PepDoRA.pt'))
    lig_data2 = torch.load(os.path.expanduser(f'path-to-your-data-folder/SMILES_GeminiMol.pt'))
    lig_data3 = torch.load(os.path.expanduser(f'path-to-your-data-folder/SMILES_MolAI.pt'))
    rec_data = torch.load(os.path.expanduser(f'path-to-your-data-folder/protein_esm2-3B_mean_each_mean.pt'))
    dataloader = get_data_loader({'PepDoRA': lig_data, 'GeminiMol': lig_data2, 'MolAI': lig_data3},
                                 {'esm2-3B': rec_data}, df, None, 16, None, 0.1, 'front', 'cpu', 16, True, 0, None)
    
    shapes = get_data_shape_from_dataset(dataloader.dataset)
    model = Arch1(shapes, hid_dim=512, pred_method=['MLP', '12', '1'], hydraformer=True, gated_sdpa=2, use_method_id=['external', 'separate_hydraformer'])
    model = init_model_parameter(model, {})
    
    model.train()
    for batch in dataloader:
        _ = batch.pop('pKa')
        _ = batch.pop('idx')
        mid = batch.pop('mid')
        out = model(batch, mid=mid)
        print(out.shape)
        break
