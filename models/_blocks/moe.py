import torch
import torch.nn as nn
import torch.nn.functional as F

@torch.compile
def gumbel_topk(logits, k, temperature=1.0):
    """
    更精确的Gumbel topk实现
    """
    # 添加Gumbel噪声到logits
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits)))
    noisy_logits = logits + gumbel_noise
    
    # 应用温度调节
    noisy_logits = noisy_logits / temperature
    
    # 使用softmax得到概率分布
    probs = F.softmax(noisy_logits, dim=-1)
    
    # 获取topk索引（这是关键区别）
    _, topk_idx = torch.topk(probs, k=k, dim=-1)
    
    # 创建topk掩码
    mask = torch.zeros_like(probs)
    mask.scatter_(-1, topk_idx, 1.0)
    
    # 使用STE技巧保持可导性
    ste_mask = mask + probs - probs.detach()
    
    return ste_mask, topk_idx


class MoEPredictor(nn.Module):
    def __init__(self, predictors: nn.ModuleList, topk: int = 1, n_head: int = 1, gatter: str = 'sum',
                 router_noise: bool = False, router_act: str = 'lambda', hid_dim: int = 256, dropout: float = 0.4,
                 forward_method: str = 'batch'):
        super().__init__()
        assert topk <= len(predictors), f'topk ({topk}) must be less than or equal to the number of predictors ({len(predictors)})'
        assert gatter in ['sum', 'mean', 'weighted', 'cat'], f'gatter ({gatter}) must be sum or mean or weighted or cat'
        assert router_act in ['lambda', 'softmax'], f'router_act ({router_act}) must be lambda or softmax'
        assert n_head > 0 and n_head <= len(predictors), f'n_head ({n_head}) must be greater than 0 and less than or equal to the number of predictors ({self.n_exp})'
        if n_head > 1 and n_head != topk:
            raise ValueError(f'n_head ({n_head}) must be equal to topk ({topk}) when n_head > 1')
        self.n_exp = len(predictors)
        self.topk = topk
        self.n_head = n_head
        self.gatter = gatter
        self.router_act = router_act
        self.hid_dim = hid_dim
        self.dropout = dropout
        self.forward_method = forward_method
        # 专家网络
        self.predictors = predictors
        
        # 门控网络
        self.gate = nn.Linear(hid_dim, self.n_exp)
        self.router_noise = router_noise
        if router_noise:
            self.noise_router = nn.Linear(hid_dim, self.n_exp)
        
        # 温度参数（可学习）
        self.temperature = nn.Parameter(torch.tensor(1.0))
        
        # 专家激活记录
        self._RECORD_EXPERT_ACTIVATION = False
        self.register_buffer('expert_activation', torch.zeros(self.n_exp))
        self.register_buffer('gate_density', torch.zeros(self.n_exp))
        self.balance_loss = 0
        
    def extra_repr(self) -> str:
        return f'n_exp={self.n_exp}, topk={self.topk}, n_head={self.n_head}, gatter={self.gatter}, router_noise={self.router_noise}, router_act={self.router_act}, hid_dim={self.hid_dim}, dropout={self.dropout}, forward_method={self.forward_method}'
        
    def reset_expert_activation(self):
        self.expert_activation.zero_()
        self.gate_density.zero_()
        
    def _moe_iter_forward(self, x, top_idx):
        # use for-iter to forward for each sample
        out_batch = []
        for i in range(x.shape[0]):
            tmp_out = []
            for j in range(self.topk):
                tmp_out.append(self.predictors[top_idx[i][j]](x[i].unsqueeze(0)))
            out_batch.append(torch.cat(tmp_out, dim=-1))
        return torch.cat(out_batch, dim=0)  # [N, ..., D -> k]
    
    def _moe_full_exp_forward(self, x, gate_scores):
        """
        并行计算所有专家的输出，然后根据gumbel_topk门控分数加权组合
        x: [N, D], gate_scores: [N, n_exp]
        output: [N, n_exp, ...], soft_mask: [N, n_exp]
        """
        # 生成专家掩码
        soft_mask, _ = gumbel_topk(gate_scores, k=self.topk, temperature=self.temperature)  # [N, n_exp]
        # 计算各专家输出
        expert_outputs = torch.stack([exp(x) for exp in self.predictors], dim=1)  # [N, n_exp, ...]
        # 只保留选中专家的输出，其他专家的输出置0
        mask_shape = list(soft_mask.shape) + [1]*(expert_outputs.dim() - soft_mask.dim())
        output = (expert_outputs * soft_mask.reshape(*mask_shape))  # [N, n_exp, ...]
        return output, soft_mask
    
    def _moe_batch_forward(self, x, topk_idx):
        batch_size = x.size(0)
        K = self.topk
        
        # 展开输入和专家索引
        x_expanded = x.repeat_interleave(K, dim=0)  # [N*K, D]
        expert_ids = topk_idx.view(-1)  # [N*K]
        
        # 创建专家路由映射
        unique_expert_ids = expert_ids.unique()
        
        # 并行计算选中专家的输出
        # 保存每个专家处理的样本索引
        expert_outputs = []
        expert_indices = []  # 保存每个专家处理的样本在原始序列中的位置
        
        # 并行计算选中专家的输出
        for eid in unique_expert_ids:
            mask = (expert_ids == eid)
            expert_outputs.append(self.predictors[int(eid)](x_expanded[mask]))
            expert_indices.append(torch.where(mask)[0])  # 保存原始位置索引
            
        # 拼接所有专家的输出
        out_shape = expert_outputs[0].shape
        concat_outputs = torch.cat(expert_outputs)
        concat_indices = torch.cat(expert_indices)
        
        # 恢复原始顺序
        ordered_outputs = torch.zeros_like(concat_outputs)
        ordered_outputs[concat_indices] = concat_outputs
    
        # 重组输出张量, NOT the [K, N, ...]
        return ordered_outputs.reshape(batch_size, K, *out_shape[1:])  # [N, K, ...]
    
    def _router(self, router: nn.Module, x: torch.Tensor):
        """process x with shape [N, 1, D] or [N, L, D]"""
        gate_scores = router(x)  # [N, ..., n_exp]
        if len(gate_scores.shape) == 3 and gate_scores.shape[1] > 1:
            gate_scores = gate_scores.mean(dim=1, keepdim=True) # [N, X, n_exp] -> [N, 1, n_exp]
        return gate_scores.squeeze(1) # [N, 1, n_exp] -> [N, n_exp]
        
    def forward(self, x: torch.Tensor):
        # 输入形状: [N, ..., hid_dim]
        if len(x.shape) == 2:
            x.unsqueeze_(1)
        # 计算门控权重
        gate_scores = self._router(self.gate, x)
        # 应用路由噪声
        if self.training and self.router_noise:
            noise = torch.randn_like(gate_scores) # [N, n_exp]
            noise_gates = self._router(self.noise_router, x) # [N, n_exp]
            noise = noise * F.softplus(noise_gates) # [N, n_exp]
            gate_scores += noise
        gate_scores = gate_scores / self.temperature
        # apply router activation
        if self.router_act == 'lambda':
            pass
        elif self.router_act == 'sigmoid':
            gate_scores = torch.sigmoid(gate_scores)
        elif self.router_act == 'softmax':
            gate_scores = F.softmax(gate_scores, dim=-1)
        elif self.router_act == 'tanh':
            gate_scores = torch.tanh(gate_scores)
        # 选择top-k专家
        if self._RECORD_EXPERT_ACTIVATION or self.forward_method in {'iter', 'batch'} or (not self.training):
            topk_val, topk_idx = torch.topk(gate_scores, k=self.topk, dim=-1)  # [N, k]
        # calculate balance loss
        if self.training:
            self.balance_loss = self.compute_balance_loss(gate_scores)
        # 更新专家激活记录
        if self._RECORD_EXPERT_ACTIVATION:
            self.topk_idx = topk_idx.detach()
            self.expert_activation += topk_idx.flatten().bincount(minlength=self.n_exp)
            self.gate_density += gate_scores.softmax(dim=-1).mean(dim=[0]).detach()
        
        if self.forward_method == 'iter':
            output = self._moe_iter_forward(x, topk_idx)
        elif self.forward_method == 'batch' or (not self.training):
            output = self._moe_batch_forward(x, topk_idx)
        else:
            output, soft_mask = self._moe_full_exp_forward(x, gate_scores) # [N, n_exp]
        
        # output: [N, K, ..., D]
        if self.n_head > 1:
            # [N, K, L, D] -> [N, L, K, D]
            N, _, L, _ = output.shape
            output = output.permute(0, 2, 1, 3)
            # [N, L, K, D] -> [N, L, K*D]
            output = output.reshape(N, L, -1)
        elif self.gatter == 'sum':
            output = output.sum(dim=1)
        elif self.gatter == 'mean':
            output = output.mean(dim=1)
        elif self.gatter == 'weighted':
            # use softmax to calcu weight
            if self.forward_method in {'iter', 'batch'}:
                weights = F.softmax(topk_val, dim=-1)  # [N, k]
            else:
                weights = F.softmax(soft_mask*gate_scores, dim=-1)  # [N, n_exp, 1]
            shape_pad = [1] * (len(output.shape) - 2)
            weights = weights.view(*weights.shape, *shape_pad)
            output = (output * weights).sum(dim=1)
        elif self.gatter == 'cat':
            output = output.squeeze()
        return output  # [N, 1]
    
    def compute_balance_loss(self, logits: torch.Tensor, indices: torch.Tensor = None):
        """负载均衡损失，防止某些专家过劳"""
        # 基于路由概率计算的每个专家的理论使用频率
        # # 基于实际选择的专家索引计算的每个专家的实际使用频率
        # mask = F.one_hot(indices, self.n_exp).float()
        # density = mask.mean(dim=[0, 1])
        # # 辅助损头：希望density和gate_density都均匀
        # loss = self.n_exp * (density * gate_density).sum()
        # # 负熵：鼓励均匀分布
        # loss = -torch.sum(gate_density * torch.log(gate_density + 1e-8))
        # # KL散度：鼓励均匀分布
        # uniform_density = torch.ones_like(gate_density) / self.n_exp
        # loss = torch.sum(gate_density * torch.log(gate_density / uniform_density + 1e-8))
        return F.cross_entropy(logits, torch.ones_like(logits))
    

class TokenMoE(nn.Module):
    def __init__(self, model: MoEPredictor) -> None:
        super().__init__()
        assert isinstance(model, MoEPredictor), 'TokenMoE only support MoEPredictor'
        self.model = model
        
    #修改forward_method时直接修改model的forward_method
    @property
    def forward_method(self):
        return self.model.forward_method
    
    @forward_method.setter
    def forward_method(self, value):
        self.model.forward_method = value
        
    @property
    def _RECORD_EXPERT_ACTIVATION(self):
        return self.model._RECORD_EXPERT_ACTIVATION
    
    @_RECORD_EXPERT_ACTIVATION.setter
    def _RECORD_EXPERT_ACTIVATION(self, value):
        self.model._RECORD_EXPERT_ACTIVATION = value
        
    @property
    def balance_loss(self):
        """获取 MoEPredictor 的 balance_loss"""
        return self.model.balance_loss
    
    @property
    def topk_idx(self):
        return self.model.topk_idx
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        N, L, D = x.shape
        # test result: no performance improvement with cuda and compile
        if mask is not None:
            # True means value, False means padding
            mask = mask.reshape(N*L).bool()
            x = x.reshape(N*L, 1, D)
            out = self.model(x[mask, :, :]) # [N*L, 1, D_out]
            out_full = torch.zeros(N*L, 1, out.size(-1), device=x.device)
            out_full[mask, :, :] = out
            return out_full.reshape(N, L, -1)
        return self.model(x.reshape(N*L, 1, D)).reshape(N, L, -1)
    
    
class MoEWithSharedExp(nn.Module):
    def __init__(self, moe: nn.Module, shared_model: nn.Module) -> None:
        super().__init__()
        assert isinstance(moe, (MoEPredictor, TokenMoE)), 'MoEWithSharedExp only support MoEPredictor and TokenMoE as moe'
        self.moe = moe
        self.shared_model = shared_model
        self.shared_is_moe = isinstance(shared_model, (MoEPredictor, TokenMoE))
        
    @property
    def balance_loss(self):
        """获取 MoEPredictor 的 balance_loss"""
        loss = self.moe.balance_loss
        if self.shared_is_moe:
            loss += self.shared_model.balance_loss
        return loss
    
    @property
    def _RECORD_EXPERT_ACTIVATION(self):
        if self.shared_is_moe:
            return self.moe._RECORD_EXPERT_ACTIVATION
        return self.moe._RECORD_EXPERT_ACTIVATION
    
    @_RECORD_EXPERT_ACTIVATION.setter
    def _RECORD_EXPERT_ACTIVATION(self, value):
        if self.shared_is_moe:
            self.shared_model._RECORD_EXPERT_ACTIVATION = value
        self.moe._RECORD_EXPERT_ACTIVATION = value
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.moe(x) + self.shared_model(x)

    
class MoMixture(MoEPredictor):
    """Mix multiple modals with learnable weights"""
    def __init__(self, hid_dim: int, n_modal: int, topk: int = 2, gatter: str = 'weighted'):
        nn.Module.__init__(self)
        assert gatter in {'sum', 'mean', 'weighted'}, f"gatter must be in {'sum', 'mean', 'weighted'}, but got {gatter}"
        self.topk = topk
        self.gatter = gatter
        self.n_modal = n_modal
        self.gate = nn.Linear(hid_dim, self.n_modal)
        self.noise_router = nn.Linear(hid_dim, self.n_modal)
        self.router_noise = True
        self.temperature = nn.Parameter(torch.ones(1))
        self.register_buffer('expert_activation', torch.zeros(self.n_modal))
        self.register_buffer('gate_density', torch.zeros(self.n_modal))
        self.balance_loss = 0
        
    def extra_repr(self) -> str:
        return f'topk={self.topk}, gatter={self.gatter}, n_modal={self.n_modal}, router_noise={self.router_noise}'
        
    def forward(self, main_feat: torch.Tensor, other_feat: torch.Tensor):
        """main_feat: [N, L, D], other_feat: [n_modal, N, L, D]"""
        if self.n_modal != len(other_feat):
            raise ValueError(f"n_modal({self.n_modal}) must be equal to len(other_feat)({len(other_feat)})")
        # 计算门控权重
        gate_scores = self._router(self.gate, main_feat)
        # 应用路由噪声
        if self.training and self.router_noise:
            noise = torch.randn_like(gate_scores) # [N, n_modal]
            noise_gates = self._router(self.noise_router, main_feat) # [N, n_modal]
            noise = noise * F.softplus(noise_gates) # [N, n_modal]
            gate_scores += noise
        gate_scores = gate_scores / self.temperature
        # 选择top-k专家
        topk_val, topk_idx = torch.topk(gate_scores, k=self.topk, dim=-1)  # [N, k]
        # calculate balance loss
        if self.training and self.router_noise:
            self.balance_loss = self.compute_balance_loss(gate_scores, topk_idx)
        # 更新专家激活记录
        self.expert_activation += topk_idx.flatten().bincount(minlength=self.n_modal).to(self.expert_activation.device)
        
        # cat selected features: -> [N, n, L, D] -> [N*k: N0...k...N0N1...k...N1, n, L, D]
        other_feat = other_feat.transpose(1, 0).repeat_interleave(self.topk, dim=0)  # [N*k, n, L, D]
        topk_idx_flatten = topk_idx.flatten()  # [N*k]
        selected_feat = other_feat[torch.arange(other_feat.shape[0]), topk_idx_flatten, :, :]  # [N*k, L, D]
        output = selected_feat.reshape(main_feat.shape[0], self.topk, *selected_feat.shape[1:])  # [N, k, L, D]
        
        # output: [N, K, ..., D]
        if self.gatter == 'sum':
            output = output.sum(dim=1)
        elif self.gatter == 'mean':
            output = output.mean(dim=1)
        elif self.gatter == 'weighted':
            # use softmax to calcu weight
            weights = F.softmax(topk_val, dim=-1)  # [N, k]
            shape_pad = [1] * (len(output.shape) - 2)
            weights = weights.view(*weights.shape, *shape_pad)
            output = (output * weights).sum(dim=1)
        return output  # [N, 1]
    
    
if __name__ == '__main__':
    n_exp, top_k = 8, 4
    moe = MoEPredictor([nn.Linear(128, 128) for _ in range(n_exp)], hid_dim=128, topk=top_k, gatter='weighted')
    moe.forward_method = 'full_exp'
    x = torch.randn(10, 1, 128)
    output = moe(x)
    print(output.shape)
    
    # batch_size, hid_dim, n_modal, topk = 4, 1, 3, 2
    # mox = MoMixture(hid_dim=hid_dim, n_modal=n_modal, topk=topk, gatter='weighted')
    # x = torch.arange(batch_size).view(batch_size, 1, 1)
    # other_feat = torch.arange(n_modal*batch_size).reshape(n_modal, batch_size, 1, hid_dim)
    # output = mox(x.float(), other_feat.float())
    # print(output.shape)
    
    import os
    from tqdm import tqdm
    import pandas as pd
    from mbapy.dl_torch.utils import init_model_parameter, set_random_seed
    from models.s1.data_loader import get_data_loader
    set_random_seed(0)
    
    df = pd.read_csv(f'./data/train.csv')
    ds = get_data_loader(torch.load(os.path.expanduser(f'path-to-your-data-folder/SMILES_PepDoRA.pt')),
                         torch.load(os.path.expanduser(f'path-to-your-data-folder/protein_esm2-3B_mean_each_mean.pt')),
                         df, None, None, 0, None, None, 'cpu', 128, False, 0, None)
    
    model = TokenMoE(MoEPredictor(nn.ModuleList([nn.Linear(384, 512) for _ in range(12)]), hid_dim=384, topk=1)).cuda()
    model = init_model_parameter(model, {})
    model = torch.compile(model)
    model.train()
    for _ in tqdm(list(range(4)), desc='with mask'):
        for mid, lig_feat, mask, prot_feat, pKa in tqdm(ds, leave=False):
            out = model(lig_feat.cuda(), mask.cuda())
    for _ in tqdm(list(range(4)), desc='without mask'):
        for mid, lig_feat, mask, prot_feat, pKa in tqdm(ds, leave=False):
            mask = mask.cuda()
            out = model(lig_feat.cuda())