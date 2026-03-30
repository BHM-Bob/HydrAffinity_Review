
import torch
import torch.nn as nn
import torch.nn.functional as F


class Linear(nn.Module):
    def __init__(self, input_dim: int, out_dim: int = 1, dropout: float = None):
        super().__init__()
        self.linear = nn.Sequential(nn.Linear(input_dim, out_dim),
                                    nn.LeakyReLU()
                                    )
    
    def forward(self, x):
        # x: [b, C]
        return self.linear(x)  # [N, out_dim]

    
def LinearDO(input_dim: int, out_dim: int = 1, dropout: float = 0.1):
    return nn.Sequential(nn.Linear(input_dim, out_dim),
                         nn.LeakyReLU(),
                         nn.Dropout(dropout)
                         )


class SimpleMLP(nn.Module):
    """Simple 2-layer MLP
    Args:
        num_classes (int): Number of classes.
        input_dim (int): Input dimension.
        dropout (float, optional): Dropout rate. Defaults to 0.4.
    """
    def __init__(self, input_dim: int, out_dim: int = 1, dropout: float = 0.4,
                 hid_dim: int = None):
        super().__init__()
        hid_dim = hid_dim or 4 * input_dim
        # Fully connected layers for output
        self.out_fc = nn.Sequential(nn.Linear(input_dim, hid_dim),
                                    nn.LeakyReLU(),
                                    nn.Dropout(dropout),  # 添加 Dropout
                                    nn.Linear(hid_dim, out_dim),
                                    nn.LeakyReLU()
                                    )
    
    def forward(self, x):
        # x: [b, C]
        return self.out_fc(x)  # [N, out_dim]
    

class SimpleMLPN(nn.Module):
    """Simple 2-layer MLP with batchnormalize1D
    Args:
        input_dim (int): Input dimension.
        out_dim (int, optional): Output dimension. Defaults to 1.
        dropout (float, optional): Dropout rate. Defaults to 0.4.
    """
    def __init__(self, input_dim: int, out_dim: int = 1, dropout: float = 0.4):
        super().__init__()
        # Fully connected layers for output
        self.out_fc1 = nn.Sequential(nn.Linear(input_dim, 4 * input_dim),
                                    nn.Dropout(dropout),
                                    nn.LeakyReLU())
        self.bn = nn.BatchNorm1d(4 * input_dim)
        self.out_fc2 = nn.Sequential(nn.Linear(4 * input_dim, out_dim),
                                    # nn.LeakyReLU()
                                    )
        
    def out_fc(self, x):
        batch_size = x.shape[0]
        x = self.out_fc1(x)
        if batch_size > 1:
            x = self.bn(x)
        x = self.out_fc2(x)
        return x
    
    def forward(self, x):
        is_reshape = len(x.shape) == 3
        if is_reshape:
            batch_size, L, D = x.shape
            x = x.reshape(batch_size*L, D)
        out = self.out_fc(x)  # [N, out_dim]
        if is_reshape:
            out = out.reshape(batch_size, L, -1)  # [N, L, out_dim]
        return out


class DeepSeekExpert(nn.Module):
    """
    from https://github.com/deepseek-ai/DeepSeek-V3/blob/main/inference/model.py
    - replace quantization Linear with torch.nn.Linear
    - replace inter_dim with input_dim // 2
    
    Expert layer for Mixture-of-Experts (MoE) models.

    Attributes:
        w1 (nn.Module): Linear layer for input-to-hidden transformation.
        w2 (nn.Module): Linear layer for hidden-to-output transformation.
        w3 (nn.Module): Additional linear layer for feature transformation.
    """
    def __init__(self, input_dim: int, inter_dim: int = None):
        """
        Initializes the Expert layer.

        Args:
            dim (int): Input and output dimensionality.
        """
        super().__init__()
        inter_dim = inter_dim or input_dim // 2
        self.input_dim = input_dim
        self.inter_dim = inter_dim
        self.w1 = nn.Linear(input_dim, inter_dim)
        self.w2 = nn.Linear(inter_dim, input_dim)
        self.w3 = nn.Linear(input_dim, inter_dim)
        
    def extra_repr(self) -> str:
        return f'input_dim={self.input_dim}, inter_dim={self.inter_dim}'

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Expert layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after expert computation.
        """
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class HighwayBase(nn.Module):
    def __init__(self, input_dim: int, out_dim: int = None, dropout: float = 0.4):
        super().__init__()
        self.highway_transform_gate = nn.Sequential(
            nn.Linear(input_dim, out_dim),
            nn.Sigmoid()
        )
        self.highway_transformation = LinearDO(input_dim, out_dim, dropout)
        # 投影层确保残差连接维度匹配
        if input_dim != out_dim:
            self.projection = nn.Linear(input_dim, out_dim)
        else:
            self.projection = nn.Identity()
    
    def forward(self, x):
        # x: [b, C]
        
        # Calculate transform gate
        gate = self.highway_transform_gate(x) # [N, out_dim]
        
        # Apply non-linear transformation
        transformed = self.highway_transformation(x) # [N, out_dim]
        
        # Apply projection for skip connection
        projected = self.projection(x) # [N, out_dim]
        
        # Apply highway mechanism: gate * transformed + (1 - gate) * projected
        output = gate * transformed + (1 - gate) * projected
        
        return output # [N, out_dim]


class HighwayMLP1(nn.Module):
    """MLP with Highway mechanism - Version 1
    在input_dim -> hid_dim层实现highway机制，然后在hid_dim -> out_dim层根据情况处理
    
    Args:
        input_dim (int): Input dimension.
        out_dim (int, optional): Output dimension. Defaults to 1.
        dropout (float, optional): Dropout rate. Defaults to 0.4.
        hid_dim (int, optional): Hidden dimension. Defaults to 4 * input_dim.
    """
    def __init__(self, input_dim: int, out_dim: int = 1, dropout: float = 0.4,
                 hid_dim: int = None):
        super().__init__()
        hid_dim = hid_dim or 4 * input_dim
        
        self.highway1 = HighwayBase(input_dim, hid_dim, dropout)
        if out_dim == 1:
            self.out_fc = Linear(hid_dim, out_dim)
        else:
            self.out_fc = HighwayBase(hid_dim, out_dim, dropout)
    
    def forward(self, x):
        # 第一个highway层
        highway_output1 = self.highway1(x)  # [N, hid_dim]
        
        # 第二个层处理
        output = self.out_fc(highway_output1)  # [N, out_dim]
        
        return output  # [N, out_dim]


class HighwayMLP2(nn.Module):
    """MLP with Highway mechanism - Version 2
    增强版本：
    
    Args:
        input_dim (int): Input dimension.
        out_dim (int, optional): Output dimension. Defaults to 1.
        dropout (float, optional): Dropout rate. Defaults to 0.4.
        hid_dim (int, optional): Hidden dimension. Defaults to 4 * input_dim.
    """
    def __init__(self, input_dim: int, out_dim: int = 1, dropout: float = 0.4,
                 hid_dim: int = None):
        super().__init__()
        hid_dim = hid_dim or 4 * input_dim
        
        # 非线性变换路径
        self.transformation = nn.Sequential(
            nn.Linear(input_dim, hid_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid_dim, out_dim)
        )
        
        # 门控机制：确保门控维度与输出维度一致
        self.transform_gate = nn.Sequential(
            nn.Linear(input_dim, out_dim),
            nn.Sigmoid()
        )
        
        # 残差连接投影层
        if input_dim != out_dim:
            self.projection = nn.Linear(input_dim, out_dim)
        else:
            self.projection = nn.Identity()
    
    def forward(self, x):
        # 计算门控值
        gate = self.transform_gate(x)  # [N, out_dim]
        
        # 应用非线性变换
        transformed = self.transformation(x)  # [N, out_dim]
        
        # 应用投影确保残差连接维度匹配
        projected = self.projection(x)  # [N, out_dim]
        
        # 应用highway机制
        output = gate * transformed + (1 - gate) * projected  # [N, out_dim]
        
        return output  # [N, out_dim]


class BSplineLayer(nn.Module):
    def __init__(self, input_dim: int, out_dim: int, *args, **kwargs) -> None:
        super().__init__()
        self.bspline = KANLinear(input_dim, out_dim)
    
    def forward(self, x):
        if len(x.shape) == 3 and x.shape[1] == 1:
            x = x.squeeze(1)
        self.bspline.update_grid(x)
        return self.bspline(x)
    
    
class BSplineLayer2(nn.Module):
    def __init__(self, input_dim: int, out_dim: int, *args, **kwargs) -> None:
        super().__init__()
        from efficient_kan import KANLinear
        self.bspline = KANLinear(input_dim, out_dim)
    
    def forward(self, x):
        if len(x.shape) == 3 and x.shape[1] == 1:
            x = x.squeeze(1)
        self.bspline.update_grid(x)
        return self.bspline(x)


class FC(nn.Module):
    def __init__(self, input_dim: int, hid_dim: int, out_dim: int, dropout: float, n_layer: int = 2):
        """
        层数： n_layer
            - 1. input_dim -> hid_dim, dropout->LeakyReLU->BatchNorm1d
            - 2. hid_dim -> hid_dim, dropout->LeakyReLU->BatchNorm1d
            - ...
            - n_layer. hid_dim -> out_dim
        """
        super(FC, self).__init__()
        self.input_dim = input_dim
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.n_layer = n_layer
        self.dropout = dropout
        self.predict = nn.ModuleList()
        for j in range(self.n_layer):
            if j == 0:
                self.predict.append(nn.Linear(input_dim, hid_dim))
                self.predict.append(nn.Dropout(dropout))
                self.predict.append(nn.LeakyReLU())
                self.predict.append(nn.BatchNorm1d(hid_dim))
            if j == self.n_layer - 1:
                self.predict.append(nn.Linear(hid_dim, out_dim))
            else:
                self.predict.append(nn.Linear(hid_dim, hid_dim))
                self.predict.append(nn.Dropout(dropout))
                self.predict.append(nn.LeakyReLU())
                self.predict.append(nn.BatchNorm1d(hid_dim))

    def forward(self, h):
        if len(h.shape) == 3 and h.shape[1] == 1:
            h = h.squeeze(1)
        for layer in self.predict:
            h = layer(h)

        return h


class MHMLP(nn.Module):
    """Multi head MLP
    Args:
        num_classes (int): Number of classes.
        input_dim (int): Input dimension.
        dropout (float, optional): Dropout rate. Defaults to 0.4.
    """
    def __init__(self, num_classes: int, input_dim: int, dropout: float = 0.4):
        super().__init__()
        # Fully connected layers for output
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.out_fcs = nn.ModuleList([SimpleMLP(input_dim, 1, dropout) for _ in range(num_classes)])
    
    def forward(self, x):
        # x: [b, C]
        return torch.cat([fc_i(x) for fc_i in self.out_fcs], dim=-1)  # [N, out_dim]


class MCMLP(nn.Module):
    """Multi channle MLP
    Args:
        num_classes (int): Number of classes.
        input_dim (int): Input dimension.
        dropout (float, optional): Dropout rate. Defaults to 0.4.
    """
    def __init__(self, num_classes: int, input_dim: int, dropout: float = 0.4):
        super().__init__()
        # Fully connected layers for output
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.out_fcs = nn.ModuleList([SimpleMLP(input_dim, 1, dropout) for _ in range(num_classes)])
    
    def forward(self, x):
        # x: [b, NC, D] -> [b, NC]
        return torch.cat([self.out_fcs[i](x[:, i, :]) for i in range(self.num_classes)], dim=-1)
    
    
class MeanPool(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
    
    def forward(self, x):
        # x: [b, NC, D] -> [b, NC, 1]
        return x.mean(dim=-1, keepdim=True)


__all__ = [
    'Linear',
    'LinearDO',
    'SimpleMLP',
    'FC',
    'MHMLP',
    'MCMLP',
    'MeanPool',
]