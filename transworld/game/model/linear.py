import torch
import torch.nn as nn
from typing import Any, Dict, List, Union, Optional, Tuple


class HomoToHeteroLinear(nn.Module):
    def __init__(
        self,
        in_size: int,
        out_size: Dict[str, int],
        activation: Optional[nn.Module] = None,
        output_activation: Optional[nn.Module] = None,
        bias: bool = True,
    ):
        """Apply linear transformations from homogeneous inputs to heterogeneous inputs.

        Args:
            in_size (int): Input feature size.
            out_size (Dict[str, int]): Output feature size for heterogeneous inputs. A key can be a string or a tuple of strings.
            activation (torch.nn.Module, optional): activative function. Defaults to None.
            bias (bool, optional): bias of network parameters. Defaults to True.
        """
        super(HomoToHeteroLinear, self).__init__()
        self.activation = activation
        self.linears = nn.ModuleDict()
        self.projector = nn.Linear(in_size, in_size // 2, bias=bias)
        assert isinstance(in_size, int), "input size should be int"
        assert isinstance(out_size, dict), "output size should be dict"
        self.output_activation = output_activation
        for typ, dim in out_size.items():
            self.linears[str(typ)] = nn.Linear(in_size // 2, dim, bias=bias)

    def forward(self, feat: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward Function

        Args:
            feat (Dict[str, torch.Tensor]): Heterogeneous input features. It maps keys to features.

        Returns:
            Dict[str, torch.Tensor]: Transformed features.
        """
        out_feat = dict()

        for typ, typ_feat in feat.items():
            out = typ_feat
            out = self.projector(out)
            if self.activation:
                out = self.activation(out)
            out = self.linears[str(typ)](out)
            if self.output_activation:
                out = self.output_activation(out)
            out_feat[typ] = out
        return out_feat


class HeteroToHomoLinear(nn.Module):
    def __init__(
        self,
        in_size: Dict[str, int],
        out_size: int,
        activation: Optional[nn.Module] = None,
        bias: bool = True,
    ):
        """Apply linear transformations on heterogeneous inputs.

        Args:
            in_size (Dict[str, int]): Input feature size for heterogeneous inputs. A key can be a string or a tuple of strings.
            out_size (int): Output feature size.
            activation (nn.Module, optional): activative function.. Defaults to None.
            bias (bool, optional): bias of network parameters. Defaults to True.

        Examples:
            >>> import torch
            >>> from dgl.nn import HeteroLinear

            >>> layer = HeteroLinear({'user': 1, ('user', 'follows', 'user'): 2}, 3)
            >>> in_feats = {'user': torch.randn(2, 1), ('user', 'follows', 'user'): torch.randn(3, 2)}
            >>> out_feats = layer(in_feats)
            >>> print(out_feats['user'].shape)
            torch.Size([2, 3])
            >>> print(out_feats[('user', 'follows', 'user')].shape)
            torch.Size([3, 3])
        """
        super(HeteroToHomoLinear, self).__init__()
        self.activation = activation
        self.linears = nn.ModuleDict()
        for typ, typ_in_size in in_size.items():
            self.linears[str(typ)] = nn.Linear(typ_in_size, out_size // 2, bias=bias)
        self.projector = nn.Linear(out_size // 2, out_size, bias=bias)

    def forward(self, feat: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward function
        Args:
            feat (Dict[str, torch.Tensor]): Heterogeneous input features. It maps keys to features.

        Returns:
            Dict[str, torch.Tensor]: Transformed features.
        """
        out_feat = dict()
        for typ, typ_feat in feat.items():
            if len(typ_feat.shape) > 2:
                typ_feat = typ_feat[
                    :, -1, :
                ]  # 由于这个模块不支持时序建模，如果输入的数据是多步时间步的，那么直接简单地取最后的一步
            out = self.linears[str(typ)](typ_feat)
            if self.activation is not None:
                out = self.activation(out)
            out = self.projector(out)
            out_feat[typ] = out
        return out_feat
