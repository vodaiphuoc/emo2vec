import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict

from .emotion2vec_base.model import Emotion2vec, PretrainOutput
from .types import ModelConfig

class E2VftModel(torch.nn.Module):
    
    def __init__(
            self,
            num_classes:int, 
            pretrain_cfg: ModelConfig,
            pretrain_state_dict: OrderedDict
        )->None:
        super().__init__()
        
        self._pretrain_model: Emotion2vec = Emotion2vec(model_conf = pretrain_cfg)
        assert isinstance(pretrain_state_dict, OrderedDict)
        self._pretrain_model.load_state_dict(pretrain_state_dict)

        self.head_pre = torch.nn.Linear(pretrain_cfg.embed_dim*281, pretrain_cfg.embed_dim)
        self.drop_out_pre = torch.nn.Dropout(p=0.3)
        self.head_inter = torch.nn.Linear(pretrain_cfg.embed_dim, pretrain_cfg.embed_dim//2)
        self.drop_out_inter = torch.nn.Dropout(p=0.15)
        self.head_out = torch.nn.Linear(pretrain_cfg.embed_dim//2, num_classes)

    def forward(
            self,
            source: torch.Tensor,
            target=None,
            id=None,
            mode=None,
            padding_mask=None,
            mask=False,
            features_only=True,
            force_remove_masked=False,
            remove_extra_tokens=True,
            precomputed_mask=None,
            **kwargs
        )->torch.Tensor:

        source = nn.functional.layer_norm(source, source.shape)

        pretrain_outputs: PretrainOutput = self._pretrain_model(
            source = source,
            target = target,
            id= id,
            mode=mode,
            padding_mask=padding_mask,
            mask=mask,
            features_only=features_only,
            force_remove_masked=force_remove_masked,
            remove_extra_tokens=remove_extra_tokens,
            precomputed_mask=precomputed_mask,
            **kwargs
        )
        print('padding_mask shape: ', pretrain_outputs.padding_mask.shape)
        B, F, D = pretrain_outputs.x.shape
        x = pretrain_outputs.x.reshape((B, F*D))
        
        x = nn.functional.leaky_relu(self.drop_out_pre(self.head_pre(x)))
        x = nn.functional.leaky_relu(self.drop_out_inter(self.head_inter(x)))
        return self.head_out(x)

