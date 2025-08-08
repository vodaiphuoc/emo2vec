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
        self.head = torch.nn.Linear(pretrain_cfg.embed_dim, num_classes)

    def forward(
            self,
            source,
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
        x = pretrain_outputs.x.mean(dim= 1)
        return self.head(x)

