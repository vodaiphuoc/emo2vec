from collections import OrderedDict
from omegaconf import OmegaConf
import os

from huggingface_hub import snapshot_download
import torch

from .types import from_dict, ModelConfig


def download_repo_from_hf(repo_id:str="emotion2vec/emotion2vec_base")->str:
    return snapshot_download(
        repo_id=repo_id, 
        local_dir= os.path.dirname(__file__).replace("model","ckpt"), 
        ignore_patterns=["*wav","*.png"],
    )

def load_pretrained_model(download_dir: str)->OrderedDict:
    with open(os.path.join(download_dir, "emotion2vec_base.pt"),"rb") as fp:
        state_dict = torch.load(fp, map_location="cpu", weights_only= False)
    return state_dict['model']


def get_pretrain_config(download_dir:str)->ModelConfig:
    r"""
    Load pretrain config from hf repo
    """
    config_path = os.path.join(download_dir,"config.yaml")

    if not os.path.isfile(config_path):
        raise FileNotFoundError
    
    config = OmegaConf.load(config_path)
    return from_dict(ModelConfig,OmegaConf.to_object(config.get("model_conf")))