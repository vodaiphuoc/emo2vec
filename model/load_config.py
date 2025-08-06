from omegaconf import OmegaConf, DictConfig
import os

def get_pretrain_config()->DictConfig:
    r"""
    Load pretrain config from hf repo
    """
    config_path = os.path.join(os.path.dirname(__file__).replace("model","ckpt"),"emotion2vec_base","config.yaml")

    if not os.path.isfile(config_path):
        raise FileNotFoundError
    
    config = OmegaConf.load(config_path)
    
    return config.get("model_conf")