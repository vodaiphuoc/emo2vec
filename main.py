from model import (
    E2VftModel,
    download_repo_from_hf,
    load_pretrained_model,
    get_pretrain_config
)


local_dir = download_repo_from_hf()
pretrain_cfg = get_pretrain_config()
pretrain_state_dict = load_pretrained_model()


E2VftModel(
    head_dim = 100, 
    num_classes = 5, 
    pretrain_cfg = pretrain_cfg,
    pretrain_state_dict = pretrain_state_dict
)