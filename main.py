from model import (
    E2VftModel,
    download_repo_from_hf,
    load_pretrained_model,
    get_pretrain_config
)
import torch
from train_utils.loader import get_dataloader
from train_utils.training_cfg import TrainingConfig

local_dir = download_repo_from_hf()
pretrain_cfg = get_pretrain_config(download_dir=local_dir)
pretrain_state_dict = load_pretrained_model(download_dir= local_dir)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = E2VftModel(
    head_dim = 100, 
    num_classes = 5, 
    pretrain_cfg = pretrain_cfg,
    pretrain_state_dict = pretrain_state_dict
).to(torch.float16).to(device)

traininig_config = TrainingConfig()

train_dl, test_dl = get_dataloader(training_config= traininig_config)

try:
    for inputs, labels in train_dl:
        inputs = {k: v.to(device) for k,v in inputs.items()}
        labels = labels.to(device)

        model(**inputs)

except Exception as e:
    print(e)