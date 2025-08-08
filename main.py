import torch.optim.adamw
from model import (
    E2VftModel,
    download_repo_from_hf,
    load_pretrained_model,
    get_pretrain_config
)
import torch
from train_utils.loader import get_dataloader
from train_utils.training_cfg import TrainingConfig
import traceback

local_dir = download_repo_from_hf()
pretrain_cfg = get_pretrain_config(download_dir=local_dir)
pretrain_state_dict = load_pretrained_model(download_dir= local_dir)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = E2VftModel( 
    num_classes = 4, 
    pretrain_cfg = pretrain_cfg,
    pretrain_state_dict = pretrain_state_dict
).to(torch.float16).to(device)

model = model.train()

traininig_config = TrainingConfig()

train_dl, test_dl = get_dataloader(training_config= traininig_config)

optimizer = torch.optim.AdamW(
    params=model.parameters(), 
    lr= traininig_config.learning_rate
)

try:
    for _ith, (inputs, labels) in enumerate(train_dl):
        inputs = {k: v.to(device) for k,v in inputs.items()}
        labels = labels.to(device)

        optimizer.zero_grad()
        predicts = model(**inputs)

        loss = torch.nn.functional.cross_entropy(predicts, labels)
        loss.backward()
        optimizer.step()

        if _ith == 10:
            print('end')
            break

except Exception as e:
    print(e)
    print(traceback.format_exc())