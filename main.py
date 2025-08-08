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
).to(torch.float32).to(device)


traininig_config = TrainingConfig()

train_dl, test_dl = get_dataloader(training_config= traininig_config)

optimizer = torch.optim.AdamW(
    params=model.parameters(), 
    lr= traininig_config.learning_rate
)

for _epoch in range(traininig_config.num_epochs):
    mean_train_loss = 0.0
    mean_train_acc = 0.0
    model = model.train()
    for _ith, (inputs, labels) in enumerate(train_dl):
        inputs = {k: v.to(device) for k,v in inputs.items()}
        labels = labels.to(device)

        optimizer.zero_grad()
        predicts = model(**inputs)

        print('predicts: ', predicts)

        loss = torch.nn.functional.cross_entropy(predicts, labels)
        loss.backward()
        optimizer.step()

        mean_train_loss += loss.item()
        mean_train_acc += len(torch.where(torch.argmax(predicts, dim= -1) == labels)[0])

    mean_train_loss /= len(train_dl)
    mean_train_acc /= len(train_dl)

    mean_test_loss = 0.0
    mean_test_acc = 0.0
    with torch.no_grad():
        model = model.eval()
        for _ith, (val_inputs, val_labels) in enumerate(test_dl):
            val_inputs = {k: v.to(device) for k,v in val_inputs.items()}
            val_labels = val_labels.to(device)

            val_predicts = model(**val_inputs)

            val_loss = torch.nn.functional.cross_entropy(val_predicts, val_labels)
            mean_test_loss += val_loss.item()
            mean_test_acc += len(torch.where(torch.argmax(val_predicts, dim= -1) == val_labels)[0])

        mean_test_loss /= len(test_dl)
        mean_test_acc /= len(test_dl)

    # report

    msg = f"""
epoch: {_epoch+1}
training:
    loss: {mean_train_loss}
    acc: {mean_train_acc}
validation:
    loss: {mean_test_loss}
    acc: {mean_test_acc}
"""
    print(msg)