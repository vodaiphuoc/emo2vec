from model.load_pretrained_model import download_weigth_from_hf


path= download_weigth_from_hf()

import torch

with open(path,"r") as fp:
    torch.load(fp, map_location="cpu")