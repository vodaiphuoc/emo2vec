from model.load_pretrained_model import download_weigth_from_hf


path= download_weigth_from_hf()

import torch
import os

with open(os.path.join(path, "emotion2vec_base.pt"),"r") as fp:
    torch.load(fp, map_location="cpu")