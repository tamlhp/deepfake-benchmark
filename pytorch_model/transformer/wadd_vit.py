

"""
Wavelet model from thesis

"""


import torch
from torch import nn
from einops import rearrange
# from efficientnet_pytorch import EfficientNet
from pytorch_model.wavelet_model.model_wavelet_add import WaveletModel
import cv2
import re
# from utils import resize
import numpy as np
from torch import einsum
from random import randint
from pytorch_model.transformer.model_vit import Transformer

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


channel_dict = {0:3*4,1:16*4,2:64*4,3:128*4,4:128,5:128}

class WADDViT(nn.Module):
    def __init__(self,selected_block=5,
                 image_size=224,patch_size=4,num_classes=1,dim=1024,
                 depth=6,heads=8,mlp_dim=2048,
                 emb_dim=32, dim_head=64,dropout=0.15,emb_dropout=0.15):
        super().__init__()

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.mlp_dim = mlp_dim
        self.emb_dim = emb_dim
        self.dim_head = dim_head
        self.dropout_value = dropout
        self.emb_dropout = emb_dropout

        assert self.image_size % self.patch_size == 0, 'image dimensions must be divisible by the patch size'

        self.selected_block = selected_block

        self.wadd = WaveletModel(in_channel=3)

        # print(wadd)
        # print(wadd.pool)
        channels = channel_dict[self.selected_block]
        patch_dim = channels * self.patch_size ** 2

        self.patch_size = self.patch_size
        # print(patch_dim)
        # self.pos_embedding = nn.Parameter(torch.randn(self.emb_dim, 1, self.dim))
        num_patches = (self.image_size // self.patch_size) * (self.image_size // self.patch_size)

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, self.dim))
        self.patch_to_embedding = nn.Linear(patch_dim, self.dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.dim))
        self.dropout = nn.Dropout(self.emb_dropout)
        self.transformer = Transformer(self.dim, self.depth, self.heads, self.dim_head, self.mlp_dim, self.dropout_value)

        self.to_cls_token = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.Linear(self.dim, self.mlp_dim),
            nn.ReLU(),
            nn.Linear(self.mlp_dim, self.num_classes)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, img, mask=None):
        p = self.patch_size
        x = self.wadd.extract_features_at_block(img,self.selected_block)  # 1280x7x7
        # x = self.features(img)

        # print(x.size())
        y = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)
        # y2 = rearrange(x2, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = p, p2 = p)
        # print(y.size())
        y = self.patch_to_embedding(y)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, y), 1)
        # shape = x.shape[0]
        # x += self.pos_embedding[0:shape]
        shape = x.shape[1]
        x += self.pos_embedding[:, 0:shape]
        x = self.dropout(x)
        x = self.transformer(x)
        x = self.to_cls_token(x[:, 0])
        x = self.mlp_head(x)
        x = self.sigmoid(x)
        return x


if __name__ == "__main__":
    model = WADDViT(image_size=256,patch_size=2)
    import torchsummary
    torchsummary.summary(model,(3,256,256))

# python train.py --train_set ../df_in_the_wild/image_jpg/train/ --val_set ../df_in_the_wild/image_jpg/test/ --batch_size 24 --niter 20 --image_size 256 --workers 24 --checkpoint wadd_1_df_inthewild_checkpoint --gpu_id 0 --lr 1e-4 --print_every 20000 waddvit