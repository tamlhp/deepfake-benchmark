

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
from pytorch_model.transformer.cross_efficient_net import MultiScaleEncoder
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


channel_dict = {0:3*4,1:16*4,2:64*4,3:128*4,4:128,5:128}

class MultiresWADDViT(nn.Module):
    def __init__(self,selected_block_sm=5,selected_block_lg=2,
                 image_size=224,patch_size_sm=4,patch_size_lg=32,num_classes=1,dim=1024,
                 depth=6,heads=8,mlp_dim=2048,
                dim_head=64,dropout=0.15,emb_dropout=0.15,
                 cross_attn_depth=2,cross_attn_heads=8, cross_attn_dim_head=64
                 ):
        super().__init__()

        self.image_size = image_size
        self.patch_size_sm = patch_size_sm
        self.patch_size_lg = patch_size_lg
        self.num_classes = num_classes
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.mlp_dim = mlp_dim
        self.dim_head = dim_head
        self.dropout_value = dropout
        self.emb_dropout = emb_dropout
        self.cross_attn_depth = cross_attn_depth
        self.cross_attn_heads = cross_attn_heads
        self.cross_attn_dim_head = cross_attn_dim_head


        assert self.image_size % self.patch_size_sm == 0, 'image dimensions must be divisible by the patch size'
        assert self.image_size % self.patch_size_lg == 0, 'image dimensions must be divisible by the patch size'

        self.selected_block_sm = selected_block_sm
        self.selected_block_lg = selected_block_lg

        self.wadd = WaveletModel(in_channel=3)

        # print(wadd)
        # print(wadd.pool)
        channels_sm = channel_dict[self.selected_block_sm]
        patch_dim_sm = channels_sm * self.patch_size_sm ** 2
        channels_lg = channel_dict[self.selected_block_lg]
        patch_dim_lg = channels_lg * self.patch_size_lg ** 2
        num_patches_lg = (image_size // patch_size_lg) * (image_size // patch_size_lg)
        num_patches_sm = (image_size // patch_size_sm) * (image_size // patch_size_sm)

        # print(patch_dim)
        self.pos_embedding_sm = nn.Parameter(torch.randn(1, num_patches_sm + 1, self.dim))
        self.pos_embedding_lg = nn.Parameter(torch.randn(1, num_patches_lg + 1, self.dim))
        self.patch_to_embedding_sm = nn.Linear(patch_dim_sm, self.dim)
        self.patch_to_embedding_lg = nn.Linear(patch_dim_lg, self.dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.dim))
        self.dropout = nn.Dropout(self.emb_dropout)

        self.multi_scale_encoder = MultiScaleEncoder(
            depth=self.depth,
            sm_dim=self.dim,
            lg_dim=self.dim,
            cross_attn_heads=self.cross_attn_heads,
            cross_attn_dim_head=self.cross_attn_dim_head,
            cross_attn_depth=self.cross_attn_depth,
            sm_enc_params=dict(
                depth=self.depth,
                heads=self.heads,
                mlp_dim=self.mlp_dim,
                dim_head=self.dim_head
            ),
            lg_enc_params=dict(
                depth=self.depth,
                heads=self.heads,
                mlp_dim=self.mlp_dim,
                dim_head=self.dim_head
            ),
            dropout=self.dropout_value
        )

        self.to_cls_token = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.Linear(self.dim, self.mlp_dim),
            nn.ReLU(),
            nn.Linear(self.mlp_dim, self.num_classes)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, img, mask=None):
        x_sm = self.wadd.extract_features_at_block(img,self.selected_block_sm)  # 1280x7x7
        x_lg = self.wadd.extract_features_at_block(img,self.selected_block_lg)  #
        # x = self.features(img)

        # print(x.size())
        y_sm = rearrange(x_sm, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.patch_size_sm, p2=self.patch_size_sm)
        y_lg = rearrange(x_lg, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=self.patch_size_lg, p2=self.patch_size_lg)
        # y2 = rearrange(x2, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = p, p2 = p)
        # print(y.size())
        y_sm = self.patch_to_embedding_sm(y_sm)
        y_lg = self.patch_to_embedding_lg(y_lg)
        cls_tokens = self.cls_token.expand(x_sm.shape[0], -1, -1)
        x_sm = torch.cat((cls_tokens, y_sm), 1)
        x_lg = torch.cat((cls_tokens, y_lg), 1)
        shape_sm = x_sm.shape[1]
        shape_lg = x_lg.shape[1]
        x_sm += self.pos_embedding_sm[:,:shape_sm]
        x_lg += self.pos_embedding_lg[:,:shape_lg]
        x_sm = self.dropout(x_sm)
        x_lg = self.dropout(x_lg)


        sm_tokens, lg_tokens = self.multi_scale_encoder(x_sm, x_lg)
        sm_cls, lg_cls = map(lambda t: t[:, 0], (sm_tokens, lg_tokens))

        x_sm = self.mlp_head(sm_cls)
        x_lg = self.mlp_head(lg_cls)
        x = self.sigmoid(x_sm+ x_lg)
        return x


if __name__ == "__main__":
    model = MultiresWADDViT(image_size=256)
    import torchsummary
    torchsummary.summary(model,(3,256,256))

# python train.py --train_set ../df_in_the_wild/image_jpg/train/ --val_set ../df_in_the_wild/image_jpg/test/ --batch_size 24 --niter 20 --image_size 256 --workers 24 --checkpoint wadd_1_df_inthewild_checkpoint --gpu_id 0 --lr 1e-4 --print_every 20000 waddvit