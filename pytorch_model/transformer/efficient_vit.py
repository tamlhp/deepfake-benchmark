

"""
https://github.com/davide-coccomini/Combining-EfficientNet-and-Vision-Transformers-for-Video-Deepfake-Detection/blob/main/efficient-vit/efficient_vit.py

Coccomini, Davide, Nicola Messina, Claudio Gennaro, and Fabrizio Falchi.
"Combining efficientnet and vision transformers for video deepfake detection."
arXiv preprint arXiv:2107.02612 (2021).



"""


import torch
from torch import nn
from einops import rearrange
# from efficientnet_pytorch import EfficientNet
from pytorch_model.efficientnet import EfficientNet
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




class EfficientViT(nn.Module):
    def __init__(self,channels=1280, selected_efficient_net=0,
                 image_size=224,patch_size=7,num_classes=1,dim=1024,
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

        self.selected_efficient_net = selected_efficient_net

        if selected_efficient_net == 0:
            self.efficient_net = EfficientNet.from_pretrained('efficientnet-b0')
        else:
            self.efficient_net = EfficientNet.from_pretrained('efficientnet-b7')
            checkpoint = torch.load("weights/final_999_DeepFakeClassifier_tf_efficientnet_b7_ns_0_23",
                                    map_location="cpu")
            state_dict = checkpoint.get("state_dict", checkpoint)
            self.efficient_net.load_state_dict({re.sub("^module.", "", k): v for k, v in state_dict.items()},
                                               strict=False)

        for i in range(0, len(self.efficient_net._blocks)):
            for index, param in enumerate(self.efficient_net._blocks[i].parameters()):
                if i >= len(self.efficient_net._blocks) - 3:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        num_patches = (7 // self.patch_size) ** 2
        patch_dim = channels * self.patch_size ** 2

        self.patch_size = self.patch_size
        # print(patch_dim)
        self.pos_embedding = nn.Parameter(torch.randn(self.emb_dim, 1, self.dim))
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
        x = self.efficient_net.extract_features(img)  # 1280x7x7
        # x = self.features(img)
        '''
        for im in img:
            image = im.cpu().detach().numpy()
            image = np.transpose(image, (1,2,0))
            cv2.imwrite("images/image"+str(randint(0,1000))+".png", image)

        x_scaled = []
        for idx, im in enumerate(x):
            im = im.cpu().detach().numpy()
            for patch_idx, patch in enumerate(im):
                patch = (255*(patch - np.min(patch))/np.ptp(patch)) 
                im[patch_idx] = patch
                #cv2.imwrite("patches/patches_"+str(idx)+"_"+str(patch_idx)+".png", patch)
            x_scaled.append(im)
        x = torch.tensor(x_scaled).cuda()   
        '''

        # x2 = self.features(img)
        # print(x.size())
        y = rearrange(x, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)
        # y2 = rearrange(x2, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = p, p2 = p)
        # print(y.size())
        y = self.patch_to_embedding(y)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, y), 1)
        shape = x.shape[0]
        x += self.pos_embedding[0:shape]
        x = self.dropout(x)
        x = self.transformer(x)
        x = self.to_cls_token(x[:, 0])
        x = self.mlp_head(x)
        x = self.sigmoid(x)
        return x


if __name__ == "__main__":
    model = EfficientViT(image_size=256,patch_size=2 )
    import torchsummary
    torchsummary.summary(model,(3,256,256))

# python train.py --train_set ../df_in_the_wild/image_jpg/train/ --val_set ../df_in_the_wild/image_jpg/test/ --batch_size 24 --niter 20 --image_size 224 --workers 24 --checkpoint efficientvit_df_inthewild_checkpoint --gpu_id 0 --lr 3e-4 --print_every 20000 efficientvit

