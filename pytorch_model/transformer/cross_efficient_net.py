

"""
https://github.com/davide-coccomini/Combining-EfficientNet-and-Vision-Transformers-for-Video-Deepfake-Detection/blob/main/cross-efficient-vit/cross_efficient_vit.py

Coccomini, Davide, Nicola Messina, Claudio Gennaro, and Fabrizio Falchi.
"Combining efficientnet and vision transformers for video deepfake detection."
arXiv preprint arXiv:2107.02612 (2021).



"""


import torch
from torch import nn, einsum
import torch.nn.functional as F
import cv2
import numpy as np
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from pytorch_model.efficientnet import EfficientNet


# helpers

def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


# pre-layernorm

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


# feedforward

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


# attention

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, kv_include_self=False):
        b, n, _, h = *x.shape, self.heads
        context = default(context, x)

        if kv_include_self:
            context = torch.cat((x, context),
                                dim=1)  # cross attention requires CLS token includes itself as key / value

        qkv = (self.to_q(x), *self.to_kv(context).chunk(2, dim=-1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


# transformer encoder, for small and large patches

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


# projecting CLS tokens, in the case that small and large patch tokens have different dimensions

class ProjectInOut(nn.Module):
    def __init__(self, dim_in, dim_out, fn):
        super().__init__()
        self.fn = fn

        need_projection = dim_in != dim_out
        self.project_in = nn.Linear(dim_in, dim_out) if need_projection else nn.Identity()
        self.project_out = nn.Linear(dim_out, dim_in) if need_projection else nn.Identity()

    def forward(self, x, *args, **kwargs):
        x = self.project_in(x)
        x = self.fn(x, *args, **kwargs)
        x = self.project_out(x)
        return x


# cross attention transformer

class CrossTransformer(nn.Module):
    def __init__(self, sm_dim, lg_dim, depth, heads, dim_head, dropout):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                ProjectInOut(sm_dim, lg_dim,
                             PreNorm(lg_dim, Attention(lg_dim, heads=heads, dim_head=dim_head, dropout=dropout))),
                ProjectInOut(lg_dim, sm_dim,
                             PreNorm(sm_dim, Attention(sm_dim, heads=heads, dim_head=dim_head, dropout=dropout)))
            ]))

    def forward(self, sm_tokens, lg_tokens):
        (sm_cls, sm_patch_tokens), (lg_cls, lg_patch_tokens) = map(lambda t: (t[:, :1], t[:, 1:]),
                                                                   (sm_tokens, lg_tokens))

        for sm_attend_lg, lg_attend_sm in self.layers:
            sm_cls = sm_attend_lg(sm_cls, context=lg_patch_tokens, kv_include_self=True) + sm_cls
            lg_cls = lg_attend_sm(lg_cls, context=sm_patch_tokens, kv_include_self=True) + lg_cls

        sm_tokens = torch.cat((sm_cls, sm_patch_tokens), dim=1)
        lg_tokens = torch.cat((lg_cls, lg_patch_tokens), dim=1)
        return sm_tokens, lg_tokens


# multi-scale encoder

class MultiScaleEncoder(nn.Module):
    def __init__(
            self,
            *,
            depth,
            sm_dim,
            lg_dim,
            sm_enc_params,
            lg_enc_params,
            cross_attn_heads,
            cross_attn_depth,
            cross_attn_dim_head=64,
            dropout=0.
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Transformer(dim=sm_dim, dropout=dropout, **sm_enc_params),
                Transformer(dim=lg_dim, dropout=dropout, **lg_enc_params),
                CrossTransformer(sm_dim=sm_dim, lg_dim=lg_dim, depth=cross_attn_depth, heads=cross_attn_heads,
                                 dim_head=cross_attn_dim_head, dropout=dropout)
            ]))

    def forward(self, sm_tokens, lg_tokens):
        for sm_enc, lg_enc, cross_attend in self.layers:
            sm_tokens, lg_tokens = sm_enc(sm_tokens), lg_enc(lg_tokens)
            sm_tokens, lg_tokens = cross_attend(sm_tokens, lg_tokens)

        return sm_tokens, lg_tokens


# patch-based image to token embedder

class ImageEmbedder(nn.Module):
    def __init__(
            self,
            *,
            dim,
            image_size,
            patch_size,
            dropout=0.,
            efficient_block=8,
            channels
    ):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        self.efficient_net = EfficientNet.from_pretrained('efficientnet-b0')
        # self.efficient_net.delete_blocks(efficient_block)
        self.efficient_block = efficient_block

        for index, (name, param) in enumerate(self.efficient_net.named_parameters()):
            param.requires_grad = True

        self.patch_size = patch_size
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(dropout)

    def forward(self, img):
        x = self.efficient_net.extract_features_at_block(img, self.efficient_block)
        '''
        x_scaled = []
        for idx, im in enumerate(x):
            im = im.cpu().detach().numpy()
            for patch_idx, patch in enumerate(im):
                patch = 2.*(patch - np.min(patch))/np.ptp(patch)-1
                im[patch_idx] = patch

            x_scaled.append(im)
        x = torch.tensor(x_scaled).cuda()    
        '''
        # x = torch.tensor(x).cuda()
        '''
        for idx, im in enumerate(x):
            im = im.cpu().detach().numpy()
            for patch_idx, patch in enumerate(im):
                cv2.imwrite("patches/patches_"+str(idx)+"_"+str(patch_idx)+".png", patch)
        '''
        x = self.to_patch_embedding(x)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]

        return self.dropout(x)


# cross ViT class

class CrossEfficientViT(nn.Module):
    def __init__(
            self,
            *,
            image_size=224,num_classes=1,sm_dim=192,sm_channels=1280,lg_dim=384,
            lg_channels=24, sm_patch_size=7,sm_enc_depth=2,sm_enc_heads=8,
            sm_enc_mlp_dim=2048,sm_enc_dim_head=64,lg_patch_size=56,lg_enc_depth=3,
            lg_enc_mlp_dim=2048,lg_enc_heads=8,lg_enc_dim_head=64,cross_attn_depth=2,
            cross_attn_heads=8,cross_attn_dim_head=64, depth=4, dropout=0.15, emb_dropout=0.15
    ):
        super().__init__()

        self.image_size = image_size
        self.num_classes = num_classes
        self.sm_dim = sm_dim
        self.sm_channels = sm_channels
        self.lg_dim = lg_dim
        self.lg_channels = lg_channels
        self.sm_patch_size = sm_patch_size
        self.sm_enc_depth = sm_enc_depth
        self.sm_enc_heads = sm_enc_heads
        self.sm_enc_mlp_dim = sm_enc_mlp_dim
        self.sm_enc_dim_head = sm_enc_dim_head
        self.lg_patch_size = lg_patch_size
        self.lg_enc_depth = lg_enc_depth
        self.lg_enc_mlp_dim = lg_enc_mlp_dim
        self.lg_enc_heads = lg_enc_heads
        self.lg_enc_dim_head = lg_enc_dim_head
        self.cross_attn_depth = cross_attn_depth
        self.cross_attn_heads = cross_attn_heads
        self.cross_attn_dim_head = cross_attn_dim_head
        self.depth = depth
        self.dropout_value = dropout
        self.emb_dropout = emb_dropout

        self.sm_image_embedder = ImageEmbedder(dim=self.sm_dim, image_size=self.image_size, patch_size=self.sm_patch_size,
                                               dropout=self.emb_dropout, efficient_block=16, channels=self.sm_channels)
        self.lg_image_embedder = ImageEmbedder(dim=self.lg_dim, image_size=self.image_size, patch_size=self.lg_patch_size,
                                               dropout=self.emb_dropout, efficient_block=1, channels=self.lg_channels)

        self.multi_scale_encoder = MultiScaleEncoder(
            depth=self.depth,
            sm_dim=self.sm_dim,
            lg_dim=self.lg_dim,
            cross_attn_heads=self.cross_attn_heads,
            cross_attn_dim_head=self.cross_attn_dim_head,
            cross_attn_depth=self.cross_attn_depth,
            sm_enc_params=dict(
                depth=self.sm_enc_depth,
                heads=self.sm_enc_heads,
                mlp_dim=self.sm_enc_mlp_dim,
                dim_head=self.sm_enc_dim_head
            ),
            lg_enc_params=dict(
                depth=self.lg_enc_depth,
                heads=self.lg_enc_heads,
                mlp_dim=self.lg_enc_mlp_dim,
                dim_head=self.lg_enc_dim_head
            ),
            dropout=self.dropout_value
        )

        self.sm_mlp_head = nn.Sequential(nn.LayerNorm(self.sm_dim), nn.Linear(self.sm_dim, self.num_classes))
        self.lg_mlp_head = nn.Sequential(nn.LayerNorm(self.lg_dim), nn.Linear(self.lg_dim, self.num_classes))
        self.sigmoid = nn.Sigmoid()

    def forward(self, img):
        sm_tokens = self.sm_image_embedder(img)
        lg_tokens = self.lg_image_embedder(img)

        sm_tokens, lg_tokens = self.multi_scale_encoder(sm_tokens, lg_tokens)

        sm_cls, lg_cls = map(lambda t: t[:, 0], (sm_tokens, lg_tokens))

        sm_logits = self.sm_mlp_head(sm_cls)
        lg_logits = self.lg_mlp_head(lg_cls)
        x = self.sigmoid(sm_logits + lg_logits)
        return x



if __name__ == "__main__":
    model = CrossEfficientViT()
    import torchsummary
    torchsummary.summary(model,(3,224,224))


