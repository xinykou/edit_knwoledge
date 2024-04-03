"""
adapt from https://github.com/lucidrains/flamingo-pytorch
"""


import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat, reduce
from einops_exts import rearrange_many, repeat_many


def exists(val):
    return val is not None

def FeedForward(dim, mult = 4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias = False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias = False)
    )

class PerceiverAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads  # 8
        inner_dim = dim_head * heads   # 512

        self.norm_media = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)   # 1024->512
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

    def forward(self, x, latents):
        """
        einstein notation
        b - batch
        t - time
        n - sequence
        d - dimension
        """
        x = self.norm_media(x)
        latents = self.norm_latents(latents)

        b, m, h = *x.shape[:2], self.heads

        q = self.to_q(latents)

        # the paper differs from Perceiver in which they also concat the key / values derived from the latents to be attended to
        kv_input = torch.cat((x, latents), dim = -2)   # todo: contexts 和 query拼接在一起作为 kv
        k, v = self.to_kv(kv_input).chunk(2, dim = -1)  # 根据最后一个维度分成 两块

        q, k, v = rearrange_many((q, k, v), 'b t n (h d) -> b h t n d', h = h)

        q = q * self.scale

        # attention

        sim = einsum('... i d, ... j d  -> ... i j', q, k)   # ... n n

        sim = sim - sim.amax(dim = -1, keepdim = True).detach()  # 取最后一个维度的最大值
        attn = sim.softmax(dim = -1)

        out = einsum('... i j, ... j d -> ... i d', attn, v)
        out = rearrange(out, 'b h t n d -> b t n (h d)', h = h)
        return self.to_out(out)

class PerceiverResampler(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        dim_head = 64,
        heads = 8,
        num_latents = 64,
        num_aug_sources = 4,
        ff_mult = 4
    ):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, dim))  # 可以学习的隐含 向量
        self.time_pos_emb = nn.Parameter(torch.randn(num_aug_sources, 1, dim))

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PerceiverAttention(dim = dim, dim_head = dim_head, heads = heads),
                FeedForward(dim = dim, mult = ff_mult)
            ]))

        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        if x.ndim == 3:
            x = rearrange(x, 'b n d -> b 1 n d')

        times = x.shape[1]  # 我们用了两个 contexts，所以这里是2，
        x = x + self.time_pos_emb[:times]  # 加入 time维度，也就是这是第几个证据 ： batch, t, seq_len, dim

        latents = repeat(self.latents, 'n d -> b m n d', b = x.shape[0], m = x.shape[1])

        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents

        return self.norm(latents)


# gated cross attention
class MaskedCrossAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8,
        only_attend_immediate_media = True
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

        # whether for text to only attend to immediate preceding image, or all images

        self.only_attend_immediate_media = only_attend_immediate_media

    def forward(
        self,
        x,
        media,
        media_locations = None,
        aug_exist_idx = None,
    ):
        b, t, m = media.shape[:3]
        # print("media: b, t, m:",b,t,m)
        h = self.heads

        x = self.norm(x)

        # print("x:", x.shape)

        q = self.to_q(x)
        
        # print("q:", q.shape)

        if len(media.shape) == 4:  # media 是 preceiver resampler输出的维度，多了一个time，是 因为为各个证据建立了不同的标志位
            media = rearrange(media, 'b t m d -> b (t m) d')   # 将新添加的维度 合并回去

        # print("media:", media.shape)

        k, v = self.to_kv(media).chunk(2, dim = -1)
        
        # print("k,v", k.shape, v.shape)
        
        q, k, v = rearrange_many((q, k, v), 'b n (h d) -> b h n d', h = h)
        
        # print("q,k,v:", q.shape, k.shape, v.shape)

        q = q * self.scale

        sim = einsum('... i d, ... j d -> ... i j', q, k)

        # print("sim:", sim.shape)

        if exists(media_locations):
            text_time = media_locations.cumsum(dim = -1) # at each boolean of True, increment the time counter (relative to media time)
            media_time = torch.arange(t, device = x.device) + 1

            # text time must equal media time if only attending to most immediate image
            # otherwise, as long as text time is greater than media time (if attending to all previous images / media)
            mask_op = torch.eq if self.only_attend_immediate_media else torch.ge

            text_to_media_mask = mask_op(rearrange(text_time, 'b i -> b 1 i 1'), repeat(media_time, 'j -> 1 1 1 (j m)', m = m))
            sim = sim.masked_fill(~text_to_media_mask, -torch.finfo(sim.dtype).max)

        # print("aug_exist_idx:", aug_exist_idx)

        if exists(aug_exist_idx):
            aug_exist_idx = aug_exist_idx.detach()
            # print(aug_exist_idx)
            expended_idx = torch.repeat_interleave(aug_exist_idx, m, dim = -1)
            # print("expended_idx:", expended_idx.shape)
            expended_idx = repeat(expended_idx, 'b d -> b h n d', b=b, h=h, n=sim.shape[2])
            assert tuple(expended_idx.shape) == tuple(sim.shape)
            # print("expended_idx:", expended_idx.shape)
            ones = torch.ones_like(expended_idx)
            attend_to_exist_aug_mask = torch.eq(ones, expended_idx)
            # print(attend_to_exist_aug_mask)
            # print("attend_to_exist_aug_mask:",attend_to_exist_aug_mask.shape)
            sim = sim.masked_fill(~attend_to_exist_aug_mask, -torch.finfo(sim.dtype).max)
            # print(attend_to_exist_aug_mask[0][0][0])
            # print(attend_to_exist_aug_mask[16][0][0])

        # print('============================================================')

        sim = sim - sim.amax(dim = -1, keepdim = True).detach()
        attn = sim.softmax(dim = -1)
        # print(sim[0][0][0])
        # print(sim[16][0][0])

        if exists(media_locations) and self.only_attend_immediate_media:
            # any text without a preceding media needs to have attention zeroed out
            text_without_media_mask = text_time == 0
            text_without_media_mask = rearrange(text_without_media_mask, 'b i -> b 1 i 1')
            attn.masked_fill(text_without_media_mask, 0.)

        out = einsum('... i j, ... j d -> ... i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class GatedCrossAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        only_attend_immediate_media = False # zhenhailong: by default attend to all retrieved examples
    ):
        super().__init__()
        self.attn = MaskedCrossAttention(dim = dim, dim_head = dim_head, heads = heads, only_attend_immediate_media = only_attend_immediate_media)
        self.attn_gate = nn.Parameter(torch.tensor([0.]))

        self.ff = FeedForward(dim, mult = ff_mult)
        self.ff_gate = nn.Parameter(torch.tensor([0.]))

    def forward(
        self,
        x,
        media,                  # media tensor, encoded by perceiver resample - (batch, time, latents, dim), 代表经过”Perceiver Resampler“模块的context中的信息
        media_locations = None,  # boolean tensor indicating positions of media - (batch, sequence)
        aug_exist_idx = None
    ):  #
        x = self.attn(x, media, media_locations = media_locations, aug_exist_idx = aug_exist_idx) * self.attn_gate.tanh() + x
        x = self.ff(x) * self.ff_gate.tanh() + x
        return x

class CrossAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        only_attend_immediate_media = False # zhenhailong: by default attend to all retrieved examples
    ):
        super().__init__()
        self.attn = MaskedCrossAttention(dim = dim, dim_head = dim_head, heads = heads, only_attend_immediate_media = only_attend_immediate_media)
        self.ff = FeedForward(dim, mult = ff_mult)

    def forward(
        self,
        x,
        media,                  # media tensor, encoded by perceiver resample - (batch, time, latents, dim)
        media_locations = None,  # boolean tensor indicating positions of media - (batch, sequence)
        aug_exist_idx = None
    ):
        x = self.attn(x, media, media_locations = media_locations, aug_exist_idx = aug_exist_idx) + x
        x = self.ff(x) + x
        return x


class Classify_for_adapter_Weight(nn.Module):
    def __init__(
        self,
        hidden_size,

    ):
        super().__init__()
        self.layer = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 2, bias=False),
            nn.ReLU(),

        )

    def forward(
        self,
        media               # media tensor, encoded by perceiver resample - (batch, time, latents, dim)
    ):
        batch, time, latents, dim = media.size()
        med = media.view(batch, -1, dim)
        mean_pooling = med.sum(1)
        x = self.layer(mean_pooling)
        return x

def set_module_requires_grad_(module, requires_grad):
    for param in module.parameters():
        param.requires_grad = requires_grad

def freeze_all_layers_(module):
    set_module_requires_grad_(module, False)

def unfreeze_all_layers_(module):
    set_module_requires_grad_(module, True)

def freeze_model_and_make_eval_(model):
    model.eval()
    freeze_all_layers_(model)

class combine(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = PerceiverResampler(
        dim = 4096,
        depth = 1,
        dim_head = 64,
        heads = 8,
        num_latents = 64,
        num_aug_sources = 4,
        ff_mult = 2
        )
        self.layer_2 = GatedCrossAttentionBlock(    # todo:
                    dim=4096,
                    dim_head=64,
                    heads=8,
                    ff_mult=4,
                    only_attend_immediate_media=False
                )

def main():
    model = combine()
    for name, param in model.named_parameters():
        if param.requires_grad:
            print("Parameter name %s", name)

    # 计算需要训练的参数数量
    total_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print()

if __name__ == "__main__":
    main()
