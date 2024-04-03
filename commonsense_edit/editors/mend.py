import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import transformers
import higher
import logging
from higher.patch import monkeypatch as make_functional
from collections import defaultdict


from .hooks import hook_model
from .nn import IDMLP
from .utils import _inner_params

LOG = logging.getLogger(__name__)


def update_counter(x, m, s, k):
    new_m = m + (x - m) / k
    new_s = s + (x - m) * (x - new_m)

    return new_m, new_s


class GradientTransform(nn.Module):
    def __init__(self, x_dim: int, delta_dim: int, cfg, n_modes = None):
        super().__init__()

        self.x_dim = x_dim  # 调用 GradientTransform的函数把*s分成两部分，第一部分是输入维度， 768
        self.delta_dim = delta_dim  # 第二部分是 3072
        self.cfg = cfg  # 代表config.mend的参数
        if cfg.combine and (cfg.one_sided or cfg.x_only or cfg.delta_only):
            raise ValueError("cfg.combine cannot be used with one-sided MEND variants")

        self.norm_init = False
        self.register_buffer("u_mean", torch.full((x_dim,), float("nan")))
        self.register_buffer("v_mean", torch.full((delta_dim,), float("nan")))
        self.register_buffer("u_std", torch.full((x_dim,), float("nan")))
        self.register_buffer("v_std", torch.full((delta_dim,), float("nan")))
        self.register_buffer("u_s", torch.full((x_dim,), float("nan")))
        self.register_buffer("v_s", torch.full((delta_dim,), float("nan")))
        self.register_buffer("k", torch.full((1,), float("nan")))

        MlpClass = IDMLP  # 得到 nn.IDMLP
        LOG.info(f"Building Gradient Transform with MLP class {MlpClass}")

        def delta_net():
            return MlpClass(delta_dim, delta_dim, delta_dim * 2, cfg.n_hidden, init=cfg.init, act=cfg.act, rank=cfg.rank, n_modes=n_modes)

        def x_net():
            return MlpClass(x_dim, x_dim, x_dim * 2, cfg.n_hidden, init=cfg.init, act=cfg.act, rank=cfg.rank, n_modes=n_modes)

        def combined_net():  # 使用的是这个
            return MlpClass(delta_dim + x_dim, delta_dim + x_dim, (delta_dim + x_dim) * 2,
                            cfg.n_hidden, init=cfg.init, act=cfg.act, rank=cfg.rank, n_modes=n_modes)

        def ID():
            return lambda x, mode=None: x

        if cfg.combine:
            self.mlp = combined_net()    # todo: 执行这里
        elif cfg.one_sided:
            if x_dim > delta_dim:
                self.mlp1, self.mlp2 = ID(), delta_net()
            else:
                self.mlp1, self.mlp2 = x_net(), ID()
        elif cfg.x_only:
            self.mlp1, self.mlp2 = x_net(), ID()
        elif cfg.delta_only:
            self.mlp1, self.mlp2 = ID(), delta_net()
        else:
            self.mlp1, self.mlp2 = x_net(), delta_net()

    def forward(self, u, v, param_idx=None):
        u, v = u.to(torch.float32), v.to(torch.float32)  # u, v 代表 线性层的输入， 线性层的 输出位置的梯度

        u_ = u.view(-1, u.shape[-1])  # batch * 线性层的参数
        v_ = v.view(-1, v.shape[-1])  # batch * 输出位置梯度的参数

        nz_mask = (u_ != 0).any(-1) * (v_ != 0).any(-1)  # Skip batch elements with zero grad
        u_ = u_[nz_mask]
        v_ = v_[nz_mask]

        if self.training:
            for idx in range(u_.shape[0]):  # 循环样本数量
                if not self.norm_init:
                    self.u_mean = u_[idx].clone().detach()
                    self.v_mean = v_[idx].clone().detach()
                    self.u_s.zero_()  # 梯度值清零
                    self.v_s.zero_()  # 梯度值清零
                    self.k[:] = 1
                    self.norm_init = True
                else:
                    self.k += 1
                    self.u_mean, self.u_s = update_counter(u_[idx], self.u_mean, self.u_s, self.k)  # 这块没有想明白
                    self.v_mean, self.v_s = update_counter(v_[idx], self.v_mean, self.v_s, self.k)

            if self.k < 2:
                raise RuntimeError(f"Can't perform normalization with only {self.k} samples so far")
            self.u_std = (self.u_s / (self.k - 1)) ** 0.5  # 标准差
            self.v_std = (self.v_s / (self.k - 1)) ** 0.5

        if self.cfg.norm:
            u_input = (u_ - self.u_mean) / (self.u_std + 1e-7)  # 正则化
            v_input = (v_ - self.v_mean) / (self.v_std + 1e-7)
        else:
            u_input = u_
            v_input = v_

        if self.cfg.combine:
            output = self.mlp(torch.cat((u_input, v_input), -1), mode=param_idx)  # todo: 真正输入编辑器的过程计算
            out1, out2 = output.split([u.shape[-1], v.shape[-1]], -1)
            return out1, out2
        else:
            return self.mlp1(u_input, mode=param_idx), self.mlp2(v_input, mode=param_idx)


class EditableModel(nn.Module):
    def __init__(self, model, config, model_constructor):
        super().__init__()

        self.pure_model = model  # 原来那个我们不更更新参数的模型
        self.config = config
        self.model_constructor = model_constructor  # 原来那个我们不更更新参数的模型 的 “副本”


class MEND(EditableModel):
    def get_shape(self, p):
        # We need to flip the shapes since OpenAI gpt2 uses convs instead of linear
        return p.shape if isinstance(self.pure_model, transformers.GPT2LMHeadModel) else (p.shape[1], p.shape[0])

    def __init__(self, model, config, model_constructor, mend=None, edit_lrs=None):
        super().__init__(model, config, model_constructor)

        if edit_lrs is None:  # todo: edit 每层的一个学习率初始化
            edit_lrs = nn.Parameter(torch.tensor([config.editor.init_edit_lr] * len(self.config.editor.inner_params)))
        self.edit_lrs = edit_lrs

        if not hasattr(self.pure_model, "handles"):   # todo: ！！！！----记录前向传播的输入/输出，反向传播的输入输出梯度---- ！！！！！
            hook_model(self.pure_model, self.config.editor.inner_params)  # 钩子 用于观察模块的输入输出
            LOG.info(f"Hooked {len(self.pure_model.handles)//2} modules")

        if config.editor.mend.shared:  # todo:统计每个形状的线性层的数----key:层形状, value:层名字
            shape_dict = defaultdict(list)
            for n, p in _inner_params(model.named_parameters(), self.config.editor.inner_params):
                shape_dict[self.get_shape(p)].append(n)
            self.shape_dict = shape_dict

        if mend is None:
            if not config.editor.mend.shared:
                self.mend = nn.ModuleDict({
                    n.replace(".", "#"): GradientTransform(*self.get_shape(p), config.editor.mend)
                    for (n, p) in _inner_params(model.named_parameters(), self.config.model.inner_params)
                })
            else:  # todo: 得到的是“编辑器”的结构
                self.mend = nn.ModuleDict({    # disilgpt2执行这里
                    str(tuple(s)): GradientTransform(*s, config.editor.mend, len(shape_dict[s]))
                    for s in shape_dict.keys()
                })
        else:
            self.mend = mend

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        state_dict = super().state_dict(prefix=prefix, keep_vars=keep_vars)  # Get default state dict
        model_keys = self.pure_model.state_dict(prefix=prefix, keep_vars=keep_vars).keys()  # Remove model params
        for k in model_keys:
            del state_dict[f"model.{k}"]
        state_dict["model_config"] = self.pure_model.config  # Include model config
        return state_dict

    def load_state_dict(self, state_dict, strict: bool = True):
        config = state_dict["model_config"]
        del state_dict["model_config"]
        if config != self.pure_model.config:
            LOG.info("Loaded model config doesn't match current model config.")
            LOG.info(f"Loaded: {config}")
            LOG.info(f"Current: {self.pure_model.config}")

        res = super().load_state_dict(state_dict, False)
        # We should only have missing keys for the model, and no unexpected keys
        assert len([k for k in res.missing_keys if not k.startswith("model.")]) == 0, "Should only have missing keys for model."
        assert len(res.unexpected_keys) == 0, "Shouldn't have any unexpected keys"
        return res

    def outer_parameters(self):
        P = list(self.mend.parameters()) + [self.edit_lrs]
        # 计算需要训练的参数数量
        total_parameters = sum(p.numel() for p in P if p.requires_grad)
        return list(self.mend.parameters()) + [self.edit_lrs]  # 编辑器的参数 + 编辑器每次的那个学习率

    def editing(self, batch, condition=None, detach_history=False):
        outputs = self.pure_model(**batch)  # 输出 batch * seq_len * voc_size
        loss = outputs.loss
        pset = set(self.config.editor.inner_params)  # 代表 要权重被mend更新的那些层的参数
        loss.backward()   # 梯度回传给不更新的那个模型
        if self.config.editor.mend.shared:  # todo: 这里执行 mend 网络
            param_idx = lambda n, p: self.shape_dict[self.get_shape(p)].index(n) if self.config.editor.mend.shared else None  # 代表该层名是该形状的 第几个
            transformed_factors = {  # param_idx 代表该尺寸下的第几个
                n: self.mend[str(tuple(self.get_shape(p)))](p.__x__, p.__delta__, param_idx(n, p))  # 使用init 中 hook_model函数可以得到“__x__”， “__delta__”,
                for n, p in _inner_params(self.pure_model.named_parameters(), self.config.editor.inner_params)  # n, p 分别是“权重名” 和 对应的“权重矩阵”
            }
        else:
            transformed_factors = {
                n: self.mend[n.replace(".", "#")](p.__x__, p.__delta__)
                for n, p in _inner_params(self.pure_model.named_parameters(), self.config.model.inner_params)
            }

        # Should be bi,bj->ji for nn.Linear, but GPT2 uses Conv1d instead...
        if isinstance(self.pure_model, transformers.GPT2LMHeadModel):
            targ = "ij"
        else:
            targ = "ji"
        mean_grads = {
            n: torch.einsum(f"bi,bj->{targ}", x, delta)  # todo: 经过超网络获取的(b,i).T * (b,j) -->(i,j) 利用矩阵乘法， 得到 权重需要更新的增量
            for n, (x, delta) in transformed_factors.items()
        }

        self.pure_model.zero_grad()  # todo: 模型的梯度 清零了， 也就是该批次下完成了梯度的记录
        assert len(self.edit_lrs) == len(list(mean_grads.items()))
        updates = {n: lr * g for lr, (n, g) in zip(self.edit_lrs, mean_grads.items())}  # todo: Alpha，用学习率表示每层的可学习参数

        edited_model = self.pure_model
        if not isinstance(edited_model, higher.patch._MonkeyPatchBase):
            # edited_model = make_functional(edited_model, in_place=True)
            edited_model = make_functional(edited_model)  # todo: 这里作用是 ???

        new_params = []
        for n, p in edited_model.named_parameters():
            if n in pset:
                new_params.append(p + updates[n])  # todo: 那个确定的线性层更新，加或者减法都可以，因为 "MEND编辑器"输出可正可负
            else:
                new_params.append(p)

        edited_model.update_params(new_params)  # todo: 调用的哪里更新 “模型” ？？？，
        # 编辑后的模型返回更新后的结果
        return MEND(edited_model, self.config, self.model_constructor, self.mend, edit_lrs=self.edit_lrs)


if __name__ == '__main__':
    import types

    model = transformers.GPT2LMHeadModel.from_pretrained("gpt2")

    config = types.SimpleNamespace()
    config.model.inner_params = [
        "transformer.h.9.mlp.c_fc.weight",
        "transformer.h.9.mlp.c_proj.weight",
        "transformer.h.10.mlp.c_fc.weight",
        "transformer.h.10.mlp.c_proj.weight",
        "transformer.h.11.mlp.c_fc.weight",
        "transformer.h.11.mlp.c_proj.weight",
    ]
    config.edit_lr = 0.0001

    config.mend = types.SimpleNamespace()
    config.mend.n_hidden = 1
    config.mend = config.mend.__dict__

    mend = MEND(model, config, lambda: copy.deepcopy(model)).cuda()
    import pdb; pdb.set_trace()
    mend.load_state_dict(torch.load("test_state.pt"))
    x = torch.arange(20).view(1, 20).cuda() + 1000
    orig_logits = mend(x)
    edited = mend.edit(x, masks=torch.ones_like(x), labels=x)
    post_logits = mend(x)

    assert torch.allclose(orig_logits, post_logits)

    orig_param = [p for (n, p) in mend.model.named_parameters() if n == config.model.inner_params[-1]][0]
    edited_param = [p for (n, p) in edited.model.named_parameters() if n == config.model.inner_params[-1]][0]

    LOG.info((orig_param - edited_param).abs().max())
    edited.eval()
    LOG.info(mend(x, labels=x).loss, edited(x, labels=x).loss, edited.edit_loss_fn(edited(x).logits, x)["nll"])
    edited2 = edited.edit(x, masks=torch.ones_like(x), labels=x)
    LOG.info(mend(x, labels=x).loss, edited(x, labels=x).loss, edited2(x, labels=x).loss)
