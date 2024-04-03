from utils import parent_module


def linear_backward_hook(mod, grad_in, grad_out):
    if not hasattr(mod, "weight"):
        print(f"{mod} has no weight!")
        return

    if hasattr(mod.weight, "__x__"):
        assert len(grad_out) == 1
        # mod.weight.__bgrad__ = grad_out[0].unsqueeze(-1) * mod.__x__[0].unsqueeze(-2)
        mod.weight.__delta__ = grad_out[0].detach()  # 代表线性层的输出位置的 “梯度”
    else:
        print(f"{mod} has no __x__")


def linear_forward_hook(mod, activations, output):
    assert len(activations) == 1
    mod.weight.__x__ = activations[0].detach()  # activations[0] 代表 这个线性层的输入，

# 用于几个模型层前向/反向传播的参数
def hook_model(model, pnames):
    handles = []
    for m in [parent_module(model, pname) for pname in pnames]:  # parent_module(model, pname) 模块的返回上一级
        handles.append(m.register_full_backward_hook(linear_backward_hook))  # 存储反向传播的 输入梯度/输出梯度
        handles.append(m.register_forward_hook(linear_forward_hook))  # 存储 前向传播的输入/输出

    model.handles = handles  # model 就是 .model
