from torch import nn

def freeze_params(model: nn.Module):
    """Set requires_grad=False for each of model.parameters()"""
    for par in model.parameters():
        par.requires_grad = False

def freeze_most_layer(model: nn.Module, config=None):
    """Set requires_grad=False for each of model.parameters()"""
    # 将模型中所有参数的 requires_grad 属性设置为 True
    trainable_list = config.model.trainable_parameters
    for name, param in model.named_parameters():
        if name in trainable_list:
            pass
        else:
            param.requires_grad = False


def unfreeze_adapter_params_encoder(model):
    for name, param in model.named_parameters():
        if ("adapter" in name or "mlp" in name or "param_gen" in name) and "encoder" in name:
            param.requires_grad = True

def unfreeze_adapter_params_llama(model):
    for name, param in model.named_parameters():
        if "adapter" in name:
            param.requires_grad = True

def unfreeze_adapter_params_decoder(model, config):
    try:
        fz_down = config.freeze_decoder_adapters_down
    except:
        fz_down = False
    try:
        fz_up = config.freeze_decoder_adapters_up
    except:
        fz_up = False

    if fz_down:
        for name, param in model.named_parameters():  #
            if ("param_gen" in name or "mlp" in name) and "decoder" in name:
                if "param_gen.decoder.weight_down" in name or "param_gen.decoder.bias_down" in name:
                    pass
                else:
                    param.requires_grad = True

    elif fz_up:
        for name, param in model.named_parameters():  #
            if ("param_gen" in name or "mlp" in name) and "decoder" in name:
                if "param_gen.decoder.weight_up" in name or "param_gen.decoder.bias_up" in name:
                    pass
                else:
                    param.requires_grad = True

    else:
        for name, param in model.named_parameters():   #
            if ("param_gen" in name or "mlp" in name) and "decoder" in name:
                param.requires_grad = True

    try:
        dec_type = config.decoder_adapter
    except:
        dec_type = None
    if dec_type == "generated":
        pass
    else:
        for name, param in model.named_parameters():
            if "adapter_layer" in name and "decoder" in name:
                param.requires_grad = True

    try:
        using_bias = config.unfreeze_decoder_adapters_bias
    except:
        using_bias = False

    if using_bias:
        for name, param in model.named_parameters():
            if "adapter_alpha" in name and "decoder" in name:
                param.requires_grad = True

def unfreeze_gated_cross_attention_layers_(model):
    for name, param in model.named_parameters():
        if "perceiver_resampler" in name:
            param.requires_grad = True

def unfreeze_perceiver_resampler_layers_(model):
    for name, param in model.named_parameters():
        if "xattn_stack" in name:
            param.requires_grad = True


def unfreeze_experts_layers_(model, config):
    for name, param in model.named_parameters():
        if "experts_for_context" in name:
            param.requires_grad = True
        if config.using_random_experts:
            if "expert_score_weight" in name:
                param.requires_grad = False


def unfreeze_vip_layers_(model):
    layer_all = ['prompt_embedding', 'vip_fc_in', 'vip_fc_out', 'sentence_encoder']
    for name, param in model.named_parameters():
        for part_name in layer_all:
            if part_name in name:
                param.requires_grad = True


def unfreeze_prefix_layers_(model):
    for name, param in model.named_parameters():
        if "prefix_layer" in name:
            param.requires_grad = True


def brackets_to_periods(name):
    return name.replace("[", ".").replace("]", "")


def param_subset(named_parameters, inner_names):
    param_dict = dict(named_parameters)
    return [param_dict[n] for n in inner_names]

def get_logits(x):
    return x.logits if hasattr(x, "logits") else x


def parent_module(model, pname):
    components = pname.split('.')
    parent = model

    for component in components[:-1]:
        if hasattr(parent, component):
            parent = getattr(parent, component)
        elif component.isdigit():
            parent = parent[int(component)]
        else:
            raise RuntimeError(f"Couldn't find child module {component}")

    if not hasattr(parent, components[-1]):
        raise RuntimeError(f"Couldn't find child module {components[-1]}")

    return parent