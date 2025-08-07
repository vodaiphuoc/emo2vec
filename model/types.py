import dataclasses
from typing import Optional
from typing import Type, TypeVar, Dict, Any

T = TypeVar('T')


class AudioDecoderConfig:
    decoder_dim:int
    decoder_groups:int
    decoder_kernel:int
    decoder_layers:int
    input_dropout:float
    add_positions_masked:bool
    add_positions_all:bool
    decoder_residual:bool
    projection_layers:float
    projection_ratio:float


@dataclasses.dataclass
class AudioConfig:
    type:str
    prenet_depth:int
    prenet_layerdrop:float
    prenet_dropout:float
    start_drop_path_rate:float
    end_drop_path_rate:float
    num_extra_tokens:float
    init_extra_token_zero:bool
    mask_noise_std:float
    mask_prob:float
    inverse_mask:bool
    mask_prob_adjust:float
    keep_masked_pct:float
    mask_length:float
    add_masks:bool
    remove_masks:bool
    mask_dropout:float
    encoder_zero_mask:bool
    mask_channel_prob:float
    mask_channel_length:float
    ema_local_encoder:bool
    local_grad_mult:float
    use_alibi_encoder:bool
    alibi_scale:float
    learned_alibi:bool
    
    learned_alibi_scale:bool
    learned_alibi_scale_per_head:bool
    learned_alibi_scale_per_layer:bool
    num_alibi_heads:int
    model_depth:int
    decoder: AudioDecoderConfig
    extractor_mode:str
    feature_encoder_spec:str
    conv_pos_width:int
    conv_pos_groups:int
    conv_pos_depth:int
    conv_pos_pre_ln:bool
    alibi_max_pos:float|None = None
    mask_prob_min:float|None = None
    

@dataclasses.dataclass
class ModalitiesConfig:
    audio: AudioConfig

@dataclasses.dataclass
class ModelConfig:
    loss_beta:float
    depth:int
    start_drop_path_rate:float
    end_drop_path_rate:float
    num_heads:int
    norm_eps:float
    norm_affine:bool
    encoder_dropout:float
    post_mlp_drop:float
    attention_dropout:float
    activation_dropout:float
    dropout_input:float
    layerdrop:float
    embed_dim:int
    mlp_ratio:float
    layer_norm_first:bool
    average_top_k_layers:int
    end_of_block_targets:bool
    clone_batch:int
    layer_norm_target_layer:bool
    batch_norm_target_layer:bool
    instance_norm_target_layer:bool
    instance_norm_targets:bool
    layer_norm_targets:bool
    ema_decay:float
    ema_same_dtype:bool
    log_norms:bool
    ema_end_decay:float
    ema_anneal_end_step:float
    ema_encoder_only:bool
    max_update:int
    extractor_mode:str
    min_target_var:float
    min_pred_var:float
    supported_modality:str
    mae_init:bool
    seed:int
    skip_ema:bool
    cls_loss:float
    recon_loss:float
    d2v_loss:float
    decoder_group:bool
    adversarial_training:bool
    adversarial_hidden_dim:float
    adversarial_weight:float
    cls_type:str
    normalize:bool
    modalities: ModalitiesConfig
    loss_scale:float|None = None
    shared_decoder:float|None = None
    vocab_size: int = -1

@dataclasses.dataclass
class MainConfig:
    model:float
    model_conf:ModelConfig


def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
    """
    Recursively initializes a dataclass from a dictionary, handling nested dataclasses.
    """
    field_values = {}
    for field in dataclasses.fields(cls):
        field_value = data.get(field.name)
        if field_value is not None:
            if dataclasses.is_dataclass(field.type):
                field_values[field.name] = from_dict(field.type, field_value)
            else:
                field_values[field.name] = field_value

    return cls(**field_values)
