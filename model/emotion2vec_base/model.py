import os
import time
import torch
import logging
import numpy as np
from functools import partial
import torch.nn.functional as F
import dataclasses

from .modules import AltBlock
from .audio import AudioEncoder
from ..utils import load_audio_text_image_video
from ..types import ModelConfig

logger = logging.getLogger(__name__)

@dataclasses.dataclass
class PretrainOutput:
    x: torch.Tensor
    padding_mask: torch.Tensor|None
    layer_results: torch.Tensor|None
    mask: torch.Tensor|None

class Emotion2vec(torch.nn.Module):
    def __init__(self, model_conf: ModelConfig):
        r"""
        Args:
            model_conf (ModelConfig): config of model
        """
        super().__init__()
        cfg = model_conf
        self.cfg = model_conf

        make_layer_norm = partial(
            torch.nn.LayerNorm, eps=cfg.norm_eps, elementwise_affine=cfg.norm_affine
        )

        def make_block(drop_path, dim=None, heads=None):
            return AltBlock(
                cfg.embed_dim if dim is None else dim,
                cfg.num_heads if heads is None else heads,
                cfg.mlp_ratio,
                qkv_bias=True,
                drop=cfg.encoder_dropout,
                attn_drop=cfg.attention_dropout,
                mlp_drop=cfg.activation_dropout,
                post_mlp_drop=cfg.post_mlp_drop,
                drop_path=drop_path,
                norm_layer=make_layer_norm,
                layer_norm_first=cfg.layer_norm_first,
                ffn_targets=not cfg.end_of_block_targets,
            )

        self.alibi_biases = {}
        self.feature_extractor = AudioEncoder(
            modality_cfg = cfg.modalities.audio,
            embed_dim = cfg.embed_dim,
            make_block = make_block,
            norm_layer = make_layer_norm,
            layer_norm_first = cfg.layer_norm_first,
            alibi_biases = self.alibi_biases,
        )
        

        self.ema = None

        self.average_top_k_layers = cfg.average_top_k_layers
        self.loss_beta = cfg.loss_beta
        self.loss_scale = cfg.loss_scale

        self.dropout_input = torch.nn.Dropout(cfg.dropout_input)

        dpr = np.linspace(
            cfg.start_drop_path_rate, cfg.end_drop_path_rate, cfg.depth
        )

        self.blocks = torch.nn.ModuleList([make_block(dpr[i]) for i in range(cfg.depth)])

        self.norm = None
        if cfg.layer_norm_first:
            self.norm = make_layer_norm(cfg.embed_dim)

        vocab_size = cfg.vocab_size
        
        self.proj = None
        if vocab_size > 0:
            self.proj = torch.nn.Linear(cfg.embed_dim, vocab_size)

    def forward(
            self,
            source,
            target=None,
            id=None,
            mode=None,
            padding_mask=None,
            mask=False,
            features_only=False,
            force_remove_masked=False,
            remove_extra_tokens=True,
            precomputed_mask=None,
            **kwargs,
        )->PretrainOutput:
        r"""
        Forward of pretrained model
        Returns:
            instant of `PretrainOutput`
        """
        mask_seeds = None

        # print('self.feature_extractor input, source shape: ', source.shape)
        extractor_out = self.feature_extractor(
            source,
            padding_mask,
            mask,
            remove_masked=not features_only or force_remove_masked,
            clone_batch=self.cfg.clone_batch if not features_only else 1,
            mask_seeds=mask_seeds,
            precomputed_mask=precomputed_mask,
        )

        x = extractor_out["x"]
        # print('self.feature_extractor, x shape: ', x.shape)
        encoder_mask = extractor_out["encoder_mask"]
        masked_padding_mask = extractor_out["padding_mask"]
        masked_alibi_bias = extractor_out.get("alibi_bias", None)
        alibi_scale = extractor_out.get("alibi_scale", None)

        if self.dropout_input is not None:
            x = self.dropout_input(x)

        layer_results = []
        for i, blk in enumerate(self.blocks):
            if (
                not self.training
                or self.cfg.layerdrop == 0
                or (np.random.random() > self.cfg.layerdrop)
            ):
                ab = masked_alibi_bias
                if ab is not None and alibi_scale is not None:
                    scale = alibi_scale[i] if alibi_scale.size(0) > 1 else alibi_scale.squeeze(0)
                    ab = ab * scale.type_as(ab)

                x, lr = blk(
                    x,
                    padding_mask=masked_padding_mask,
                    alibi_bias=ab,
                )
                # print('blk, x shape: ', x.shape)
                if features_only:
                    layer_results.append(lr)


        if self.norm is not None:
            x = self.norm(x)

        if features_only:
            if remove_extra_tokens:
                x = x[:, self.feature_extractor.modality_cfg.num_extra_tokens :]
                if masked_padding_mask is not None:
                    masked_padding_mask = masked_padding_mask[
                        :, self.feature_extractor.modality_cfg.num_extra_tokens :
                    ]

            return PretrainOutput(**{
                "x": x,
                "padding_mask": masked_padding_mask,
                "layer_results": layer_results,
                "mask": encoder_mask,
            })
        else:
            # print('features_only branch')
            return PretrainOutput(**{
                "x": x,
                "padding_mask": None,
                "layer_results": None,
                "mask": None,
            })

    def extract_features(
        self, source, mode=None, padding_mask=None, mask=False, remove_extra_tokens=True
    ):
        res = self.forward(
            source,
            mode=mode,
            padding_mask=padding_mask,
            mask=mask,
            features_only=True,
            remove_extra_tokens=remove_extra_tokens,
        )
        return res

    def inference(
        self,
        data_in,
        data_lengths=None,
        key: list = None,
        tokenizer=None,
        frontend=None,
        **kwargs,
    ):

        # if source_file.endswith('.wav'):
        #     wav, sr = sf.read(source_file)
        #     channel = sf.info(source_file).channels
        #     assert sr == 16e3, "Sample rate should be 16kHz, but got {}in file {}".format(sr, source_file)
        #     assert channel == 1, "Channel should be 1, but got {} in file {}".format(channel, source_file)
        granularity = kwargs.get("granularity", "utterance")
        extract_embedding = kwargs.get("extract_embedding", True)
        if self.proj is None:
            extract_embedding = True
        meta_data = {}
        # extract fbank feats
        time1 = time.perf_counter()
        audio_sample_list = load_audio_text_image_video(
            data_in,
            fs=16000,
            audio_fs=kwargs.get("fs", 16000),
            data_type=kwargs.get("data_type", "sound"),
            tokenizer=tokenizer,
        )

        time2 = time.perf_counter()
        meta_data["load_data"] = f"{time2 - time1:0.3f}"
        meta_data["batch_data_time"] = len(audio_sample_list[0]) / kwargs.get("fs", 16000)

        results = []
        output_dir = kwargs.output_dir
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        for i, wav in enumerate(audio_sample_list):
            source = wav.to(device=kwargs["device"])
            if self.cfg.normalize:
                source = F.layer_norm(source, source.shape)
            source = source.view(1, -1)

            feats = self.extract_features(source, padding_mask=None)
            x = feats["x"]
            feats = feats["x"].squeeze(0).cpu().numpy()
            if granularity == "frame":
                feats = feats
            elif granularity == "utterance":
                feats = np.mean(feats, axis=0)

            if output_dir and extract_embedding:
                np.save(os.path.join(output_dir, "{}.npy".format(key[i])), feats)

            labels = tokenizer.token_list if tokenizer is not None else []
            scores = []
            if self.proj:
                x = x.mean(dim=1)
                x = self.proj(x)
                for idx, lab in enumerate(labels):
                    x[:,idx] = -np.inf if lab.startswith("unuse") else x[:,idx]
                x = torch.softmax(x, dim=-1)
                scores = x[0].tolist()

            select_label = [lb for lb in labels if not lb.startswith("unuse")]
            select_score = [scores[idx] for idx, lb in enumerate(labels) if not lb.startswith("unuse")]

            # result_i = {"key": key[i], "labels": labels, "scores": scores}
            result_i = {"key": key[i], "labels": select_label, "scores": select_score}

            if extract_embedding:
                result_i["feats"] = feats
            results.append(result_i)

        return results, meta_data

    def export(self, **kwargs):
        from .export_meta import export_rebuild_model

        models = export_rebuild_model(model=self, **kwargs)
        return models
