# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch.nn as nn

from vllm.config import VllmConfig, replace
from vllm.distributed.parallel_state import get_pp_group
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model
from vllm.v1.worker.gpu.spec_decode.eagle.utils import (
    _should_share,
    get_target_lm_head,
)

logger = init_logger(__name__)


def _has_materialized_weight(module: nn.Module | None) -> bool:
    return module is not None and getattr(module, "weight", None) is not None


def _validate_pp_placement(target_inner: nn.Module, layer_ids: tuple[int, ...]) -> None:
    start_layer = getattr(target_inner, "start_layer", None)
    end_layer = getattr(target_inner, "end_layer", None)
    if start_layer is None or end_layer is None:
        raise RuntimeError("DSpark+PP target does not expose its local layer range")
    missing = [i for i in layer_ids if not start_layer <= i < end_layer]
    if missing:
        raise RuntimeError(
            "DSpark+PP requires every dspark_target_layer_id on the last PP "
            f"stage; local range is [{start_layer}, {end_layer}), missing {missing}. "
            "Adjust VLLM_PP_LAYER_PARTITION."
        )
    logger.info(
        "DSpark drafter placement: last PP stage owns target layers [%d, %d) "
        "and DSpark taps %s",
        start_layer,
        end_layer,
        layer_ids,
    )


def load_dspark_model(target_model: nn.Module, vllm_config: VllmConfig) -> nn.Module:
    speculative_config = vllm_config.speculative_config
    assert speculative_config is not None
    draft_model_config = speculative_config.draft_model_config

    from vllm.compilation.backends import set_model_tag
    from vllm.model_executor.models.qwen3_dflash import dflash_has_any_non_causal
    from vllm.model_executor.models.utils import get_draft_quant_config

    # None re-runs backend auto-selection for the draft, which can pick a
    # different attention class than the target; fall back to the target's.
    draft_attention_backend = (
        speculative_config.attention_backend or vllm_config.attention_config.backend
    )

    draft_vllm_config = replace(
        vllm_config,
        attention_config=replace(
            vllm_config.attention_config,
            use_non_causal=dflash_has_any_non_causal(draft_model_config.hf_config),
            backend=draft_attention_backend,
        ),
        cache_config=(
            replace(
                vllm_config.cache_config,
                cache_dtype=speculative_config.kv_cache_dtype,
            )
            if speculative_config.kv_cache_dtype is not None
            else vllm_config.cache_config
        ),
    )
    # VllmConfig post-init restores the target's quant config because the target
    # config is retained for DSpark's target-layer metadata, so we must override it.
    draft_vllm_config.quant_config = get_draft_quant_config(vllm_config)

    with set_model_tag("dspark_head"):
        draft_model = get_model(
            vllm_config=draft_vllm_config, model_config=draft_model_config
        )

    pp_group = get_pp_group()
    is_pp = pp_group.world_size != 1
    if is_pp and not pp_group.is_last_rank:
        raise RuntimeError("DSpark drafter must be instantiated on the last PP stage")

    target_language_model = (
        target_model.get_language_model()
        if hasattr(target_model, "get_language_model")
        else target_model
    )
    target_inner = target_language_model.model
    draft_inner = draft_model.model
    if is_pp:
        layer_ids = tuple(draft_model_config.hf_config.dspark_target_layer_ids)
        _validate_pp_placement(target_inner, layer_ids)

    target_embed = getattr(target_inner, "embed_tokens", None)
    draft_embed = getattr(draft_inner, "embed_tokens", None)
    if _has_materialized_weight(target_embed) and _should_share(
        draft_model, "has_own_embed_tokens", draft_embed, target_embed
    ):
        if draft_embed is not None:
            del draft_inner.embed_tokens
        draft_inner.embed_tokens = target_embed

    target_lm_head = get_target_lm_head(target_model, target_language_model)
    draft_lm_head = getattr(draft_model, "lm_head", None)
    if _has_materialized_weight(target_lm_head) and _should_share(
        draft_model, "has_own_lm_head", draft_lm_head, target_lm_head
    ):
        if draft_lm_head is not None:
            del draft_model.lm_head
        draft_model.lm_head = target_lm_head
    elif is_pp:
        raise RuntimeError(
            "DSpark+PP requires a materialized target lm_head on the last PP stage"
        )

    return draft_model
