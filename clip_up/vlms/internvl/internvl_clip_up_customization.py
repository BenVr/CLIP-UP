from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch.nn import CrossEntropyLoss
from transformers import (GenerationConfig)
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen2.modeling_qwen2 import KwargsForCausalLM, QWEN2_INPUTS_DOCSTRING, _CONFIG_FOR_DOC, \
    apply_rotary_pos_emb, eager_attention_forward
from transformers.processing_utils import Unpack
from transformers.utils import (
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.utils import logging
from transformers.utils.deprecation import deprecate_kwarg

logger = logging.get_logger(__name__)


def add_new_clip_up_embedding_internvl(self, new_clip_up_embedding, input_embeds, attention_mask, position_ids, labels,
                                   input_ids_reshaped):
    # Add new_clip_up_embedding to input_embeds and update attention_mask, position_ids, labels
    new_input_embeds = []
    new_attention_masks = []
    new_position_ids = []
    new_labels = []
    for i, single_input_id in enumerate(input_ids_reshaped):
        curr_location_for_new_embedding = (single_input_id == self.image_end_token_id).nonzero(as_tuple=True)[0]
        assert len(curr_location_for_new_embedding) == 1
        curr_single_new_input_embeds = torch.cat(
            [input_embeds[i, :curr_location_for_new_embedding, :], new_clip_up_embedding[i, :, :],
             input_embeds[i, curr_location_for_new_embedding:, :]], dim=0)
        new_input_embeds.append(curr_single_new_input_embeds.unsqueeze(0))

        curr_single_new_attention_mask = torch.cat([attention_mask[i, :curr_location_for_new_embedding],
                                                    torch.tensor([True], dtype=attention_mask.dtype,
                                                                 device=attention_mask.device),
                                                    attention_mask[i, curr_location_for_new_embedding:]], dim=0)
        new_attention_masks.append(curr_single_new_attention_mask.unsqueeze(0))

        if position_ids is not None:
            last_part_position_ids = position_ids[i, curr_location_for_new_embedding:].clone()
            last_part_position_ids[last_part_position_ids != 0] = last_part_position_ids[
                                                                      last_part_position_ids != 0] + 1

            curr_single_new_position_ids = torch.cat([position_ids[i, :curr_location_for_new_embedding],
                                                      torch.tensor(
                                                          [position_ids[i, curr_location_for_new_embedding] + 1],
                                                          dtype=position_ids.dtype, device=position_ids.device),
                                                      last_part_position_ids], dim=0)
            new_position_ids.append(curr_single_new_position_ids.unsqueeze(0))

        if labels is not None:
            curr_single_labels = torch.cat([labels[i, :curr_location_for_new_embedding],
                                            torch.tensor([-100], dtype=labels.dtype, device=labels.device),
                                            labels[i, curr_location_for_new_embedding:]], dim=0)
            new_labels.append(curr_single_labels.unsqueeze(0))

    input_embeds = torch.cat(new_input_embeds, dim=0)
    attention_mask = torch.cat(new_attention_masks, dim=0)

    if position_ids is not None:
        position_ids = torch.cat(new_position_ids, dim=0)

    if labels is not None:
        labels = torch.cat(new_labels, dim=0)

    return input_embeds, attention_mask, position_ids, labels


def forward_InternVLChatModel_for_clip_up_emb(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        image_flags: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        clip_signal: Optional[torch.Tensor] = None,
) -> Union[Tuple, CausalLMOutputWithPast]:
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    image_flags = image_flags.squeeze(-1)
    input_embeds = self.language_model.get_input_embeddings()(input_ids).clone()

    vit_embeds, new_clip_up_embedding = self.extract_feature(pixel_values, clip_signal)
    vit_embeds = vit_embeds[image_flags == 1]
    vit_batch_size = pixel_values.shape[0]

    B, N, C = input_embeds.shape
    input_embeds = input_embeds.reshape(B * N, C)

    if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
        print(f'dynamic ViT batch size: {vit_batch_size}, images per sample: {vit_batch_size / B}, dynamic token length: {N}')

    input_ids = input_ids.reshape(B * N)
    selected = (input_ids == self.img_context_token_id)
    try:
        input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds.reshape(-1, C)
    except Exception as e:
        vit_embeds = vit_embeds.reshape(-1, C)
        print(f'warning: {e}, input_embeds[selected].shape={input_embeds[selected].shape}, '
              f'vit_embeds.shape={vit_embeds.shape}')
        n_token = min(selected.sum(), vit_embeds.size(0))
        input_embeds[selected][:n_token] = input_embeds[selected][:n_token] * 0.0 + vit_embeds[:n_token]

    input_embeds = input_embeds.reshape(B, N, C)

    if new_clip_up_embedding is not None:
        input_ids_reshaped = input_ids.reshape(B, N)
        input_embeds, attention_mask, position_ids, labels = \
            self.add_new_clip_up_embedding(new_clip_up_embedding, input_embeds=input_embeds, attention_mask=attention_mask,
                                       position_ids=position_ids, labels=labels, input_ids_reshaped=input_ids_reshaped)

    injected_lora_kwargs = {"clip_signal": clip_signal} if self.with_injected_lora else {}

    outputs = self.language_model(
        inputs_embeds=input_embeds,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        **injected_lora_kwargs
    )
    logits = outputs.logits

    loss = None
    if labels is not None:
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.language_model.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    output = CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )
    output['labels'] = labels
    return output


def extract_feature_InternVLChatModel_for_clip_up_emb(self, pixel_values, clip_signal):
    if self.select_layer == -1:
        vit_embeds = self.vision_model(
            pixel_values=pixel_values,
            output_hidden_states=False,
            return_dict=True).last_hidden_state
    else:
        vit_embeds = self.vision_model(
            pixel_values=pixel_values,
            output_hidden_states=True,
            return_dict=True).hidden_states[self.select_layer]
    vit_embeds = vit_embeds[:, 1:, :]

    h = w = int(vit_embeds.shape[1] ** 0.5)
    vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
    vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
    vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
    from clip_up.clip_up_wrapper_internals import ExtendedVisionProjection
    if type(self.mlp1) == ExtendedVisionProjection:
        vit_embeds, new_clip_up_embedding = self.mlp1(vit_embeds, clip_signal=clip_signal)
    else:
        vit_embeds = self.mlp1(vit_embeds)
        new_clip_up_embedding = None

    return vit_embeds, new_clip_up_embedding


@torch.no_grad()
def generate_InternVLChatModel_for_clip_up_emb(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        input_ids: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        visual_features: Optional[torch.FloatTensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        output_hidden_states: Optional[bool] = None,
        clip_signal: Optional[torch.Tensor] = None,
        **generate_kwargs,
) -> torch.LongTensor:

    assert self.img_context_token_id is not None
    if pixel_values is not None:
        if visual_features is not None:
            vit_embeds = visual_features
        else:
            vit_embeds, new_clip_up_embedding = self.extract_feature(pixel_values, clip_signal)

        input_embeds = self.language_model.get_input_embeddings()(input_ids)
        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)

        input_ids = input_ids.reshape(B * N)
        selected = (input_ids == self.img_context_token_id)
        assert selected.sum() != 0
        input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)

        input_embeds = input_embeds.reshape(B, N, C)

        if new_clip_up_embedding is not None:
            input_ids_reshaped = input_ids.reshape(B, N)
            input_embeds, attention_mask, _, _ = \
                self.add_new_clip_up_embedding(new_clip_up_embedding, input_embeds=input_embeds,
                                           attention_mask=attention_mask,
                                           position_ids=None, labels=None, input_ids_reshaped=input_ids_reshaped)

    else:
        input_embeds = self.language_model.get_input_embeddings()(input_ids)

    if self.with_injected_lora:
        generate_kwargs["clip_signal"] = clip_signal

    outputs = self.language_model.generate(
        inputs_embeds=input_embeds,
        attention_mask=attention_mask,
        generation_config=generation_config,
        output_hidden_states=output_hidden_states,
        use_cache=True,
        **generate_kwargs,
    )

    return outputs

@deprecate_kwarg("num_logits_to_keep", version="4.50", new_name="logits_to_keep")
@add_start_docstrings_to_model_forward(QWEN2_INPUTS_DOCSTRING)
@replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
def forward_Qwen2ForCausalLM_for_clip_up_inj_lora(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    logits_to_keep: Union[int, torch.Tensor] = 0,
    clip_signal = None,
    **kwargs: Unpack[KwargsForCausalLM],
) -> Union[Tuple, CausalLMOutputWithPast]:
    r"""
    Args:
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        logits_to_keep (`int` or `torch.Tensor`, *optional*):
            If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
            `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
            token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
            If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
            This is useful when using packed tensor format (single dimension for batch and sequence length).

    Returns:

    Example:

    ```python
    >>> from transformers import AutoTokenizer, Qwen2ForCausalLM

    >>> model = Qwen2ForCausalLM.from_pretrained("meta-qwen2/Qwen2-2-7b-hf")
    >>> tokenizer = AutoTokenizer.from_pretrained("meta-qwen2/Qwen2-2-7b-hf")

    >>> prompt = "Hey, are you conscious? Can you talk to me?"
    >>> inputs = tokenizer(prompt, return_tensors="pt")

    >>> # Generate
    >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
    >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
    ```"""
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
        clip_signal=clip_signal,
        **kwargs,
    )

    hidden_states = outputs[0]
    # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    logits = self.lm_head(hidden_states[:, slice_indices, :])

    loss = None
    if labels is not None:
        loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


@add_start_docstrings_to_model_forward(QWEN2_INPUTS_DOCSTRING)
def forward_Qwen2Model_for_clip_up_inj_lora(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    clip_signal = None,
    **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
) -> Union[Tuple, BaseModelOutputWithPast]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if self.gradient_checkpointing and self.training and use_cache:
        logger.warning_once(
            "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
        )
        use_cache = False

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if use_cache and past_key_values is None:
        past_key_values = DynamicCache()

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )

    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    causal_mask = self._update_causal_mask(
        attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
    )

    hidden_states = inputs_embeds

    # create position embeddings to be shared across the decoder layers
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None

    for decoder_layer in self.layers[: self.config.num_hidden_layers]:
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                decoder_layer.__call__,
                hidden_states,
                causal_mask,
                position_ids,
                past_key_values,
                output_attentions,
                use_cache,
                cache_position,
                position_embeddings,
                clip_signal
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                clip_signal=clip_signal,
                **flash_attn_kwargs,
            )

        hidden_states = layer_outputs[0]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    output = BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values if use_cache else None,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )
    return output if return_dict else output.to_tuple()


def forward_Qwen2DecoderLayer_for_clip_up_inj_lora(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
    clip_signal=None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
    residual = hidden_states

    hidden_states = self.input_layernorm(hidden_states)

    # Self Attention
    hidden_states, self_attn_weights = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
        cache_position=cache_position,
        position_embeddings=position_embeddings,
        clip_signal=clip_signal,
        **kwargs,
    )
    hidden_states = residual + hidden_states

    # Fully Connected
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states, clip_signal=clip_signal)
    hidden_states = residual + hidden_states

    outputs = (hidden_states,)
    if output_attentions:
        outputs += (self_attn_weights,)

    return outputs


def forward_Qwen2Attention_for_clip_up_inj_lora(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_value: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    clip_signal=None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states, clip_signal).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states, clip_signal).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states, clip_signal).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    sliding_window = None
    if (
        self.config.use_sliding_window
        and getattr(self.config, "sliding_window", None) is not None
        and self.layer_idx >= self.config.max_window_layers
    ):
        sliding_window = self.config.sliding_window

    attention_interface: Callable = eager_attention_forward
    if self.config._attn_implementation != "eager":
        if self.config._attn_implementation == "sdpa" and kwargs.get("output_attentions", False):
            logger.warning_once(
                "`torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to "
                'eager attention. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
        else:
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=sliding_window,  # main diff with Llama
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output, clip_signal)
    return attn_output, attn_weights


def forward_Qwen2MLP_for_clip_up_inj_lora(self, x, clip_signal):
    down_proj = self.down_proj(self.act_fn(self.gate_proj(x, clip_signal)) * self.up_proj(x, clip_signal), clip_signal)
    return down_proj
