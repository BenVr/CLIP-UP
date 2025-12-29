from functools import partial
from omegaconf import OmegaConf

from clip_up.vlms.internvl.clip_up_internvl_utils import is_internvl_model_name, InternVLDataset, collate_func_internvl, \
    get_internvl_generation_args, get_internvl_validation_output, get_internvl_loss, \
    set_internvl_model, add_clip_up_to_internvl, run_internvl_inference_for_single_item
from clip_up.vlms.llava.clip_up_llava_utils import collate_func_llava, LLaVADataset, \
    get_llava_generation_args, get_llava_validation_output, get_llava_loss, \
    add_clip_up_to_llava, run_llava_inference_for_single_item
from clip_up.vlms.llava.clip_up_llava_utils import is_llava_model_name, set_llava_model


VLM_PATHS = {"llava-1.5-7b": "liuhaotian/llava-v1.5-7b", "internvl3-1b": "OpenGVLab/InternVL3-1B",
             "internvl3-8b": "OpenGVLab/InternVL3-8B"}
VLM_NAMES = list(VLM_PATHS.keys())


def get_vlm_path_and_config(model_name):
    model_path = VLM_PATHS[model_name]

    vlm_config = OmegaConf.load(f"clip_up/vlms/configs/{model_name}.yaml")
    base_config = OmegaConf.load("clip_up/vlms/configs/base.yaml")
    model_config = OmegaConf.merge(base_config, vlm_config)

    return model_path, model_config


def init_vlm(model_name, model_path, model_config, is_clip_up_emb_lora_injection, is_for_open_ended_questions, device):
    # Get lora rank and alpha
    lora_rank, lora_alpha = None, None
    if is_clip_up_emb_lora_injection:
        if is_for_open_ended_questions:
            lora_rank, lora_alpha = model_config.injected_lora_open_ended_rank, model_config.injected_lora_open_ended_alpha
        else:
            lora_rank, lora_alpha = model_config.injected_lora_multiple_choice_rank, model_config.injected_lora_multiple_choice_alpha

    # Init VLM
    if is_llava_model_name(model_name):
        tokenizer, model, image_processor, conv_mode = set_llava_model(model_path=model_path,
        load_with_lora_for_training=is_clip_up_emb_lora_injection, device=device, lora_rank=lora_rank,
        lora_alpha=lora_alpha)
    elif is_internvl_model_name(model_name):
        tokenizer, model = set_internvl_model(model_path=model_path, model_name=model_name, device=device,
                                               load_with_lora_for_training=is_clip_up_emb_lora_injection,
                                               lora_rank=lora_rank, lora_alpha=lora_alpha)
        conv_mode, image_processor = None, None
    else:
        assert False

    return model, image_processor, tokenizer, conv_mode


def add_clip_up_to_vlm(model, model_name, model_config, language_model_dim, is_for_open_ended_questions, device, torch_dtype, with_injected_lora):
    if is_llava_model_name(model_name):
        extended_vision_projection = add_clip_up_to_llava(model, model_config, language_model_dim, is_for_open_ended_questions, device, torch_dtype, with_injected_lora=with_injected_lora)
    elif is_internvl_model_name(model_name):
        extended_vision_projection = add_clip_up_to_internvl(model, model_config, language_model_dim, is_for_open_ended_questions, device, torch_dtype, with_injected_lora=with_injected_lora)
    else:
        assert False

    return extended_vision_projection


def get_vlm_dataset_class_and_extra_kwargs(model_name, clip_up_wrapper):
    if is_llava_model_name(model_name):
        dataset_class = LLaVADataset
        extra_kwargs = {'vision_encoder_dtype': clip_up_wrapper.get_model().model.get_vision_tower().dtype,
                        'conv_mode': clip_up_wrapper.conv_mode}
    elif is_internvl_model_name(model_name):
        dataset_class = InternVLDataset
        extra_kwargs = {"num_image_token": clip_up_wrapper.model.num_image_token,
                        "internvl_max_num": clip_up_wrapper.model.internvl_max_num,
                        "conv_template": clip_up_wrapper.model.conv_template.copy()}
    else:
        assert False

    return dataset_class, extra_kwargs


def get_vlm_collate_function(model_name, tokenizer):
    if is_llava_model_name(model_name):
        collate_func = partial(collate_func_llava, tokenizer=tokenizer)
    elif is_internvl_model_name(model_name):
        collate_func = collate_func_internvl
    else:
        assert False

    return collate_func


def get_generation_args(model_name, clip_up_wrapper):
    if is_llava_model_name(model_name):
        generation_args = get_llava_generation_args(conv_mode=clip_up_wrapper.conv_mode)
    elif is_internvl_model_name(model_name):
        generation_args = get_internvl_generation_args(eos_token_id=clip_up_wrapper.unwrapped_model.eos_token_id,
                                                          pad_token_id=clip_up_wrapper.unwrapped_model.pad_token_id)
    else:
        assert False

    return generation_args


def get_vlm_generation_output(model_name, model, tokenizer, val_batch, use_clip_signal, generation_args):
    if is_llava_model_name(model_name):
        output = get_llava_validation_output(unwrapped_model=model, tokenizer=tokenizer, val_batch=val_batch,
                                             use_clip_signal=use_clip_signal, **generation_args)
    elif is_internvl_model_name(model_name):
        output = get_internvl_validation_output(
            unwrapped_model=model,
            tokenizer=tokenizer,
            val_batch=val_batch,
            use_clip_signal=use_clip_signal, **generation_args)
    else:
        assert False

    return output


def run_vlm_forward(model_name, model, batch, clip_up_wrapper):
    # For all models, we recompute the loss such that it sums over samples rather than averaging over samples (this is needed for clip-up training loop)
    if is_llava_model_name(model_name):
        batch = {k: v for k, v in batch.items() if k in ['input_ids', 'attention_mask', 'images', 'labels', 'image_sizes', 'clip_signal']}
        output = model(**batch)
        loss = get_llava_loss(logits=output.logits, labels=output.labels,
                              vocab_size=clip_up_wrapper.get_unwrapped_model().config.vocab_size,
                              num_items_in_batch=1)
    elif is_internvl_model_name(model_name):
        batch = {k: v for k, v in batch.items() if k in ['input_ids', 'labels', 'attention_mask', 'position_ids', 'pixel_values', 'image_flags', 'clip_signal']}
        output = model(**batch)
        loss = get_internvl_loss(logits=output.logits, labels=output.labels,
                                  vocab_size=clip_up_wrapper.get_unwrapped_model().language_model.config.vocab_size,
                                  reduction='sum')
    else:
        assert False

    # Please note that the output.labels may be different from batch['label'] due to clip-up signal insertion

    return output.logits, output.labels, loss


def compute_vlm_per_sample_loss(model_name, logits, labels, clip_up_wrapper):
    batch_size = logits.shape[0]
    losses_per_sample = [None] * batch_size

    if is_llava_model_name(model_name):
        for i in range(batch_size):
            losses_per_sample[i] = get_llava_loss(logits=logits[i, :].unsqueeze(0), labels=labels[i, :].unsqueeze(0), vocab_size=clip_up_wrapper.get_unwrapped_model().config.vocab_size).item()
    elif is_internvl_model_name(model_name):
        for i in range(batch_size):
            losses_per_sample[i] = get_internvl_loss(logits=logits[i, :].unsqueeze(0), labels=labels[i, :].unsqueeze(0),
                                                      vocab_size=clip_up_wrapper.get_unwrapped_model().language_model.config.vocab_size, reduction='mean').item()
    else:
        assert False
    return losses_per_sample


def run_vlm_inference_for_single_item(model, model_name, conv_mode, tokenizer, image_processor, upd_setting, upd_type,
                                      question, hint, options, image, clip_signal):
    if is_llava_model_name(model_name):
        output, full_prompt = \
            run_llava_inference_for_single_item(model=model, tokenizer=tokenizer, image_processor=image_processor,
                                                model_name=model_name, conv_mode=conv_mode, question=question, hint=hint,
                                                options=options, image=image, clip_signal=clip_signal, upd_setting=upd_setting,
                                                upd_type=upd_type)
    elif is_internvl_model_name(model_name):
        output, full_prompt = \
            run_internvl_inference_for_single_item(model=model, tokenizer=tokenizer, question=question, hint=hint,
                                                   options=options, image=image, clip_signal=clip_signal,
                                                   upd_setting=upd_setting, upd_type=upd_type)
    else:
        assert False

    return output, full_prompt
