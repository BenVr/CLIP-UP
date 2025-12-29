import copy
import os
import shutil
import torch
import warnings
from typing import List, Optional, Tuple, Union, Dict, Any
from PIL import Image
from peft import LoraConfig, get_peft_model
from peft import PeftModel
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
from transformers.loss.loss_utils import ForCausalLMLoss
from transformers.modeling_outputs import CausalLMOutputWithPast

from llava.constants import (DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX,
                             IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN)
from llava.conversation import conv_templates, SeparatorStyle
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token, KeywordsStoppingCriteria
from llava.model import LlavaMptForCausalLM, LlavaMistralForCausalLM
from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
from llava.model.llava_arch import LlavaMetaForCausalLM
from llava.train.train import find_all_linear_names

from clip_up.clip_up_utils import add_injected_lora_support_to_modules
from clip_up.clip_up_wrapper_internals import ExtendedVisionProjection
from clip_up.vlms.llava.llava_clip_up_customization import forward_LlamaModel_for_clip_up_inj_lora, \
    forward_LlamaDecoderLayer_for_clip_up_inj_lora, forward_LlamaAttention_for_clip_up_inj_lora, \
    forward_LlamaMLP_for_clip_up_inj_lora
from clip_up.vlms.llava.llava_clip_up_customization import forward_LlavaLlamaForCausalLM_for_clip_up_emb, \
    generate_LlavaLlamaForCausalLM_for_clip_up_emb, \
    encode_images_LlavaMetaForCausalLM_for_clip_up_emb, \
    prepare_inputs_labels_for_multimodal_LlavaMetaForCausalLM_for_clip_up_emb
from prompting.prompting_utils import all_options, get_options, build_choices, get_question_string, get_target_str


def is_llava_1_5_7b(model_name):
    return model_name == 'llava-1.5-7b'


def is_llava_model_name(model_name):
    return is_llava_1_5_7b(model_name)


# Function based on llava.model.builder.load_pretrained_model
def load_pretrained_llava_model(model_path, model_base, model_name, load_with_lora_for_training,
                          load_8bit=False, load_4bit=False, device_map="auto", device="cuda", use_flash_attn=False,
                          lora_rank=None, lora_alpha=None, **kwargs):
    kwargs = {"device_map": device_map, **kwargs}

    if device != "cuda":
        kwargs['device_map'] = {"": device}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.bfloat16 if load_with_lora_for_training else torch.float16

    if use_flash_attn:
        kwargs['attn_implementation'] = 'flash_attention_2'

    if 'llava' in model_name.lower():
        # Load LLaVA model
        if 'lora' in model_name.lower() and model_base is None:
            warnings.warn('There is `lora` in model name but no `model_base` is provided. If you are loading a LoRA model, please provide the `model_base` argument. Detailed instruction: https://github.com/haotian-liu/LLaVA#launch-a-model-worker-lora-weights-unmerged.')
        if 'lora' in model_name.lower() and model_base is not None:
            from llava.model.language_model.llava_llama import LlavaConfig
            lora_cfg_pretrained = LlavaConfig.from_pretrained(model_path)
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            print('Loading LLaVA from base model...')
            model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=lora_cfg_pretrained, **kwargs)
            token_num, tokem_dim = model.lm_head.out_features, model.lm_head.in_features
            if model.lm_head.weight.shape[0] != token_num:
                model.lm_head.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))
                model.model.embed_tokens.weight = torch.nn.Parameter(torch.empty(token_num, tokem_dim, device=model.device, dtype=model.dtype))

            print('Loading additional LLaVA weights...')
            if os.path.exists(os.path.join(model_path, 'non_lora_trainables.bin')):
                non_lora_trainables = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
            else:
                # this is probably from HF Hub
                from huggingface_hub import hf_hub_download
                def load_from_hf(repo_id, filename, subfolder=None):
                    cache_file = hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        subfolder=subfolder)
                    return torch.load(cache_file, map_location='cpu')
                non_lora_trainables = load_from_hf(model_path, 'non_lora_trainables.bin')
            non_lora_trainables = {(k[11:] if k.startswith('base_model.') else k): v for k, v in non_lora_trainables.items()}
            if any(k.startswith('model.model.') for k in non_lora_trainables):
                non_lora_trainables = {(k[6:] if k.startswith('model.') else k): v for k, v in non_lora_trainables.items()}
            model.load_state_dict(non_lora_trainables, strict=False)

            from peft import PeftModel
            print('Loading LoRA weights...')
            model = PeftModel.from_pretrained(model, model_path)
            print('Merging LoRA weights...')
            model = model.merge_and_unload()
            print('Model is loaded...')
        elif model_base is not None:
            # this may be mm projector only
            print('Loading LLaVA from base model...')
            if 'mpt' in model_name.lower():
                if not os.path.isfile(os.path.join(model_path, 'configuration_mpt.py')):
                    shutil.copyfile(os.path.join(model_base, 'configuration_mpt.py'), os.path.join(model_path, 'configuration_mpt.py'))
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=True)
                cfg_pretrained = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
                model = LlavaMptForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
                cfg_pretrained = AutoConfig.from_pretrained(model_path)
                model = LlavaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)

            mm_projector_weights = torch.load(os.path.join(model_path, 'mm_projector.bin'), map_location='cpu')
            mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items()}
            model.load_state_dict(mm_projector_weights, strict=False)
        else:
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = LlavaMptForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
            elif 'mistral' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = LlavaMistralForCausalLM.from_pretrained(
                    model_path,
                    low_cpu_mem_usage=True,
                    **kwargs
                )
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = LlavaLlamaForCausalLM.from_pretrained(
                    model_path,
                    low_cpu_mem_usage=True,
                    **kwargs
                )
    else:
        # Load language model
        if model_base is not None:
            # PEFT model
            from peft import PeftModel
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, **kwargs)
            print(f"Loading LoRA weights from {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
            print(f"Merging weights")
            model = model.merge_and_unload()
            print('Convert to FP16...')
            model.to(torch.float16)
        else:
            use_fast = False
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

    image_processor = None

    # Make Peft
    if load_with_lora_for_training:
        all_linear_names = find_all_linear_names(model)
        all_linear_names = [name for name, module in model.named_modules() if 'vision_tower' not in name and
                                 any([linear_name in name for linear_name in all_linear_names])]

        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=all_linear_names,
            lora_dropout=0.05,
            bias='none',
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

    if 'llava' in model_name.lower():
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))

        vision_tower = model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model(device_map=device_map)
        if device_map != 'auto':
            vision_tower.to(device=device_map, dtype=torch.float16)
        image_processor = vision_tower.image_processor

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len


def process_image_for_llava(image, image_processor, device, dtype):
    pixel_values = image_processor.preprocess(image, return_tensors='pt')['pixel_values']
    pixel_values = pixel_values.to(device, dtype=dtype)
    return pixel_values


def set_llava_model(model_path, device, load_with_lora_for_training=False, lora_rank=None, lora_alpha=None):
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_llava_model(model_path, None, model_name,
                                                                           load_with_lora_for_training, device=device,
                                                                           device_map=None, lora_rank=lora_rank, lora_alpha=lora_alpha)
    model = model.to(device)

    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"

    # Override forward with an identical function that gets 'cache_position' as input
    # (required for it to work with current transformers version)
    non_peft_model = model.base_model.model if isinstance(model, PeftModel) else model
    old_LlavaLlamaForCausalLM_forward = model.forward
    def new_LlavaLlamaForCausalLM_forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            images: Optional[torch.FloatTensor] = None,
            image_sizes: Optional[List[List[int]]] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return old_LlavaLlamaForCausalLM_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            images=images,
            image_sizes=image_sizes,
            return_dict=return_dict,
        )
    non_peft_model.forward = new_LlavaLlamaForCausalLM_forward.__get__(non_peft_model, non_peft_model.__class__)

    return tokenizer, model, image_processor, conv_mode


def add_clip_up_to_llava(model, model_config, language_model_dim, is_for_open_ended_questions, device, torch_dtype,
                        with_injected_lora):
    # 1. Add CLIP-UP embedding injection projection layer
    model.model.mm_projector = ExtendedVisionProjection(original_projection=model.model.mm_projector,
                                                        language_model_dim=language_model_dim,
                                                        structure_clip_dim=model_config.structure_clip_dim,
                                                        is_for_open_ended_questions=is_for_open_ended_questions,
                                                        device=device,
                                                        torch_dtype=torch_dtype)
    extended_vision_projection = model.model.mm_projector

    # 2. Modify functions to support embedding injection
    model.forward = forward_LlavaLlamaForCausalLM_for_clip_up_emb.__get__(model, model.__class__)
    model.generate = generate_LlavaLlamaForCausalLM_for_clip_up_emb.__get__(model, model.__class__)
    model.encode_images = encode_images_LlavaMetaForCausalLM_for_clip_up_emb.__get__(model, model.__class__)
    model.prepare_inputs_labels_for_multimodal = prepare_inputs_labels_for_multimodal_LlavaMetaForCausalLM_for_clip_up_emb.__get__(
        model, model.__class__)

    if with_injected_lora:
        # 3. Add Injected LoRA projection layers
        target_modules_name = ['k_proj', 'o_proj', 'up_proj', 'down_proj', 'q_proj', 'v_proj', 'gate_proj']
        target_modules = {name: module for name, module in model.named_modules() if 'vision_tower' not in name and any(name.endswith(x) for x in target_modules_name)}
        lora_rank = model.peft_config['default'].r
        add_injected_lora_support_to_modules(target_modules, lora_rank=lora_rank, torch_dtype=torch_dtype, signal_dim=extended_vision_projection.in_clip_up_dim)

        # 4. Modify functions to support Injected LoRA
        model.model.forward = forward_LlamaModel_for_clip_up_inj_lora.__get__(model.model, model.model.__class__)
        for curr_layer in model.model.layers:
            curr_layer.forward = forward_LlamaDecoderLayer_for_clip_up_inj_lora.__get__(curr_layer, curr_layer.__class__)
            curr_layer.self_attn.forward = forward_LlamaAttention_for_clip_up_inj_lora.__get__(curr_layer.self_attn, curr_layer.self_attn.__class__)
            curr_layer.mlp.forward = forward_LlamaMLP_for_clip_up_inj_lora.__get__(curr_layer.mlp, curr_layer.mlp.__class__)

    return extended_vision_projection


def get_llava_loss(logits, labels, vocab_size, num_items_in_batch=None):
    loss = ForCausalLMLoss(logits=logits, labels=labels, vocab_size=vocab_size, num_items_in_batch=num_items_in_batch)
    return loss


def get_llava_generation_args(conv_mode):
    conv = conv_templates[conv_mode].copy()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    return {'keywords': keywords}


def get_llava_validation_output(unwrapped_model, tokenizer, val_batch, use_clip_signal, keywords):
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, val_batch['input_ids'])
    target_len = [len(t[t != IGNORE_INDEX]) for t in val_batch['labels']]
    input_ids = val_batch['input_ids']
    masks = val_batch['attention_mask']
    max_len = max(target_len)
    input_ids = torch.stack([in_id[:-max_len] for in_id in input_ids])
    attention_masks = torch.stack([mask[:-max_len] for mask in masks])

    with torch.inference_mode():
        generated_ids = unwrapped_model.generate(inputs=input_ids, images=val_batch['images'],
                                                 image_sizes=val_batch['image_sizes'],
                                                 clip_signal=val_batch['clip_signal'] if use_clip_signal else None,
                                                 attention_mask=attention_masks, stopping_criteria=[stopping_criteria],
                                                 do_sample=False, max_new_tokens=512)
    output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    output = [out.strip() for out in output]
    return output


def get_question_string_llava(question, hint, options, upd_setting, upd_type):
    full_question_str = get_question_string(question, hint, options, upd_setting, upd_type)
    full_question_str_formatted = DEFAULT_IMAGE_TOKEN + '\n' + full_question_str
    return full_question_str_formatted


def run_llava_inference_for_single_item(model, tokenizer, image_processor, model_name, conv_mode, question, hint, options,
                                        image, clip_signal, upd_setting, upd_type):
    # 1. Get prompt
    full_question_str_formatted = get_question_string_llava(question, hint, options, upd_setting=upd_setting, upd_type=upd_type)

    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], full_question_str_formatted)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    # 2. Encode prompt and image
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    image = image.convert("RGB") if image.mode != 'RGB' else image
    pixel_values = process_image_for_llava(image, image_processor, device=model.device,
                                           dtype=model.model.get_vision_tower().dtype)

    clip_signal_kwargs = {"clip_signal": clip_signal.unsqueeze(0).cuda()} if clip_signal is not None else {}

    # 3. Run llava
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=pixel_values,
            image_sizes=[image.size],
            do_sample=False,
            max_new_tokens=1024,
            use_cache=True,
            **clip_signal_kwargs)

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    return outputs, prompt


class LLaVADataset(Dataset):
    def __init__(self,
                 inputs,
                 tokenizer,
                 image_processor,
                 model_config,
                 upd_setting,
                 upd_type,
                 conv_mode,
                 model_name,
                 vision_encoder_dtype,
                 ):

        self.inputs = inputs
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.upd_setting = upd_setting
        self.upd_type = upd_type
        self.conv_mode = conv_mode
        self.model_name = model_name
        self.vision_encoder_dtype = vision_encoder_dtype
        self.device = 'cpu'
        return

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.inputs[idx]

        # 1. Encode the input ids and labels
        sample = {}
        options = get_options(item, all_options)
        question = item['question']
        hint = item['hint']
        gt_answer = item['answer']
        full_question_str_formatted = get_question_string_llava(question, hint, options, upd_setting=self.upd_setting,
                                                                upd_type=self.upd_type)
        target_str = get_target_str(gt_answer=gt_answer, upd_type=self.upd_type, item=item,
                                    eos_token=self.tokenizer.eos_token)
        input_ids, targets = self.encode_instruction_and_target(input_text=full_question_str_formatted,
                                                                target_text=target_str)

        sample['input_ids'] = input_ids.squeeze(0)
        sample['labels'] = targets.squeeze(0)

        # 2. Encode the image
        image = Image.open(item["image_path"])
        sample['images'] = process_image_for_llava(image=image, image_processor=self.image_processor,
                                                   device=self.device, dtype=self.vision_encoder_dtype).squeeze(0)

        # 3. Set CLIP-UP signal
        sample['clip_signal'] = item['clip_signal'].to(self.device)

        # Add info for debugging and evaluation
        sample['question_str'] = question
        sample['gt_answer_str'] = target_str
        sample['type'] = item['type']
        sample['options'] = options
        sample['choices'] = build_choices(item)
        sample['original_gt_answer'] = gt_answer
        sample['question_id'] = item['index']
        sample['upd_challenge_type'] = item['upd_challenge_type']
        sample['masked_answer'] = item['masked_answer'] if 'masked_answer' in item else None
        sample['image_sizes'] = image.size

        return sample

    def encode_instruction_and_target(self, input_text, target_text):
        # 1. Encode input ids
        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], input_text)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt() + conv.sep + target_text
        input_ids = tokenizer_image_token(
            prompt,
            self.tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors='pt'
        ).unsqueeze(0).to(self.device)

        # 2. Get targets
        sep = conv.sep + conv.roles[1] + ": "
        parts = prompt.split(sep)
        parts[0] += sep
        instruction_len = len(tokenizer_image_token(parts[0], self.tokenizer)) - 2

        target_ids = tokenizer_image_token(
            parts[1],
            self.tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors='pt'
        ).unsqueeze(0).to(self.device)

        # For the target, need to set everything up to the target equal to IGNORE_INDEX
        targets = copy.deepcopy(input_ids)
        targets[:, :1] = IGNORE_INDEX
        for i in range(target_ids.shape[0]):
            targets[i, 1:1 + instruction_len] = IGNORE_INDEX

        return input_ids, targets


def collate_func_llava(batch, tokenizer):
    joined_batch = {}

    joined_batch['input_ids'] = torch.nn.utils.rnn.pad_sequence(
        [b['input_ids'] for b in batch],
        batch_first=True,
        padding_value=tokenizer.pad_token_id
    )
    joined_batch['attention_mask'] = joined_batch['input_ids'].ne(tokenizer.pad_token_id)
    images = [b['images'] for b in batch]
    if all(x is not None and x.shape == images[0].shape for x in images):
        joined_batch['images'] = torch.stack(images)
    else:
        joined_batch['images'] = images

    joined_batch['labels'] = torch.nn.utils.rnn.pad_sequence(
        [b['labels'] for b in batch],
        batch_first=True,
        padding_value=IGNORE_INDEX
    )
    joined_batch['image_sizes'] = [b['image_sizes'] for b in batch]

    joined_batch['clip_signal'] = torch.stack([b['clip_signal'] for b in batch])

    for k in ['question_str', 'gt_answer_str', 'type', 'options', 'choices', 'original_gt_answer', 'question_id',
              'masked_answer', 'upd_challenge_type']:
        joined_batch[k] = [b[k] for b in batch]

    return joined_batch
