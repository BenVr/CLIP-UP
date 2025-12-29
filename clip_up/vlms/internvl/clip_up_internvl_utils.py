import copy
from typing import Dict, Any
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import transformers
from PIL import Image
from peft import LoraConfig, get_peft_model
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer, AutoModel

from clip_up.clip_up_utils import add_injected_lora_support_to_modules
from clip_up.clip_up_wrapper_internals import ExtendedVisionProjection
from clip_up.vlms.internvl.internvl_clip_up_customization import forward_Qwen2ForCausalLM_for_clip_up_inj_lora, \
    forward_Qwen2Model_for_clip_up_inj_lora, \
    forward_Qwen2DecoderLayer_for_clip_up_inj_lora, forward_Qwen2Attention_for_clip_up_inj_lora, \
    forward_Qwen2MLP_for_clip_up_inj_lora, \
    forward_InternVLChatModel_for_clip_up_emb, generate_InternVLChatModel_for_clip_up_emb, \
    extract_feature_InternVLChatModel_for_clip_up_emb, add_new_clip_up_embedding_internvl
from prompting.prompting_utils import all_options, get_options, build_choices, get_question_string, \
    get_target_str

IMG_START_TOKEN = '<img>'
IMG_END_TOKEN = '</img>'
IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IGNORE_TOKEN_ID = -100


def is_internvl_model_name(model_name):
    return is_internvl3_1b_model_name(model_name) or is_internvl3_8b_model_name(model_name)


def is_internvl3_1b_model_name(model_name):
    return model_name == 'internvl3-1b'


def is_internvl3_8b_model_name(model_name):
    return model_name == 'internvl3-8b'


# Function taken from https://huggingface.co/OpenGVLab/InternVL3-1B
def build_transform_internvl(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


# Function taken from https://huggingface.co/OpenGVLab/InternVL3-1B
def load_image_internvl(image, input_size=448, max_num=None):
    transform = build_transform_internvl(input_size=input_size)
    images = dynamic_preprocess_internvl(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


# Function taken from https://huggingface.co/OpenGVLab/InternVL3-1B
def dynamic_preprocess_internvl(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio_internvl(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


# Function taken from https://huggingface.co/OpenGVLab/InternVL3-1B
def find_closest_aspect_ratio_internvl(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def wrap_llm_lora_internvl(model, r=128, lora_alpha=256, lora_dropout=0.05):
    # Determine the target modules based on the architecture of the language model
    llm_arch_name = model.config.llm_config.architectures[0]
    if llm_arch_name == 'InternLM2ForCausalLM':
        target_modules = ['attention.wqkv', 'attention.wo', 'feed_forward.w1', 'feed_forward.w2', 'feed_forward.w3']
    elif llm_arch_name == 'Phi3ForCausalLM':
        target_modules = ['mlp.down_proj', 'mlp.gate_up_proj', 'self_attn.o_proj', 'self_attn.qkv_proj']
    elif llm_arch_name in ['Qwen2ForCausalLM', 'LlamaForCausalLM']:
        target_modules = ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj',
                          'mlp.gate_proj', 'mlp.down_proj', 'mlp.up_proj']
    else:
        raise NotImplemented
    lora_config = LoraConfig(
        r=r,
        target_modules=target_modules,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        task_type='CAUSAL_LM'
    )
    model.language_model = get_peft_model(model.language_model, lora_config)
    model.language_model.enable_input_require_grads()
    return model


def set_internvl_model(model_path, model_name, device, load_with_lora_for_training,
                        lora_rank=None, lora_alpha=None):
    if is_internvl3_1b_model_name(model_name):
        revision = '06cddba9140fdb73a47951480b6e9cec04970559'
    elif is_internvl3_8b_model_name(model_name):
        revision = '24dc81a234a6e1901f3314eeadaa2813f2b78038'
    else:
        assert False

    model = AutoModel.from_pretrained(model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
                                       use_flash_attn=True, revision=revision, trust_remote_code=True).to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_path, revision=revision, trust_remote_code=True, use_fast=False)

    if load_with_lora_for_training:
        model = wrap_llm_lora_internvl(model=model, r=lora_rank, lora_alpha=lora_alpha)
        model.config.use_llm_lora = lora_rank

    # Set model properties
    model.internvl_max_num = 6
    model.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    model.eos_token_id = tokenizer.convert_tokens_to_ids(model.conv_template.copy().sep.strip())
    model.pad_token_id = model.eos_token_id
    model.image_end_token_id = tokenizer.convert_tokens_to_ids(IMG_END_TOKEN)

    return tokenizer, model


def add_clip_up_to_internvl(model, model_config, language_model_dim, is_for_open_ended_questions, device, torch_dtype,
                            with_injected_lora):
    # 1. Add CLIP-UP embedding injection projection layer
    model.mlp1 = ExtendedVisionProjection(original_projection=model.mlp1, language_model_dim=language_model_dim,
                                          structure_clip_dim=model_config.structure_clip_dim,
                                          is_for_open_ended_questions=is_for_open_ended_questions, device=device,
                                          torch_dtype=torch_dtype)
    extended_vision_projection = model.mlp1

    # 2. Modify functions to support embedding injection
    model.forward = forward_InternVLChatModel_for_clip_up_emb.__get__(model, model.__class__)
    model.extract_feature = extract_feature_InternVLChatModel_for_clip_up_emb.__get__(model, model.__class__)
    model.generate = generate_InternVLChatModel_for_clip_up_emb.__get__(model, model.__class__)
    model.add_new_clip_up_embedding = add_new_clip_up_embedding_internvl.__get__(model, model.__class__)
    model.with_injected_lora = with_injected_lora

    if with_injected_lora:
        # 3. Add Injected LoRA projection layers
        target_modules_name = ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'self_attn.o_proj',
                          'mlp.gate_proj', 'mlp.down_proj', 'mlp.up_proj']
        target_modules = {name: module for name, module in model.named_modules() if any(name.endswith(x) for x in target_modules_name)}
        lora_rank = model.language_model.peft_config['default'].r
        add_injected_lora_support_to_modules(target_modules, lora_rank=lora_rank, torch_dtype=torch_dtype, signal_dim=extended_vision_projection.in_clip_up_dim)

        # 4. Modify functions to support Injected LoRA
        model.language_model.base_model.model.forward = forward_Qwen2ForCausalLM_for_clip_up_inj_lora.__get__(
            model.language_model.base_model.model, model.language_model.base_model.model.__class__)
        model.language_model.base_model.model.model.forward = forward_Qwen2Model_for_clip_up_inj_lora.__get__(
            model.language_model.base_model.model.model, model.language_model.base_model.model.model.__class__)

        for curr_layer in model.language_model.base_model.model.model.layers:
            curr_layer.forward = forward_Qwen2DecoderLayer_for_clip_up_inj_lora.__get__(curr_layer, curr_layer.__class__)
            curr_layer.self_attn.forward = forward_Qwen2Attention_for_clip_up_inj_lora.__get__(curr_layer.self_attn, curr_layer.self_attn.__class__)
            curr_layer.mlp.forward = forward_Qwen2MLP_for_clip_up_inj_lora.__get__(curr_layer.mlp, curr_layer.mlp.__class__)

    return extended_vision_projection


def get_internvl_loss(logits, labels, vocab_size, reduction):
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    # Flatten the tokens
    loss_fct = CrossEntropyLoss(reduction=reduction)
    shift_logits = shift_logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)
    loss = loss_fct(shift_logits, shift_labels)
    return loss


def get_internvl_generation_args(eos_token_id, pad_token_id):
    generation_args = {"eos_token_id": eos_token_id, "pad_token_id": pad_token_id}
    return generation_args


def get_internvl_validation_output(unwrapped_model, tokenizer, val_batch, use_clip_signal, eos_token_id, pad_token_id):
    target_len = [len(t[t != IGNORE_TOKEN_ID]) for t in val_batch['labels']]

    # the last token is '\n' which we don't learn, the before last token should and part of the answer
    if all([t[-1] == IGNORE_TOKEN_ID and t[-2] != IGNORE_TOKEN_ID for t in val_batch['labels']]):
        target_len = [t + 1 for t in target_len]
    else:
        assert False

    input_ids = val_batch['input_ids']
    masks = val_batch['attention_mask']
    max_len = max(target_len)
    input_ids = torch.stack([in_id[:-max_len] for in_id in input_ids])
    attention_mask = torch.stack([mask[:-max_len] for mask in masks])

    with torch.inference_mode():
        generation_output = unwrapped_model.generate(
            input_ids=input_ids,
            pixel_values=val_batch['pixel_values'],
            clip_signal=val_batch['clip_signal'] if use_clip_signal else None,
            attention_mask=attention_mask,
            do_sample=False,
            max_new_tokens=512,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
        )

    output = tokenizer.batch_decode(generation_output, skip_special_tokens=True)
    output = [out.strip() for out in output]
    return output


def get_question_string_internvl(question, hint, options, upd_setting, upd_type):
    full_question_str = get_question_string(question, hint, options, upd_setting, upd_type)
    full_question_str_formatted = '<image>\n' + full_question_str
    return full_question_str_formatted


def run_internvl_inference_for_single_item(model, tokenizer, question, hint, options, image, clip_signal, upd_setting, upd_type):
    # 1. Encode image
    image = image.convert('RGB')
    pixel_values = load_image_internvl(image, max_num=model.internvl_max_num).to(torch.bfloat16).to(model.device)
    num_patches_list = [pixel_values.shape[0]]

    # 2. Encode prompt
    full_question_str_formatted = get_question_string_internvl(question, hint, options, upd_setting=upd_setting, upd_type=upd_type)
    template = model.conv_template.copy()
    template.append_message(template.roles[0], full_question_str_formatted)
    template.append_message(template.roles[1], None)
    prompt = template.get_prompt()
    prompt_to_return = prompt

    for num_patches in num_patches_list:
        image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * model.num_image_token * num_patches + IMG_END_TOKEN
        prompt = prompt.replace('<image>', image_tokens, 1)

    model_inputs = tokenizer(prompt, return_tensors='pt')
    input_ids = model_inputs['input_ids'].to(model.device)
    attention_mask = model_inputs['attention_mask'].to(model.device)

    clip_signal_kwargs = {"clip_signal": clip_signal.unsqueeze(0).cuda()} if clip_signal is not None else {}

    # 3. Run InternVL
    with torch.inference_mode():
        output_ids = model.generate(input_ids=input_ids,
                                       pixel_values=pixel_values,
                                       attention_mask=attention_mask,
                                       do_sample=False,
                                       max_new_tokens=512,
                                       eos_token_id=model.eos_token_id,
                                       pad_token_id=model.pad_token_id,
                                       **clip_signal_kwargs)

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    return outputs, prompt_to_return


class InternVLDataset(Dataset):
    def __init__(self,
                 inputs,
                 tokenizer,
                 image_processor,
                 model_config,
                 upd_setting,
                 upd_type,
                 model_name,
                 num_image_token,
                 internvl_max_num,
                 conv_template,
                 ):

        self.inputs = inputs
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.model_config = model_config
        self.upd_setting = upd_setting
        self.upd_type = upd_type
        self.model_name = model_name
        self.num_image_token = num_image_token
        self.internvl_max_num = internvl_max_num
        self.conv_template = conv_template
        self.device = 'cpu'
        return

    def __len__(self):
        return len(self.inputs)

    # Function taken from https://github.com/OpenGVLab/InternVL/blob/2410d1dbf208f0e799459aff9376e5747dbf41a2/internvl_chat/internvl/train/dataset.py#L711
    def preprocess_internvl2_5(self, sources,
                               tokenizer: transformers.PreTrainedTokenizer,
                               num_image_token_list: list,
                               text_only: bool = False,
                               group_by_length: bool = False,
                               use_packed_ds: bool = False,
                               ds_name: str = None,
                               num_image: int = 1) -> Dict:
        assert len(sources) == 1, 'process only the first conversations'
        conversations = sources[0]

        if conversations[0]['from'] == 'system':
            system_prompt = conversations[0]['value']
            conversations = conversations[1:]  # remove system prompt
        else:
            system_prompt = self.conv_template.copy().system_message

        if not text_only:
            new_conversations = []
            current_image_idx = 0
            for conversation in conversations:
                if conversation['from'] == 'human':
                    image_cnt = conversation['value'].count('<image>')
                    for i in range(image_cnt):
                        if current_image_idx == num_image:
                            break
                        image_tokens = f'{IMG_START_TOKEN}{IMG_CONTEXT_TOKEN * num_image_token_list[current_image_idx]}{IMG_END_TOKEN}'
                        conversation['value'] = conversation['value'].replace('<image>', image_tokens, 1)
                        current_image_idx += 1
                new_conversations.append(conversation)
            conversations = new_conversations
            assert current_image_idx == num_image, f'{current_image_idx} != {num_image}'

        batches, roles = [], []
        if system_prompt is not None:
            batches.append(f'<|im_start|>system\n{system_prompt}<|im_end|>\n')
            roles.append('system')
        for conversation in conversations:
            if conversation['from'] == 'human':
                batches.append(f'<|im_start|>user\n{conversation["value"]}<|im_end|>\n')
                roles.append('human')
            elif conversation['from'] == 'gpt':
                batches.append(f'<|im_start|>assistant\n{conversation["value"]}<|im_end|>\n')
                roles.append('gpt')
            else:
                raise NotImplementedError

        add_bos_token = getattr(tokenizer, 'add_bos_token', False)
        if add_bos_token:  # for InternLM series
            batches[0] = tokenizer.bos_token + batches[0]

        # Tokenize conversations
        input_ids = tokenizer(
            batches,
            return_tensors='np',
            padding=False,
            max_length=tokenizer.model_max_length,
            truncation=False,
        ).input_ids

        if add_bos_token:  # for InternLM series
            input_ids = [item[1:] for item in input_ids]

        final_input_ids, final_targets = [], []
        ignore_ids = tokenizer('<|im_start|>assistant\n', return_tensors='np').input_ids[0]
        ignore_len = ignore_ids.shape[0] - 1 if add_bos_token else ignore_ids.shape[0]
        for role, input_id in zip(roles, input_ids):
            final_input_ids.append(input_id)
            if role == 'system' or role == 'human':
                final_targets.append(np.full(input_id.shape, IGNORE_TOKEN_ID))  # ignore
            elif role == 'gpt':
                target = input_id.copy()
                target[:ignore_len] = IGNORE_TOKEN_ID  # ignore loss for `<|im_start|>assistant\n`
                target[-1:] = IGNORE_TOKEN_ID  # ignore loss for `\n`
                final_targets.append(target)
            else:
                raise NotImplementedError
        input_ids = torch.tensor(np.concatenate(final_input_ids))[:tokenizer.model_max_length]
        targets = torch.tensor(np.concatenate(final_targets))[:tokenizer.model_max_length]

        padding = False if group_by_length or use_packed_ds else True
        if padding:
            current_length = input_ids.size(0)
            padding_length = tokenizer.model_max_length - current_length
            input_ids = F.pad(input_ids, (0, padding_length), value=tokenizer.pad_token_id)
            targets = F.pad(targets, (0, padding_length), value=IGNORE_TOKEN_ID)

        input_ids = input_ids.unsqueeze(0)
        targets = targets.unsqueeze(0)

        return dict(
            input_ids=input_ids.to(self.device),
            labels=targets.to(self.device),
            attention_mask=input_ids.ne(tokenizer.pad_token_id).to(self.device),
        )

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.inputs[idx]

        # 1. Process the image
        sample = {}
        image_path = item["image_path"]
        image = Image.open(image_path).convert('RGB')
        pixel_values = load_image_internvl(image, max_num=self.internvl_max_num).to(torch.bfloat16).to(self.device)
        num_patches = pixel_values.size(0)

        # 2. Get the question and answer
        options = get_options(item, all_options)
        question = item['question']
        hint = item['hint']
        full_question_str_formatted = get_question_string_internvl(question, hint, options, upd_setting=self.upd_setting, upd_type=self.upd_type)
        gt_answer = item['answer']
        # The eos token will be added in preprocess_internvl2_5
        target_str = get_target_str(gt_answer=gt_answer, upd_type=self.upd_type, item=item, eos_token=None)

        conversations = [
            {
                "from": "human",
                "value": full_question_str_formatted
            },
            {
                "from": "gpt",
                "value": target_str
            }
        ]

        # 3. Preprocess the conversations and generate the return dictionary
        ret = self.preprocess_internvl2_5(sources=[copy.deepcopy(conversations)],
                                          tokenizer=self.tokenizer, num_image_token_list=[self.num_image_token * num_patches],
                                          group_by_length=True,
                                          use_packed_ds=False)

        # 4. Calculate position_ids
        position_ids = ret['attention_mask'].long().cumsum(-1) - 1
        position_ids.masked_fill_(ret['attention_mask'] == 0, 1)
        image_end_token_id = self.tokenizer.convert_tokens_to_ids(IMG_END_TOKEN)
        assert (ret['input_ids'][0] == image_end_token_id).sum() == 1

        # 5. Update the sample dictionary
        sample.update(dict(
            input_ids=ret['input_ids'][0],
            labels=ret['labels'][0],
            attention_mask=ret['attention_mask'][0],
            position_ids=position_ids[0],
            pixel_values=pixel_values,
            image_flags=torch.tensor([1] * num_patches, dtype=torch.long)
        ))

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


# Function is based on https://github.com/OpenGVLab/InternVL/blob/2410d1dbf208f0e799459aff9376e5747dbf41a2/internvl_chat/internvl/patch/pad_data_collator.py#L57
def collate_func_internvl(batch):
    joined_batch = {}

    first = batch[0]

    batch_lens = [feat['input_ids'].shape for feat in batch]
    max_item_length = max(batch_lens)[0]
    pad_id = 0

    for idx in range(len(batch)):
        feat = batch[idx]
        temp_input_ids = torch.LongTensor([pad_id] * max_item_length)
        temp_input_ids[:feat['input_ids'].shape[0]] = feat['input_ids']
        feat['input_ids'] = temp_input_ids
        temp_labels = torch.LongTensor([IGNORE_TOKEN_ID] * max_item_length)
        temp_labels[:feat['labels'].shape[0]] = feat['labels']
        feat['labels'] = temp_labels
        feat['attention_mask'] = feat['input_ids'].ne(pad_id)

        if 'position_ids' in feat:
            temp_position_ids = torch.LongTensor([pad_id] * max_item_length)
            temp_position_ids[:feat['position_ids'].shape[0]] = feat['position_ids']
            feat['position_ids'] = temp_position_ids

        if 'loss_weight' in feat:
            temp_loss_weight = torch.FloatTensor([pad_id] * max_item_length)
            temp_loss_weight[:feat['loss_weight'].shape[0]] = feat['loss_weight']
            feat['loss_weight'] = temp_loss_weight

    other_keys = []
    for k, v in first.items():
        # In cases not defined in InternVl - will we hande them later
        if k not in ['input_ids', 'labels', 'attention_mask', 'position_ids', 'pixel_values', 'image_flags']:
            other_keys.append(k)
            continue
        if k not in ('label', 'label_ids', 'pixel_values', 'image_flags') and v is not None and not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                joined_batch[k] = torch.stack([f[k] for f in batch])
            elif isinstance(v, np.ndarray):
                joined_batch[k] = torch.tensor(np.stack([f[k] for f in batch]))
            else:
                joined_batch[k] = torch.tensor([f[k] for f in batch])
        if k in ('pixel_values', 'image_flags'):
            if isinstance(v, torch.Tensor):
                joined_batch[k] = torch.concat([f[k] for f in batch])
            elif isinstance(v, np.ndarray):
                joined_batch[k] = torch.concat(np.stack([f[k] for f in batch]))
            else:
                joined_batch[k] = torch.concat([f[k] for f in batch])

    for k in other_keys:
        if k == 'clip_signal':
            joined_batch['clip_signal'] = torch.stack([b['clip_signal'] for b in batch])
        elif k in ['image_sizes', 'question_str', 'gt_answer_str', 'type', 'options', 'choices',
                   'original_gt_answer', 'question_id', 'masked_answer', 'upd_challenge_type']:
            joined_batch[k] = [b[k] for b in batch]
        else:
            assert False

    return joined_batch
