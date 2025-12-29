import os
import pickle
import clip
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Sampler
from tqdm import tqdm
from pathlib import Path

from clip_up.clip_up_general_utils import load_image_from_base64
from clip_up.clip_up_utils import get_question_with_answers_list_from_example
from prompting.prompting_utils import all_options, get_options


class StructureClipSignalExtractor:
    def __init__(self, device, torch_dtype, structure_clip_checkpoint_path, accelerator=None):
        self.device = device
        self.accelerator = accelerator
        self.torch_dtype = torch_dtype

        self.structure_clip_model, self.structure_clip_preprocess = clip.load("ViT-L/14@336px", device=device)
        loaded_cpkts = torch.load(structure_clip_checkpoint_path, map_location=torch.device(device), weights_only=True)
        self.structure_clip_model.load_state_dict(loaded_cpkts, strict=True)
        self.structure_clip_model.eval()
        self.structure_clip_model.requires_grad_(False)
        return

    def process_structure_clip_inputs(self, questions, image):
        image = image.convert("RGB")
        image = self.structure_clip_preprocess(image).unsqueeze(0).to(self.device)
        text_tokens = clip.tokenize(questions, truncate=True).to(self.device)
        clip_inputs = {'image': image, 'text_tokens': text_tokens}
        return clip_inputs

    def extract_clip_signal(self, questions, image):
        inputs = self.process_structure_clip_inputs(questions=questions, image=image)
        signal = self.get_structure_clip_signal(inputs).to(self.torch_dtype).to('cpu')
        return signal

    def get_structure_clip_signal(self, clip_inputs):
        with torch.inference_mode():
            image_features = self.structure_clip_model.encode_image(clip_inputs['image'])
            image_features = image_features / image_features.norm(dim=1, keepdim=True)

            text_features = self.structure_clip_model.encode_text(clip_inputs['text_tokens'])
            text_features = text_features / text_features.norm(dim=1, keepdim=True)

            clip_signal = (image_features * text_features).detach()

        return clip_signal

    @staticmethod
    def get_clip_signal_cache_key(image_path, questions_list_str):
        return image_path + ';;' + questions_list_str

    @staticmethod
    def get_clip_signal_for_one_example(example, clip_signal_extractor, should_use_cache, cached_clip_up_signals_dict):
        # 1. Get texts
        question_with_answers_list = get_question_with_answers_list_from_example(question=example['question'],
                                                                                 options=get_options(example, all_options))

        # 2. Get image and image_cache_key
        if 'image' in example:
            image_base_64 = example['image']
            image_cache_key = image_base_64
            image = load_image_from_base64(image_base_64)
        else:
            curr_image_path = example["image_path"]
            image_cache_key = curr_image_path
            image = Image.open(curr_image_path)

        # 3. Get key for cache dict
        clip_up_signal_cache_key = StructureClipSignalExtractor.get_clip_signal_cache_key(image_cache_key, str(question_with_answers_list))

        # 4. Extract signal or get from cache
        if not should_use_cache:
            clip_up_signal = clip_signal_extractor.extract_clip_signal(questions=question_with_answers_list, image=image).flatten()
        else:
            clip_up_signal = cached_clip_up_signals_dict[clip_up_signal_cache_key]

        return clip_up_signal, clip_up_signal_cache_key

    @staticmethod
    def add_clip_up_signals_to_data(clip_signal_extractor, data_list, force_run_clip, signals_cache_path, tqdm_desc):
        # 1. Load signals from cache
        cached_clip_up_signals_dict = None
        should_use_cache = not force_run_clip and os.path.exists(signals_cache_path)
        if should_use_cache:
            with open(signals_cache_path, 'rb') as file:
                cached_clip_up_signals_dict = pickle.load(file)

        # 2. Run loop
        clip_signals_dict = {}
        for curr_example in tqdm(data_list, desc=tqdm_desc):
            curr_clip_signal, curr_clip_signal_key = \
                StructureClipSignalExtractor.get_clip_signal_for_one_example(curr_example, clip_signal_extractor,
                                                                            should_use_cache, cached_clip_up_signals_dict)

            curr_example['clip_signal'] = curr_clip_signal

            if not should_use_cache:
                clip_signals_dict[curr_clip_signal_key] = curr_example['clip_signal']

        # 3. Save signals to pickle file
        if not should_use_cache:
            if clip_signal_extractor.accelerator is None or clip_signal_extractor.accelerator.is_local_main_process:
                os.makedirs(Path(signals_cache_path).parent, exist_ok=True)
                with open(signals_cache_path, 'wb') as file:
                    pickle.dump(clip_signals_dict, file)

        return data_list


class LinearSignalProjection(torch.nn.Module):
    def __init__(self, in_dim, out_dim, device, torch_dtype, bias=True):
        super().__init__()
        self.linear_projection = nn.Linear(in_dim, out_dim, device=device, dtype=torch_dtype, bias=bias)
        return

    def forward(self, x):
        x = self.linear_projection(x)
        return x


class ExtendedVisionProjection(nn.Module):
    def __init__(self, original_projection, language_model_dim, structure_clip_dim, is_for_open_ended_questions,
                 device, torch_dtype):
        super().__init__()
        self.original_projection = original_projection
        self.device = device
        self.torch_dtype = torch_dtype

        self.in_clip_up_dim = structure_clip_dim if is_for_open_ended_questions else structure_clip_dim * 4
        self.out_dim = language_model_dim

        self.clip_up_projection = LinearSignalProjection(self.in_clip_up_dim, self.out_dim, device=self.device,
                                                     torch_dtype=self.torch_dtype)
        self.clip_up_projection.requires_grad_(True)

        return

    def forward(self, x, clip_signal):
        # 1. Run original projection
        original_out = self.original_projection(x.to(self.original_projection[0].weight.dtype) if hasattr(self.original_projection[0], 'weight') else x)

        if clip_signal is None:
            return original_out, None

        # 2. Project clip info into the LM space
        clip_up_embedding = self.clip_up_projection(clip_signal).unsqueeze(1)

        return original_out, clip_up_embedding


class ClipUpSampler(Sampler):
    def __init__(self, train_upd_datasets, train_standard_datasets):
        super().__init__()
        num_upd_total_samples = sum([len(s) for s in train_upd_datasets])
        num_standard_total_samples = sum([len(s) for s in train_standard_datasets])
        assert num_upd_total_samples == num_standard_total_samples

        self.num_upd_or_standard_samples = num_upd_total_samples
        self.num_all_samples = num_upd_total_samples + num_standard_total_samples
        return

    def __iter__(self):
        indices_upd = np.random.permutation(self.num_upd_or_standard_samples)
        indices_standard = indices_upd + self.num_upd_or_standard_samples
        indices_structure = np.stack([indices_upd, indices_standard], axis=1)
        indices_structure = indices_structure.flatten()

        return iter(indices_structure)

    def __len__(self):
        return self.num_all_samples
