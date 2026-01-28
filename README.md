# CLIP-UP: CLIP-Based Unanswerable Problem Detection for Visual Question Answering (WACV 2026 Oral Presentation)

[![arXiv](https://img.shields.io/badge/arXiv-2501.01371-b31b1b)](https://arxiv.org/abs/2501.01371)
[![Project](https://img.shields.io/badge/Project-Website-red)](https://benvr.github.io/CLIP-UP)

# Abstract

> Vision-Language Models (VLMs) demonstrate remarkable capabilities in visual understanding and reasoning, such as in Visual Question Answering (VQA), where the model is asked a question related to a visual input. 
Still, these models can make distinctly unnatural errors, for example, providing (wrong) answers to unanswerable VQA questions, such as questions asking about objects that do not appear in the image.
>
> To address this issue, we propose CLIP-UP: CLIP-based Unanswerable Problem detection, a novel lightweight method for equipping VLMs with the ability to withhold answers to unanswerable questions. 
CLIP-UP leverages CLIP-based similarity measures to extract question-image alignment information to detect unanswerability, requiring efficient training of only a few additional layers, while keeping the original VLMs' weights unchanged.
>
> Tested across several models, CLIP-UP achieves significant improvements on benchmarks assessing unanswerability in both multiple-choice and open-ended VQA, surpassing other methods, while preserving original performance on other tasks.

# Table of Contents

- [Requirements](#requirements)
- [Training](#training)
- [Inference](#inference)
- [License](#license)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)

# Requirements

## Installation

To install the conda environment, please run:
```
conda env create -f environment/environment.yml
conda activate clip_up

# Install LLaVA after conda environment, without clashing dependencies
pip install --no-deps git+https://github.com/haotian-liu/LLaVA.git@v1.2.2.post1
```

## Data

We share the following datasets:
1. Our novel multiple-choice VQA training and validation dataset. The data is available in [this link](https://drive.google.com/file/d/1Rt7kqq5H7Te7ZHy-LBbFqCjzjSr3vkiY/view?usp=sharing).
2. The dataset used for open-ended VQA training and validation, based on the TDIUC dataset. The data is available in [this link](https://drive.google.com/file/d/1jSdvJHqxVmAd2ouhBJUa-YUrjmd1ObaF/view?usp=sharing).
3. The dataset used for open-ended VQA testing, based on the RGQA dataset. The data is available in [this link](https://drive.google.com/file/d/1VB6CzuLzGJ28xCpDCDaSLbSJUNvBR3Tt/view?usp=sharing).

The multiple-choice VQA testing dataset is [MM-UPD](https://huggingface.co/datasets/MM-UPD/MM-UPD), and is automatically downloaded in code.

The data should be organized as follows:
```
data_root
├── multiple_choice_train_and_val
├── open_ended_train_and_val
└── open_ended_test
```

Each folder should contain the corresponding extracted zip file (downloaded from the above link). The data root path should be set in `data_root` in `clip_up/vlms/configs/base.yaml`.

Please note that only the datasets required for the intended experiments need to be downloaded. For example, if running only multiple-choice VQA, the open-ended datasets are not required."

## Other Dependencies

Use [this link](https://drive.google.com/file/d/1b7GYEf9kRSJ0S9zmFIQeCi_5Br91jZkw/view?usp=sharing) to download the Structure-CLIP checkpoint. 
The checkpoint path should be set in `structure_clip_checkpoint_path` in `clip_up/vlms/configs/base.yaml`

The code uses GPT-3.5 for multiple-choice and open-ended evaluation when simple string-matching evaluation is insufficient.
For this, an OpenAI API key is required. Please see the [OpenAI documentation](https://platform.openai.com/docs/quickstart) for instructions to create an API key.
The key should be set in `open_ai_api_key` in `clip_up/vlms/configs/base.yaml`.

# Training

Use `train.py` to train CLIP-UP for unanswerable question detection. The code supports the following options:
1. Training for multiple-choice or open-ended VQA (via the `vqa_type` argument).
2. CLIP-UP-Emb or CLIP-UP-Emb-LoRA injection types (via the `injection_type` argument). CLIP-UP-Emb trains only a single projection layer, while CLIP-UP-Emb-LoRA additionally trains Injected LoRA layers.
3. Three different VLMs: LLaVA-1.5-7B, InternVL3-1B, and InternVL3-8B (via the `vlm` argument).

The training code uses [🤗 Accelerate](https://github.com/huggingface/accelerate), which enables straightforward multi-GPU execution.
Below are example `train.py` commands:

```
# Train CLIP-UP-Emb-LoRA for multiple-choice VQA, on LLaVA-1.5-7B, with two GPUs
CUDA_VISIBLE_DEVICES=0,1 python -m accelerate.commands.launch --num_processes=2 --multi_gpu --num_machines 1 --mixed_precision no --dynamo_backend no train.py --vlm llava-1.5-7b --output_root_dir train_outputs --vqa_type multiple_choice --injection_type clip_up_emb_lora

# Train CLIP-UP-Emb-LoRA for multiple-choice VQA, on LLaVA-1.5-7B, with one GPU
CUDA_VISIBLE_DEVICES=0 python -m accelerate.commands.launch --num_processes=1 --num_machines 1 --mixed_precision no --dynamo_backend no train.py --vlm llava-1.5-7b --output_root_dir train_outputs --vqa_type multiple_choice --injection_type clip_up_emb_lora

# Train CLIP-UP-Emb-LoRA for open-ended VQA, on LLaVA-1.5-7B, with one GPU
CUDA_VISIBLE_DEVICES=0 python -m accelerate.commands.launch --num_processes=1 --num_machines 1 --mixed_precision no --dynamo_backend no train.py --vlm llava-1.5-7b --output_root_dir train_outputs --vqa_type open_ended --injection_type clip_up_emb_lora

# Train CLIP-UP-Emb for multiple-choice VQA, on InternVL3-8B, with one GPU
CUDA_VISIBLE_DEVICES=0 python -m accelerate.commands.launch --num_processes=1 --num_machines 1 --mixed_precision no --dynamo_backend no train.py --vlm internvl3-8b --output_root_dir train_outputs --vqa_type multiple_choice  --injection_type clip_up_emb

# Train CLIP-UP-Emb for multiple-choice VQA, on InternVL3-1B, with one GPU
CUDA_VISIBLE_DEVICES=0 python -m accelerate.commands.launch --num_processes=1 --num_machines 1 --mixed_precision no --dynamo_backend no train.py --vlm internvl3-1b --output_root_dir train_outputs --vqa_type multiple_choice --injection_type clip_up_emb
```

To change hyperparameters, modify `clip_up/vlms/configs/base.yaml` or the VLM-specific configuration files in `clip_up/vlms/configs/`.

# Inference

The `inference.py` Python script enables inference and evaluation of VLMs equipped with CLIP-UP. This script does not use Accelerate.

The `checkpoint` argument should point to the CLIP-UP checkpoint produced during training (e.g., `checkpoint_epoch_2.pt` after three epochs). Note that full multiple-choice VQA inference requires running the script three times, once for each multiple-choice type (AAD, IASD, and IVQD).

Below are a few example commands:

```
# Inference on LLaVA-1.5-7B with CLIP-UP-Emb-LoRA for AAD multiple-choice VQA
CUDA_VISIBLE_DEVICES=0 python inference.py --vlm llava-1.5-7b --output_root_dir inference_outputs --use_clip_up --checkpoint <clip_up_training_checkpoint_path> --upd_type aad --injection_type clip_up_emb_lora

# Inference on LLaVA-1.5-7B with CLIP-UP-Emb-LoRA for IASD multiple-choice VQA
CUDA_VISIBLE_DEVICES=0 python inference.py --vlm llava-1.5-7b --output_root_dir inference_outputs --use_clip_up --checkpoint <clip_up_training_checkpoint_path> --upd_type iasd --injection_type clip_up_emb_lora

# Inference on LLaVA-1.5-7B with CLIP-UP-Emb-LoRA for IVQD multiple-choice VQA
CUDA_VISIBLE_DEVICES=0 python inference.py --vlm llava-1.5-7b --output_root_dir inference_outputs --use_clip_up --checkpoint <clip_up_training_checkpoint_path> --upd_type ivqd --injection_type clip_up_emb_lora

# Inference on LLaVA-1.5-7B with CLIP-UP-Emb-LoRA for open-ended VQA
CUDA_VISIBLE_DEVICES=0 python inference.py --vlm llava-1.5-7b --output_root_dir inference_outputs --use_clip_up --checkpoint <clip_up_training_checkpoint_path> --upd_type open_ended --injection_type clip_up_emb_lora

# Inference on InternVL3-8B with CLIP-UP-Emb for open-ended VQA
CUDA_VISIBLE_DEVICES=0 python inference.py --vlm internvl3-8b --output_root_dir inference_outputs --use_clip_up --checkpoint <clip_up_training_checkpoint_path> --upd_type open_ended --injection_type clip_up_emb
```

The code also supports inference and evaluation for plain VLMs without CLIP-UP, using the prompt-engineering baselines reported in the main experiments of the paper. Below are a few example commands:

```
# Inference on LLaVA-1.5-7B for IVQD multiple-choice VQA, with the original multiple-choice prompt setting
CUDA_VISIBLE_DEVICES=0 python inference.py --vlm llava-1.5-7b --output_root_dir inference_outputs --upd_type ivqd --upd_setting original

# Inference on LLaVA-1.5-7B for IVQD multiple-choice VQA, with the base multiple-choice prompt setting
CUDA_VISIBLE_DEVICES=0 python inference.py --vlm llava-1.5-7b --output_root_dir inference_outputs --upd_type ivqd --upd_setting base

# Inference on LLaVA-1.5-7B for IVQD multiple-choice VQA, with the additional-option multiple-choice prompt setting
CUDA_VISIBLE_DEVICES=0 python inference.py --vlm llava-1.5-7b --output_root_dir inference_outputs --upd_type ivqd --upd_setting option

# Inference on LLaVA-1.5-7B for IVQD multiple-choice VQA, with the additional-instruction multiple-choice prompt setting
CUDA_VISIBLE_DEVICES=0 python inference.py --vlm llava-1.5-7b --output_root_dir inference_outputs --upd_type ivqd --upd_setting inst

# Inference on LLaVA-1.5-7B for IVQD open-ended VQA, with the original open-ended setting (note that the open-ended original setting is called 'base' in code)
CUDA_VISIBLE_DEVICES=0 python inference.py --vlm llava-1.5-7b --output_root_dir inference_outputs --upd_type open_ended --upd_setting base

# Inference on LLaVA-1.5-7B for IVQD open-ended VQA, with the "prompt engineering" open-ended setting (note that the "prompt engineering" setting is called 'inst' in code)
CUDA_VISIBLE_DEVICES=0 python inference.py --vlm llava-1.5-7b --output_root_dir inference_outputs --upd_type open_ended --upd_setting inst
```

# License 

1. The CLIP-UP code is released under the MIT license.
2. The novel multiple-choice training and validation dataset builds on several source datasets (see Appendix B.1. in the paper) and is therefore subject to the licenses of the original data sources. Consequently, this dataset is intended for non-commercial research and educational use only. The same applies to the open-ended training, validation, and test datasets.
3. The [Structure-CLIP code](https://github.com/zjukg/Structure-CLIP) used to generate the Structure-CLIP weights is released under the MIT license, and its use in this project complies with the terms of that license.

# Acknowledgements 

Our code builds on many existing repositories. We specifically acknowledge the following projects:
1. [Unsolvable Problem Detection: Robust Understanding Evaluation for Large Multimodal Models](https://github.com/AtsuMiyai/UPD/) for multiple-choice VQA evaluation and baselines.
2. [Structure-CLIP: Towards Scene Graph Knowledge to Enhance Multi-modal Structured Representations](https://github.com/zjukg/Structure-CLIP) for Structure-CLIP training code.
3. [LLaVA](https://github.com/haotian-liu/LLaVA) and [InternVL3](https://huggingface.co/OpenGVLab/InternVL3-1B) for the underlying VLM implementations and weights.

# Citation
If you use our work, please cite:
```
@misc{vardi2025clipupclipbasedunanswerableproblem,
      title={CLIP-UP: CLIP-Based Unanswerable Problem Detection for Visual Question Answering}, 
      author={Ben Vardi and Oron Nir and Ariel Shamir},
      year={2025},
      eprint={2501.01371},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2501.01371}, 
}
```
