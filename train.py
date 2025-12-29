import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"  # Allow determinism
os.environ["TOKENIZERS_PARALLELISM"] = "false" # Set to disable warning
import argparse
from accelerate import Accelerator

from clip_up.vlms.clip_up_vlms import init_vlm, get_vlm_path_and_config, VLM_NAMES
from clip_up.clip_up_utils import ValidationSetInfo, TrainingSetInfo, InjectionType
from clip_up.clip_up_wrapper import ClipUpWrapper
from clip_up.clip_up_general_utils import seed_everything, \
    print_args, now, set_output_dir_and_logging, read_jsonl


def main(args):
    # 1. Get model path and config
    model_name = args.vlm
    model_path, model_config = get_vlm_path_and_config(model_name)

    # 2. Set accelerator, set seed, and outputs
    accelerator = Accelerator()
    seed_everything(seed=42)

    run_name = args.run_name if len(args.run_name) > 0 else f'{now()}'
    output_dir = f'{args.output_root_dir}/{model_name}/{args.vqa_type}_train_results/{run_name}'
    set_output_dir_and_logging(output_dir)

    if accelerator.is_local_main_process:
        print(f"Accelerator number of processes: {accelerator.num_processes}")
        print_args(args)

    # 3. Get model
    device = 'cuda'
    is_for_open_ended_questions = args.vqa_type == "open_ended"
    model, image_processor, tokenizer, conv_mode = (
        init_vlm(model_name=model_name, model_path=model_path, model_config=model_config, is_clip_up_emb_lora_injection=args.injection_type == InjectionType.CLIP_UP_EMB_LORA,
             is_for_open_ended_questions=is_for_open_ended_questions, device=device))

    # 4. Init wrapper
    clip_up_wrapper = ClipUpWrapper(model=model, model_name=model_name, model_config=model_config, conv_mode=conv_mode, tokenizer=tokenizer,
                                image_processor=image_processor, device=device,
                                output_dir=output_dir, accelerator=accelerator,
                                is_for_open_ended_questions=is_for_open_ended_questions,
                                injection_type=args.injection_type)
    clip_up_wrapper.model.gradient_checkpointing_enable()

    # 4. Prepare data
    train_sets_infos, val_sets_infos = get_train_and_vals_sets_infos(vqa_type=args.vqa_type, data_root=model_config.data_root)

    train_dataloader, val_sets_infos, train_dataloader_for_warmup = \
        clip_up_wrapper.prepare_train_and_val_data(train_sets_infos=train_sets_infos, val_sets_infos=val_sets_infos)

    # 4. Train
    clip_up_wrapper.train(train_dataloader=train_dataloader, val_sets_infos=val_sets_infos, train_dataloader_for_warmup=train_dataloader_for_warmup)

    return


def get_train_and_vals_sets_infos(vqa_type, data_root):
    data_folder_for_vqa_type = os.path.join(data_root, f'{vqa_type}_train_and_val')

    upd_types = ["aad", "iasd", "ivqd"] if vqa_type == "multiple_choice" else ["open_ended"]

    train_sets_infos = []
    for curr_upd_type in upd_types:
        curr_train_dataset_file = os.path.join(data_folder_for_vqa_type, f'train_{curr_upd_type}_annotations.jsonl')
        curr_train_items = read_jsonl(curr_train_dataset_file)
        for curr_example in curr_train_items:
            curr_example['image_path'] = os.path.join(data_folder_for_vqa_type, curr_example['image_path'])
        curr_train_info_set = TrainingSetInfo(curr_upd_type, curr_train_items)
        train_sets_infos.append(curr_train_info_set)

    val_sets_infos = []
    for curr_upd_type in upd_types:
        curr_val_dataset_file = os.path.join(data_folder_for_vqa_type, f'val_{curr_upd_type}_annotations.jsonl')
        curr_val_items = read_jsonl(curr_val_dataset_file)
        for curr_example in curr_val_items:
            curr_example['image_path'] = os.path.join(data_folder_for_vqa_type, curr_example['image_path'])
        curr_val_info_set = ValidationSetInfo(curr_upd_type, curr_val_items)
        val_sets_infos.append(curr_val_info_set)

    return train_sets_infos, val_sets_infos


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--vlm', type=str, choices=VLM_NAMES, required=True)
    parser.add_argument("--injection_type", type=InjectionType, choices=list(InjectionType), required=True)
    parser.add_argument('--vqa_type', type=str, choices=["multiple_choice", "open_ended"], required=True)
    parser.add_argument("--output_root_dir", type=str, required=True)
    parser.add_argument('--run_name', type=str, default='')

    args = parser.parse_args()
    main(args)
