import os
os.environ["TOKENIZERS_PARALLELISM"] = "false" # Set to disable warning
import json
import argparse
import time
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset

from clip_up.vlms.clip_up_vlms import init_vlm, get_vlm_path_and_config, VLM_NAMES, run_vlm_inference_for_single_item
from eval.multiple_choice_eval import evaluate_multiple_choice_results
from eval.open_ended_eval import evaluate_open_ended_results
from clip_up.clip_up_wrapper import StructureClipSignalExtractor, ClipUpWrapper
from prompting.prompting_utils import all_options, get_options, build_choices, get_circular_item_index
from clip_up.clip_up_general_utils import now, seed_everything, print_args, load_image_from_base64, disable_torch_init, \
    set_output_dir_and_logging, read_jsonl, set_open_ai_client
from clip_up.clip_up_utils import output_clip_up_projection_t_sne, InjectionType


def main(args):
    # 1. Get model path and config
    model_name = args.vlm
    model_path, model_config = get_vlm_path_and_config(model_name)
    upd_setting = args.upd_setting

    # 2. Set seed and outputs
    seed_everything(42, with_accelerate=False)
    output_dir = f'{args.output_root_dir}/{model_name}/{args.upd_type}_results/{now()}{args.run_name_suffix}'
    set_output_dir_and_logging(output_dir)
    print_args(args)

    # 3. Verify input validity
    if args.use_clip_up:
        assert args.checkpoint is not None, 'Must load some checkpoint if using CLIP-UP'
        assert args.injection_type is not None, 'Must define injection type if using CLIP-UP'
        assert upd_setting is None, 'upd_setting is meant for running plain (non-CLIP-UP) model'
    else:
        assert upd_setting is not None, 'upd_setting is should be defined for running plain (non-CLIP-UP) model'

    if upd_setting:
        assert args.checkpoint is None and args.injection_type is None and not args.use_clip_up, 'upd_setting is meant for running plain (non-CLIP-UP) model'

    # 4. Get model
    disable_torch_init()
    device = 'cuda'
    is_for_open_ended_questions = 'open_ended' == args.upd_type
    model, image_processor, tokenizer, conv_mode = (
        init_vlm(model_name=model_name, model_path=model_path, model_config=model_config,
                 is_clip_up_emb_lora_injection=args.injection_type == InjectionType.CLIP_UP_EMB_LORA,
                 is_for_open_ended_questions=is_for_open_ended_questions, device=device))

    # 5. Init wrapper and load checkpoint
    if args.use_clip_up:
        clip_up_wrapper = ClipUpWrapper(model=model, model_name=model_name, model_config=model_config, conv_mode=conv_mode, tokenizer=tokenizer,
                                      image_processor=image_processor, device=device,
                                      output_dir=output_dir,
                                      accelerator=None,
                                      is_for_open_ended_questions=is_for_open_ended_questions,
                                      injection_type=args.injection_type)

        upd_setting = clip_up_wrapper.upd_setting
        clip_up_wrapper.load_checkpoint(checkpoint_path=args.checkpoint)
        open_ai_client = clip_up_wrapper.open_ai_client
    else:
        open_ai_client = set_open_ai_client(model_config.open_ai_api_key)

    model.requires_grad_(False)
    model.eval()

    # 6. Prepare data
    data_list = get_test_data(upd_type=args.upd_type, upd_setting=upd_setting, data_root=model_config.data_root)
    print(f'Running on {len(data_list)} samples')

    # 7. Add CLIP-UP signals to data and output CLIP-UP projection t-SNE
    if args.use_clip_up:
        data_list = add_clip_up_signals_to_test_data(data_list, clip_up_wrapper=clip_up_wrapper, upd_type=args.upd_type)
        output_clip_up_projection_t_sne(data_list, clip_up_wrapper.get_clip_up_projection(), args.upd_type, output_dir=output_dir)

    # 8. Run inference
    run_inference(data_list, model, model_name, conv_mode, tokenizer, image_processor, output_dir,
                  upd_setting, args.upd_type, open_ai_client=open_ai_client)

    return


def get_test_data(upd_type, upd_setting, data_root):
    if upd_type in ['aad', 'iasd', 'ivqd']:
        # Load MM-UPD
        dataset_name = 'base' if upd_setting in ['inst', 'base', 'original'] else upd_setting  # Instruction and original use the base dataset
        upd_type_and_setting = f'{upd_type}_{dataset_name}'
        data_list = load_dataset("MM-UPD/MM-UPD", name=f'mm{upd_type_and_setting}', trust_remote_code=True)['test']
        data_list = list(data_list)
    elif upd_type in ['open_ended']:
        # Load open-ended data
        data_folder_for_upd_type = os.path.join(data_root, f'{upd_type}_test')
        test_dataset_file = os.path.join(data_folder_for_upd_type, f'test_{upd_type}_annotations.jsonl')
        data_list = read_jsonl(test_dataset_file)
        for curr_example in data_list:
            curr_example['image_path'] = os.path.join(data_folder_for_upd_type, curr_example['image_path'])
    else:
        assert False

    data_list = sorted(data_list, key=lambda x: get_circular_item_index(x['index']))
    return data_list


def add_clip_up_signals_to_test_data(data_list, clip_up_wrapper, upd_type):
    clip_signal_extractor = StructureClipSignalExtractor(device=clip_up_wrapper.device,
                                                         torch_dtype=clip_up_wrapper.torch_dtype,
                                                         structure_clip_checkpoint_path=clip_up_wrapper.model_config.structure_clip_checkpoint_path)

    signals_cache_path = f'clip_up_cached_signals/test/test_{upd_type}.pkl'
    data_list = StructureClipSignalExtractor.add_clip_up_signals_to_data(clip_signal_extractor, data_list,
                                                                         force_run_clip=False,
                                                                         signals_cache_path=signals_cache_path,
                                                                         tqdm_desc="Extracting CLIP-UP signals")
    return data_list



def run_inference(data_list, model, model_name, conv_mode, tokenizer, image_processor,
                  output_dir, upd_setting, upd_type, open_ai_client):
    # 1. Create results file
    results_file = open(f'{output_dir}/results.jsonl', "w")

    # 2. Run inference
    print(f'*** Start of VLM inference ***')
    results_standard, results_upd = [], []
    start_time = time.time()
    for curr_sample in tqdm(data_list, total=len(data_list)):
        curr_sample_dict = run_inference_for_single_item(sample=curr_sample, model=model, model_name=model_name,
                                                         conv_mode=conv_mode, tokenizer=tokenizer,
                                                         image_processor=image_processor,
                                                         upd_setting=upd_setting, upd_type=upd_type)

        results_file.write(json.dumps(curr_sample_dict) + "\n")
        results_file.flush()
        if curr_sample['type'] == 'standard':
            results_standard.append(curr_sample_dict)
        elif curr_sample['type'] == 'upd':
            results_upd.append(curr_sample_dict)
        else:
            assert False

    # 3. Print time and close file
    end_time = time.time()
    execution_time_seconds = end_time - start_time
    print(f"Time taken to perform inference (total): {execution_time_seconds / 60:.2f} minutes")
    results_file.close()
    print(f'*** End of VLM inference ***')

    # 4. Evaluate standard and upd accuracies
    print(f'*** Start of evaluation of VLM inference results ***')
    results_eval_info_file_path = f'{output_dir}/results_with_eval_info.jsonl'
    if upd_type in ['aad', 'iasd', 'ivqd']:
        standard_accuracy, circ_standard_accuracy, upd_accuracy, circ_upd_accuracy, dual_accuracy, num_gpt_calls_to_gpt = (
            evaluate_multiple_choice_results(open_ai_client=open_ai_client, standard_results=results_standard,
                                             upd_results=results_upd, upd_type=upd_type,
                                             upd_setting=upd_setting, epoch=None, results_eval_info_file_path=results_eval_info_file_path))
        print(f'Evaluating: there were {num_gpt_calls_to_gpt} calls to GPT.')
    elif upd_type in ['open_ended']:
        standard_accuracy, upd_accuracy, dual_accuracy, num_calls_to_lave = evaluate_open_ended_results(open_ai_client=open_ai_client, standard_results=results_standard,
        upd_results=results_upd, upd_type=upd_type, epoch=None, results_eval_info_file_path=results_eval_info_file_path)
        print(f'Evaluating: there were {num_calls_to_lave} calls to LAVE.')
    else:
        assert False
    print(f'*** End of evaluation of VLM inference results ***')

    return


def run_inference_for_single_item(sample, model, model_name, conv_mode, tokenizer, image_processor,
                                  upd_setting, upd_type):
    # 1. Get sample data
    options = get_options(sample, all_options)
    question = sample['question']
    hint = sample['hint']
    image = Image.open(sample['image_path']) if 'image_path' in sample else load_image_from_base64(sample['image'])
    clip_signal = sample["clip_signal"] if "clip_signal" in sample else None

    # 2. Run model
    output, full_prompt = run_vlm_inference_for_single_item(model=model, model_name=model_name, conv_mode=conv_mode,
                                                            tokenizer=tokenizer, image_processor=image_processor,
                                                            upd_setting=upd_setting,
                                                            upd_type=upd_type, question=question, hint=hint,
                                                            options=options, image=image, clip_signal=clip_signal)

    # 3. Return result
    sample_dict = {"question_id": sample['index'],
                    "type": sample['type'],
                    "question_str": question,
                    "output": output,
                    "options": options,
                    "option_char": all_options[:len(options)],
                    "model_name": model_name,
                    "prompt_detail": full_prompt,
                    "original_gt_answer": sample['answer'],
                    "choices": build_choices(sample),
                    "category": sample['category'],
                    "l2-category": sample['l2-category'],
                    'masked_answer': sample['masked_answer']}

    return sample_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--vlm', type=str, choices=VLM_NAMES, required=True)
    parser.add_argument("--injection_type", type=InjectionType, choices=list(InjectionType), default=None)
    parser.add_argument('--upd_type', type=str, choices=["aad", "iasd", "ivqd", "open_ended"], required=True)
    parser.add_argument("--use_clip_up", action="store_true")
    parser.add_argument("--upd_setting", type=str, choices=["original", "base", "option", "inst"], default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output_root_dir", type=str, required=True)
    parser.add_argument('--run_name_suffix', type=str, default='')

    args = parser.parse_args()
    main(args)

