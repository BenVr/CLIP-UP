import contextlib
import os
from collections import defaultdict
from pathlib import Path

import math
import torch
from torch.utils.data import DataLoader, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import get_scheduler

from clip_up.clip_up_general_utils import is_peft_model, set_open_ai_client
from clip_up.clip_up_utils import SupConLoss
from clip_up.clip_up_utils import get_clip_up_learnable_params_groups, get_learnable_lora_and_clip_up_params, \
    InjectionType
from clip_up.vlms.clip_up_vlms import add_clip_up_to_vlm, get_vlm_dataset_class_and_extra_kwargs, get_vlm_collate_function, \
    get_generation_args, get_vlm_generation_output, run_vlm_forward, compute_vlm_per_sample_loss
from clip_up.clip_up_wrapper_internals import StructureClipSignalExtractor, ClipUpSampler
from eval.multiple_choice_eval import evaluate_multiple_choice_results
from eval.open_ended_eval import evaluate_open_ended_results
from prompting.prompting_utils import get_circular_item_index


class ClipUpWrapper:
    def __init__(self, model, model_name, model_config, conv_mode, tokenizer, image_processor, device,
                 output_dir, accelerator, is_for_open_ended_questions, injection_type):
        self.model = model
        self.unwrapped_model = model
        self.is_peft_model = is_peft_model(self.model)

        self.model_name = model_name
        self.model_config = model_config

        self.conv_mode = conv_mode
        self.tokenizer = tokenizer
        self.image_processor = image_processor

        self.device = device
        self.torch_dtype = torch.bfloat16
        self.output_path = output_dir
        self.accelerator = accelerator
        self.upd_setting = 'base'

        self.is_for_open_ended_questions = is_for_open_ended_questions

        self.injection_type = injection_type
        for n, p in self.model.named_parameters():
            if 'lora_' in n:
                p.requires_grad = True
            else:
                p.requires_grad = False

        num_processes = self.accelerator.num_processes if self.accelerator is not None else 1
        self.train_batch_size_per_device = self.model_config.effective_batch_size // num_processes
        self.warmup_train_batch_size_per_device = self.model_config.effective_warmup_batch_size // num_processes

        self.extended_vision_projection = add_clip_up_to_vlm(model=self.get_model(), model_name=self.model_name, model_config=self.model_config,
                                                             language_model_dim=self.model_config.language_model_dim,
                                                             is_for_open_ended_questions=is_for_open_ended_questions,
                                                             device=self.device, torch_dtype=self.torch_dtype,
                                                             with_injected_lora=self.injection_type == InjectionType.CLIP_UP_EMB_LORA)

        self.model.train()

        self.open_ai_client = set_open_ai_client(self.model_config.open_ai_api_key)

        if self.accelerator is not None and self.accelerator.is_local_main_process:
            tensorboard_dir = f'{self.output_path}/tensorboard'
            os.makedirs(tensorboard_dir, exist_ok=True)
            self.tensorboard_writer = SummaryWriter(tensorboard_dir)

        return

    def __del__(self):
        if self.accelerator is not None and self.accelerator.is_local_main_process:
            self.tensorboard_writer.close()

    def get_model(self):
        if self.is_peft_model:
            return self.model.base_model.model
        else:
            return self.model

    def get_unwrapped_model(self):
        if self.is_peft_model:
            return self.unwrapped_model.base_model.model
        else:
            return self.unwrapped_model

    def get_clip_up_projection(self):
        return self.extended_vision_projection.clip_up_projection

    def prepare_train_and_val_data(self, train_sets_infos, val_sets_infos):
        # 1. Set clip_signal_extractor
        signal_extractor = StructureClipSignalExtractor(device=self.device, torch_dtype=self.torch_dtype,
        structure_clip_checkpoint_path=self.model_config.structure_clip_checkpoint_path, accelerator=self.accelerator)

        # 2. Set training sets infos
        for curr_train_set_info in train_sets_infos:
            curr_pickle_file_path_train = f'clip_up_cached_signals/train/train_{curr_train_set_info.train_upd_type}.pkl'
            curr_train_set_info.items = StructureClipSignalExtractor.add_clip_up_signals_to_data(signal_extractor, curr_train_set_info.items,
                                                                                        force_run_clip=False,
                                                                                        signals_cache_path=curr_pickle_file_path_train,
                                                                                        tqdm_desc=f"Extracting CLIP-UP signals for {curr_train_set_info.train_upd_type.upper()} train data")

            curr_train_set_info.upd_items = [example for example in curr_train_set_info.items if example["type"] == 'upd']
            curr_train_set_info.upd_items = sorted(curr_train_set_info.upd_items, key=lambda x: x["index"])

            curr_train_set_info.standard_items = [example for example in curr_train_set_info.items if example["type"] == 'standard']
            curr_train_set_info.standard_items = sorted(curr_train_set_info.standard_items, key=lambda x: x["index"])

        # 3. Set validation sets infos
        for curr_val_set_info in val_sets_infos:
            curr_val_set_info.items = sorted(curr_val_set_info.items, key=lambda x: get_circular_item_index(x['index']))

            curr_pickle_file_path_val = f'clip_up_cached_signals/val/val_{curr_val_set_info.val_upd_type}.pkl'
            curr_val_set_info.items = \
                StructureClipSignalExtractor.add_clip_up_signals_to_data(signal_extractor, curr_val_set_info.items,
                                                                         force_run_clip=False,
                                                                         signals_cache_path=curr_pickle_file_path_val,
                                                                         tqdm_desc=f"Extracting CLIP-UP signals for val data ({curr_val_set_info.val_upd_type.upper()})")

        train_dataset, val_sets_infos, data_sampler = self.get_train_and_val_datasets(train_sets_infos=train_sets_infos,
                                                                                     val_sets_infos=val_sets_infos)

        train_dataloader, val_sets_infos, train_dataloader_for_warmup = \
            self.get_train_and_val_dataloaders(train_dataset=train_dataset, val_sets_infos=val_sets_infos, sampler=data_sampler)

        return train_dataloader, val_sets_infos, train_dataloader_for_warmup

    def get_train_and_val_datasets(self, train_sets_infos, val_sets_infos):
        # 1. Get dataset kwargs and class
        dataset_kwargs = dict(
            tokenizer=self.tokenizer,
            image_processor=self.image_processor,
            model_config=self.model.config,
            upd_setting=self.upd_setting,
            model_name=self.model_name,
        )
        dataset_class, extra_kwargs = get_vlm_dataset_class_and_extra_kwargs(model_name=self.model_name, clip_up_wrapper=self)
        dataset_kwargs.update(extra_kwargs)

        # 2. For each train set, create a dataset instance for standard and UPD questions
        train_standard_datasets, train_upd_datasets = [], []
        for curr_train_info in train_sets_infos:
            curr_train_dataset_upd = dataset_class(
                inputs=curr_train_info.upd_items,
                upd_type=curr_train_info.train_upd_type,
                **dataset_kwargs
            )
            curr_train_dataset_standard = dataset_class(
                inputs=curr_train_info.standard_items,
                upd_type=curr_train_info.train_upd_type,
                **dataset_kwargs
            )
            train_upd_datasets.append(curr_train_dataset_upd)
            train_standard_datasets.append(curr_train_dataset_standard)

        # 3. Create a concatenated training dataset, and a sampler which samples batches of corresponding upd and standard items
        train_sets_list = train_upd_datasets + train_standard_datasets
        train_dataset = ConcatDataset(train_sets_list)
        data_sampler = ClipUpSampler(train_upd_datasets, train_standard_datasets)

        # 4. Create a dataset instance for each val set
        for curr_val_info in val_sets_infos:
            curr_val_info.dataset = dataset_class(
                inputs=curr_val_info.items,
                upd_type=curr_val_info.val_upd_type,
                **dataset_kwargs
            )

        return train_dataset, val_sets_infos, data_sampler

    def get_train_and_val_dataloaders(self, train_dataset, val_sets_infos, sampler):
        # 1. Get collate function
        collate_func = get_vlm_collate_function(model_name=self.model_name, tokenizer=self.tokenizer)

        # 2. Set training dataloader and warmup training dataloader
        train_dataloader = DataLoader(train_dataset, batch_size=self.train_batch_size_per_device, shuffle=False,
                                      num_workers=0, sampler=sampler, collate_fn=collate_func)

        train_dataloader_for_warmup = DataLoader(train_dataset, batch_size=self.warmup_train_batch_size_per_device,
                                                 shuffle=False, num_workers=0, sampler=sampler,
                                                 collate_fn=collate_func)

        # 3. Set validation dataloaders
        for curr_val_info in val_sets_infos:
            curr_val_info.dataloader = DataLoader(curr_val_info.dataset, batch_size=1, shuffle=False,
                                                  num_workers=0, collate_fn=collate_func)

        return train_dataloader, val_sets_infos, train_dataloader_for_warmup

    def train_warmup_epoch(self, clip_up_projection_layer, train_dataloader_for_warmup, warmup_optimizer, warmup_epoch):
        criterion = SupConLoss()
        pbar_iterations = tqdm(train_dataloader_for_warmup, desc="Warmup iterations", disable=not self.accelerator.is_local_main_process)

        losses_list = []
        warmup_optimizer.zero_grad()

        for batch in pbar_iterations:
            # Run projection
            output_embeddings = clip_up_projection_layer.forward(batch['clip_signal']).unsqueeze(1)

            # Get labels
            upd_or_standard_labels = torch.zeros(output_embeddings.shape[0]).to(output_embeddings.device).to(torch.int64)
            for curr_type_str_idx, curr_type_str in enumerate(batch['type']):
                upd_or_standard_labels[curr_type_str_idx] = {'upd': 0, 'standard': 1}[curr_type_str]

            # Compute loss, backward and step
            loss = criterion(output_embeddings, upd_or_standard_labels)
            self.accelerator.backward(loss)
            warmup_optimizer.step()
            warmup_optimizer.zero_grad()

            curr_losses = self.accelerator.gather_for_metrics([loss.item()])
            losses_list.extend(curr_losses)

            pbar_iterations.set_description(f"Warmup epoch {warmup_epoch}: Loss: {loss.item():0.3f}", refresh=False)

        if self.accelerator.is_local_main_process:
            print(f'Warmup epoch {warmup_epoch}: average loss = {sum(losses_list) / len(losses_list):0.3f}')

        self.accelerator.wait_for_everyone()
        return

    def train_epoch(self, train_dataloader, optimizers, schedulers, epoch, tensorboard_steps_count_dict):
        # 1. Prepare for training
        learning_stats_dict = defaultdict(list)
        self.model.train()
        _ = [opt.zero_grad() for opt in optimizers]
        learnable_params = [p for p in self.model.parameters() if p.requires_grad]

        # 2. Train (the below code is based on https://huggingface.co/docs/accelerate/en/usage_guides/gradient_accumulation#skeleton-code)
        training_iterator = iter(train_dataloader)
        num_samples_in_epoch = len(train_dataloader)
        remainder = num_samples_in_epoch % self.model_config.gradient_accumulation_steps
        remainder = remainder if remainder != 0 else self.model_config.gradient_accumulation_steps
        total_updates = math.ceil(num_samples_in_epoch / self.model_config.gradient_accumulation_steps)

        grad_accum_index = 0
        pbar_iterations = tqdm(range(total_updates), disable=not self.accelerator.is_local_main_process)

        for update_step in pbar_iterations:
            # In order to correctly the total number of non-padded tokens on which we'll compute the cross-entropy loss
            # we need to pre-load the full local batch - i.e the next per_device_batch_size * accumulation_steps samples
            batch_samples = []
            num_batches_in_step = self.model_config.gradient_accumulation_steps if update_step != (total_updates - 1) else remainder
            for _ in range(num_batches_in_step):
                batch_samples += [next(training_iterator)]

            # get local num items in batch
            num_items_in_batch = sum([(batch["labels"].ne(-100)).sum() for batch in batch_samples])

            # to compute it correctly in a multi-device DDP training, we need to gather the total number of items in the full batch.
            num_items_in_batch = self.accelerator.gather(num_items_in_batch).sum().item()

            for i, batch in enumerate(batch_samples):
                # if we perform gradient accumulation in a multi-devices set-up, we want to avoid unnecessary communications when accumulating
                # cf: https://muellerzr.github.io/blog/gradient_accumulation.html
                if (i < len(batch_samples) - 1 and self.accelerator.num_processes > 1):
                    ctx = self.model.no_sync
                else:
                    ctx = contextlib.nullcontext

                with ctx():
                    # Run forward
                    output_logits, output_labels, loss = run_vlm_forward(model_name=self.model_name, model=self.model, batch=batch, clip_up_wrapper=self)

                    # Compute losses per sample (for logging only)
                    losses_per_sample = compute_vlm_per_sample_loss(model_name=self.model_name, logits=output_logits, labels=output_labels, clip_up_wrapper=self)
                    del output_logits, output_labels

                    # We multiply by num_processes because the DDP calculates the average gradient across all devices whereas dividing by num_items_in_batch already takes into account all devices
                    # Same reason for gradient_accumulation_steps, but this times it's Accelerate that calculate the average gradient across the accumulated steps
                    loss = (loss * self.model_config.gradient_accumulation_steps * self.accelerator.num_processes) / num_items_in_batch
                    loss_raw = loss.item()

                    self.accelerator.backward(loss)

                    # Compute grad norm and get learning rate
                    grad_norm = torch.cat([p.grad.flatten().to(learnable_params[0].device) for p in learnable_params]).norm().detach().item()
                    current_lrs = [sched.get_last_lr()[0] for sched in schedulers]

                    # Get stats from all processes
                    losses_per_sample_all = self.accelerator.gather_for_metrics(losses_per_sample)
                    type_strs_all = self.accelerator.gather_for_metrics(batch['type'])
                    upd_challenge_types_all = self.accelerator.gather_for_metrics(batch['upd_challenge_type'])
                    loss_raw_list_all = self.accelerator.gather_for_metrics([loss_raw])
                    grad_norms_all = self.accelerator.gather_for_metrics([grad_norm])
                    current_lr_all = self.accelerator.gather_for_metrics([current_lrs[0]])

                    # Update stats in learning_stats_dict and log to tensorboard
                    standard_upd_losses = defaultdict(list)
                    for curr_sample_loss, curr_sample_type, curr_upd_challenge_type in zip(losses_per_sample_all, type_strs_all, upd_challenge_types_all):
                        standard_upd_losses[curr_sample_type].append(curr_sample_loss)
                        learning_stats_dict[f'{curr_sample_type}_losses'].append(curr_sample_loss)
                        # learning_stats_dict[f'{curr_upd_challenge_type}_{curr_sample_type}_losses'].append(curr_sample_loss) # for stats per UPD type

                    learning_stats_dict['all_losses'].extend(loss_raw_list_all)

                    if self.accelerator.is_local_main_process:
                        self.log_to_tensorboard(entries=current_lr_all, tag=f'training/lr per step', count_dict=tensorboard_steps_count_dict)
                        self.log_to_tensorboard(entries=grad_norms_all, tag=f'training/grad_norms per step', count_dict=tensorboard_steps_count_dict)
                        self.log_to_tensorboard(entries=standard_upd_losses['standard'], tag=f'training/standard_losses per step', count_dict=tensorboard_steps_count_dict)
                        self.log_to_tensorboard(entries=standard_upd_losses['upd'], tag=f'training/upd_losses per step', count_dict=tensorboard_steps_count_dict)

                    pbar_iterations.set_description(f"Epoch {epoch}: loss = {loss_raw:0.3f}, "
                                                    f"grad norm = {grad_norm:0.12f}, "
                                                    f"losses per sample = [{', '.join(f'{num:.4f}' for num in losses_per_sample)}]",
                                                    refresh=False)
                    grad_accum_index += 1

            if self.model_config.gradient_clipping_max_norm:
                self.accelerator.clip_grad_norm_(learnable_params, max_norm=self.model_config.gradient_clipping_max_norm, norm_type=2)

            for opt in optimizers:
                opt.step()
                opt.zero_grad()

            _ = [sched.step() for sched in schedulers]

        self.accelerator.wait_for_everyone()

        if self.accelerator.is_local_main_process:
            all_losses_avg = sum(learning_stats_dict['all_losses']) / len(learning_stats_dict['all_losses'])
            print(f'Epoch {epoch}: all_losses_avg = {all_losses_avg:.4f}')
            self.log_to_tensorboard(entries=[all_losses_avg], tag='training/Average raw loss per epoch', count_dict=tensorboard_steps_count_dict)

            upd_losses_avg = sum(learning_stats_dict['upd_losses']) / len(learning_stats_dict['upd_losses'])
            print(f'Epoch {epoch}: upd_losses_avg = {upd_losses_avg:.4f}')
            self.log_to_tensorboard(entries=[upd_losses_avg], tag='training/upd_losses_avg', count_dict=tensorboard_steps_count_dict)

            standard_losses = sum(learning_stats_dict['standard_losses']) / len(learning_stats_dict['standard_losses'])
            print(f'Epoch {epoch}: standard_losses = {standard_losses:.4f}')
            self.log_to_tensorboard(entries=[standard_losses], tag='training/standard_losses_avg', count_dict=tensorboard_steps_count_dict)

        return

    def set_optimizers_and_schedulers(self, train_dataloader):
        num_samples_in_epoch = len(train_dataloader)
        total_training_steps = math.ceil(
            num_samples_in_epoch / self.model_config.gradient_accumulation_steps) * self.model_config.num_train_epochs

        clip_up_learnable_params, lora_learnable_params = \
            get_learnable_lora_and_clip_up_params(dict(self.model.named_parameters()))

        if self.injection_type == InjectionType.CLIP_UP_EMB:
            assert len(lora_learnable_params) == 0

        optimizers, schedulers = [], []

        # 1. Set optimizer and scheduler for LoRA params
        if len(lora_learnable_params) > 0:
            optimizer_regular_lora = torch.optim.AdamW(lora_learnable_params,
                                                       lr=self.model_config.injected_lora_scheduler.start_lr,
                                                       weight_decay=1e-4)

            scheduler_regular_lora = get_scheduler(
                name="cosine",
                optimizer=optimizer_regular_lora,
                num_warmup_steps=int(self.model_config.injected_lora_scheduler.warm_up_ratio * total_training_steps),
                num_training_steps=total_training_steps,
            )

            optimizers.append(optimizer_regular_lora)
            schedulers.append(scheduler_regular_lora)

        # 2. Set optimizer and scheduler for CLIP-UP projection params
        if len(clip_up_learnable_params) > 0:
            optimizer_clip_up = torch.optim.AdamW(clip_up_learnable_params,
                                              lr=self.model_config.clip_up_emb_scheduler.start_lr, weight_decay=1e-4)

            scheduler_clip_up = get_scheduler(
                name=self.model_config.clip_up_emb_scheduler.type,
                optimizer=optimizer_clip_up,
                num_warmup_steps=int(self.model_config.clip_up_emb_scheduler.warm_up_ratio * total_training_steps),
                num_training_steps=total_training_steps,
            )

            optimizers.append(optimizer_clip_up)
            schedulers.append(scheduler_clip_up)

        assert len(schedulers) == len(optimizers)

        # 3. Convert to accelerator
        optimizers = self.accelerator.prepare(*optimizers)
        if not isinstance(optimizers, tuple):
            optimizers = (optimizers,)
        schedulers = self.accelerator.prepare(*schedulers)
        if not isinstance(schedulers, tuple):
            schedulers = (schedulers,)

        return optimizers, schedulers

    def warmup_train(self, train_dataloader_for_warmup):
        # 1. Prepare warmup optimizer and dataloader
        clip_up_projection_layer = self.get_clip_up_projection()
        warmup_learnable_params = [p for n, p in clip_up_projection_layer.named_parameters()]
        warmup_optimizer = torch.optim.AdamW(warmup_learnable_params, lr=self.model_config.multiple_choice_warmup_lr, weight_decay=1e-4)
        warmup_optimizer, train_dataloader_for_warmup = self.accelerator.prepare(warmup_optimizer, train_dataloader_for_warmup)

        if self.accelerator.is_local_main_process:
            print(f'*** Start of warmup training ***')

        # 2. Train warmup
        for curr_warmup_epoch in range(self.model_config.num_warmup_epochs):
            self.train_warmup_epoch(clip_up_projection_layer, train_dataloader_for_warmup, warmup_optimizer, warmup_epoch=curr_warmup_epoch)

        if self.accelerator.is_local_main_process:
            print(f'*** End of warmup training ***')

        return

    def train(self, train_dataloader, val_sets_infos, train_dataloader_for_warmup):
        # 1. Set optimizers and schedulers
        optimizers, schedulers = self.set_optimizers_and_schedulers(train_dataloader)

        # 2. Call accelerator prepare
        self.model = self.accelerator.prepare(self.model, device_placement=[True])
        train_dataloader = self.accelerator.prepare(train_dataloader)
        self.unwrapped_model = self.accelerator.unwrap_model(self.model)

        for curr_val_set_info in val_sets_infos:
            curr_val_set_info.dataloader = self.accelerator.prepare(curr_val_set_info.dataloader)

        # 3. Validate before training
        if self.model_config.should_run_validation_before_training:
            for i, curr_val_set_info in enumerate(val_sets_infos):
                self.validate(curr_val_set_info, epoch=-1, use_clip_signal=False)

        # 4. Warmup training
        if not self.is_for_open_ended_questions:
            self.warmup_train(train_dataloader_for_warmup)

        # 5. Main training
        if self.accelerator.is_local_main_process:
            print(f'*** Start of main training ***')

        epochs_pbar = tqdm(range(self.model_config.num_train_epochs), desc="Epochs", disable=not self.accelerator.is_local_main_process)
        tensorboard_steps_count_dict = defaultdict(int)

        for epoch in epochs_pbar:
            # 5.1. Run training epoch
            self.accelerator.wait_for_everyone()
            self.train_epoch(train_dataloader, optimizers, schedulers, epoch=epoch,
                             tensorboard_steps_count_dict=tensorboard_steps_count_dict)
            torch.cuda.empty_cache()
            self.accelerator.wait_for_everyone()

            # 5.2. Validate
            if self.should_validate(epoch):
                self.do_validation_after_epoch(epoch, val_sets_infos, tensorboard_steps_count_dict)
            self.accelerator.wait_for_everyone()

            # 5.3. Save checkpoint
            if self.accelerator.is_local_main_process:
                if self.should_save_checkpoint(epoch):
                    self.save_checkpoint(epoch)

        if self.accelerator.is_local_main_process:
            print(f'*** End of main training ***')

        self.accelerator.wait_for_everyone()

        return

    def do_validation_after_epoch(self, epoch, val_sets_infos, tensorboard_steps_count_dict):
        for curr_val_set_info in val_sets_infos:
            val_results_dict = self.validate(curr_val_set_info, epoch)
            if self.accelerator.is_local_main_process:
                self.log_validation_results(val_results_dict, upd_type=curr_val_set_info.val_upd_type.upper(),
                                            tensorboard_steps_count_dict=tensorboard_steps_count_dict)

        return val_sets_infos

    def should_validate(self, epoch):
        return (epoch + 1) % self.model_config.validation_freq == 0

    def should_save_checkpoint(self, epoch):
        return ((epoch + 1) % self.model_config.checkpoint_save_freq == 0) or (epoch + 1 == self.model_config.num_train_epochs)

    def save_checkpoint(self, epoch):
        checkpoints_dict = {
            'epoch': epoch,
            'config': {'injection_type': self.injection_type.value},
            'model_state_dict': {}
        }

        embedding_injection_learnable_params, regular_lora_learnable_params, injected_lora_learnable_params = \
            get_clip_up_learnable_params_groups(dict(self.unwrapped_model.named_parameters()), return_dict=True)

        # Save embedding injection params
        checkpoints_dict['model_state_dict'].update(embedding_injection_learnable_params)
        assert len(embedding_injection_learnable_params) > 0

        # Save injected lora params
        state_dict_lora = {**regular_lora_learnable_params, **injected_lora_learnable_params}
        checkpoints_dict['model_state_dict'].update(state_dict_lora)
        if self.injection_type == InjectionType.CLIP_UP_EMB_LORA:
            assert len(regular_lora_learnable_params) > 0 and len(injected_lora_learnable_params) > 0

        torch.save(checkpoints_dict, Path(self.output_path) / f'checkpoint_epoch_{epoch}.pt')

        return

    def load_checkpoint(self, checkpoint_path):
        # 1. Load checkpoints dict
        print(f'Loading checkpoint from {checkpoint_path}')
        checkpoint_dict = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        assert checkpoint_dict['config']['injection_type'] == self.injection_type.value

        # 2. Verify checkpoint
        embedding_injection_learnable_params, regular_lora_learnable_params, injected_lora_learnable_params = \
            get_clip_up_learnable_params_groups(checkpoint_dict['model_state_dict'], return_dict=True)
        assert len(embedding_injection_learnable_params) > 0
        if self.injection_type == InjectionType.CLIP_UP_EMB_LORA:
            assert len(regular_lora_learnable_params) > 0 and len(injected_lora_learnable_params) > 0

        # 3. Load state dict
        missing_keys, unexpected_keys = self.model.load_state_dict(checkpoint_dict['model_state_dict'], strict=False)
        assert len(unexpected_keys) == 0
        assert len([k for k in missing_keys if 'lora' in k or 'clip_up' in k]) == 0

        return

    def validate(self, val_set_info, epoch, use_clip_signal=True):
        self.model.eval()

        # 1. Run generation
        generation_args = get_generation_args(model_name=self.model_name, clip_up_wrapper=self)
        standard_results, upd_results = [], []
        for val_batch in tqdm(val_set_info.dataloader, desc=f"Validation for epoch {epoch}, for {val_set_info.val_upd_type.upper()}",
                              disable=not self.accelerator.is_local_main_process):
            output = get_vlm_generation_output(model_name=self.model_name, model=self.unwrapped_model,
                                               tokenizer=self.tokenizer, val_batch=val_batch,
                                               use_clip_signal=use_clip_signal, generation_args=generation_args)

            output = self.accelerator.gather_for_metrics([output])
            val_batch = self.accelerator.gather_for_metrics([val_batch])

            for process_output, process_batch in zip(output, val_batch):
                for i, curr_out in enumerate(process_output):
                    res_dict = {'output': curr_out}
                    res_dict.update({k: process_batch[k][i] for k in ['type', 'gt_answer_str', 'question_str', 'options',
                                                                   'choices', 'original_gt_answer', 'question_id',
                                                                   'masked_answer', 'upd_challenge_type']})

                    if process_batch['type'][i] == 'standard':
                        standard_results.append(res_dict)
                    elif process_batch['type'][i] == 'upd':
                        upd_results.append(res_dict)
                    else:
                        assert False

        # 2. Evaluate scores
        val_results_dict = {}
        if self.accelerator.is_local_main_process:
            print('')
            if val_set_info.val_upd_type in ['aad', 'iasd', 'ivqd']:
                standard_accuracy, circ_standard_accuracy, upd_accuracy, circ_upd_accuracy, dual_accuracy, _ = (
                    evaluate_multiple_choice_results(open_ai_client=self.open_ai_client, standard_results=standard_results,
                                                     upd_results=upd_results, upd_type=val_set_info.val_upd_type,
                                                     upd_setting=self.upd_setting, epoch=epoch))
                val_results_dict = {'standard_accuracy': standard_accuracy, 'circ_standard_accuracy': circ_standard_accuracy,
                                    'upd_accuracy': upd_accuracy, 'circ_upd_accuracy': circ_upd_accuracy, 'dual_accuracy': dual_accuracy}
            else:
                standard_accuracy, upd_accuracy, dual_accuracy, _ = (
                    evaluate_open_ended_results(self.open_ai_client, standard_results=standard_results,
                                                upd_results=upd_results, upd_type=val_set_info.val_upd_type,
                                                epoch=epoch))
                val_results_dict = {'standard_accuracy': standard_accuracy,
                                    'upd_accuracy': upd_accuracy, 'dual_accuracy': dual_accuracy}
            print('-' * 100)

        return val_results_dict

    def log_validation_results(self, results_dict, upd_type, tensorboard_steps_count_dict):
        for metric_name, metric_value in results_dict.items():
            self.log_to_tensorboard(entries=[metric_value], tag=f'validation/{upd_type}_{metric_name}',
                                    count_dict=tensorboard_steps_count_dict)
        return

    def log_to_tensorboard(self, entries, tag, count_dict):
        for curr_entry in entries:
            self.tensorboard_writer.add_scalar(tag, curr_entry, global_step=count_dict[tag])
            count_dict[tag] += 1
        return
