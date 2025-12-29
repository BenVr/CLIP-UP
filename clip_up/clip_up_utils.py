import copy
from enum import Enum
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import peft
import torch
from sklearn.manifold import TSNE
from tqdm import tqdm

from clip_up.clip_up_general_utils import show_tsne


class ValidationSetInfo:
    def __init__(self, upd_type, items):
        self.val_upd_type = upd_type
        self.items = items
        self.dataset = None
        self.dataloader = None


class TrainingSetInfo:
    def __init__(self, upd_type, items):
        self.train_upd_type = upd_type
        self.items = items
        self.upd_items = None
        self.standard_items = None


class InjectionType(str, Enum):
    CLIP_UP_EMB = "clip_up_emb"
    CLIP_UP_EMB_LORA = "clip_up_emb_lora"


# Implementation taken from https://github.com/HobbitLong/SupContrast
class SupConLoss(torch.nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point.
        # Edge case e.g.:-
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan]
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


def get_question_with_answers_list(question, options):
    question_with_answers_list = [question + ' ' + ans for ans in options]
    return question_with_answers_list


def complete_question_with_answers_list_to_four(question_with_answers_list):
    question_with_answers_list += [''] * (4 - len(question_with_answers_list))
    return question_with_answers_list


def get_question_with_answers_list_from_example(question, options):
    if len(options) == 0:
        question_with_answers_list = [question]
    else:
        question_with_answers_list = get_question_with_answers_list(question, options)
        question_with_answers_list = complete_question_with_answers_list_to_four(question_with_answers_list)
        assert len(question_with_answers_list) == 4
    return question_with_answers_list


def output_clip_up_projection_t_sne(data_list, clip_up_projection_layer, upd_type, output_dir):
    # For nice open ended t-sne
    if upd_type == 'open_ended':
        data_list = data_list[::-1]

    # 1. Get all clip-up embedding vectors
    all_clip_up_embedding_vectors, all_types = [], []

    for i, row in tqdm(enumerate(data_list), total=len(data_list), desc='Creating embeddings for t-SNE'):
        curr_embedding_vector = clip_up_projection_layer(row["clip_signal"].unsqueeze(0).cuda())
        all_clip_up_embedding_vectors.append(curr_embedding_vector.squeeze(0).detach().cpu().float().numpy())

        if row['type'] == 'standard':
            all_types.append('Standard')
        elif row['type'] == 'upd':
            if upd_type in ['open_ended']:
                all_types.append('Unanswerable')
            else:
                all_types.append(upd_type.upper())
        else:
            assert False

    # 2. Create t-SNE
    all_clip_up_embedding_vectors = np.array(all_clip_up_embedding_vectors)
    try:
        tsne_results = TSNE(random_state=0, max_iter=1000).fit_transform(all_clip_up_embedding_vectors)
        df_tsne = pd.DataFrame(tsne_results, columns=['TSNE1', 'TSNE2'])

        # 3. Print t-SNE
        df_tsne['Class Name'] = all_types
        show_tsne(df_tsne, col_labels='Class Name', file_path=Path(output_dir) / f't_sne_{upd_type}.jpg')
    except ValueError as e:
        print(f"Caught a ValueError: {e}")

    return


def get_clip_up_learnable_params_groups(model_named_parameters, return_dict=False):
    embedding_injection_learnable_params = {n: p for n, p in model_named_parameters.items() if p.requires_grad and 'lora' not in n}
    regular_lora_learnable_params = {n: p for n, p in model_named_parameters.items() if p.requires_grad and 'lora' in n and 'lora_clip_up_projection' not in n}
    injected_lora_learnable_params = {n: p for n, p in model_named_parameters.items() if p.requires_grad and 'lora' in n and 'lora_clip_up_projection' in n}
    if return_dict:
        return embedding_injection_learnable_params, regular_lora_learnable_params, injected_lora_learnable_params
    else:
        return list(embedding_injection_learnable_params.values()), list(regular_lora_learnable_params.values()), list(injected_lora_learnable_params.values())


def get_learnable_lora_and_clip_up_params(model_named_parameters):
    embedding_injection_learnable_params, regular_lora_learnable_params, injected_lora_learnable_params = \
        get_clip_up_learnable_params_groups(dict(model_named_parameters))

    clip_up_params = embedding_injection_learnable_params + injected_lora_learnable_params
    lora_params = regular_lora_learnable_params
    return clip_up_params, lora_params


def custom_lora_linear_forward(self, x: torch.Tensor, clip_signal=None, *args: Any, **kwargs: Any) -> torch.Tensor:
    self._check_forward_args(x, *args, **kwargs)
    adapter_names = kwargs.pop("adapter_names", None)

    if self.disable_adapters:
        if self.merged:
            self.unmerge()
        result = self.base_layer(x, *args, **kwargs)
    elif adapter_names is not None:
        result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)
    elif self.merged:
        result = self.base_layer(x, *args, **kwargs)
    else:
        result = self.base_layer(x, *args, **kwargs)
        torch_result_dtype = result.dtype
        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A.keys():
                continue
            lora_A = self.lora_A[active_adapter]
            lora_B = self.lora_B[active_adapter]
            dropout = self.lora_dropout[active_adapter]
            scaling = self.scaling[active_adapter]
            x = x.to(lora_A.weight.dtype)

            if not self.use_dora[active_adapter]:
                if clip_signal is not None:
                    clip_up_projected_signal = self.lora_clip_up_projection(clip_signal.to(self.lora_clip_up_projection.linear_projection.weight.device))
                    clip_up_projected_signal = clip_up_projected_signal.view(-1, self.r[active_adapter], self.r[active_adapter])
                    result = result + lora_B(torch.matmul(lora_A(dropout(x)), clip_up_projected_signal + self.lora_residual_matrix.weight)) * scaling
                else:
                    result = result + lora_B(lora_A(dropout(x))) * scaling
            else:
                x = dropout(x)
                result = result + self._apply_dora(x, lora_A, lora_B, scaling, active_adapter)

        result = result.to(torch_result_dtype)

    return result


def add_injected_lora_support_to_modules(target_modules, lora_rank, torch_dtype, signal_dim):
    assert all(isinstance(module, peft.tuners.lora.Linear) for module in target_modules.values())

    out_dim = lora_rank ** 2
    for name, module in target_modules.items():
        module.forward = custom_lora_linear_forward.__get__(module, module.__class__)

        from clip_up.clip_up_wrapper_internals import LinearSignalProjection
        module.lora_clip_up_projection = LinearSignalProjection(signal_dim, out_dim, device=module.weight.device, torch_dtype=torch_dtype, bias=False)
        module.lora_clip_up_projection.requires_grad_(True)

        module.lora_residual_matrix = torch.nn.Linear(lora_rank, lora_rank, device=module.weight.device, dtype=torch_dtype, bias=False)
        module.lora_residual_matrix.requires_grad_(True)

    return
