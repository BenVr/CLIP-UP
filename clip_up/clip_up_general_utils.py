import base64
import datetime
import json
import os
import random
import sys
from io import BytesIO

import accelerate
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from PIL import Image
from openai import OpenAI
from peft import PeftModel


# This class makes printing to be both to console and to a file
# (taken from https://stackoverflow.com/a/14906787/23048953)
class DualScreenAndFileLogger(object):
    def __init__(self, out, output_file):
        self.terminal = out
        self.log = open(output_file, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.log.flush()


def set_output_dir_and_logging(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    output_log_file = f'{output_dir}/screen_log.txt'
    sys.stdout = DualScreenAndFileLogger(out=sys.stdout, output_file=output_log_file)
    sys.stderr = DualScreenAndFileLogger(out=sys.stderr, output_file=output_log_file)
    return


def now():
    return datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")


def read_jsonl(input_path):
    data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.rstrip('\n|\r')))
    return data


def seed_everything(seed: int, with_accelerate=True):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if with_accelerate:
        accelerate.utils.set_seed(seed, deterministic=True)
    return


def print_args(args):
    print("Arguments passed:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")


def show_tsne(df_tsne, col_labels, file_path):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.set_style("white")
    scatter = sns.scatterplot(data=df_tsne, x='TSNE1', y='TSNE2', hue=col_labels, palette='hls', alpha=0.85)
    scatter.legend(title='', prop={'size': 19})

    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.axis('equal')
    plt.savefig(file_path, bbox_inches='tight')
    plt.close()
    return


def is_peft_model(model):
    return isinstance(model, PeftModel)


def load_image_from_base64(image):
    return Image.open(BytesIO(base64.b64decode(image)))


def capitalize_only_first_letter(in_string):
    return in_string[0].upper() + in_string[1:]


def disable_torch_init():
    # Disable the redundant torch default initialization to accelerate model creation.
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)
    return


def set_open_ai_client(open_ai_api_key):
    if open_ai_api_key is not None:
        open_ai_client = OpenAI(api_key=open_ai_api_key)
    else:
        open_ai_client = None
        print(f'OpenAI key was not given. Run is expected to crash when calling GPT')
    return open_ai_client
