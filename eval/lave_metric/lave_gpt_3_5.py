import torch
import re

from eval.lave_metric.lave import LaveBase
from typing import Any, List, Union
from eval.eval_utils import call_open_ai_gpt_3_5


class LaveGPT_3_5(LaveBase):
    def __init__(self, open_ai_client, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.open_ai_client = open_ai_client

    def generate(self, prompt: Union[str, List[str]], **generate_kwargs) -> torch.Tensor:
        answer_from_gpt = call_open_ai_gpt_3_5(self.open_ai_client, prompt)
        return answer_from_gpt

    def postprocess(self, text):
        match = re.search(r"rating=(\d+)$", text.strip())
        if match and text.strip().endswith(match.group(0)):
            rating = int(match.group(1))
        else:
            # In case no match was found, LAVE defaults to rating of 2
            rating = 2

        assert rating in [1, 2, 3]
        rating = ((rating - 1.) / 2.)

        return rating

    def compute(
        self,
        prediction: str,
        references: List[str],
        question: str,
        caption: str = None
    ):
        prompt = self.build_prompt(prediction, references, question, caption)
        answer_from_gpt = self.generate(prompt)
        score = self.postprocess(answer_from_gpt)
        return score, answer_from_gpt

