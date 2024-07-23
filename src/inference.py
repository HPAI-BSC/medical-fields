import os
import json
import re
import sys
from omegaconf import DictConfig
from typing import List, Tuple, Dict, Optional
from vllm import LLM, SamplingParams
from vllm.transformers_utils.tokenizer import get_tokenizer
from utils import load_txt_as_array

class Inference(object):
    def __init__(self, vllm_params: dict, sampling_params: dict, prompt: DictConfig, options_file: str, progress_file: str) -> None:
        """Initialize LogLikelihoodInference."""
        self.model = LLM(**vllm_params)
        self.tokenizer = get_tokenizer(vllm_params["model"], trust_remote_code=True)
        self.sampling_params = SamplingParams(**sampling_params, logprobs=1)
        self.options = load_txt_as_array(options_file)
        prompt.system_prompt = prompt.system_prompt.replace("MEDICAL_FIELDS", ",".join(self.options))
        self.pattern = re.compile(prompt.regex)
        self.prompt = prompt
        self.progress_file = progress_file

    def generate_chat_template(self, question: str) -> List[Dict[str, str]]:
        """Create a chat template."""
        chat_template = []
        chat_template.append({"role": "system", "content": self.prompt.system_prompt})
        for shot in self.prompt.fewshot_examples:
            chat_template.append({"role": "user", "content": shot.question})
            chat_template.append({"role": "assistant", "content": shot.answer})

        chat_template.append({"role": "user", "content": question})
        return chat_template

    def get_tokens(self, text: str) -> Dict[str, int]:
        """Get tokens for a given text."""
        return self.tokenizer(text)["input_ids"]

    def get_tokens_with_chat_template(self, question: str) -> Dict[str, int]:
        """Apply chat template and tokenize."""
        chat_template = self.generate_chat_template(question)
        return self.tokenizer.apply_chat_template(chat_template, tokenize=True, add_generation_prompt=True)

    def generate_inputs(self, questions: List[str]) -> List[List[int]]:
        """Generate prompts for the model."""
        return [self.get_tokens_with_chat_template(question) for question in questions]

    def load_progress(self) -> Tuple[List[str], List[float], List[str], int]:
        """Load progress from the temporary file if it exists."""
        if os.path.exists(self.progress_file):
            with open(self.progress_file, 'r') as file:
                data = json.load(file)
                print (f"Loading progress of {len(data)} samples")
                return data['preds'], data['logprobs'], data['cot'], data['processed']
        return [], [], [], 0

    def save_progress(self, preds: List[str], logprobs: List[float], cot: List[str], processed: int) -> None:
        """Save progress to the temporary file."""
        with open(self.progress_file, 'w') as file:
            json.dump({'preds': preds, 'logprobs': logprobs, 'cot': cot, 'processed': processed}, file)

    def remove_process(self) -> None:
        """Remove the temporary file."""
        if os.path.exists(self.progress_file):
            os.remove(self.progress_file)

    def predict(self, input_data: List[str], batch_size: int = 5) -> Tuple[List[str], List[float], List[str]]:
        """Generate predictions until a condition is met."""
        preds, logprobs, cot, processed = self.load_progress()
        remaining_inputs = input_data[processed:]

        total_inputs = len(remaining_inputs)

        for i in range(0, total_inputs, batch_size):
            batch_inputs = remaining_inputs[i:i+batch_size]
            inputs = self.generate_inputs(batch_inputs)
            outputs = self.model.generate(prompt_token_ids=inputs, sampling_params=self.sampling_params)
            batch_preds, batch_logprobs, batch_cot = zip(*[(self.find_answer(output.outputs[0]),
                                                                        output.outputs[0].cumulative_logprob, 
                                                                        output.outputs[0].text)  for output in outputs])


            preds.extend(batch_preds)
            logprobs.extend(batch_logprobs)
            cot.extend(batch_cot)
            processed += len(batch_inputs)
            # Save progress
            self.save_progress(preds, logprobs, cot, processed)

        self.remove_process()

        return preds, logprobs, cot

    def find_answer(self, output: List[str]) -> Optional[str]:
        """Find the answer in the outputs."""
        last_match = None
        for match in re.finditer(self.pattern, output.text):
            last_match = match
        if last_match:
            answer = last_match.group('category')
            if answer in self.options:
                return answer
        return 'None'

