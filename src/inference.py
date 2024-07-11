
from omegaconf import DictConfig
from typing import List, Tuple, Dict, Optional
from vllm import LLM, SamplingParams
from vllm.transformers_utils.tokenizer import get_tokenizer
from utils import  load_txt_as_array
import re

class Inference(object):
    def __init__(self, vllm_params: dict, sampling_params: dict, prompt: DictConfig, options_file: str, name="generate_until") -> None:
        """Initialize LogLikelihoodInference."""
        self.model = LLM(**vllm_params)
        self.tokenizer = get_tokenizer(vllm_params["model"], trust_remote_code=True)
        self.sampling_params = SamplingParams(**sampling_params, logprobs=1)
        self.options = load_txt_as_array(options_file)
        prompt.system_prompt = prompt.system_prompt.replace("MEDICAL_FIELDS", ",".join(self.options))
        self.pattern = re.compile(prompt.regex)
        self.prompt = prompt

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
    
    def predict(self, input_data: List[str], save_file: str = None) -> Tuple[List[str], List[float]]:
        """Generate predictions until a condition is met."""
        inputs = self.generate_inputs(input_data)
        outputs = self.model.generate(prompt_token_ids=inputs, sampling_params=self.sampling_params)
        preds, logprobs = self.preds_postprocess(outputs)
        cot = [output.outputs[0].text for output in outputs]
        return preds, logprobs, cot
    

    def find_answer_logprob(self, output: List[str]) -> Optional[str]:
        """Find the answer in the outputs."""
        last_match = None
        for match in re.finditer(self.pattern, output.text):
            last_match = match
        if last_match:
            answer = last_match.group('category') 
            logprob = output.cumulative_logprob
            if answer in self.options:
                return answer, logprob
        return 'None', None
    
    def preds_postprocess(self,  outputs: List[str]) -> Tuple[List[str], List[float]]:
        preds_logprobs = [self.find_answer_logprob(output.outputs[0])  for output in outputs]
        return zip(*preds_logprobs)    
    
