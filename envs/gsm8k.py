'''From https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb'''

from envs.base import BaseEnv, BaseState
from dataclasses import dataclass, replace
import numpy as np

def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip().replace(",", "").replace("$", "")

def has_formatting(text: str) -> bool:
    return "<answer>" in text and "</answer>" in text and "<reasoning>" in text and "</reasoning>" in text

def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        raise ValueError("Expected text to contain '####' for answer extraction.")
    return text.split("####")[1].strip().replace(",", "").replace("$", "")

SYSTEM_PROMPT = """
Respond in the following format. Put a single number in the <answer> tag. Think for a maximum of three sentences.
<reasoning>
...
</reasoning>
<answer>
...
</answer>"""

@dataclass(frozen=True)
class GSMState(BaseState):
    tokens: list
    correct_answer: str
    
class GSM8KEnv(BaseEnv):
    def __init__(self, tokenizer):
        super().__init__()
        self.tokens_per_action = 128
        self.data_dict = {}
        self.tokenizer = tokenizer
        from datasets import load_dataset
        self.ds = load_dataset('openai/gsm8k', 'main')['train']

    def reset(self):
        rand_idx = np.random.randint(0, len(self.ds))
        output_tokens = self.tokenizer.apply_chat_template([
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {"role": "user", "content": self.ds[rand_idx]['question']},
            ],
            add_generation_prompt=True,
            enable_thinking=False
        )
        state = GSMState(tokens=output_tokens, correct_answer=extract_hash_answer(self.ds[rand_idx]['answer']))
        return state, output_tokens

    def render(self, state):
        return self.tokenizer.decode(state.tokens) + "\nCorrect answer: " + state.correct_answer

    def step(self, state, action_tokens):
        action_tokens = self.clean_action(action_tokens, self.tokenizer.get_eos_token_id())
        action_msg = self.tokenizer.decode(action_tokens)
        reward = 0.0
        if has_formatting(action_msg):
            reward += 0.5
        if extract_xml_answer(action_msg) == state.correct_answer:
            reward += 0.5
        state = replace(state, tokens=state.tokens + action_tokens)
        return state, [], reward, True, {
            'has_formatting': has_formatting(action_msg),
            'correct_answer': extract_xml_answer(action_msg) == state.correct_answer,
        }