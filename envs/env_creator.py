
from lmpo.envs.poem_length import PoemLengthEnv
from lmpo.envs.gsm8k import GSM8KEnv
from lmpo.envs.countdown import CountdownEnv
from lmpo.envs.deepscaler import DeepscalerEnv
from lmpo.envs.aime import AimeEnv

def create_env(env_name, tokenizer):
    env_name = env_name.lower()
    if env_name == 'poem':
        env = PoemLengthEnv(tokenizer)
    elif env_name == 'gsm8k':
        env = GSM8KEnv(tokenizer)
    elif env_name == 'gsm8k-test':
        env = GSM8KEnv(tokenizer, train=False)
    elif env_name == 'countdown':
        env = CountdownEnv(tokenizer)
    elif env_name == 'deepscaler':
        env = DeepscalerEnv(tokenizer)
    elif env_name == 'aime':
        env = AimeEnv(tokenizer)
    else:
        raise ValueError(f"Unknown environment name: {env_name}")
    return env