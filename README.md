# Language Model Policy Optimization

To setup this repo, first download the Qwen checkpoints and convert into a Jax checkpoint.
```
python models/download_model.py --model_id Qwen/Qwen3-1.7B --dest_root_path ~/checkpoints/
python models/hf_to_jax.py

# On a multi-host machine, use:
TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 TPU_PROCESS_BOUNDS=1,1,1 TPU_VISIBLE_DEVICES=0,1,2,3 python models/hf_to_jax.py
```

To run inference, try:
```
python inference/write_poem.py
```

To train GRPO, try:
```
python algs/grpo.py --wandb_name GRPO-1.7B --env_name Countdown --model_dir /nfs/gcs/jaxconverted/Qwen3-1.7B/ --groups_per_batch 64
```