## Eval model

First serve the model using vllm:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve /model-checkpoints/sft-qwen3-14b-boot/checkpoint-165/ --tensor-parallel-size 4  --chat-template configs/chat_template.jinja --generation-config vllm --max-model-len 16k --served-model-name qwen3-14b-sft
```

Then run matching the `served-model-name`:

```
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve /model-checkpoints/sft-qwen3-14b-boot/checkpoint-165/ --tensor-parallel-size 4 --served-model-name qwen3-14b-sft --chat-template configs/chat_template.jinja --generation-config vllm --max-model-len 16k
```

For qwen3 models:

```
CUDA_VISIBLE_DEVICES=0,1,2,3 vllm serve /model-checkpoints/sft-qwen3-14b-boot-instruct2/ --tensor-parallel-size 4 --served-model-name qwen3-14b-boot-instruct2 --generation-config vllm --max-model-len 16k --enable-reasoning --reasoning-parser deepseek_r1
```