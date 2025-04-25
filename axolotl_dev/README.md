# Running all this

## vLLM server needs to run with some GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 axolotl vllm-serve config_14b.yaml  --tensor-parallel-size 4
```

## Rewards server

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 python server.py
```

## Training

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 axolotl train config_14b.yaml --deepspeed deepspeed_configs/zero2.json
```