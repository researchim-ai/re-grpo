# Minimal GRPO implementation

Goal: Working toy implementation of llama-3.2-3b locally RL training with GRPO. Understanding the algorithm & hyper parameters. Just running everything locally on a single node.

### Setup

1. Create conda env

```
conda create --name grpo python=3.12 -y
conda activate grpo
```

2. Install dependencies

```
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
pip install accelerate
```

3. Play with the source

Run standard training:

```
python train.py
```

or use the accelerate script:

First, configure accelerate:

```
accelerate config
```

Select CPU offloading options when prompted.

Then, launch training:

```
accelerate launch train_accelerate.py
```

The `train_accelerate.py` script leverages Hugging Face Accelerate for CPU offloading to optimize memory usage and allow training larger models.

### Inspiration

- [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF)
- [Spinning Up in Deep RL](https://spinningup.openai.com/en/latest/)


### References

- [DeepSeek-R1 tech report](https://github.com/deepseek-ai/DeepSeek-R1/blob/main/DeepSeek_R1.pdf)
- [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)
