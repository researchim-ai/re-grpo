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

Each experiment is now located in its own directory.

**1. Base Experiment (Single GPU, no Accelerate)**

To run the basic training script:
```bash
python experiment_base/train.py
```

**2. Accelerate Experiment (Single/Multi-GPU with Accelerate)**

First, configure Accelerate if you haven't already:
```bash
accelerate config
```
Follow the prompts. For single GPU, you might choose "This machine" and "No distributed training". For multi-GPU, configure accordingly.

Then, launch the training:
```bash
accelerate launch experiment_accelerate/train_accelerate.py
```

**3. Accelerate DDP Experiment (Multi-GPU with Accelerate DDP)**

Ensure Accelerate is configured for DDP (Distributed Data Parallel).
```bash
accelerate config
```
Choose "This machine" and select the number of GPUs you want to use for distributed training.

Then, launch the training:
```bash
accelerate launch experiment_accelerate_ddp/train_accelerate_ddp.py
```

**4. Multiturn Calculator Tool Calling Experiment**

This script trains a model for multi-turn interactions involving a calculator tool.
```bash
python experiment_multiturn_tool_calling/train_multiturn_calc_tool_calling.py
```
This script accepts command-line arguments. Use `--help` to see available options:
```bash
python experiment_multiturn_tool_calling/train_multiturn_calc_tool_calling.py --help
```

**5. Multiturn QA with Search Tool Experiment**

This script trains a model for multi-turn QA that can use a search tool.
```bash
python experiment_multiturn_qa/train_multiturn_qa.py
```
This script also accepts command-line arguments. Use `--help` to see available options:
```bash
python experiment_multiturn_qa/train_multiturn_qa.py --help
```

**6. Dr. GRPO Experiment (GRPO with Debiasing Techniques)**

This directory `experiment_drgrpo` contains an implementation of GRPO with modifications inspired by "Dr. GRPO" to mitigate certain biases.
The key changes are:
- **Length Bias Removal**: The critic's value aggregation is modified to use a masked sum with a constant normalizer (based on `generate_max_length`) instead of a simple masked mean. This is controlled by the `critic_type="drgrpo"` parameter in the `GRPOLoss` class and related configuration in `train.py`.
- **Difficulty Bias Removal**: The Monte Carlo advantage calculation is adjusted to not normalize by the standard deviation of rewards when `critic_type="drgrpo"`. This change is implemented in the `group_advantages` function in `train.py`.

These modifications aim to improve the robustness and performance of the GRPO algorithm.
To run this experiment:
```bash
python experiment_drgrpo/train.py
```

### Inspiration

- [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF)
- [Spinning Up in Deep RL](https://spinningup.openai.com/en/latest/)


### References

- [DeepSeek-R1 tech report](https://github.com/deepseek-ai/DeepSeek-R1/blob/main/DeepSeek_R1.pdf)
- [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300)
