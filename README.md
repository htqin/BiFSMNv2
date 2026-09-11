# BiFSMNv2: Pushing Binary Neural Networks for Keyword Spotting to Real-Network Performance

**IEEE TNNLS 35(8):10674–10686 · 2024 (Early Access 2023)**

Haotong Qin, Xudong Ma, Yifu Ding, Xiaoyang Li, Yang Zhang, Zejun Ma, Jiakai Wang, Jie Luo, Xianglong Liu

[Published paper](https://doi.org/10.1109/TNNLS.2023.3243259) | [arXiv](https://arxiv.org/abs/2211.06987) | [Citation](#citation)

**BiFSMNv2 is a binary keyword-spotting network combining dual-scale thinnable architecture, Frequency Independent Distillation (FID), and a learnable binarizer (LPB).** It retains speech information during low-bit training and supports an accuracy/latency trade-off at deployment.

## Published results and scope

Speech Commands classification accuracy (%) on the 12-class task. W/A denotes weights/activations in the binary network; sensitive full-precision operations are retained. The full, half, and quarter settings refer to the paper's thinnable architecture, not to weight bit-width.

| Method / runtime setting | W/A | V1-12 accuracy | V2-12 accuracy |
| --- | --- | --- | --- |
| Full-precision reference | 32/32 | 97.93 | 98.05 |
| BiFSMN, full | 1/1 | 95.03 | 94.86 |
| BiFSMN, half | 1/1 | 94.87 | 94.73 |
| BiFSMN, quarter | 1/1 | 94.48 | 94.63 |
| BiFSMNv2, full | 1/1 | 96.42 | 96.45 |
| BiFSMNv2, half | 1/1 | 96.23 | 95.87 |
| BiFSMNv2, quarter | 1/1 | 94.65 | 94.25 |

Source: Table III of the linked manuscript. The full BiFSMNv2 is 1.51 percentage points below the full-precision V1-12 reference. The maximum **25.1× speedup** belongs to the **quarter** variant, not this highest-accuracy configuration. Figure 10 reports 26.8/18.9/15.0 ms for full/half/quarter with the proposed kernel, versus 377 ms for full precision. The reported **20.2× storage saving** concerns model storage, not peak working memory.

### What this paper supports

- Dual-scale thinnable architecture improves binary representation and permits runtime depth selection (Section III-B; Table I).
- FID separately distills low- and high-frequency information to reduce teacher/student mismatch (Section III-C; Table I).
- LPB learns the binarization behavior in forward and backward propagation (Section III-D; Table I).
- The full model improves V1-12 accuracy from BiFSMN's 95.03% to 96.42% under the reported settings (Table III).
- ARMv8 deployment with the optimized kernel yields the latency/storage trade-offs in Figure 10 (Section IV-C).

The deployment experiment uses Raspberry Pi 3B+ with a 1.2 GHz ARMv8 Cortex-A53 CPU as described in the paper. These results depend on the optimized binary kernel and do not imply the same speedup on GPU/PyTorch. Training uses speech data and a full-precision teacher; it is not data-free PTQ. The tables above are paper results, not newly reproduced measurements.

Created by [Haotong Qin](https://htqin.github.io/), Xudong Ma, [Yifu Ding](https://yifu-ding.github.io/), Xiaoyang Li, Yang Zhang, Zejun Ma, Jiakai Wang, Jie Luo, and [Xianglong Liu](https://xlliu-beihang.github.io/) from Beihang University, Bytedance AI Lab, and Zhongguancun Laboratory.

![loading-ag-172](./overview.png)

## Datasets and Pretrained Models

We train and test BiFSMNv2 on Google Speech Commands V1 and V2 datasets, which can be downloaded in the reference document:

- https://pytorch.org/audio/stable/_modules/torchaudio/datasets/speechcommands.html#SPEECHCOMMANDS

Distillation requires a full-precision teacher checkpoint for the matching Speech Commands task.

## Execution

Set `teacher_model_checkpoint_path` to your local full-precision teacher checkpoint before running. This README currently has no downloadable teacher-checkpoint link. The reference training command is:

```shell
python3 train_speech_commands.py \
    --gpu=0 \
    --model=BiDfsmn_thinnable --dfsmn_with_bn \
    --method=lpb \
    --distill \
    --distill_alpha=0.01 \
    --select_pass=fid \
    --J=1 \
    --pretrained \
    --teacher_model=BiDfsmn_thinnable_pre \
    --teacher_model_checkpoint=${teacher_model_checkpoint_path} \
    --version=speech_commands_v0.01 \
    --num_classes=12 \
    --lr-scheduler=cosin \
    --opt=sgd \
    --lr=5e-3 \
    --weight-decay=1e-4 \
    --epoch=300
```

## Citation

Please cite the published paper below. Open paper versions are linked at the top of this README.

```bibtex
@article{qin2024bifsmnv2,
  title = {BiFSMNv2: Pushing Binary Neural Networks for Keyword Spotting to Real-Network Performance},
  author = {Haotong Qin and Xudong Ma and Yifu Ding and Xiaoyang Li and Yang Zhang and Zejun Ma and Jiakai Wang and Jie Luo and Xianglong Liu},
  journal = {IEEE Transactions on Neural Networks and Learning Systems},
  year = {2024},
  volume = {35},
  number = {8},
  pages = {10674--10686},
  doi = {10.1109/TNNLS.2023.3243259},
  url = {https://doi.org/10.1109/TNNLS.2023.3243259}
}
```
