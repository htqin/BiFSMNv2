# *BiFSMNv2: Pushing Binary Neural Networks for Keyword Spotting to Real-Network Performance*

## Introduction

This project is the official implementation of our paper *BiFSMNv2: Pushing Binary Neural Networks for Keyword Spotting to Real-Network Performance*.

## Datasets and Pretrained Models

We train and test BiFSMNv2 on Google Speech Commands V1 and V2 datasets, which can be downloaded in the reference document:

- https://pytorch.org/audio/stable/_modules/torchaudio/datasets/speechcommands.html#SPEECHCOMMANDS

And we also release a pretrained model on Speech Commands V1-12 task for our distillation.

## Execution

Our experiments are based on the fine-tuned full-precision BiFSMN_pre, which can be found here. Complete running scripts is provided as follow

```shell
python3 train_speech_commands.py \
    --gpu=0 \
    --model=BiDfsmn_thinnable --dfsmn_with_bn \
    --method=Vanilla \
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
    --epoch=300 \
```
