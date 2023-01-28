from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
import torchaudio

from core.registry import CONFIG

import basic
from models.torch.FSMN import FSMN
from models.torch.dfsmn import DfsmnModel
from models.torch.bidfsmn import BiDfsmnModel, BiDfsmnModel_thinnable, DfsmnModel_pre

from speech_commands.dataset.speech_commands import SpeechCommandV1
from speech_commands.dataset.transform import ChangeAmplitude, \
    FixAudioLength, ChangeSpeedAndPitchAudio, TimeshiftAudio

from torch_utils import mixup

from pytorch_wavelets import DWTForward, DWTInverse

def att_map_r2b(A):
    a = torch.abs(A)
    Q = a * a
    return Q

def r2b_loss(Q_s, Q_t):
    Q_s = att_map_r2b(Q_s)
    Q_t = att_map_r2b(Q_t)
    Q_s_norm = Q_s / torch.norm(Q_s, p=2)
    Q_t_norm = Q_t / torch.norm(Q_t, p=2)
    tmp = Q_s_norm - Q_t_norm
    loss = torch.norm(tmp, p=2)
    return loss

def pass_filter(x, select_pass, J=1, wave='haar', mode='zero'):
    xfm = DWTForward(J=J, mode=mode, wave=wave) # Accepts all wave types available to PyWavelets
    ifm = DWTInverse(mode=mode, wave=wave)
    if x.is_cuda:
        xfm, ifm = xfm.cuda(), ifm.cuda()

    if len(x.shape) == 3:
        yl, yh = xfm(x.unsqueeze(1))
    elif len(x.shape) == 4:
        yl, yh = xfm(x)
    else:
        assert(False) # error

    if select_pass == 'high':
        yl.zero_()
    elif select_pass == 'low':
        for i in range(J): # lowpass
            yh[i].zero_()
    elif select_pass == '':
        return
    
    y = ifm((yl, yh))
    if len(x.shape) == 3:
        y = y.squeeze(1)
    return y


def get_model2(model_type: str, in_channels=1, **kwargs):
    if model_type == 'fsmn':
        return FSMN(in_channels=in_channels, **kwargs)
    elif model_type == 'Dfsmn':
        return DfsmnModel(in_channels=in_channels, **kwargs)
    elif model_type == 'BiDfsmn':
        return BiDfsmnModel(in_channels=in_channels, **kwargs)
    elif model_type == 'BiDfsmn_thinnable_pre':
        return DfsmnModel_pre(in_channels=in_channels, **kwargs)
    elif model_type == 'BiDfsmn_thinnable':
        return BiDfsmnModel_thinnable(in_channels=in_channels, **kwargs)
    else:
        raise RuntimeError('unsupport model type: ', model_type)
    

def get_model(model_type: str, in_channels=1, method="no", bits=1, teacher=False, **kwargs):
    if method == "no" or teacher:
        model = get_model2(model_type, in_channels, **kwargs)
        return model
    else:
        model = get_model2(model_type, in_channels, **kwargs)
        try:
            import os
            fp_path = os.path.join(kwargs['saveroot'], kwargs['teacher_model_checkpoint'])
            chpk = torch.load(fp_path)
            model.load_state_dict(chpk['state_dict'], strict=False)
            print('load fp model ok!')
        except:
            pass
        model.method = method
        from basic import Count, Modify
        cnt = Count(model)
        model, _ = Modify(model, bits=bits, method=method, id=0, first=1, last=cnt)
        return model


def create_dataloader(dataset_type, configs, use_gpu, version):
    train_transform = Compose([
        ChangeAmplitude(),
        ChangeSpeedAndPitchAudio(),
        TimeshiftAudio(),
        FixAudioLength(),
        torchaudio.transforms.MelSpectrogram(sample_rate=16000,
                                             n_fft=2048,
                                             hop_length=512,
                                             n_mels=configs.n_mels,
                                             normalized=True),
        torchaudio.transforms.AmplitudeToDB(),
    ])
    valid_transform = Compose([
        FixAudioLength(),
        torchaudio.transforms.MelSpectrogram(sample_rate=16000,
                                             n_fft=2048,
                                             hop_length=512,
                                             n_mels=configs.n_mels,
                                             normalized=True),
        torchaudio.transforms.AmplitudeToDB(),
    ])

    dataset_train = SpeechCommandV1(configs.dataroot,
                                    subset='training',
                                    download=True,
                                    transform=train_transform,
                                    num_classes=configs.num_classes,
                                    noise_ratio=0.3,
                                    noise_max_scale=0.3,
                                    cache_origin_data=False,
                                    version=version)

    dataset_valid = SpeechCommandV1(configs.dataroot,
                                    subset='validation',
                                    download=True,
                                    transform=valid_transform,
                                    num_classes=configs.num_classes,
                                    cache_origin_data=True,
                                    version=version)

    dataset_test = SpeechCommandV1(configs.dataroot,
                                   subset='testing',
                                   download=True,
                                   transform=valid_transform,
                                   num_classes=configs.num_classes,
                                   cache_origin_data=True,
                                    version=version)

    dataset_dict = {
        'training': dataset_train,
        'validation': dataset_valid,
        'testing': dataset_test
    }
    return DataLoader(dataset_dict[dataset_type],
                      batch_size=configs.batch_size,
                      shuffle=dataset_type == 'training',
                      sampler=None,
                      pin_memory=use_gpu,
                      num_workers=16,
                      persistent_workers=True)


def create_lr_schedule(configs, optimizer):
    if configs.lr_scheduler == 'plateau':
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=configs.lr_scheduler_patience,
            factor=configs.lr_scheduler_gamma)
    elif configs.lr_scheduler == 'step':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=configs.lr_scheduler_stepsize,
            gamma=configs.lr_scheduler_gamma,
            last_epoch=configs.epoch - 1)
    elif configs.lr_scheduler == 'cosin':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=configs.epoch)
    else:
        raise RuntimeError('unsupported lr schedule type: ',
                           configs.lr_scheduler)
    return lr_scheduler


def create_optimizer(configs, model):
    if configs.optim == 'sgd':
        optimizer = torch.optim.SGD([
            {'params': [y for x, y in model.named_parameters() if 'channel_threshold' in x or 'alpha' in x], 'lr': configs.lr * 0.1}, 
            {'params': [y for x, y in model.named_parameters() if 'channel_threshold' not in x and 'alpha' not in x]}], 
            lr=configs.lr,
            momentum=0.9,
            weight_decay=configs.weight_decay)
    elif configs.optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=configs.lr,
                                     weight_decay=configs.weight_decay)

    return optimizer


weights = [1, 0.5, 0.25]
distillation_pred = torch.nn.MSELoss()
pred = True

def train_epoch(model: nn.Module,
                teacher_model: nn.Module,
                optimizer,
                criterion,
                data_loader: data.DataLoader,
                epoch,
                with_gpu,
                log_iter=10,
                distill_alpha=0,
                distill_conv=False,
                select_pass='no',
                J=1,
                num_classes=None,
                r2b=True):
    """
    training one epoch
    """
    print('r2b =', r2b, 'weight =', weights)
    model.train()
    if with_gpu:
        model = model.cuda()

    epoch_size = len(data_loader)
   
    thin_n = model.thin_n if hasattr(model, 'thin_n') else 1
    running_loss = 0
    i = 0
    for inputs, target in data_loader:
        if with_gpu:
            inputs, target = inputs.cuda(), target.cuda()

        loss = 0
       
        for op in range(thin_n):
            weight = 1 if thin_n == 1 else weights[op]
            if model.__class__.__name__[-9:] != 'thinnable':
                out = model(inputs)
            else:
                out = model(inputs, op)

            loss_one_hot = criterion(out, target)
            loss = loss + loss_one_hot * weight

            if distill_alpha != 0:
                teacher_out = teacher_model(inputs)
                distill_op = 'layer_output'
                teacher_feature = basic._get_attr(teacher_model, distill_op)
                student_feature = basic._get_attr(model, distill_op)
                
                if select_pass == 'fid':
                    teacher_features = [
                        [pass_filter(f, select_pass='high', J=J) for f in teacher_feature],
                        [pass_filter(f, select_pass='low', J=J) for f in teacher_feature],
                    ]
                    student_features = [
                        [pass_filter(f, select_pass='high', J=J) for f in student_feature],
                        [pass_filter(f, select_pass='low', J=J) for f in student_feature],
                    ]

                    loss_distill = None
                    for teacher_feature, student_feature in zip(teacher_features, student_features):
                        for j in range(len(student_feature)):
                            if (j + 1) % (2 ** op) == 0:
                                if loss_distill == None:
                                    if r2b:
                                        loss_distill = r2b_loss(student_feature[j], teacher_feature[j])
                                    else:
                                        loss_distill = torch.norm(student_feature[j] - teacher_feature[j], p=2)
                                else:
                                    if r2b:
                                        loss_distill = loss_distill + r2b_loss(student_feature[j], teacher_feature[j])
                                    else:
                                        loss_distill = loss_distill + torch.norm(student_feature[j] - teacher_feature[j], p=2)
                    loss = loss + loss_distill * distill_alpha * weight
                    if pred:
                        loss_pred = distillation_pred(out, teacher_out)
                        loss = loss + loss_pred * distill_alpha * weight
                else:
                    if select_pass == 'no':
                        teacher_features = [teacher_feature]
                    elif select_pass == 'high' or select_pass == 'low':
                        teacher_feature = [f1 / torch.std(f1) + f2 / torch.std(f2) for f1, f2 in [(pass_filter(f, select_pass=select_pass, J=J), f) for f in teacher_feature]]
                        teacher_features = [teacher_feature]

                    feature = [student_feature[i] for i in range(len(student_feature)) if (i + 1) % (2 ** op) == 0]
                    for teacher_feature in teacher_features:
                        if len(teacher_feature) % len(feature) == 0:
                            loss_distill = None
                            for k in range(len(feature)):
                                j = int((k + 1) * len(teacher_feature) / len(feature) - 1)
                                if loss_distill == None:
                                    loss_distill = r2b_loss(feature[k], teacher_feature[j])
                                else:
                                    loss_distill = loss_distill + r2b_loss(feature[k], teacher_feature[j])
                            loss = loss + loss_distill * distill_alpha * weight
                            if pred:
                                loss_pred = distillation_pred(out, teacher_out)
                                loss = loss + loss_pred * distill_alpha * weight
                        else:
                            print ('Distiilation Error: teacher {}, student {}!'.format(len(teacher_feature), len(feature))) 


        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        i += 1
        if i % (epoch_size // 5) == 0:
            print('[epoch %d]  [iter %d]  train loss: %.10f' % (epoch, i, loss.item()), flush=True)
        
    running_loss /= i

    return running_loss

def valid_epoch(model: nn.Module,
                criterion,
                data_loader: data.DataLoader,
                epoch,
                with_gpu,
                log_iter=10):
    """
    valid on dataset
    """
    model.eval()
    if with_gpu:
        model = model.cuda()

    # pbar = tqdm(data_loader, unit="audios", unit_scale=data_loader.batch_size)
    epoch_size = len(data_loader)

    thin_n = model.thin_n if hasattr(model, 'thin_n') else 1
    running_loss = 0
    running_acc = [0 for op in range(thin_n)]
    
    cnt = 0
    with torch.no_grad():
        for i, (feat, target) in enumerate(data_loader):
            cnt += target.size(0)
            for op in range(thin_n):
                if with_gpu:
                    feat, target = feat.cuda(), target.cuda()
    
                # forward
                if model.__class__.__name__[-9:] != 'thinnable':
                    out = model(feat)
                else:
                    out = model(feat, op)
                
                loss = criterion(out, target)

                pred = out.max(1, keepdim=True)[1]
                acc = pred.eq(target.view_as(pred)).sum()

                running_loss += loss.item() * target.size(0)
                running_acc[op] += acc.item()

            if i % (epoch_size // 5) == 0:
                print('[epoch %d]  [iter %d]  test loss: %.10f' % (epoch, i, loss.item()))

        running_acc = [acc / cnt for acc in running_acc]
        running_loss /= cnt

        log_acc = ' - '.join(['%.4f%%' % (acc * 100) for acc in running_acc])
        print('[epoch %d] test loss: %.10f, test acc: %s' % (epoch, running_loss, log_acc))
        
        if len(running_acc) == 1:   running_acc = running_acc[0]
        return running_loss, running_acc

