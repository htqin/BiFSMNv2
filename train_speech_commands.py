import argparse
import random
import warnings
import os

try:
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))
except:
    print('no resource')
# sudo sh -c "ulimit -n 65535 && exec su $LOGNAME"

import torch
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp

from train_valid_test import train_epoch, valid_epoch, \
    create_lr_schedule, create_optimizer, get_model, create_dataloader



def parse_args():
    """
    Parse input arguments
    """
    # general args
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--saveroot',
                        help='set root folder for log and checkpoint',
                        type=str,
                        default='save_models')
    parser.add_argument('--dataroot',
                        help='set root folder for dataset',
                        type=str,
                        default='/mnt/data/sata/lin/datasets/SpeechCommands')
    parser.add_argument('--checkpoint',
                        help='choose a checkpoint to resume',
                        type=str,
                        default=None)
    parser.add_argument(
        '--test',
        action='store_true',
        help='test accuracy with input checkpoint',
    )

    # model args
    parser.add_argument('--n_mels',
                        type=int,
                        default=32,
                        help='mel feature size')
    parser.add_argument(
        '--model',
        type=str,
        default='Dfsmn')
    parser.add_argument('--dfsmn_with_bn',
                        action='store_true',
                        help='use BatchNorm for Dfsmn model')
    parser.add_argument('--num_layer',
                        type=int,
                        default=8,
                        help='num_layer for  Dfsmn model')
    parser.add_argument('--frondend_channels',
                        type=int,
                        default=16,
                        help='frondend_channels for  Dfsmn model')
    parser.add_argument('--frondend_kernel_size',
                        type=int,
                        default=5,
                        help='frondend_kernel_size for  Dfsmn model')
    parser.add_argument('--hidden_size',
                        type=int,
                        default=256,
                        help='hidden_size for  Dfsmn model')
    parser.add_argument('--backbone_memory_size',
                        type=int,
                        default=128,
                        help='backbone_memory_size for  Dfsmn model')
    parser.add_argument('--left_kernel_size',
                        type=int,
                        default=2,
                        help='left_kernel_size for  Dfsmn model')
    parser.add_argument('--right_kernel_size',
                        type=int,
                        default=2,
                        help='right_kernel_size for  Dfsmn model')

    # args for training hyper parameters
    parser.add_argument("--epoch", type=int, default=300, help='total epochs')
    parser.add_argument("--batch-size", type=int, default=96, help='batch size')
    parser.add_argument("--lr", type=float, default=1e-3, help='learning rate')
    parser.add_argument("--lr-scheduler",
                        choices=['plateau', 'step', 'cosin'],
                        default='cosin',
                        help='method to adjust learning rate')
    parser.add_argument("--weight-decay",
                        type=float,
                        default=1e-2,
                        help='weight decay')
    parser.add_argument(
        "--lr-scheduler-patience",
        type=int,
        default=5,
        help='lr scheduler plateau: Number of epochs with no improvement '
        'after which learning rate will be reduced')
    parser.add_argument(
        "--lr-scheduler-stepsize",
        type=int,
        default=5,
        help='lr scheduler step: number of epochs of learning rate decay.')
    parser.add_argument(
        "--lr-scheduler-gamma",
        type=float,
        default=0.1,
        help='learning rate is multiplied by the gamma to decrease it')
    parser.add_argument("--optim",
                        choices=['sgd', 'adam'],
                        default='sgd',
                        help='choices of optimization algorithms')
    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=0,
        help='label_smoothing (float, optional): A float in [0.0, 1.0].')

    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
    parser.add_argument('--seed',
                        default=None,
                        type=int,
                        help='seed for initializing training. ')

    # args for distill/thinnable
    parser.add_argument('--num_classes', type=int, default=12, choices=[12, 20, 35], help='num_classes for dataset')
    parser.add_argument('--version', default="speech_commands_v0.01", choices=["speech_commands_v0.01", "speech_commands_v0.02"], type=str, help='dataset version')
    parser.add_argument('--thin_n', type=int, default=3, choices=[1, 2, 3, 4], help='ways for BiDfsmn_thinnable')
    parser.add_argument("--distill", action='store_true', help='disitll')
    parser.add_argument("--distill_conv", action='store_true', help='disitll conv1d and conv2d')
    parser.add_argument("--distill_alpha", type=float, default=0, help='disitll alpha.')
    parser.add_argument("--teacher_model", choices=['Vgg19Bn', 'Mobilenetv1', 'Mobilenetv2', 'BCResNet', 'Dfsmn', 'BiDfsmn', 'BiDfsmn_thinnable', 'BiDfsmn_thinnable_pre', 'fsmn'], type=str, default='Dfsmn', help='teacher model')
    parser.add_argument('--teacher_model_checkpoint', type=str, help='teacher pretrained model path: saveroot + teacher_model_checkpoint')
    parser.add_argument('--pretrained', action='store_true', help='load the pre-trained teacher model')
    parser.add_argument("--select_pass", type=str, default='no', help='high-pass or low-pass for wavelet.')
    parser.add_argument("--J", type=int, default=1, help='scale of wavelet.')
    parser.add_argument("--method", type=str, default='no', help='bi method.')
    parser.add_argument("--bits", type=int, default=1, help='bi method.')
    parser.add_argument("--mse", action='store_true', help='mse')

    parsed_args = parser.parse_args()
    return parsed_args


def test_speech_commands(configs, gpu_id=None, model=None):
    if model == None:
        model = get_model(configs.model,
                        in_channels=1,
                        **(vars(configs)))
        print(model)
        nparams = sum(p.numel() for p in model.parameters() if p.requires_grad)
        names_params = {
            n: p.numel() * 1e-6
            for n, p in model.named_parameters() if p.requires_grad
        }
        sorted_names_params = sorted(names_params.items(),
                                    key=lambda kv: kv[1],
                                    reverse=True)
        print(sorted_names_params)
        print('create model: {}, with {} M Params(With BN param)'.format(
                configs.model, nparams * 1e-6))

        if configs.checkpoint is None:
            raise RuntimeError('test mode must provider checkpoint')

        chpk = torch.load(configs.checkpoint)
        model.load_state_dict(chpk['state_dict'])
        if gpu_id is not None:
            model.cuda(gpu_id)

    dataloader_test = create_dataloader('testing',
                                        configs,
                                        use_gpu=gpu_id is not None,
                                        version=configs.version)

    criterion = torch.nn.CrossEntropyLoss()
    
    valid_loss, accuracy = valid_epoch(model, criterion, dataloader_test,
                                        0, gpu_id is not None, 10)
    if not isinstance(accuracy, list):
        accuracy = [accuracy]
    print('checkpoint: {}, loss: {}, accuracy: {}'.format(
        configs.checkpoint, valid_loss, ['%.4f%%' % (x * 100) for x in accuracy]))
    
def train_speech_commands(configs, gpu_id=None):
    best_accuracy = 0
    best_accuracys = None
    epoch = 0

    use_gpu = torch.cuda.is_available()
    if gpu_id is not None:
        torch.cuda.set_device(gpu_id)

    model = get_model(configs.model,
                      in_channels=1,
                      **(vars(configs)))
    print(model)
    nparams = sum(p.numel() for p in model.parameters() if p.requires_grad)
    names_params = {
        n: p.numel() * 1e-6
        for n, p in model.named_parameters() if p.requires_grad
    }
    sorted_names_params = sorted(names_params.items(),
                                 key=lambda kv: kv[1],
                                 reverse=True)
    print(sorted_names_params)
    print('create model: {}, with {} M Params(With BN param)'.format(
            configs.model, nparams * 1e-6))
            
    teacher_model = None
    if configs.distill:
        teacher_model = get_model(configs.teacher_model,
                      in_channels=1,
                      teacher=True,
                      **(vars(configs)))
        print('teacher_model !!!!!!!')
        print(teacher_model)
        chpk = torch.load(os.path.join(configs.saveroot, configs.teacher_model_checkpoint), map_location='cpu')
        teacher_model.load_state_dict(chpk['state_dict'])
    if configs.pretrained:
        chpk = torch.load(os.path.join(configs.saveroot, configs.teacher_model_checkpoint), map_location='cpu')
        model.load_state_dict(chpk['state_dict'], strict=False)
        
    criterion = torch.nn.CrossEntropyLoss()

    optimizer = create_optimizer(configs, model)    
    if configs.checkpoint is not None:
        chpk = torch.load(configs.checkpoint, map_location='cpu')
        best_accuracy = chpk['accuracy']
        epoch = chpk['epoch']
        model.load_state_dict(chpk['state_dict'])
        optimizer.load_state_dict(chpk['optimizer'])

    lr_scheduler = create_lr_schedule(configs, optimizer)

    dataloader_train = create_dataloader('training', configs, use_gpu, version=configs.version)

    dataloader_valid = create_dataloader('validation', configs, use_gpu, version=configs.version)

    if gpu_id is not None:
        model = model.cuda(gpu_id)
        if teacher_model != None:
            teacher_model = teacher_model.cuda(gpu_id)

    # train
    for cur_epoch in range(epoch, configs.epoch):
        print("runing on epoch: {}, learning_rate: {}, lr: {}, wd:{}".format(
            cur_epoch, optimizer.param_groups[0]['lr'],  configs.lr, configs.weight_decay, ), flush=True)
        green = lambda x: '\033[32m' + x + '\033[0m'
        print(green('{}, v{}-{}, {}, lr={}, wd={}, alpha={}'.format(args.method, args.version[-1:], args.num_classes, args.optim, args.lr, args.weight_decay, args.distill_alpha)))

        distill_conv = True if configs.distill_conv else False
        train_loss = train_epoch(model,
                                teacher_model,
                                optimizer,
                                criterion,
                                dataloader_train,
                                epoch=cur_epoch,
                                with_gpu=use_gpu,
                                log_iter=10,
                                distill_alpha=configs.distill_alpha,
                                distill_conv=distill_conv,
                                select_pass=configs.select_pass,
                                J=configs.J,
                                num_classes=configs.num_classes,
                                r2b=not configs.mse)
        valid_loss, accuracy = valid_epoch(model, criterion, dataloader_valid,
                                        cur_epoch, use_gpu, 10)
       
        # valid_loss, accuracy = 0, 0
        if configs.lr_scheduler == 'plateau':
            lr_scheduler.step(metrics=valid_loss)
        else:
            lr_scheduler.step()
       

        if not isinstance(accuracy, list):
            accuracy = [accuracy]
        if best_accuracys == None:
            best_accuracys = accuracy
        avg_accuracy = accuracy[0]
        if avg_accuracy > best_accuracy and (len(accuracy) == 1 or (min([x - y for x, y in zip(accuracy[:-1], accuracy[1:])]) > 0)):
            best_accuracy = avg_accuracy
            best_accuracys = accuracy
            print("Got better checkpointer, epoch: {}, accuracy: {}, valid loss: {}"
                .format(cur_epoch, best_accuracy, valid_loss))
            checkpoint = {
                'epoch': cur_epoch,
                'state_dict': model.cpu().state_dict(),
                'accuracy': best_accuracy,
                'optimizer': optimizer.state_dict(),
            }
            pth_name = '{}_{}_lr_{}_wd_{}_lrscheudle_{}_v{}_{}'.format(
                    configs.model, configs.method, configs.lr, configs.weight_decay,
                    configs.lr_scheduler, int(configs.version[-1:]), int(configs.num_classes))
            if configs.distill:
                if configs.distill_conv:
                    pth_name = pth_name + '_distill_conv_{}'.format(configs.distill_alpha)
                else:
                    pth_name = pth_name + '_distill_{}'.format(configs.distill_alpha)
            if configs.select_pass != 'no':
                pth_name = pth_name + '_' + configs.select_pass + '_J_{}'.format(configs.J)
            pth_name = pth_name + '_best.pth'
            best_checkpoint_path = os.path.join(
                configs.saveroot,
                pth_name)
            torch.save(checkpoint, best_checkpoint_path)
            configs.checkpoint = best_checkpoint_path
        
        print('train loss: ', train_loss,
                    ', valid: best_accuracy: ', best_accuracy,
                    ', cur_accuracy: ', ['%.4f%%' % (x * 100) for x in accuracy],
                    ', best_accuracys', ['%.4f%%' % (x * 100) for x in best_accuracys],
                    ', valid loss: ', valid_loss)

    
        checkpoint = {
            'epoch': cur_epoch,
            'state_dict': model.cpu().state_dict(),
            'accuracy': accuracy[0],
            'optimizer': optimizer.state_dict(),
        }
        pth_name = '{}_{}_small_lr_{}_wd_{}_lrscheudle_{}_v{}_{}'.format(
                configs.model, configs.method, configs.lr, configs.weight_decay,
                configs.lr_scheduler, int(configs.version[-1:]), int(configs.num_classes))
        if configs.distill:
            if configs.distill_conv:
                pth_name = pth_name + '_distill_conv_{}'.format(configs.distill_alpha)
            else:
                pth_name = pth_name + '_distill_{}'.format(configs.distill_alpha)
        if configs.select_pass != 'no':
            pth_name = pth_name + '_' + configs.select_pass + '_J_{}'.format(configs.J)
        pth_name = pth_name + '_last.pth'
        last_checkpoint_path = os.path.join(
            configs.saveroot,
            pth_name)
        torch.save(checkpoint, last_checkpoint_path)

    test_speech_commands(configs, gpu_id)

if __name__ == "__main__":
    with mp.Pool(40) as pool:
        # mp.set_start_method('spawn')
        args = parse_args()

        assert((args.distill and args.distill_alpha != 0) or (args.distill_alpha == 0 and not args.distill))

        if args.test:
            test_speech_commands(args, args.gpu)
        else:
            if args.seed is not None:
                random.seed(args.seed)
                torch.manual_seed(args.seed)
                cudnn.deterministic = True
                warnings.warn('You have chosen to seed training. '
                            'This will turn on the CUDNN deterministic setting, '
                            'which can slow down your training considerably! '
                            'You may see unexpected behavior when restarting '
                            'from checkpoints.')

            if args.gpu is not None:
                warnings.warn(
                    'You have chosen a specific GPU. This will completely '
                    'disable data parallelism.')

            # build model
            os.makedirs(args.saveroot, exist_ok=True)

            train_speech_commands(args, gpu_id=args.gpu)
        
        green = lambda x: '\033[32m' + x + '\033[0m'
        print(green('{}, v{}-{}, {}, lr={}, wd={}, alpha={}'.format(args.method, args.version[-1:], args.num_classes, args.optim, args.lr, args.weight_decay, args.distill_alpha)))

