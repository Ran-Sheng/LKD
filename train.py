import copy
import os
import torch
import torch.nn as nn
import logging
import wandb
import time
import random
import numpy as np
from scipy.stats import spearmanr
from reparam_module import ReparamModule
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
import lib
from lib.models.builder import build_model
from lib.models.losses import CrossEntropyLabelSmooth, \
    SoftTargetCrossEntropy
from lib.dataset.builder import build_dataloader
from lib.utils.optim import build_optimizer
from lib.utils.scheduler import build_scheduler
from lib.utils.args import parse_args
from lib.utils.dist_utils import init_dist, init_logger
from lib.utils.misc import accuracy, AverageMeter, \
    CheckpointManager, AuxiliaryOutputBuffer
from lib.utils.model_ema import ModelEMA
from lib.utils.measure import get_params, get_flops

torch.backends.cudnn.benchmark = True

'''init logger'''
logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S')
logger = logging.getLogger()
logger.setLevel(logging.INFO)

writer = SummaryWriter('/root/tf-logs')


def main():
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'
    global kd_loss_ema
    args, args_text = parse_args()

    # 初始化wandb
    wandb.init(project=args.experiment, entity="sheng-ran", config=args)

    args.exp_dir = f'experiments/{args.experiment}'

    '''distributed'''
    init_dist(args)
    init_logger(args)

    # save args
    if args.rank == 0:
        with open(os.path.join(args.exp_dir, 'args.yaml'), 'w') as f:
            f.write(args_text)

    '''fix random seed'''
    seed = args.seed + args.rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.deterministic = True

    '''build dataloader'''
    train_dataset, val_dataset, train_loader, val_loader = build_dataloader(args)

    '''build model'''
    if args.mixup > 0. or args.cutmix > 0 or args.cutmix_minmax is not None:
        loss_fn = SoftTargetCrossEntropy()
    elif args.smoothing == 0.:
        loss_fn = nn.CrossEntropyLoss().cuda()
    else:
        loss_fn = CrossEntropyLabelSmooth(num_classes=args.num_classes,
                                          epsilon=args.smoothing).cuda()
    val_loss_fn = loss_fn

    model = build_model(args, args.model)
    logger.info(model)
    logger.info(
        f'Model {args.model} created, params: {get_params(model) / 1e6:.3f} M, '
        f'FLOPs: {get_flops(model, input_shape=args.input_shape) / 1e9:.3f} G')

    # Diverse Branch Blocks
    if args.dbb:
        # convert 3x3 convs to dbb blocks
        from lib.models.utils.dbb_converter import convert_to_dbb
        convert_to_dbb(model)
        logger.info(model)
        logger.info(
            f'Converted to DBB blocks, model params: {get_params(model) / 1e6:.3f} M, '
            f'FLOPs: {get_flops(model, input_shape=args.input_shape) / 1e9:.3f} G')

    model.cuda()
    model = DDP(model,
                device_ids=[args.local_rank],
                find_unused_parameters=False)

    # knowledge distillation
    if args.kd != '':
        # build teacher model
        teacher_model = build_model(args, args.teacher_model, args.teacher_pretrained, args.teacher_ckpt)
        logger.info(
            f'Teacher model {args.teacher_model} created, params: {get_params(teacher_model) / 1e6:.3f} M, '
            f'FLOPs: {get_flops(teacher_model, input_shape=args.input_shape) / 1e9:.3f} G')
        teacher_model.cuda()
        test_metrics = validate_teacher(args, 0, teacher_model, val_loader, val_loss_fn, log_suffix=' (teacher)')
        logger.info(f'Top-1 accuracy of teacher model {args.teacher_model}: {test_metrics["top1"]:.2f}')

        # build kd loss
        from lib.models.losses.kd_loss import KDLoss
        loss_fn = KDLoss(model, teacher_model, loss_fn, args.kd, args.student_module,
                         args.teacher_module, args.ori_loss_weight, args.kd_loss_weight)
        if args.kd_loss_ema:
            kd_loss_ema = ModelEMA(loss_fn.kd_loss, decay=args.kd_loss_ema_decay)
        else:
            kd_loss_ema = None

    if args.model_ema:
        model_ema = ModelEMA(model, decay=args.model_ema_decay)
    else:
        model_ema = None

    '''build optimizer'''
    optimizer = build_optimizer(args.opt,
                                model.module,
                                args.lr,
                                eps=args.opt_eps,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                filter_bias_and_bn=not args.opt_no_filter,
                                nesterov=not args.sgd_no_nesterov,
                                sort_params=args.dyrep)
    # Build optimizer for kd_loss
    optimizer_kd_loss = build_optimizer(args.kd_opt,
                                        model.module,
                                        args.kd_lr,
                                        eps=args.kd_opt_eps,
                                        momentum=args.kd_momentum,
                                        weight_decay=args.kd_weight_decay,
                                        filter_bias_and_bn=not args.opt_no_filter,
                                        nesterov=not args.sgd_no_nesterov,
                                        sort_params=args.dyrep)

    '''build scheduler'''
    steps_per_epoch = len(train_loader)
    warmup_steps = args.warmup_epochs * steps_per_epoch
    decay_steps = args.decay_epochs * steps_per_epoch
    total_steps = args.epochs * steps_per_epoch
    scheduler = build_scheduler(args.sched,
                                optimizer,
                                warmup_steps,
                                args.warmup_lr,
                                decay_steps,
                                args.decay_rate,
                                total_steps,
                                steps_per_epoch=steps_per_epoch,
                                decay_by_epoch=args.decay_by_epoch,
                                min_lr=args.min_lr)
    scheduler_kd_loss = build_scheduler(args.sched,
                                        optimizer_kd_loss,
                                        warmup_steps,
                                        args.warmup_lr,
                                        decay_steps,
                                        args.decay_rate,
                                        total_steps,
                                        steps_per_epoch=steps_per_epoch,
                                        decay_by_epoch=args.decay_by_epoch,
                                        min_lr=args.min_lr)

    '''dyrep'''
    if args.dyrep:
        from lib.models.utils.dyrep import DyRep
        from lib.models.utils.recal_bn import recal_bn
        dyrep = DyRep(
            model.module,
            optimizer,
            recal_bn_fn=lambda m: recal_bn(model.module, train_loader,
                                           args.dyrep_recal_bn_iters, m),
            filter_bias_and_bn=not args.opt_no_filter)
        logger.info('Init DyRep done.')
    else:
        dyrep = None

    '''amp'''
    if args.amp:
        loss_scaler = torch.cuda.amp.GradScaler()
    else:
        loss_scaler = None

    '''resume'''
    ckpt_manager = CheckpointManager(model,
                                     optimizer,
                                     ema_model=model_ema,
                                     save_dir=args.exp_dir,
                                     rank=args.rank,
                                     additions={
                                         'scaler': loss_scaler,
                                         'dyrep': dyrep
                                     })

    if args.resume:
        start_epoch = ckpt_manager.load(args.resume) + 1
        if start_epoch > args.warmup_epochs:
            scheduler.finished = True
        scheduler.step(start_epoch * len(train_loader))
        if args.dyrep:
            model = DDP(model.module,
                        device_ids=[args.local_rank],
                        find_unused_parameters=True)
        logger.info(
            f'Resume ckpt {args.resume} done, '
            f'start training from epoch {start_epoch}'
        )
    else:
        start_epoch = 0

    '''auxiliary tower'''
    if args.auxiliary:
        auxiliary_buffer = AuxiliaryOutputBuffer(model, args.auxiliary_weight)
    else:
        auxiliary_buffer = None

    batch_kd_losses = []
    batch_top1_accs = []

    '''train & val'''
    for epoch in range(start_epoch, args.epochs):
        # torch.autograd.set_detect_anomaly(True)
        train_loader.loader.sampler.set_epoch(epoch)

        if args.drop_path_rate > 0. and args.drop_path_strategy == 'linear':
            # update drop path rate
            if hasattr(model.module, 'drop_path_rate'):
                model.module.drop_path_rate = \
                    args.drop_path_rate * epoch / args.epochs

        # train
        metrics = train_epoch(args, epoch, model, model_ema, train_loader,
                              optimizer, loss_fn, scheduler, auxiliary_buffer,
                              dyrep, loss_scaler)
        if args.kd == 'learnable_kd':
            kd_train_loader = iter(train_loader)

            for i in range(20):
                weights_files = [f for f in os.listdir(args.exp_dir) if f.startswith('student_weights_epoch_')]
                # 如果有保存的权重，则随机选择一个加载到副本
                if weights_files:
                    chosen_file = random.choice(weights_files)
                    student_weights_path = os.path.join(args.exp_dir, chosen_file)
                else:
                    student_weights_path = None

                train_one_epoch_learnable_kd_loss(args, epoch, teacher_model, model, loss_fn.kd_loss, optimizer_kd_loss,
                                                  train_loader, scheduler_kd_loss, kd_train_loader, student_weights_path)

        # validate
        test_metrics = validate(args, epoch, teacher_model, model, val_loader, val_loss_fn, loss_fn.kd_loss,
                                batch_kd_losses=batch_kd_losses,
                                batch_top1_accs=batch_top1_accs)
        if model_ema is not None:
            test_metrics = validate(args,
                                    epoch,
                                    model_ema.module,
                                    val_loader,
                                    loss_fn,
                                    log_suffix='(EMA)')

        # dyrep
        if dyrep is not None:
            if epoch < args.dyrep_max_adjust_epochs:
                if (epoch + 1) % args.dyrep_adjust_interval == 0:
                    # adjust
                    logger.info('DyRep: adjust model.')
                    dyrep.adjust_model()
                    logger.info(
                        f'Model params: {get_params(model)/1e6:.3f} M, FLOPs: {get_flops(model, input_shape=args.input_shape)/1e9:.3f} G'
                    )
                    # re-init DDP
                    model = DDP(model.module,
                                device_ids=[args.local_rank],
                                find_unused_parameters=True)
                    test_metrics = validate(args, epoch, model, val_loader, val_loss_fn)
                elif args.dyrep_recal_bn_every_epoch:
                    logger.info('DyRep: recalibrate BN.')
                    recal_bn(model.module, train_loader, 200)
                    test_metrics = validate(args, epoch, model, val_loader, val_loss_fn)

        metrics.update(test_metrics)
        ckpts = ckpt_manager.update(epoch, metrics)
        logger.info('\n'.join(['Checkpoints:'] + [
            '        {} : {:.3f}%'.format(ckpt, score) for ckpt, score in ckpts
        ]))
    spearman_corr, _ = spearmanr(batch_kd_losses, batch_top1_accs)
    # np.save('LKD_batch_kd_losses.npy', batch_kd_losses)
    # np.save('LKD_batch_top1_accs.npy', batch_top1_accs)
    logger.info(f"Spearman's rank correlation coefficient for batches: {spearman_corr}")
    wandb.finish()


def train_epoch(args,
                epoch,
                model,
                model_ema,
                loader,
                optimizer,
                loss_fn,
                scheduler,
                auxiliary_buffer=None,
                dyrep=None,
                loss_scaler=None):
    loss_m = AverageMeter(dist=True)
    data_time_m = AverageMeter(dist=True)
    batch_time_m = AverageMeter(dist=True)
    start_time = time.time()

    model.train()
    for batch_idx, (input, target) in enumerate(loader):
        data_time = time.time() - start_time
        data_time_m.update(data_time)

        # optimizer.zero_grad()
        # use optimizer.zero_grad(set_to_none=False) for speedup
        for p in model.parameters():
            p.grad = None

        if not args.kd:
            output = model(input)
            loss = loss_fn(output, target)
        else:
            if args.kd_loss_ema:
                loss_fn.set_kd_loss_ema(kd_loss_ema.module)
                loss = loss_fn(input, target)
            else:
                loss = loss_fn(input, target)

        if auxiliary_buffer is not None:
            loss_aux = loss_fn(auxiliary_buffer.output, target)
            loss += loss_aux * auxiliary_buffer.loss_weight

        if loss_scaler is None:
            loss.backward()
        else:
            # amp
            loss_scaler.scale(loss).backward()
        if args.clip_grad_norm:
            if loss_scaler is not None:
                loss_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           args.clip_grad_max_norm)

        if dyrep is not None:
            # record states of model in dyrep
            dyrep.record_metrics()
            
        if loss_scaler is None:
            optimizer.step()
        else:
            loss_scaler.step(optimizer)
            loss_scaler.update()

        if model_ema is not None:
            model_ema.update(model)

        loss_m.update(loss.item(), n=input.size(0))
        batch_time = time.time() - start_time
        batch_time_m.update(batch_time)
        if batch_idx % args.log_interval == 0 or batch_idx == len(loader) - 1:
            logger.info('Train: {} [{:>4d}/{}] '
                        'Loss: {loss.val:.3f} ({loss.avg:.3f}) '
                        'LR: {lr:.3e} '
                        'Time: {batch_time.val:.2f}s ({batch_time.avg:.2f}s) '
                        'Data: {data_time.val:.2f}s'.format(
                            epoch,
                            batch_idx,
                            len(loader),
                            loss=loss_m,
                            lr=optimizer.param_groups[0]['lr'],
                            batch_time=batch_time_m,
                            data_time=data_time_m))
        scheduler.step(epoch * len(loader) + batch_idx + 1)
        start_time = time.time()

    # 保存学生网络权重
    if (epoch + 1) % 5 == 0:
        student_weights_path = os.path.join(args.exp_dir, f'student_weights_epoch_{epoch + 1}.pth')
        torch.save(model.module.state_dict(), student_weights_path)
        logger.info(f'Student weights saved at epoch {epoch + 1} to {student_weights_path}')

    return {'train_loss': loss_m.avg}


def train_one_epoch_learnable_kd_loss(args, epoch, teacher_net, student_net, kd_loss, optimizer_kd_loss, ori_train_loader,
                                      scheduler, train_loader, student_weights_path):
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 创建学生网络副本
    student_net_copy = copy.deepcopy(student_net.module).to(args.device)
    # 获取保存的学生网络权重文件列表
    #weights_files = [f for f in os.listdir(args.exp_dir) if f.startswith('student_weights_epoch_')]
    # 如果有保存的权重，则随机选择一个加载到副本
    #if weights_files:
    #    chosen_file = random.choice(weights_files)
    #    student_weights_path = os.path.join(args.exp_dir, chosen_file)
    if student_weights_path is not None:
        student_net_copy.load_state_dict(torch.load(student_weights_path))
        logger.info(f'Loaded student weights from {student_weights_path} for Learnable KD loss training.')

    student_net_copy = ReparamModule(student_net_copy)
    kd_loss.train()
    student_net_copy.train()
    teacher_net.eval()

    total_ce_loss = 0
    batch_list = []
    criterion = nn.CrossEntropyLoss().to(args.device)

    student_params = [torch.cat([p.data.to(args.device).reshape(-1) for p in student_net_copy.parameters()],
                                dim=0).requires_grad_(True)]

    # # 定义参数扰动函数
    # def perturb_student_params(params, noise_scale=0.01):
    #     noise = torch.randn_like(params) * noise_scale
    #     return params + noise  # 返回新的扰动后的张量
    #
    # student_params[-1] = perturb_student_params(student_params[-1], noise_scale=0.01)

    # Update student network parameters
    # for step in range(args.update_student_params_steps):
    random_steps = int(np.round(np.random.normal(25, 2.55)))
    for step in range(random_steps):
        try:
            x_batch1, y_batch1 = next(train_loader)
        except StopIteration:
            train_loader = iter(ori_train_loader)
            x_batch1, y_batch1 = next(train_loader)
        batch_list.append((x_batch1, y_batch1))
        x_batch1, y_batch1 = x_batch1.to(args.device), y_batch1.to(args.device)

        # Forward pass with teacher and student networks
        logit_teacher = teacher_net(x=x_batch1)
        logit_student = student_net_copy(x=x_batch1, flat_param=student_params[-1])

        # Calculate learnable KD loss
        # loss, weight, T = kd_loss(logit_student, logit_teacher, y_batch1)
        loss, T = kd_loss(logit_student, logit_teacher, y_batch1)
        loss = loss + criterion(logit_student, y_batch1)
        # 每10个epoch的最后一个step记录一次统计量
        if epoch % 10 == 0 and step == random_steps-1:
            # 计算统计量
            # weight_max = torch.max(weight).item()
            # weight_min = torch.min(weight).item()
            # weight_variance = torch.var(weight).item()

            T_max = torch.max(T).item()
            T_min = torch.min(T).item()
            T_variance = torch.var(T).item()

            # 记录日志
            logger.info(f'Epoch {epoch}, Step {step}, '
                        # f'Weight Max: {weight_max}, Weight Min: {weight_min}, '
                        # f'Weight Variance: {weight_variance}, '
                        f'T Max: {T_max}, T Min: {T_min}, '
                        f'T Variance: {T_variance}')
        # loss, weight, T = loss.to(args.device), weight.to(args.device), T.to(args.device)
        loss, T = loss.to(args.device), T.to(args.device)

        # Compute gradients
        gradients = torch.autograd.grad(loss,
                                        student_params[-1],
                                        create_graph=True,
                                        allow_unused=True)

        # Ensure at least one gradient tensor is used
        updated_params = student_params[-1] - 0.01 * gradients[0]
        student_params.append(updated_params)

    # Calculate classification loss and update parameters
    for x, y in batch_list:
        x, y = x.to(args.device), y.to(args.device)
        val_student = student_net_copy(x=x, flat_param=student_params[-1]).to(args.device)

        # Calculate classification loss
        ce_loss = criterion(val_student, y).to(args.device)
        total_ce_loss = total_ce_loss + ce_loss

    # Calculate average classification loss
    average_ce_loss = total_ce_loss / len(batch_list)
    writer.add_scalar("average_ce_loss", average_ce_loss, epoch)
    wandb.log({"average_ce_loss": average_ce_loss, "epoch": epoch})

    # Backward pass and optimization
    optimizer_kd_loss.zero_grad()
    average_ce_loss.backward()
    # nn.utils.clip_grad_norm_(parameters=kd_loss.parameters(), max_norm=0.5)
    optimizer_kd_loss.step()
    if args.kd_loss_ema:
        kd_loss_ema.update(kd_loss)

    # Scheduler step
    # scheduler.step()

    # writer.add_histogram('weights_distribution', weight, epoch)
    writer.add_histogram('T_distribution', T, epoch)
    # weight_np = weight.detach().cpu().numpy()
    T_np = T.detach().cpu().numpy()
    # wandb.log({"weights_distribution": wandb.Histogram(weight_np), "epoch": epoch})
    wandb.log({"T_distribution": wandb.Histogram(T_np), "epoch": epoch})

    if epoch == args.epochs - 1:
        kd_loss_weights_path = os.path.join(args.exp_dir, 'LKD_Loss_cifar100.pth')
        torch.save(kd_loss.state_dict(), kd_loss_weights_path)
        logger.info(f'KD loss weights saved to {kd_loss_weights_path}')

    return {'average_ce_loss': average_ce_loss.item()}


def validate(args, epoch, teacher_model, model, loader, loss_fn, kd_loss_fn, log_suffix='',
             batch_kd_losses=None, batch_top1_accs=None):
    loss_m = AverageMeter(dist=True)
    kd_loss_m = AverageMeter(dist=True)
    top1_m = AverageMeter(dist=True)
    top5_m = AverageMeter(dist=True)
    batch_time_m = AverageMeter(dist=True)
    start_time = time.time()

    teacher_model.eval()
    model.eval()
    for batch_idx, (input, target) in enumerate(loader):
        with torch.no_grad():
            teacher_output = teacher_model(input)
            student_output = model(input)
            if args.kd == 'learnable_kd':
                kd_loss, _ = kd_loss_fn(student_output, teacher_output, target)
            else:
                kd_loss = kd_loss_fn(student_output, teacher_output)
            loss = loss_fn(student_output, target)

        top1, top5 = accuracy(student_output, target, topk=(1, 5))
        kd_loss_m.update(kd_loss.item(), n=input.size(0))
        loss_m.update(loss.item(), n=input.size(0))
        top1_m.update(top1 * 100, n=input.size(0))
        top5_m.update(top5 * 100, n=input.size(0))

        batch_time = time.time() - start_time
        batch_time_m.update(batch_time)

        batch_kd_losses.append(kd_loss.item())
        batch_top1_accs.append(top1)

        if batch_idx % args.log_interval == 0 or batch_idx == len(loader) - 1:
            logger.info('Test{}: {} [{:>4d}/{}] '
                        'Loss: {loss.val:.3f} ({loss.avg:.3f}) '
                        'KD Loss: {kd_loss.val:.3f} ({kd_loss.avg:.3f}) '
                        'Top-1: {top1.val:.3f}% ({top1.avg:.3f}%) '
                        'Top-5: {top5.val:.3f}% ({top5.avg:.3f}%) '
                        'Time: {batch_time.val:.2f}s'.format(
                            log_suffix,
                            epoch,
                            batch_idx,
                            len(loader),
                            loss=loss_m,
                            kd_loss=kd_loss_m,
                            top1=top1_m,
                            top5=top5_m,
                            batch_time=batch_time_m))
        start_time = time.time()

    return {'test_loss': loss_m.avg, 'val_kd_loss': kd_loss_m.avg, 'top1': top1_m.avg, 'top5': top5_m.avg}


def validate_teacher(args, epoch, teacher_model, loader, loss_fn, log_suffix=''):
    loss_m = AverageMeter(dist=True)
    top1_m = AverageMeter(dist=True)
    top5_m = AverageMeter(dist=True)
    batch_time_m = AverageMeter(dist=True)
    start_time = time.time()

    teacher_model.eval()
    for batch_idx, (input, target) in enumerate(loader):
        with torch.no_grad():
            output = teacher_model(input)
            loss = loss_fn(output, target)

        top1, top5 = accuracy(output, target, topk=(1, 5))
        loss_m.update(loss.item(), n=input.size(0))
        top1_m.update(top1 * 100, n=input.size(0))
        top5_m.update(top5 * 100, n=input.size(0))

        batch_time = time.time() - start_time
        batch_time_m.update(batch_time)
        if batch_idx % args.log_interval == 0 or batch_idx == len(loader) - 1:
            logger.info('Test{}: {} [{:>4d}/{}] '
                        'Loss: {loss.val:.3f} ({loss.avg:.3f}) '
                        'Top-1: {top1.val:.3f}% ({top1.avg:.3f}%) '
                        'Top-5: {top5.val:.3f}% ({top5.avg:.3f}%) '
                        'Time: {batch_time.val:.2f}s'.format(
                            log_suffix,
                            epoch,
                            batch_idx,
                            len(loader),
                            loss=loss_m,
                            top1=top1_m,
                            top5=top5_m,
                            batch_time=batch_time_m))
        start_time = time.time()

    return {'test_loss': loss_m.avg, 'top1': top1_m.avg, 'top5': top5_m.avg}


if __name__ == '__main__':
    main()
