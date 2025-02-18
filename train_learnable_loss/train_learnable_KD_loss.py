import os
import argparse
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from utils import get_dataset, get_network, get_eval_pool, get_time
from lib.models import nas_model
from lib.models.cifar.vgg import *
from learnable_loss.learnable_KD_loss import LearnableKDLoss
from reparam_module import ReparamModule
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

writer = SummaryWriter('/root/tf-logs')

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4'


def main(args):
    if args.max_experts is not None and args.max_files is not None:
        args.total_experts = args.max_experts * args.max_files

    print("CUDNN STATUS: {}".format(torch.backends.cudnn.enabled))

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    channel, im_size, num_classes, train_loader, val_loader, test_loader = get_dataset(dataset=args.dataset,
                                                                                       data_path=args.data_path,
                                                                                       subset=args.subset)
    model_eval_pool = get_eval_pool(args.eval_mode, args.student_model, args.student_model)

    args.im_size = im_size

    accs_all_exps = dict()  # record performances of all experiments
    for key in model_eval_pool:
        accs_all_exps[key] = []

    print('Hyper-parameters: \n', parser.parse_args())
    print('Evaluation model pool: ', model_eval_pool)

    ''' training '''
    kd_loss = LearnableKDLoss(num_classes=num_classes, T=args.T).to(args.device)
    kd_loss.train()

    optimizer_kd_loss = torch.optim.SGD(kd_loss.parameters(), lr=args.lr_kd_loss, momentum=0.9, weight_decay=1e-4)
    lr_scheduler = CosineAnnealingLR(optimizer=optimizer_kd_loss, T_max=args.Epochs)

    criterion = nn.CrossEntropyLoss().to(args.device)
    print('%s training begins' % get_time())

    expert_dir = os.path.join(args.buffer_path, args.dataset)
    if args.dataset == "ImageNet":
        expert_dir = os.path.join(expert_dir, args.subset, str(args.res))
    if args.dataset in ["CIFAR10", "CIFAR100"]:
        expert_dir = os.path.join(expert_dir, args.student_model)
    print("Expert Dir: {}".format(expert_dir))

    buffer = []
    n = 0

    while os.path.exists(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n))):
        buffer = buffer + torch.load(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n)))
        n += 1
    if n == 0:
        raise AssertionError("No buffers detected at {}".format(expert_dir))

    v_loader = iter(val_loader)
    t_loader = iter(train_loader)

    # get a teacher model
    teacher_net = vgg16_bn().to(args.device)
    checkpoint = torch.load('experiments/vgg16_cifar10_exp/best.pth.tar')
    teacher_net.load_state_dict(checkpoint['model'])
    teacher_net.eval()

    # get a random student model
    student_net = get_network(args.student_model, channel, num_classes, im_size, dist=False).to(args.device)
    student_net = ReparamModule(student_net)
    student_net.train()

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        teacher_net = nn.DataParallel(teacher_net)

    for epoch in range(args.Epochs):
        print('%s iter = %04d' % (get_time(), epoch))

        log = {
            "correct_before": 0.0,
            "total_before": 0.0,
            "correct_after": 0.0,
            "total_after": 0.0,
            "loss": 0.0,
        }

        if len(buffer) > 0:
            random_index = np.random.randint(0, len(buffer))
            expert_trajectory = buffer[random_index]
        else:
            # 处理 buffer 为空的情况，例如提供默认值或引发异常
            raise ValueError("Buffer is empty, cannot select expert_trajectory.")

        start_epoch = np.random.randint(0, args.max_start_epoch)
        starting_params = expert_trajectory[start_epoch]
        student_params = [torch.cat([p.data.to(args.device).reshape(-1)
                                     for p in starting_params], dim=0).requires_grad_(True)]

        with torch.no_grad():
            for x_test, y_test in test_loader:
                x_test, y_test = x_test.to(args.device), y_test.to(args.device)
                out_test = student_net(x_test, flat_param=student_params[-1])

                log["correct_before"] += torch.sum(torch.eq(torch.argmax(out_test, dim=1), y_test))
                log["total_before"] += x_test.size(0)

        print("acc_before: {}".format(log["correct_before"] / log["total_before"]))

        for step in range(args.update_student_params_steps):
            try:
                x_batch1, y_batch1 = next(t_loader)
            except StopIteration:
                # 删除迭代器iter_test
                del t_loader
                # 重新生成迭代器iter_test
                t_loader = iter(train_loader)
                x_batch1, y_batch1 = next(t_loader)
            x_batch1, y_batch1 = x_batch1.to(args.device), y_batch1.to(args.device)
            logit_1, logit_2 = teacher_net(x=x_batch1), student_net(x=x_batch1, flat_param=student_params[-1])
            logit_1, logit_2 = logit_1.to(args.device), logit_2.to(args.device)
            loss, weight = kd_loss(logit_1, logit_2)
            loss, weight = loss.to(args.device), weight.to(args.device)
            gradients = torch.autograd.grad(loss, student_params[-1], create_graph=True, allow_unused=True)

            # 确保至少使用一个梯度张量
            updated_params = student_params[-1] - 0.01 * gradients[0]
            student_params.append(updated_params)

        # x, y = next(v_loader)
        try:
            x, y = next(v_loader)
        except StopIteration:
            # 删除迭代器iter_test
            del v_loader
            # 重新生成迭代器iter_test
            v_loader = iter(val_loader)
            x, y = next(v_loader)
        x, y = x.to(args.device), y.to(args.device)
        val_logit_1 = student_net(x=x, flat_param=student_params[-1]).to(args.device)
        ce_loss = criterion(val_logit_1, y).to(args.device)
        writer.add_scalar("ce_loss", ce_loss, epoch)

        optimizer_kd_loss.zero_grad()
        ce_loss.backward()
        optimizer_kd_loss.step()
        lr_scheduler.step()

        with torch.no_grad():
            for x_test, y_test in test_loader:
                x_test, y_test = x_test.to(args.device), y_test.to(args.device)
                out_test = student_net(x_test, flat_param=student_params[-1])

                log["correct_after"] += torch.sum(torch.eq(torch.argmax(out_test, dim=1), y_test))
                log["total_after"] += x_test.size(0)

        print("acc_after: {}".format(log["correct_after"] / log["total_after"]))
        writer.add_histogram('weights_distribution', weight, epoch)
        writer.add_scalar("acc_after - acc_before",
                          (log["correct_after"]/log["total_after"]) - (log["correct_before"]/log["total_before"]),
                          epoch)

    torch.save(kd_loss.state_dict(), 'learnable_kd_loss_cifar10.pt')
    writer.close()
    torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')

    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')

    parser.add_argument('--subset', type=str, default='imagenette', help='ImageNet subset. '
                                                                         'This only does anything '
                                                                         'when --dataset=ImageNet')

    parser.add_argument('--student_model', type=str, default='ConvNet', help='model')

    parser.add_argument('--eval_mode', type=str, default='S',
                        help='eval_mode, check utils.py for more info')

    parser.add_argument('--Epochs', type=int, default=600, help='how many distillation steps to perform')

    parser.add_argument('--T', type=int, default=1, help='KD temperature')
    parser.add_argument('--lr_kd_loss', type=float, default=1e-3, help='learning rate for updating synthetic images')

    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--buffer_path', type=str, default='./buffers', help='buffer path')

    parser.add_argument('--update_student_params_steps', type=int, default=60, help='how many steps to take on val data')
    parser.add_argument('--max_start_epoch', type=int, default=25, help='max epoch we can start at')

    # parser.add_argument('--load_all', action='store_true', help="only use if you can fit "
    #                                                             "all expert trajectories into RAM")

    parser.add_argument('--max_files', type=int, default=None, help='number of expert files to read '
                                                                    '(leave as None unless doing ablations)')
    parser.add_argument('--max_experts', type=int, default=None, help='number of experts to read per file '
                                                                      '(leave as None unless doing ablations)')

    args = parser.parse_args()

    main(args)
