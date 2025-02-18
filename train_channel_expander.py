import argparse
import yaml
import torch
from torch import nn
from learmable_hint_loss import ChannelExpander, ExtraNetwork
from lib.models import nas_model
from train_teacher_net import get_dataset
from utils import get_time

# Load YAML file
with open('configs/models/VGG/vgg16_half_cifar10.yaml', 'r') as file:
    config = yaml.safe_load(file)


def main(args):
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    _, _, num_classes, train_loader, test_loader = get_dataset(dataset=args.dataset,
                                                               data_path=args.data_path,
                                                               subset=args.subset,
                                                               args=args)
    print('Hyper-parameters: \n', parser.parse_args())

    ''' training '''
    # student_model with pretrained weights
    student_model = nas_model.gen_nas_model(config).to(args.device)
    checkpoint = torch.load('experiments/vgg16_half_cifar10_exp/best.pth.tar')
    student_model.load_state_dict(checkpoint['model'])
    student_model.eval()  # Set the model to evaluation mode, as you don't want to update its parameters

    partial_student_model = nn.Sequential(
        student_model.features.conv0,
        student_model.features.conv1,
        student_model.features.pool1,
        student_model.features.conv2,
        student_model.features.conv3,
        student_model.features.pool2,
        student_model.features.conv4,
        student_model.features.conv5,
        student_model.features.conv6,
        student_model.features.pool3,
        student_model.features.conv7,
        student_model.features.conv8,
        student_model.features.conv9,
        student_model.features.pool4,
        student_model.features.conv10,
        student_model.features.conv11,
        student_model.features.conv12,
    )

    # 冻结这一部分的参数，使其不可训练
    for param in partial_student_model.parameters():
        param.requires_grad = False

    # Assuming extra_network is an additional network to be appended to ChannelExpander
    extra_network = ExtraNetwork(in_features=512, out_features=num_classes, hidden_features=None)

    # Initialize ChannelExpander with the desired parameters
    channel_expander = ChannelExpander(in_channels=256, out_channels=512)

    # Combine student_model, channel_expander, and extra_network into a single model
    pretrain_model = nn.Sequential(partial_student_model, channel_expander, extra_network)

    # 打印模型结构
    print(pretrain_model)

    # Define optimizer for the combined model (excluding student_model parameters)
    pretrain_optimizer = torch.optim.SGD(params=pretrain_model.parameters(),
                                         lr=args.pretrain_lr,
                                         momentum=args.pretrain_momentum,
                                         weight_decay=args.pretrain_weight_decay)
    pretrain_loss_criterion = nn.CrossEntropyLoss().to(args.device)

    for epoch in range(args.pretrain_epochs):
        log = {"correct": 0.0, "total": 0.0, "loss": 0.0}

        for x_train, y_train in train_loader:
            x_train, y_train = x_train.to(args.device), y_train.to(args.device)

            # Forward pass through student_model, channel_expander, and extra_network
            output = pretrain_model(x_train)

            # Assuming you have a pretrain_loss_criterion for the combined model
            loss = pretrain_loss_criterion(output, y_train)

            pretrain_optimizer.zero_grad()
            loss.backward()
            pretrain_optimizer.step()

            # Update the loss value
            log["loss"] += loss.item()

        # Evaluation on the validation set
        pretrain_model.eval()
        with torch.no_grad():
            for x_val, y_val in test_loader:
                x_val, y_val = x_val.to(args.device), y_val.to(args.device)
                out_val = pretrain_model(x_val)
                log["correct"] += torch.sum(torch.eq(torch.argmax(out_val, dim=1), y_val))
                log["total"] += x_val.size(0)

        accuracy = log["correct"] / log["total"] * 100
        print('%s epoch = %04d' % (get_time(), epoch))
        print("loss: {}, acc: {}%".format(log["loss"], accuracy))

    # Separate the trained ChannelExpander from the combined model
    trained_channel_expander = pretrain_model[1]
    torch.save(trained_channel_expander.state_dict(), 'channel_expander_vgg16_cifar10.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--subset', type=str, default='imagenette', help='ImageNet subset. '
                                                                         'This only does anything '
                                                                         'when --dataset=ImageNet')
    parser.add_argument('--pretrain_epochs', type=int, default=200, help='how many distillation steps to perform')
    parser.add_argument('--pretrain_lr', type=float, default=1e-3, help='learning rate for updating synthetic images')
    parser.add_argument('--pretrain_momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--wd', '--pretrain-weight-decay', default=1e-4, type=float, metavar='W',
                        help='weight decay (default: 1e-4)', dest='pretrain_weight_decay')
    args = parser.parse_args()

    main(args)