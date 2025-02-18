import torch.nn as nn
import torch.nn.functional as F
import torch

''' Learnable Feature Loss '''


class ChannelExpander(nn.Module):
    def __init__(self, in_channels, out_channels=None, act_layer=nn.GELU, layer_norm=True):
        super().__init__()
        self.layer_norm = layer_norm
        out_channels = out_channels or in_channels
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.act = act_layer()
        if self.layer_norm:
            self.norm = nn.LayerNorm(out_channels)

    def forward(self, x):
        x = self.conv1x1(x)
        x = self.act(x)
        if self.layer_norm:
            x = self.norm(x)
        return x


class ExtraNetwork(nn.Module):
    def __init__(self, in_features, out_features, hidden_features=None, act_layer=nn.ReLU, dropout=0.0):
        super(ExtraNetwork, self).__init__()
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act1 = act_layer()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = x.mean([2, 3])  # Global average pooling or use a different pooling strategy
        x = self.fc1(x)
        x = self.act1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        return x


class HintLoss(nn.Module):
    def __init__(self):
        super(HintLoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, student_hint, teacher_hint):
        # Calculate MSE loss between student and teacher hints
        loss = self.mse_loss(student_hint, teacher_hint)
        return loss


class LearnableHintLoss(nn.Module):
    def __init__(self, dim=512, hidden_dims=(256, 128)):
        super(LearnableHintLoss, self).__init__()

        # Convolutional blocks for student hint
        self.conv_student = nn.ModuleList([
            nn.Conv2d(dim, hidden_dims[i], kernel_size=3, padding=1) for i in range(len(hidden_dims))
        ])

        # Convolutional blocks for teacher hint
        self.conv_teacher = nn.ModuleList([
            nn.Conv2d(dim, hidden_dims[i], kernel_size=3, padding=1) for i in range(len(hidden_dims))
        ])

        # Convolutional block for combining hints
        self.conv_combine1 = nn.Conv2d(hidden_dims[-1] * 2, dim, kernel_size=3, padding=1)
        self.conv_combine2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)

    def forward(self, student_hint, teacher_hint):
        student_conv = student_hint.clone()
        teacher_conv = teacher_hint.clone()
        # Apply convolutional blocks to student and teacher hints
        for conv_student, conv_teacher in zip(self.conv_student, self.conv_teacher):
            student_conv = F.relu(conv_student(student_conv))
            teacher_conv = F.relu(conv_teacher(teacher_conv))

        # Concatenate the processed hints
        combined_conv = torch.cat((student_conv, teacher_conv), dim=1)

        # Apply convolutional blocks for combining hints
        combined_conv = F.relu(self.conv_combine1(combined_conv))
        # combined_conv = F.relu(self.conv_combine2(combined_conv))
        combined_conv = self.conv_combine2(combined_conv)

        # Calculate learnable weight using sigmoid activation
        weight = torch.sigmoid(combined_conv)

        # Scale student and teacher hints using the learnable weight
        # weighted_student_hint = weight * student_hint
        # weighted_teacher_hint = weight * teacher_hint

        # Calculate MSE loss between weighted hints
        # loss = F.mse_loss(weighted_student_hint, weighted_teacher_hint)
        loss = F.mse_loss(student_hint, teacher_hint, reduction='none')
        loss = (weight * loss).mean()

        return loss, weight
