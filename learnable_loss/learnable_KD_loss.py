import torch.nn as nn
import torch.nn.functional as F
import torch
#from lib.models.losses.dist_kd import pearson_correlation

''' Learnable KD Loss '''


def cosine_similarity(a, b, eps=1e-8):
    return (a * b) / (a.norm(dim=1).unsqueeze(1) * b.norm(dim=1).unsqueeze(1) + eps)


def pearson_correlation(a, b, eps=1e-8):
    return cosine_similarity(a - a.mean(1).unsqueeze(1),
                             b - b.mean(1).unsqueeze(1), eps)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., layer_norm=True):
        super().__init__()
        self.layer_norm = layer_norm
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        if self.layer_norm:
            self.norm = nn.LayerNorm(out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        if self.layer_norm:
            x = self.norm(x)
        x = self.drop(x)
        return x


class LearnableKDLoss(nn.Module):
    def __init__(self, num_classes, dim=512, act_layer=nn.GELU, drop=0.):
        super(LearnableKDLoss, self).__init__()
        self.num_classes = num_classes

        # Additional layer to output the scalar temperature parameter
        self.T_layer = Mlp(in_features=dim,
                           hidden_features=dim * 4,
                           out_features=dim,
                           act_layer=act_layer,
                           drop=drop)

        # Deeper and wider network for x_t and x_s
        self.proj = nn.Linear(self.num_classes, dim)
        self.mlp_pre1 = Mlp(in_features=dim,
                            hidden_features=dim * 4,
                            out_features=dim,
                            act_layer=act_layer,
                            drop=drop)

        # Deep and wide network for concatenated outputs
        self.proj2 = nn.Linear(dim*2, dim)
        self.mlp1 = Mlp(in_features=dim,  # x_t and x_s concatenated
                        hidden_features=dim * 4,
                        out_features=dim,
                        act_layer=act_layer,
                        drop=drop)

        self.fc_T = nn.Linear(dim, 1)
        self.fc_out = nn.Linear(dim, self.num_classes)

        self.cls_embed = nn.Embedding(num_classes, dim)

    def forward(self, x_s_logit, x_t_logit, label):

        x_t = self.proj(x_t_logit)
        x_t = self.mlp_pre1(x_t)

        x_s = self.proj(x_s_logit)
        x_s = self.mlp_pre1(x_s)

        # Concatenate the outputs and pass through the deep network
        out = torch.cat((x_t, x_s), 1)
        out = self.proj2(out)
 
        # add cls embedding
        out = out + self.cls_embed(label)

        # get learnable T
        T = self.T_layer(out)
        T = self.fc_T(T)
        T = T.sigmoid() * 8

        # out = self.mlp1(out)
        # weight = torch.sigmoid(self.fc_out(out))

        # p_t = F.softmax(x_t_logit / T, dim=-1)
        # p_s = F.softmax(x_s_logit / T, dim=-1)
        # kl_div = p_t * torch.log(p_t / p_s + 1e-5) * T ** 2

        p_s = F.log_softmax(x_s_logit / T, dim=-1)
        p_t = F.softmax(x_t_logit / T, dim=-1)
        kl_div = F.kl_div(p_s, p_t, size_average=False) * (T ** 2) / x_s_logit.shape[0]

        #kl_div = 1 - pearson_correlation(p_t, p_s) * T ** 2

        # loss = (kl_div * weight).sum(-1).mean(0)
        loss = kl_div.sum(-1).mean(0)

        # return loss, weight, T
        return loss, T

