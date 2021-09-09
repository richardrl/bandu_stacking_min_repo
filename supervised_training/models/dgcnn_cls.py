import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from .utils.dgcnn_util import get_graph_feature


class DGCNNCls(nn.Module):
    def __init__(self,
                 n_knn=20,
                 num_class=40,
                 normal_channel=False,
                 num_channels=3,
                 use_dropout=True,
                 use_xcoord_first=False):
        """

        DGCNN with Classification Head

        :param n_knn:
        :param num_class: Output_dim. For producing quaternions, this would be set to 4
        :param normal_channel:
        :param num_channels: Num input channels
        :param use_dropout:
        """
        super().__init__()

        self.use_xcoord_first = use_xcoord_first

        self.n_knn = n_knn

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(1024)

        self.conv1 = nn.Sequential(nn.Conv2d(num_channels*2, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, 1024, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(1024*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.linear3 = nn.Linear(256, num_class)
        self.use_dropout = use_dropout

        if use_dropout:
            self.dp1 = nn.Dropout(p=0.5)
            self.dp2 = nn.Dropout(p=0.5)

    def forward(self, x):
        # -> nB, 4, num_points
        batch_size = x.size(0)

        # move to own device
        model_device = next(self.parameters()).device

        # it seems we need .float() because the dataloader makes it a double
        x = x.to(model_device).float()

        # -> nB, num_points, num_neighbors, num_dims * 2
        x = get_graph_feature(x, x_coord=x[:, :3, :] if self.use_xcoord_first else None, k=self.n_knn, device=model_device)

        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.n_knn, device=model_device)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.n_knn, device=model_device)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.n_knn, device=model_device)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)

        if self.use_dropout:
            x = self.dp1(x)

        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)

        if self.use_dropout:
            x = self.dp2(x)

        x = self.linear3(x)

        trans_feat = None
        # return x, trans_feat
        return x