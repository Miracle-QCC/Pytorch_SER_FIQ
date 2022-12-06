from .backbone import MobileNetV3_Small
import torch
from torch import nn
from torch.optim import Adam
import torch.nn.functional as F
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

"""
    使用pytorch实现SER_FIQ，backbone改为Mobilenetv3_Small

"""


class SER_FIQ(nn.Module):
    def __init__(self, ):
        """
            采用10个子网络转换embedding,两两计算欧式距离，用于评估人脸质量
        """
        super(SER_FIQ, self).__init__()

        self.backbone = MobileNetV3_Small()  # 1 * 3 * 112 * 112 -> 1 * 2304
        self.fc_0 = nn.Linear(2304, 512)
        self.fc_1 = nn.Linear(2304, 512)
        self.fc_2 = nn.Linear(2304, 512)
        self.fc_3 = nn.Linear(2304, 512)
        self.fc_4 = nn.Linear(2304, 512)
        self.fc_5 = nn.Linear(2304, 512)
        self.fc_6 = nn.Linear(2304, 512)
        self.fc_7 = nn.Linear(2304, 512)
        self.fc_8 = nn.Linear(2304, 512)
        self.fc_9 = nn.Linear(2304, 512)

        self.dropout = nn.Dropout(p=0.8)
        self._initialize_weights()

    def _initialize_weights(self):
        '''
        This method is to initialize model weights
        '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def euclidean_dist_loss(self, x):
        b, m, c = x.shape
        out = torch.zeros(b, 1)
        count = 0
        for i in range(m):
            for j in range(i + 1, m):
                out += F.smooth_l1_loss(x[:,i,:],x[:,j,:])
                count += 1

        out /= count
        return out

    def forward(self, x, weight):
        feature1 = self.backbone(x)
        # feature100 = feature1.repeat(1,100,1)
        emb_0 = self.fc_0(self.dropout(feature1))
        emb_1 = self.fc_0(self.dropout(feature1))
        emb_2 = self.fc_0(self.dropout(feature1))
        emb_3 = self.fc_0(self.dropout(feature1))
        emb_4 = self.fc_0(self.dropout(feature1))
        emb_5 = self.fc_0(self.dropout(feature1))
        emb_6 = self.fc_0(self.dropout(feature1))
        emb_7 = self.fc_0(self.dropout(feature1))
        emb_8 = self.fc_0(self.dropout(feature1))
        emb_9 = self.fc_0(self.dropout(feature1))

        # embs = torch.stack([emb_0, emb_1, emb_2,
        #                        emb_3, emb_4, emb_5,
        #                        emb_6, emb_7, emb_8,
        #                        emb_9], dim=1)

        #normalize
        emb_0 = emb_0 / torch.norm(emb_0, p = 2)
        emb_1 = emb_1 / torch.norm(emb_1, p = 2)
        emb_2 = emb_2 / torch.norm(emb_2, p = 2)
        emb_3 = emb_3 / torch.norm(emb_3, p = 2)
        emb_4 = emb_4 / torch.norm(emb_4, p = 2)
        emb_5 = emb_5 / torch.norm(emb_5, p = 2)
        emb_6 = emb_6 / torch.norm(emb_6, p = 2)
        emb_7 = emb_7 / torch.norm(emb_7, p = 2)
        emb_8 = emb_8 / torch.norm(emb_8, p = 2)
        emb_9 = emb_9 / torch.norm(emb_9, p = 2)

        # norm_out = nn.functional.normalize(embs, p=2, dim=1, eps=1e-12, out=None)

        out_emb = torch.stack([emb_0, emb_1, emb_2,
                               emb_3, emb_4, emb_5,
                               emb_6, emb_7, emb_8,
                               emb_9], dim=1)

        # norm_out = torch.nn.functional.normalize(out, p=1,dim=1)
        # distance_emb = self.euclidean_dist(norm)
        # score = 2 * (1 / (1 + torch.exp(distance)))
        # cal distance
        ecul_dis = self.euclidean_dist_loss(out_emb, weight)
        # score = 2 * F.sigmoid(-ecul_dis-1)
        return ecul_dis

if __name__ == '__main__':
    model = SER_FIQ()
    optimizer = Adam(model.parameters(), lr=1e-3)
    model.eval()
    model.dropout.train()
    input = torch.rand(1, 3, 112, 112)
    model(input)
