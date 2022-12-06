import argparse
import os.path

from model.train_model import SER_FIQ
import torch
from torch import nn
from torch.optim import Adam,SGD
import torch.nn.functional as F
from torch.optim.lr_scheduler import  CosineAnnealingWarmRestarts,CosineAnnealingLR,StepLR
from dataset.Cus_DataLoader import Face_Quality_Dataset,data_transforms
from torch.utils.data import DataLoader
import time
from utils.data_augs import random_rotation,random_horflip,random_verflip

def parse_args():
    parser = argparse.ArgumentParser(
        description='Pytorch SER_FIQ train a model')
    # parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', default='wout', help='output folder')
    parser.add_argument('--save-preds', action='store_true', help='save results')
    parser.add_argument('--debug', action='store_true', help='debug flag')
    parser.add_argument('--batch_size', default=32, help='set batch size ')
    parser.add_argument('--val_seq',default=1,help='frequency of validating model')
    parser.add_argument('--save_seq',default=2,help='frequency of save model')
    parser.add_argument('--device',default='cpu',help='train model cpu / cuda')

    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--show-score-thr',
        type=float,
        default=0.3,
        help='score threshold (default: 0.3)')
    parser.add_argument(
        '--thr',
        type=float,
        default=0.02,
        help='score threshold')
    parser.add_argument('--mode', type=int, default=0)
    args = parser.parse_args()

    return args

def save_chcekpoint(model,epoch,optimizer,scheduler):
    out_state = {
        'model':model,
        "net":model.state_dict(),
        'epoch':epoch,
        'optimizer':optimizer,
        "scheduler":scheduler,
    }
    now = int(round(time.time() * 1000))
    save_name = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(now / 1000))
    if not os.path.exists('./checkpoints'):
        os.makedirs('./checkpoints')
    torch.save(out_state,'checkpoints/' + save_name + ".pth")




"使用DRQ的思想对数据进行增强，用于训练backbone"
def drq_aug(model,aug_net,data):
    b,c,w,h = data.shape
    data_aug1 = random_rotation(data,p=1)
    data_aug2 = random_horflip(data,p=1)
    data_aug3 = random_verflip(data, p=1)
    net_data = torch.cat([data,
                          data_aug1,
                          data_aug2,
                          data_aug3],dim=0)

    feature = model.backbone(net_data)
    emb_out = aug_net(feature)
    loss = 0
    count = 0
    for i in range(4):
        for j in range(i+1,4):
            loss += F.smooth_l1_loss(emb_out[(b * i):(b*(i+1))],
                                     emb_out[(b * j):(b*(j+1))])
            count += 1
    return loss / 4.0


def val(model):
    pass


if __name__ == '__main__':
    args = parse_args()
    model = SER_FIQ()
    # 用于增强训练，提高backbone的特征提取能力
    Augment_Net = nn.Sequential(
        nn.Linear(2304, 512)
    )
    MAX_EPOCH = 25
    optimizer = Adam(model.parameters(),lr=0.01)
    scheduler = CosineAnnealingLR(optimizer,T_max=MAX_EPOCH)

    train_dataset = Face_Quality_Dataset(root_dir=r'D:\Code\face_quality_assessment\face_quality\asian_celeb\ASIAN_FQ_DATASET',
                                         ann_file=r'D:\Code\face_quality_assessment\face_quality\asian_celeb\labels.txt',
                                         transform=data_transforms["train"])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,shuffle=True)
    val_loader = DataLoader(train_dataset, batch_size=args.batch_size,shuffle=False)
    epoch = 0
    for epoch in range(MAX_EPOCH):
        for img,label in train_loader:
            img = img.to(args.device)
            drq_loss = drq_aug(model,aug_net=Augment_Net,data=img)
            dis_loss = model(img)

            loss = dis_loss *
        #     print(img.shape)
        # if epoch != 0 and epoch % args.val_seq == 0:
        #     val(model,val_loader)

        if epoch % args.save_seq == 0:
            save_chcekpoint(model,epoch,optimizer,scheduler)

    save_chcekpoint(model,epoch,optimizer,scheduler)
