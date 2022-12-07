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
MAX_EPOCH = 50
def parse_args():
    parser = argparse.ArgumentParser(
        description='Pytorch SER_FIQ train a model')
    parser.add_argument('root_dir', help='img data path')
    parser.add_argument('ann_file', help='annotation txt path')
    parser.add_argument('--debug', action='store_true', help='debug flag')
    parser.add_argument('--batch_size', default=32, help='set batch size ')
    parser.add_argument('--val_seq',default=1,help='frequency of validating model')
    parser.add_argument('--save_seq',default=2,help='frequency of save model')
    parser.add_argument('--device',default='cpu',help='train model cpu / cuda')

    args = parser.parse_args()
    return args

def save_checkpoint(model,epoch,optimizer,scheduler):
    out_state = {
        'model':model,
        "net":model.state_dict(),
        'epoch':epoch,
        'optimizer':optimizer.state_dict(),
        "scheduler":scheduler.state_dict(),
    }
    now = int(round(time.time() * 1000))
    save_name = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(now / 1000))
    if not os.path.exists('./checkpoints'):
        os.makedirs('./checkpoints')
    torch.save(out_state,'checkpoints/' + save_name + ".pth")

def load_checkpoint(model_path):
    model_dict = torch.load(model_path)
    model = model_dict["model"]
    model.load_state_dict(model_dict["net"])
    optimizer = Adam(model.parameters(),lr=0.01)
    optimizer.load_state_dict(model_dict["optimizer"])
    scheduler = CosineAnnealingLR(optimizer,T_max=MAX_EPOCH)
    scheduler.load_state_dict(model_dict["scheduler"])
    return model,optimizer,scheduler




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
    optimizer = Adam(model.parameters(),lr=0.01)
    scheduler = CosineAnnealingLR(optimizer,T_max=MAX_EPOCH)
    #
    train_dataset = Face_Quality_Dataset(root_dir=args.root_dir,
                                         ann_file=args.ann_file,
                                         transform=data_transforms["train"])
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,shuffle=True)
    # val_loader = DataLoader(train_dataset, batch_size=args.batch_size,shuffle=False)
    epoch = 0
    for epoch in range(MAX_EPOCH):
        for img,label in train_loader:
            img = img.to(args.device)
            drq_loss = drq_aug(model,aug_net=Augment_Net,data=img)
            dis_loss = model(img)
    #
            loss = dis_loss + drq_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            now = int(round(time.time() * 1000))
            time_name = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime(now / 1000))
            print(time_name + " total loss:",loss, "  drq loss: ",drq_loss,"  dis loss:",dis_loss)

    #     #     print(img.shape)
    #     # if epoch != 0 and epoch % args.val_seq == 0:
    #     #     val(model,val_loader)
    #
        if epoch % args.save_seq == 0:
            save_checkpoint(model,epoch,optimizer,scheduler)

    # save_checkpoint(model,epoch,optimizer,scheduler)
