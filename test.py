
import cv2
import torch
from torch.nn import functional as F

img = torch.rand((2,2))
zero = torch.zeros((2,2))
print(img)

print(F.l1_loss(img,zero))



