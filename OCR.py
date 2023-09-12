import torch.nn as nn
import torch
import cv2
import numpy as np
from torchvision.models.vgg import vgg16_bn
device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
class OCRmodel(nn.Module):
  def __init__(self,ks=3,st=1,pad=1,pool=2, drop=0.2,vocab_len=70):
    super().__init__()
    self.vocap=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's',
                't', 'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L',
                'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '.', '-', "'",
                '1', '9', '5', '8', '3', '4', '0', ',', '2', '7', '6', '/', '*', '?', '"']
    self.code_char = {i + 1: x for i, x in enumerate(self.vocap)}
    self.code_char.update({0: '-'})
    # self.model=nn.Sequential(
    #     nn.Conv2d(1, 128, kernel_size=ks, stride=st, padding=pad),
    #     nn.BatchNorm2d(128, momentum=0.3),
    #     nn.ReLU(inplace=True),
    #     nn.MaxPool2d(pool),
    #     nn.Dropout2d(drop),
    #     nn.Conv2d(128, 128, kernel_size=ks, stride=st, padding=pad),
    #     nn.BatchNorm2d(128, momentum=0.3),
    #     nn.ReLU(inplace=True),
    #     nn.MaxPool2d(pool),
    #     nn.Dropout2d(drop),
    #     nn.Conv2d(128, 256, kernel_size=ks, stride=st, padding=pad),
    #     nn.BatchNorm2d(256, momentum=0.3),
    #     nn.ReLU(inplace=True),
    #     nn.MaxPool2d((4,2)),
    #     nn.Dropout2d(drop)
    # )
    self.model = vgg16_bn(pretrained=True).features.to(device)
    self.model[0] = nn.Conv2d(1, 64, kernel_size=ks, stride=st, padding=pad)
    self.model[-1] = nn.Sequential(nn.Conv2d(512, 256, kernel_size=ks, stride=st, padding=pad),
                                   nn.Upsample(scale_factor=(1, 2), mode='nearest'),
                                   nn.BatchNorm2d(256, momentum=0.3),
                                   nn.ReLU(inplace=True))
    self.rnn=nn.Sequential(
        nn.LSTM(256, 256, num_layers=2, dropout=0.2, bidirectional=True)
    )
    self.classification = nn.Sequential(
        nn.Linear(512, vocab_len+1),
        nn.LogSoftmax(-1),
    )
  def forward(self,x):
    x=self.model(x)
    x=x.reshape(-1, 256, 32)
    x=x.permute(2,0,1)
    x,ls=self.rnn(x)
    x=self.classification(x)
    return x

  def img_reshape(self,img):

      shape = (32, 128)
      target = np.ones(shape) * 255
      H, W = shape
      h, w = img.shape
      fx = H / h
      fy = W / w
      f = min(fx, fy)
      _h = int(h * fx)
      _w = int(w * fy)
      _img = cv2.resize(img, (_w, _h),interpolation=cv2.INTER_LANCZOS4)
      target[:_h, :_w] = _img
      return target
  def decode(self, pred):
      decoded = ""
      last = ""
      pred = pred.cpu().detach().numpy()
      for i in range(len(pred)):
          k = np.argmax(pred[i])
          if k > 0 and self.code_char[k] != last:
              last = self.code_char[k]
              decoded = decoded + last
          elif k > 0 and self.code_char[k] == last:
              continue
          else:
              last = ""
      return decoded.replace(" ","")

  @torch.no_grad()
  def predict(self,img):
      self.eval()
      image=img
      try:
          row_indices, col_indices = np.where(image < 100)
          image=image[max(0,min(row_indices)-1):min(max(row_indices)+1,32),min(col_indices):]
          image=self.img_reshape(image)
      except:
          pass

      if np.min(image)==255:
          return ''
      preds=self.forward(torch.tensor(image).float().to(device)[None,None])[:,0,:]
      decodded_str=self.decode(preds)
      return decodded_str
