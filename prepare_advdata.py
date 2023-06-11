"""
This implementation is used to create adversarial dataset.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets,models,transforms,utils
from PIL import Image
import os

from deeprobust.image.attack.pgd import PGD
# from deeprobust.image.attack.fgsm import FGSM
# from deeprobust.image.attack.entropy_loss import FGSM
from deeprobust.image.attack.deepfool import DeepFool
from deeprobust.image.attack.onepixel import Onepixel
from deeprobust.image.attack.cw import CarliniWagner
from deeprobust.image.attack.Nattack import NATTACK
from deeprobust.image.attack.lbfgs import LBFGS

import deeprobust.image.netmodels.resnet as resnet
import matplotlib.pyplot as plt

import argparse
from torchvision import transforms
from numpy import linalg as LA
from deeprobust.image.attack.base_attack import BaseAttack
##-------------------------------------------------------##
##---------------------FGSM------------------------------##
##-------------------------------------------------------##


class FGSM(BaseAttack):
    """
    FGSM attack is an one step gradient descent method.

    """
    def __init__(self, model, device = 'cuda'):

        super(FGSM, self).__init__(model, device)

    def generate(self, image, label, **kwargs):

        label = label.type(torch.FloatTensor)
        ## check and parse parameters for attack
        assert self.check_type_device(image, label)
        assert self.parse_params(**kwargs)

        return fgm(self.model,
                   self.image,
                   self.label,
                   self.epsilon,
                   self.order,
                   self.clip_min,
                   self.clip_max,
                   self.device)

    def parse_params(self,
                     epsilon = 0.1,
                     order = np.inf,
                     clip_max = None,
                     clip_min = None):
        
        self.epsilon = epsilon
        self.order = order
        self.clip_max = clip_max
        self.clip_min = clip_min
        return True

#添加全局参数，传进来要添加的扰动大小，一个数据集应该是一样的
def fgm(model, image, label, epsilon, order, clip_min, clip_max, device,):
    imageArray = image.cpu().detach().numpy()
    X_fgsm = torch.tensor(imageArray).to(device)
    X_fgsm.requires_grad = True

    opt = optim.SGD([X_fgsm], lr=1e-3) #loss.backward（）之后模型的参数就更新，让loss变得更小
    opt.zero_grad()
    
    

    # t1 = X_fgsm.softmax(1)
    # t2 = X_fgsm.log_softmax(1)
    # t3 = -(X_fgsm.softmax(1) * X_fgsm.log_softmax(1)).sum(1)
    
    # x = torch.ones(64,64,requires_grad=True).cuda()
    # loss = -(X_fgsm.softmax(1) * X_fgsm.log_softmax(1)).sum(1).mean(0)
    # loss.backward(torch.ones_like(x))
    loss = nn.CrossEntropyLoss()(model(X_fgsm), label)
    loss.backward()
    if order == np.inf:
        d = epsilon * X_fgsm.grad.data.sign()
    #print('d:',d)
    x_adv = X_fgsm - d # 让输入远离决策边界
    #print(x_adv)
    if clip_max == None and clip_min == None:
        clip_max = np.inf
        clip_min = -np.inf

    x_adv = torch.clamp(x_adv,clip_min, clip_max)
    #print(x_adv)
    return x_adv



##-------------------------------------------------------##
##---------------------main------------------------------##
##-------------------------------------------------------##

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--epsilon', type=float, default = 0.03)
    parser.add_argument('--method', type=str, default='fgsm', choices=['fgsm', 'pgd','cw', 'deepfool',
                                                             'l2_attack', 'lbfgs', 'onepixel'])
    args = parser.parse_args()
    if args.method =='fgsm':
        kwargs = {'epsilon': args.epsilon, #  沿着梯度的步长系数
        'order': np.inf,
        'clip_max': None,
        'clip_min': None}

    #Load Model.
    import torchvision.models as models
    model = models.resnet50(pretrained=False, num_classes=200).cuda()
    print("Load network")

    model.load_state_dict(torch.load("/home/majc/DeepRobust/checkpoint/resnet50_tinyimagenet_tune30.pt"))
    model.eval()
   
    # model = models.resnet50(pretrained=True).to('cuda')
    # model.eval()

    test_loader = torch.utils.data.DataLoader(datasets.ImageFolder('/home/majc/data/Tiny-ImageNet-C/zoom_blur/4', transforms.Compose([
                    transforms.Resize(64), transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])), batch_size=1, shuffle=False)
    
    # test_loader = torch.utils.data.DataLoader(datasets.ImageFolder('/home/majc/data/tiny-imagenet-200/val', transforms.Compose([
    #                 transforms.Resize(64), transforms.ToTensor(),
    #                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])), batch_size=1, shuffle=False)
    
    #follow the setup in contrastive autoeval
    # test_loader = torch.utils.data.DataLoader(datasets.ImageFolder('/home/majc/data/tiny-imagenet-200/val',transforms.Compose([
    #                   transforms.RandomResizedCrop(64),transforms.RandomHorizontalFlip(),transforms.ToTensor(),
    #                   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])), batch_size=1, shuffle=False)
    
    normal_data, adv_data = None, None
    adversary = FGSM(model)

    train_acc = 0
    adv_acc = 0
    train_n = 0
    new_data = []
    label = []
    pseudo_label = []
    for x, y in test_loader:
        x,  y= x.cuda(), y.cuda()
        train_n += y.size(0)
        logits = model(x)
        # 实现softmax之后得到confidence
        softmax= torch.nn.Softmax(dim=1)
        output = softmax(logits)

        label_conf = output[:,y].reshape(-1)       # 求label的 confidence
        # prob, pred = output.topk(1, 1, True, True)# top1 confidence 

        # 求模型pred的confidence，比较攻击之前label的confidence上升了吗？？模型预测对了吗（这种攻击方法有没有效果）？
        # 那么有没有效果是怎么评价的？
        ## 1、label的confidece变高了吗？是最高的吗？
        # 2、pred的对了吗？如果不对、有什么区别

        y_pred = logits.argmax(dim = 1, keepdim = True)
        pred_conf = output[:,y_pred].reshape(-1)

        train_acc += y_pred.eq(y.view_as(y_pred)).sum().item() # <fix>
        y_pred = y_pred.reshape(-1)
        #####------------如果label和pred不一致，再尝试攻击-----------######


        x_adv = adversary.generate(x, y, **kwargs)
        print('finish攻击第',train_n,'张图片')

        y_adv = model(x_adv)
        output_adv = softmax(y_adv)
        label_conf_adv = output_adv[:,y].reshape(-1)

        y_adv = y_adv.argmax(dim = 1, keepdim = True)  # <fix> 求模型pred的label和confidence
        pred_conf_adv = output_adv[:,y_adv].reshape(-1)

        adv_acc += y_adv.eq(y.view_as(y_adv)).sum().item() # <fix>
        y_adv = y_adv.reshape(-1)
        print('比较攻击之前label的confidence上升了吗?\n之前label\pred: {},{}\n之后label\pred :{},{}'.format(label_conf.item(),pred_conf.item(),label_conf_adv.item(),pred_conf_adv.item()))
        x, x_adv = x.data, x_adv.data
        # clone tensor to cpu
        y = y.cpu().clone().item()
        y_adv = y_adv.cpu().clone().item()
        image = x_adv.cpu().clone().squeeze(0)

        unloader = transforms.ToPILImage()
        image = unloader(image)
        
        #对齐metaset
        data = np.array(image) 
        new_data.append(data)
        label.append(y)
        pseudo_label.append(y_adv)

        # visualize the adv samples
        #utils.save_image(x,'/home/majc/DeepRobust/data/entropy_0.2_num{}.jpg'.format(train_n),normalize=True)
        #utils.save_image(x_adv,'/home/majc/DeepRobust/data/{}_{}_{}.jpg'.format(args.method, str(kwargs['epsilon']),train_n), normalize=True)
        
        #test the set format
        # if len(label) == 4:
        #     new_data = np.stack(new_data, axis=3)
        #     dirs = '/home/majc/data/metasets/tinyimagenet_adv_entropy/{}'.format(str(kwargs['epsilon']))
        #     if not os.path.exists(dirs):
        #         os.makedirs(dirs)
        #     np.save(os.path.join(dirs,'test_data.npy'), new_data)
        #     np.save(os.path.join(dirs,'test_label.npy'), label)
        #     np.save(os.path.join(dirs,'pseudo_label.npy'), pseudo_label)

   
    print("Accuracy(normal) {:.6f}, Accuracy({}) {:.6f}".format(train_acc / train_n * 100, args.method, adv_acc / train_n * 100))
    # print()
   
    # save as .npy
    new_data = np.stack(new_data, axis=3)
    dirs = '/home/majc/data/fgsm_4autoeval/tinyimagenet_fgsm_4autoeval/{}'.format(str(kwargs['epsilon']))
    if not os.path.exists(dirs):
        os.makedirs(dirs)

    np.save(os.path.join(dirs,'test_data.npy'), new_data)
    np.save(os.path.join(dirs,'test_label.npy'), label)
    # np.save(os.path.join(dirs,'pseudo_label.npy'), pseudo_label)

    # print("Accuracy(normal) {:.6f}, Accuracy({}) {:.6f}".format(train_acc / train_n * 100, args.method, adv_acc / train_n * 100))



