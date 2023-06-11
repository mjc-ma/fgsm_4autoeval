import argparse
import torch
import torch.backends.cudnn as cudnn
from torchvision import models
from data_aug.contrastive_learning_dataset import ContrastiveLearningDataset
from models.densenet_simclr import DenseNetSimCLR  # <fix>
from models.lenet_simclr import LeNetSimCLR  # <fix>
from models.resnet_simclr import ResNetSimCLR
from models.vgg_simclr import VggSimCLR  # <fix>
from utils import set_seed_torch  # <fix>
from simclr import SimCLR
import os  # <fix>
from tqdm import trange
import numpy as np
from scipy import stats
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

######### Refer to SimCLR-master ########
model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name])) + ["lenet", "densenet40-12"]  # <fix>

parser = argparse.ArgumentParser(description='PyTorch SimCLR')
parser.add_argument('--data', metavar='DIR', default='/home/majc/data/',
                    help='path to dataset')
parser.add_argument('--test-dataset-name', default='mnist',
                    help='Unseen target dataset name', choices=['svhn', 'usps', 'cifar10_1', 'cifar10_c',
                                                                'cifar100', 'cifar100_c', 'caltech', 'pascal',
                                                                'imagenet', 'tinyimagenet_c'])  # <fix>
parser.add_argument('--val-dataset-name', default='tinyimagenet_val',
                    help='validation dataset name', choices=['mnist', 'mnist_raw','fashion_mnist', 'k_mnist',
                                                             'cifar10', 'cifar100', 'stl10', 'coco', 'tinyimagenet','tinyimagenet_val'])  # <fix>                                                                
parser.add_argument('--meta-dataset-name', default='tinyimagenet',
                    help='Meta-set dataset name',
                    choices=['mnist', 'cifar10', 'cifar100', 'coco', 'tinyimagenet'])  # <fix>
parser.add_argument('--metaset-dir', metavar='DIR', default='/home/majc/data/metasets/TinyImageNet200/dataset_default/',
                    help='path to save the generated meta-set')
parser.add_argument('--metaset-numLim', default=2000, type=int, metavar='N',
                    help='the range of selected meta-set')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet50)')
parser.add_argument('--pretrained', action='store_true', default=False, help='Use the pretrained cnn')
parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('-test_b', '--test-batch-size', default=36, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('-meta_b', '--meta-batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--seed', default=0, type=int,
                    help='seed for initializing training.')  # <fix>
parser.add_argument('--disable-cuda', action='store_true',
                    help='Disable CUDA')
parser.add_argument('--fp16-precision', action='store_true',
                    help='Whether or not to use 16-bit precision GPU training.')

parser.add_argument('--out-dim', default=1024, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--log-every-n-steps', default=100, type=int,
                    help='Log every n steps')
parser.add_argument('--temperature', default=0.07, type=float,
                    help='softmax temperature (default: 0.07)')
parser.add_argument('--n-views', default=2, type=int, metavar='N',
                    help='Number of views for contrastive learning training.')
parser.add_argument('--gpu-index', default=7, type=int, help='Gpu index.')

######## Refer to Accuracy-estimation-with-self-supervision-main #########
parser.add_argument('--start-epoch', default=0, type=int,
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--layers', default=40, type=int,
                    help='total number of DenseNet layers (default: 40)')
parser.add_argument('--growth', default=12, type=int,
                    help='number of new channels per layer (default: 12)')
parser.add_argument('--droprate', default=0, type=float,
                    help='dropout probability (default: 0.0)')
parser.add_argument('--reduce', default=1.0, type=float,
                    help='compression rate in transition stage (default: 1.0)')
parser.add_argument('--no-bottleneck', dest='bottleneck', default=True, action='store_false',
                    help='To not use bottleneck block')
parser.add_argument('--no-augment', dest='augment', default=True, action='store_false',
                    help='whether to use standard augmentation (default: True)')
parser.set_defaults(bottleneck=True)
parser.set_defaults(augment=True)

# fmt: on
parser.add_argument('--num-classes', default=200, type=int,
                    help='total number of classes (default: 10)')
parser.add_argument('--save-dir', metavar='DIR', default='/home/majc/Contrastive_AutoEval-1/checkpoints/TinyImageNet/metaset200',
                    help='path to save checkpoints')
parser.add_argument('--restore-file', default='/home/majc/Contrastive_AutoEval-1/checkpoints/checkpoint_best.pth',
                    help='filename from which to load checkpoint '
                         '(default: <save-dir>/checkpoint_0005.pth')  # <fix>, the storage of loaded pre-trained model
parser.add_argument('--cl-model', default='SimCLR', help='the name of contrastive learning framework',
                    choices=['SimCLR', 'MoCo_V1', 'MoCo_V2', 'BYOL'])  # <fix>
parser.add_argument('--brightness', default=0.8, type=float, help='brightness value in ColorJitter')
parser.add_argument('--contrast', default=0.8, type=float, help='contrast value in ColorJitter')
parser.add_argument('--saturation', default=0.8, type=float, help='saturation value in ColorJitter')
parser.add_argument('--hue', default=0.2, type=float, help='hue value in ColorJitter')
parser.add_argument('--ResizedCropScale', default='(0.08, 1)', metavar='B',
                    help='the scale for transforms.RandomResizedCrop')
parser.add_argument('--data-setup', default='tinyimagenet1', help='the processed data setup now',
                    choices=['none', 'mnist', 'mnist1', 'cifar', 'cifar1',
                             'coco', 'coco1', 'coco2', 'tinyimagenet',
                             'tinyimagenet1'])  # <fix>
parser.add_argument('--claLoss-weight', default=1., type=float, metavar='D',
                    help='the weight for the classification loss')
parser.add_argument('--conLoss-weight', default=0.01, type=float, metavar='D',
                    help='the weight for the contrstive loss')


def load_model(args):
    # <fix>, For MNIST
    if args.arch.startswith("lenet"):
        model = LeNetSimCLR(args.num_classes, args.out_dim)

    # <fix>, For CIFAR10, num_classes: 10, Other Backbones: ResNet-18, ResNet-34, VGG-11, VGG-19; For CIFAR100, num_classes: 10
    elif args.arch.startswith("densenet"):
        model = DenseNetSimCLR(args.layers, args.num_classes, args.growth, out_dim=args.out_dim,
                               reduction=args.reduce, bottleneck=args.bottleneck, dropRate=args.droprate)

    elif args.arch.startswith("vgg"):
        model = VggSimCLR(base_model=args.arch, pretrained=args.pretrained, num_classes=args.num_classes,
                          out_dim=args.out_dim)

    # <fix>, For COCO
    elif args.arch.startswith("resnet"):
        model = ResNetSimCLR(base_model=args.arch, pretrained=args.pretrained, num_classes=args.num_classes,
                             out_dim=args.out_dim)

    return model


def main():
    args = parser.parse_args()
    assert args.n_views == 2, "Only two view training is supported. Please use --n-views 2."
    # check if gpu training is available
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1
    # set_seed_torch(args.seed)   # <fix>, fixed random seeds

    model = load_model(args)
    # get the number of model parameters
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))

    if args.restore_file:
        if os.path.isfile(args.restore_file):
            print("=> loading checkpoint '{}'".format(args.restore_file))
            checkpoint = torch.load(args.restore_file)
            args.start_epoch = checkpoint['epoch']
            cla_acc = checkpoint['cla_acc']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.restore_file, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.restore_file))

    #  It’s a no-op if the 'gpu_index' argument is a negative integer or None.
    simclr = SimCLR(model=model, optimizer=None, scheduler=None, dataset_name=args.test_dataset_name, args=args)

    epoch = checkpoint['epoch']
    print(f'\nEpoch:{epoch}')

    # use to two acc of sample sets to fit the regression model
    dataset = ContrastiveLearningDataset(args)  # <fix>, use the cl-model type to choose the transformation type

    con_acc = []
    cla_acc = []
    max_mean_conf = []
    thre_acc = []
    con_val = 0
    cla_val = 0
    threshold = 0
    
    ###########compute train max_mean——conf############### 
    with torch.no_grad():
        val_dataset = dataset.get_val_dataset(args.val_dataset_name,
                                                args.n_views,
                                                args.data_setup, args.augment,
                                                train_trans=False)  # <fix>, whether or not to transform the original image
        val_loader = torch.utils.data.DataLoader(
            dataset=val_dataset, batch_size=args.meta_batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True)

        con_val, cla_val, threshold ,thre_val = simclr.test(val_loader)


        f_list = os.listdir(args.metaset_dir)
        f_list.sort()
        if args.metaset_numLim >= len(f_list):
            assert "Out of list index"

        for i in trange(0, len(f_list)):  # <fix>
            meta_dataset = dataset.get_meta_dataset(args.metaset_dir + "/" + f_list[i], args.meta_dataset_name,
                                                    args.n_views,
                                                    args.data_setup, args.augment,
                                                    train_trans=False)  # <fix>, whether or not to transform the original image
            metaset_loader = torch.utils.data.DataLoader(
                dataset=meta_dataset, batch_size=args.meta_batch_size, shuffle=True,
                num_workers=args.workers, pin_memory=True)

            con_top1, cla_top1, conf, thre_ = simclr.test(metaset_loader,threshold)
            con_acc.append(con_top1)
            cla_acc.append(cla_top1)
            max_mean_conf.append(conf)
            thre_acc.append(thre_)


    con_acc = np.array(con_acc)
    cla_acc = np.array(cla_acc)
    max_mean_conf = np.array(max_mean_conf)
    thre_acc = np.array(thre_acc)

    #plot these pair in one pic
    plt.figure(figsize=(10,8),dpi=100) #设置画布大小，像素
    plt.scatter(con_acc,cla_acc,label='con_acc vs cla_acc') #画散点图并指定图片标签
    plt.legend() #显示图片中的标签
    plt.savefig('/home/majc/Contrastive_AutoEval-1/checkpoints/TinyImageNet/metaset200/scatter1.jpg')#保存图片

    plt.figure(figsize=(10,8),dpi=100) #设置画布大小，像素
    plt.scatter(max_mean_conf,cla_acc,label='max_mean_conf vs cla_acc') #画散点图并指定图片标签
    plt.legend() #显示图片中的标签
    plt.savefig('/home/majc/Contrastive_AutoEval-1/checkpoints/TinyImageNet/metaset200/scatter2.jpg')#保存图片

    plt.figure(figsize=(10,8),dpi=100) #设置画布大小，像素
    plt.scatter(thre_acc,cla_acc,label='ATC vs cla_acc') #画散点图并指定图片标签
    plt.legend() #显示图片中的标签
    plt.savefig('/home/majc/Contrastive_AutoEval-1/checkpoints/TinyImageNet/metaset200/scatter3.jpg')#保存图片


    # filter outliers
    # index = np.array([35, 139, 180])
    # cla_acc = np.array([val for i, val in enumerate(cla_acc) if all(i != index)])
    # con_acc = np.array([val for i, val in enumerate(con_acc) if all(i != index)])
    np.save(f'{args.save_dir}/accuracy_cla{epoch}.npy', cla_acc)  # {args.meta_dataset_name}
    np.save(f'{args.save_dir}/accuracy_con{epoch}.npy', con_acc)
    np.save(f'{args.save_dir}/accuracy_conf{epoch}.npy', max_mean_conf)
    np.save(f'{args.save_dir}/accuracy_threshold{epoch}.npy', thre_acc)
    # # load the regression model
    # cla_acc = np.load(f'{args.save_dir}/accuracy_cla{epoch}.npy')#/home/majc/Contrastive_AutoEval-1/checkpoints/TinyImageNet/metaset200_sampleset10k/tinyimagenet_c
    # con_acc = np.load(f'{args.save_dir}/accuracy_con{epoch}.npy')#accuracy_cla28.npy accuracy_con28.npy
    # conf = np.load(f'{args.save_dir}/accuracy_conf{epoch}.npy')

    # the statistical correlation value on conf
    rho, pval = stats.spearmanr(thre_acc, cla_acc)
    print('############# conf and cl_acc #############\n')
    print('\nSpearman\'s Rank correlation-rho', rho)
    print('Spearman\'s Rank correlation-pval', pval)

    rho, pval = stats.pearsonr(thre_acc, cla_acc)
    print('\nPearsons correlation-rho', rho)
    print('Pearsons correlation-pval', pval)

    rho, pval = stats.kendalltau(thre_acc, cla_acc)
    print('\nKendall\'s Rank correlation-rho', rho)
    print('Kendall\'s correlation-pval', pval)

    r2 = r2_score(cla_acc, thre_acc)
    print('\nCoefficients of Determination-r2', r2)

    # the statistical correlation value
    print('############# con_acc and cl_acc #############\n')
    rho, pval = stats.spearmanr(con_acc, cla_acc)
    print('\nSpearman\'s Rank correlation-rho', rho)
    print('Spearman\'s Rank correlation-pval', pval)

    rho, pval = stats.pearsonr(con_acc, cla_acc)
    print('\nPearsons correlation-rho', rho)
    print('Pearsons correlation-pval', pval)

    rho, pval = stats.kendalltau(con_acc, cla_acc)
    print('\nKendall\'s Rank correlation-rho', rho)
    print('Kendall\'s correlation-pval', pval)

    r2 = r2_score(cla_acc, con_acc)
    print('\nCoefficients of Determination-r2', r2)

if __name__ == "__main__":
    main()