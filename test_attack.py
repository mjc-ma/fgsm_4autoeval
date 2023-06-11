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

######### Refer to SimCLR-master ########
model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name])) + ["lenet", "densenet40-12"]  # <fix>

parser = argparse.ArgumentParser(description='PyTorch SimCLR')
parser.add_argument('--data', metavar='DIR', default='/home/majc/data/Tiny-ImageNet-C/zoom_blur/4/',
                    help='path to dataset')
parser.add_argument('--data_adv', metavar='DIR', default='/home/majc/data/fgsm_4autoeval/tinyimagenet_fgsm_4autoeval/0.03/',
                    help='path to dataset')
# parser.add_argument('--data', metavar='DIR', default='/home/majc/data/metasets/tinyimagenet_adv_fgsm/0.01/',
#                     help='path to dataset')
parser.add_argument('--test-dataset-name', default='tinyimagenet_test',
                    help='Unseen target dataset name', choices=['svhn', 'usps', 'cifar10_1', 'cifar10_c',
                                                                'cifar100', 'cifar100_c', 'caltech', 'pascal',
                                                                'imagenet', 'tinyimagenet_c','imagenet_a', 'tinyimagenet_adv', 'tinyimagenet_test', 'tinyimagenet_adv_fgsm'])  # <fix>
parser.add_argument('--meta-dataset-name', default='tinyimagenet',
                    help='Meta-set dataset name',
                    choices=['mnist', 'cifar10', 'cifar100', 'coco', 'tinyimagenet'])  # <fix>
parser.add_argument('--val-dataset-name', default='tinyimagenet_val',
                    help='validation dataset name', choices=['mnist', 'mnist_raw','fashion_mnist', 'k_mnist',
                                                             'cifar10', 'cifar100', 'stl10', 'coco', 'tinyimagenet','tinyimagenet_val'])  # <fix>  

parser.add_argument('--metaset-dir', metavar='DIR', default='/home/majc/data/metasets/TinyImageNet200/dataset_default',
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
parser.add_argument('-test_b', '--test-batch-size', default=64, type=int,
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
parser.add_argument('--gpu-index', default=0, type=int, help='Gpu index.')

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
    print(f'\nTest batch size:{args.test_batch_size}')

    # eval on unseen test set
    print(f'Test on {args.test_dataset_name}')
    test_con_acc = 0
    test_cla_acc = 0
    test_conf_acc = 0
    test_num_acc = 0

    dataset = ContrastiveLearningDataset(args)  # <fix>, use the cl-model type to choose the transformation type
    if args.test_dataset_name == 'tinyimagenet_c':
        corrupution_types = os.listdir(args.data+ "Tiny-ImageNet-C")
        for i in range(0, len(corrupution_types)):  # <fix>, each corrupution_types
            test_con_acc_i = 0
            test_cla_acc_i = 0
            test_conf_acc_i = 0
            test_num_acc_i = 0

            for j in range(1, 6):
                test_dataset = dataset.get_test_dataset(os.path.join(args.data, "Tiny-ImageNet-C", corrupution_types[i], f"{j}") , args.test_dataset_name,
                                                        args.n_views, args.data_setup, args.augment, train_trans=False)  # <fix>, whether or not to transform the original image
                test_loader = torch.utils.data.DataLoader(
                    test_dataset, batch_size=args.test_batch_size, shuffle=True,
                    num_workers=args.workers, pin_memory=True)

                # evaluate on semantic classification and contrastive learning (Calculate both acc)
                test_con_acc_ij , test_cla_acc_ij , test_conf_acc_ij ,test_num_acc_ij = simclr.test(test_loader) # <fix>
                test_con_acc_i += test_con_acc_ij
                test_cla_acc_i += test_cla_acc_ij
                test_conf_acc_i += test_conf_acc_ij
                test_num_acc_i += test_num_acc_ij

            test_con_acc_i /= 5
            test_cla_acc_i /= 5
            test_conf_acc_i /= 5
            test_num_acc_i /= 5

            print(f'\nSemantic classification accuracy on {args.test_dataset_name}/{corrupution_types[i]}: %.2f' %  (test_cla_acc_i))
            print(f'Contrastive learning accuracy on {args.test_dataset_name}/{corrupution_types[i]}: %.2f' %  (test_con_acc_i))
            print(f'max mean confidence on {args.test_dataset_name}/{corrupution_types[i]}: %.2f' %  (test_conf_acc_i))
            print(f'ATC on {args.test_dataset_name}/{corrupution_types[i]}: %.2f' %  (test_num_acc_i))

            test_con_acc += test_con_acc_i
            test_cla_acc += test_cla_acc_i
            test_con_acc += test_con_acc_i
            test_cla_acc += test_cla_acc_i

        test_con_acc /= len(corrupution_types)
        test_cla_acc /= len(corrupution_types)
        test_conf_acc /= len(corrupution_types)
        test_num_acc /= len(corrupution_types)
        ## predict for all corrupution_types
        print(f'\nSemantic classification accuracy on {args.test_dataset_name}: %.2f' % (test_cla_acc))
        print(f'Contrastive learning accuracy on {args.test_dataset_name}: %.2f' % (test_con_acc))
        print(f'max mean confidence on {args.test_dataset_name}: %.2f' % (test_conf_acc))
        print(f'ATC on {args.test_dataset_name}: %.2f' % (test_num_acc))


    elif args.test_dataset_name == 'tinyimagenet_test':
        meta_dataset = dataset.get_test_dataset(args.data_adv,'tinyimagenet_adv', args.n_views, args.data_setup, args.augment,
                                                    train_trans=False)  # <fix>, whether or not to transform the original image

        test_loader = torch.utils.data.DataLoader(
                dataset=meta_dataset, batch_size=args.meta_batch_size, shuffle=True,
                num_workers=args.workers, pin_memory=True)
        test_con_acc , test_cla_acc , test_conf_acc ,test_num_acc = simclr.test(test_loader,threshold= 0.77) # <fix>


        ## predict for all corrupution_types
        print('\n####----------------This is fgsm_4autoeval test on autoeval-------------####')
        print(f'\nSemantic classification accuracy on {args.test_dataset_name}: %.2f' % (test_cla_acc))
        print(f'Contrastive learning accuracy on {args.test_dataset_name}: %.2f' % (test_con_acc))
        print(f'max mean confidence on {args.test_dataset_name}: %.2f' % (test_conf_acc))
        print(f'ATC on {args.test_dataset_name}: %.2f' % (test_num_acc))



        # meta_dataset = dataset.get_test_dataset(args.data , args.test_dataset_name,
        #                                             args.n_views,
        #                                             args.data_setup, args.augment,
        #                                             train_trans=False)  # <fix>, whether or not to transform the original image
        # test_loader = torch.utils.data.DataLoader(
        #         dataset=meta_dataset, batch_size=args.meta_batch_size, shuffle=True,
        #         num_workers=args.workers, pin_memory=True)
        # test_con_acc , test_cla_acc , test_conf_acc ,test_num_acc = simclr.test(test_loader,threshold= 0.771) # <fix>

        # ## predict for all corrupution_types
        # print('\n###----------------This is origin test on autoeval-------------####')
        # print(f'\nSemantic classification accuracy on {args.test_dataset_name}: %.2f' % (test_cla_acc))
        # print(f'Contrastive learning accuracy on {args.test_dataset_name}: %.2f' % (test_con_acc))
        # print(f'max mean confidence on {args.test_dataset_name}: %.2f' % (test_conf_acc))
        # print(f'ATC on {args.test_dataset_name}: %.2f' % (test_num_acc))



    elif args.test_dataset_name == 'tinyimagenet_val':
        meta_dataset = dataset.get_val_dataset(args.val_dataset_name, args.n_views, args.data_setup, 
                                            args.augment, train_trans=False)  # <fix>, whether or not to transform the original image

        test_loader = torch.utils.data.DataLoader(
                dataset=meta_dataset, batch_size=args.meta_batch_size, shuffle=True,
                num_workers=args.workers, pin_memory=True)
        test_con_acc , test_cla_acc , test_conf_acc ,test_num_acc = simclr.test(test_loader) # <fix>

        ## predict for all corrupution_types
        print(f'\nSemantic classification accuracy on {args.test_dataset_name}: %.2f' % (test_cla_acc))
        print(f'Contrastive learning accuracy on {args.test_dataset_name}: %.2f' % (test_con_acc))
        print(f'max mean confidence on {args.test_dataset_name}: %.2f' % (test_conf_acc))
        print(f'ATC on {args.test_dataset_name}: %.2f' % (test_num_acc))


    elif args.test_dataset_name == 'imagenet_a':
        test_dataset = dataset.get_test_dataset(os.path.join(args.data, "ImageNet-A") , args.test_dataset_name,
                                                        args.n_views, args.data_setup, args.augment, train_trans=False)  # <fix>, whether or not to transform the original image
        test_loader = torch.utils.data.DataLoader(
                    test_dataset, batch_size=args.test_batch_size, shuffle=True,
                    num_workers=args.workers, pin_memory=True)
        test_con_acc , test_cla_acc, test_conf_acc ,test_num_acc = simclr.test(test_loader)
        ## predict for all corrupution_types
        print(f'\nSemantic classification accuracy on {args.test_dataset_name}: %.2f' % (test_cla_acc))
        print(f'Contrastive learning accuracy on {args.test_dataset_name}: %.2f' % (test_con_acc))



    elif args.test_dataset_name == 'cifar10_c':
        corrupution_types = [os.path.splitext(i)[0] for i in os.listdir(args.data+"CIFAR-10-C")
                             if os.path.splitext(i)[0] != "labels" and os.path.splitext(i)[1] == ".npy"]
        for i in range(0, len(corrupution_types)):
            test_dataset = dataset.get_test_dataset(os.path.join(args.data, "CIFAR-10-C", corrupution_types[i]+".npy"), args.test_dataset_name,
                                                    args.n_views, args.data_setup, args.augment, train_trans=False)  # <fix>, whether or not to transform the original image
            test_loader = torch.utils.data.DataLoader(
                test_dataset, batch_size=args.test_batch_size, shuffle=True,
                num_workers=args.workers, pin_memory=True)

            # evaluate on semantic classification and contrastive learning (Calculate both acc)
            test_con_acc_i, test_cla_acc_i = simclr.test(test_loader)
            # print(f'\nSemantic classification accuracy on {args.test_dataset_name}/{corrupution_types[i]}: %.2f' % (test_cla_acc_i))
            # print(f'Contrastive learning accuracy on {args.test_dataset_name}/{corrupution_types[i]}: %.2f' % (test_con_acc_i))

            test_con_acc += test_con_acc_i
            test_cla_acc += test_cla_acc_i
        test_con_acc /= len(corrupution_types)
        test_cla_acc /= len(corrupution_types)
        ## predict for all corrupution_types
        print(f'\nSemantic classification accuracy on {args.test_dataset_name}: %.2f' % (test_cla_acc))
        print(f'Contrastive learning accuracy on {args.test_dataset_name}: %.2f' % (test_con_acc))



    elif args.test_dataset_name == 'cifar100_c':
        corrupution_types = [os.path.splitext(i)[0] for i in os.listdir(args.data+"CIFAR-100-C")
                             if os.path.splitext(i)[0] != "labels" and os.path.splitext(i)[1] == ".npy"]
        for i in range(0, len(corrupution_types)):
            test_dataset = dataset.get_test_dataset(os.path.join(args.data, "CIFAR-100-C", corrupution_types[i]+".npy"), args.test_dataset_name,
                                                    args.n_views, args.data_setup, args.augment, train_trans=False)  # <fix>, whether or not to transform the original image
            test_loader = torch.utils.data.DataLoader(
                test_dataset, batch_size=args.test_batch_size, shuffle=True,
                num_workers=args.workers, pin_memory=True)

            # evaluate on semantic classification and contrastive learning (Calculate both acc)
            test_con_acc_i, test_cla_acc_i = simclr.test(test_loader)
            # print(f'\nSemantic classification accuracy on {args.test_dataset_name}/{corrupution_types[i]}: %.2f' % (test_cla_acc_i))
            # print(f'Contrastive learning accuracy on {args.test_dataset_name}/{corrupution_types[i]}: %.2f' % (test_con_acc_i))

            test_con_acc += test_con_acc_i
            test_cla_acc += test_cla_acc_i
        test_con_acc /= len(corrupution_types)
        test_cla_acc /= len(corrupution_types)
        ## predict for all corrupution_types
        print(f'\nSemantic classification accuracy on {args.test_dataset_name}: %.2f' % (test_cla_acc))
        print(f'Contrastive learning accuracy on {args.test_dataset_name}: %.2f' % (test_con_acc))

    else:
        test_dataset = dataset.get_test_dataset(args.data, args.test_dataset_name, args.n_views, args.data_setup,
                                                args.augment, train_trans=False)  # <fix>, whether or not to transform the original image
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.test_batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True)

        # evaluate on semantic classification and contrastive learning (Calculate both acc)
        test_con_acc, test_cla_acc = simclr.test(test_loader)
        print(f'\nSemantic classification accuracy on {args.test_dataset_name}: %.2f' % (test_cla_acc))
        print(f'Contrastive learning accuracy on {args.test_dataset_name}: %.2f' % (test_con_acc))

    # load the regression model
    cla_acc = np.load(f'{args.save_dir}/accuracy_cla{epoch}.npy')#/home/majc/Contrastive_AutoEval-1/checkpoints/TinyImageNet/metaset200_sampleset10k/tinyimagenet_c
    con_acc = np.load(f'{args.save_dir}/accuracy_con{epoch}.npy')#accuracy_cla28.npy accuracy_con28.npy
    conf_acc = np.load(f'{args.save_dir}/accuracy_conf{epoch}.npy')#accuracy_cla28.npy accuracy_con28.npy
    num_acc = np.load(f'{args.save_dir}/accuracy_threshold{epoch}.npy')#accuracy_cla28.npy accuracy_con28.npy

    # # the statistical correlation value
    # rho, pval = stats.spearmanr(con_acc, cla_acc)
    # print('\nSpearman\'s Rank correlation-rho', rho)
    # print('Spearman\'s Rank correlation-pval', pval)

    # rho, pval = stats.pearsonr(con_acc, cla_acc)
    # print('\nPearsons correlation-rho', rho)
    # print('Pearsons correlation-pval', pval)

    # rho, pval = stats.kendalltau(con_acc, cla_acc)
    # print('\nKendall\'s Rank correlation-rho', rho)
    # print('Kendall\'s correlation-pval', pval)

    # r2 = r2_score(cla_acc, con_acc)
    # print('\nCoefficients of Determination-r2', r2)

    ### using regression model to predict preformance of unseen target sets
    from sklearn.linear_model import LinearRegression, HuberRegressor
    from sklearn.metrics import mean_squared_error

    slr1 = LinearRegression()
    slr1.fit(np.array(con_acc.reshape(-1, 1)), np.array(cla_acc.reshape(-1, 1)))
    pred = slr1.predict(np.array(test_con_acc).reshape(-1, 1)) #<fix>估计的得分
    error = mean_squared_error(pred, np.array(test_cla_acc).reshape(-1, 1), squared=False)  # squared=False returns RMSE value
    print('\nLinear regression model predicts %4f ,its true acc is %4f,the absolute error is %4f' % (pred, test_cla_acc, error))

    slr2 = LinearRegression()
    slr2.fit(np.array(conf_acc.reshape(-1, 1)), np.array(cla_acc.reshape(-1, 1)))
    pred = slr2.predict(np.array(test_conf_acc).reshape(-1, 1)) #<fix>估计的得分
    error = mean_squared_error(pred, np.array(test_cla_acc).reshape(-1, 1), squared=False)  # squared=False returns RMSE value
    print('\nLinear regression model predicts %4f ,its true acc is %4f,the absolute error is %4f' % (pred, test_cla_acc, error))

    slr2 = LinearRegression()
    slr2.fit(np.array(num_acc.reshape(-1, 1)), np.array(cla_acc.reshape(-1, 1)))
    pred = slr2.predict(np.array(test_num_acc).reshape(-1, 1)) #<fix>估计的得分
    error = mean_squared_error(pred, np.array(test_cla_acc).reshape(-1, 1), squared=False)  # squared=False returns RMSE value
    print('\nLinear regression model predicts %4f ,its true acc is %4f,the absolute error is %4f' % (pred, test_cla_acc, error))
    
    # robust_reg = HuberRegressor()
    # robust_reg.fit(np.array(con_acc.reshape(-1, 1)), np.array(cla_acc.reshape(-1)))
    # robust_pred = robust_reg.predict(np.array(test_con_acc).reshape(-1, 1))
    # robust_error = mean_squared_error(robust_pred, np.array(test_cla_acc).reshape(-1, 1), squared=False)
    # print('Robust Linear regression model predicts %4f and its absolute error is %4f' % (robust_pred, robust_error))


if __name__ == "__main__":
    main()