import numpy as np
from torchvision.transforms import transforms  # <fix>

np.random.seed(0)  # <fix>


class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2, data_setup=None, augment=True, train_trans=True):    # <fix>
        self.base_transform = base_transform
        self.n_views = n_views
        self.augment = augment    # <fix>
        self.train_trans = train_trans

        self.train_transform = None
        self.test_transform = None

        ## MNIST forgets that all the original images of sets should be normalized without enhancement, just like 'mnist1'
        if data_setup == "mnist":  # <fix>, MNIST Setup
            self.train_transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            self.test_transform = transforms.Compose([
                transforms.ToTensor(),
            ])

        elif data_setup == "mnist1":  # <fix>, MNIST Setup
            self.normalize = transforms.Normalize((0.5,), (0.5,))  # transforms.Normalize((0.1307,), (0.3081,))
            self.train_transform = transforms.Compose([
                transforms.ToTensor(),
                self.normalize,
            ])
            self.test_transform = transforms.Compose([
                transforms.ToTensor(),
                self.normalize,
            ])

        ## CIFAR10/100 forgets that the transformation of the original image of the train and val/metaset/unseen test is different, just like 'cifar1'
        elif data_setup == "cifar":  # <fix>, CIFAR Setup
            self.normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                             std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
            self.train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                self.normalize,
            ])
            self.test_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                self.normalize,
            ])

        elif data_setup == "cifar1": # <fix>, CIFAR Setup
            self.normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                             std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
            self.train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                self.normalize,
            ])
            self.test_transform = transforms.Compose([
                transforms.ToTensor(),
                self.normalize,
            ])

        ## COCO forgot that the transformation to be done in the original image of train and val/metaset/unseen test
        # is not the same, and the follow-up should be changed to the following
        elif data_setup == "coco":  # <fix>, COCO Setup, the pretrained transforms sets
            self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
            self.train_transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                self.normalize
            ])
            self.test_transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                self.normalize
            ])

        elif data_setup == "coco1":  # <fix>, COCO Setup, the pretrained transforms sets
            self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
            self.train_transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                self.normalize
            ])
            self.test_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                self.normalize
            ])

        elif data_setup == "coco2":  # <fix>, COCO Setup1, the pretrained transforms sets
            self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
            self.train_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                self.normalize
                ])
            self.test_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                self.normalize
            ])

        elif data_setup == "tinyimagenet":  # <fix>, TinyImageNet Setup, the pretrained transforms sets
            self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
            self.train_transform = transforms.Compose([
                transforms.RandomResizedCrop(64),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                self.normalize
            ])
            self.test_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(64),
                transforms.ToTensor(),
                self.normalize
            ])

        elif data_setup == "tinyimagenet1":  # <fix>, TinyImageNet Setup1, the pretrained transforms sets
            self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])
            self.train_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                self.normalize
                ])
            self.test_transform = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                self.normalize
            ])

    def __call__(self, x):
        if self.augment:          # <fix>, whether or not to transform the original image
            if self.train_trans:  # <fix>, whether current dataset is train set
                return [self.base_transform(x) for _ in range(self.n_views-1)] + [self.train_transform(x)]
            else:
                return [self.base_transform(x) for _ in range(self.n_views - 1)] + [self.test_transform(x)]
        else:
            # MNIST原先是这样做的
            return [self.base_transform(x) for _ in range(self.n_views-1)] + [transforms.ToTensor()(x)]
            # return [self.base_transform(x) for _ in range(self.n_views - 1)] + [transforms.Compose([transforms.ToTensor(), self.normalize])(x)]
