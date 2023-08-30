import math
import os
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

import display_progress


class Trainer:
    def __init__(self, data_filepath, dataset_filepath, sigma=None, weights=None):
        self.workers = 0
        self.epochs = 30
        self.batch_size = 64
        self.learning_rate = 0.01
        self.momentum = 0.9
        self.weight_decay = 1e-4
        self.print_freq = 10
        self.sigma = sigma
        self.dataset_filepath = dataset_filepath
        self.data_filepath = data_filepath
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.model = models.efficientnet_b0(weights=weights, num_classes=6).to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(),
                                         lr=self.learning_rate,
                                         momentum=self.momentum,
                                         weight_decay=self.weight_decay)
        self.train_loader = None
        self.val_loader = None
        self.__init_loaders__()
        self.best_acc1 = 0

        self.train_progress = display_progress.ProgressMeter(len(self.train_loader),
                                                 batch_time=display_progress.AverageMeter('Time', ':6.3f'),
                                                 data_time=display_progress.AverageMeter('Data', ':6.3f'),
                                                 losses=display_progress.AverageMeter('Loss', ':.4e'),
                                                 top1=display_progress.AverageMeter('Acc@1', ':6.2f'),
                                                 top5=display_progress.AverageMeter('Acc@5', ':6.2f'))

        self.val_progress = display_progress.ProgressMeter(len(self.val_loader),
                                               batch_time=display_progress.AverageMeter(
                                                   'Time', ':6.3f',display_progress.Summary.NONE),
                                               data_time=display_progress.AverageMeter(
                                                   'Data', ':6.3f', display_progress.Summary.NONE),
                                               losses=display_progress.AverageMeter(
                                                   'Loss', ':.4e', display_progress.Summary.NONE),
                                               top1=display_progress.AverageMeter('Acc@1', ':6.2f'),
                                               top5=display_progress.AverageMeter('Acc@5', ':6.2f'),
                                               prefix='Val: ')

        self.writer = SummaryWriter(self.data_filepath)

    def __init_loaders__(self):
        traindir = os.path.join(self.dataset_filepath, 'train')
        valdir = os.path.join(self.dataset_filepath, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        if self.sigma is not None:
            train_dataset = datasets.ImageFolder(
                traindir,
                transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Lambda(lambda image: torch.clip(image + self.sigma * torch.randn(image.shape), 0, 1)),
                    normalize,
                ]))
        else:
            train_dataset = datasets.ImageFolder(
                traindir,
                transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]))

        val_dataset = datasets.ImageFolder(
            valdir,
            transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ]))

        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=self.workers, pin_memory=True)

        self.val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=self.workers, pin_memory=True)

    def train_model(self):
        for epoch in range(self.epochs):
            self.__train_epoch__(epoch)
            self.__validate__(epoch)
            if self.val_progress.top1.avg > self.best_acc1:
                self.best_acc1 = self.val_progress.top1.avg
                torch.save(self.model.state_dict(), os.path.join(self.data_filepath, 'model.pth.tar'))
                if math.isclose(self.best_acc1, 100.0, abs_tol=0.001):
                    print('100% Accuracy on Validation Set')
                    exit()

    def __train_epoch__(self, epoch):
        self.train_progress.reset()
        self.train_progress.prefix = "Epoch: [{}]".format(epoch)

        self.model.train()

        end = time.time()
        for i, (images, target) in enumerate(self.train_loader):
            self.train_progress.data_time.update(time.time() - end)

            images = images.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True)

            output = self.model(images)
            loss = self.criterion(output, target)

            acc1, acc5 = self.__accuracy__(output, target, topk=(1, 5))
            self.train_progress.losses.update(loss.item(), images.size(0))
            self.train_progress.top1.update(acc1[0], images.size(0))
            self.train_progress.top5.update(acc5[0], images.size(0))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.train_progress.batch_time.update(time.time() - end)

            if i % self.print_freq == 0:
                self.train_progress.display(i + 1)

            end = time.time()

        self.writer.add_scalar('Loss/train', self.train_progress.losses.avg, epoch)
        self.writer.add_scalar('Accuracy/train', self.train_progress.top1.avg, epoch)

    def __validate__(self, epoch):
        self.val_progress.reset()
        self.model.eval()

        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(self.val_loader):
                self.val_progress.data_time.update(time.time() - end)

                images = images.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True)
                output = self.model(images)
                loss = self.criterion(output, target)

                acc1, acc5 = self.__accuracy__(output, target, topk=(1, 5))
                self.val_progress.losses.update(loss.item(), images.size(0))
                self.val_progress.top1.update(acc1[0], images.size(0))
                self.val_progress.top5.update(acc5[0], images.size(0))

                self.val_progress.batch_time.update(time.time() - end)

                if i % self.print_freq == 0:
                    self.val_progress.display(i + 1)

                end = time.time()

        self.val_progress.display_summary()
        self.writer.add_scalar('Loss/val', self.val_progress.losses.avg, epoch)
        self.writer.add_scalar('Accuracy/val', self.val_progress.top1.avg, epoch)

    @staticmethod
    def __accuracy__(output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""

        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res


if __name__ == '__main__':
    trainer = Trainer('../Models/Planet/Texture', 'G:/Datasets/Planet/Texture', sigma=None)
    trainer.train_model()
