import logging
import os
import sys

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import save_config_file, accuracy, save_checkpoint
import time

torch.manual_seed(0)   # <fix>


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class SimCLR(object):

    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.writer = SummaryWriter(log_dir=os.path.join(self.args.save_dir, kwargs['dataset_name']))  # <fix>.register the logging file
        logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)

    def info_nce_loss(self, features, batch_size):

        labels = torch.cat([torch.arange(batch_size) for i in range(self.args.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.args.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.bz, self.args.n_views * self.args.bz)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)

        logits = logits / self.args.temperature
        return logits, labels

    def run(self, train_loader, val_loader, best_acc=0):

        scaler = GradScaler(enabled=self.args.fp16_precision)

        # save config file
        save_config_file(self.writer.log_dir, self.args)

        n_iter = 0
        logging.info(f"Start SimCLR training for {self.args.epochs} epochs.")
        logging.info(f"Training with gpu: {not self.args.disable_cuda}.")

        for epoch_counter in range(self.args.start_epoch, self.args.epochs):
            # train for one epoch
            n_iter = self.train(scaler, train_loader, epoch_counter, n_iter)

            # <fix>, decay the learning rate
            if self.scheduler is not None:
                self.scheduler.step()

            # evaluate on validation set for semantic classification and contrastive learning
            con_acc, cla_acc = self.test(val_loader, n_iter)

            # save model checkpoints
            is_best = (cla_acc+con_acc) > best_acc
            best_acc = max(cla_acc+con_acc, best_acc)
            checkpoint_name = 'checkpoint_{:04d}.pth'.format(epoch_counter)
            save_checkpoint({
                'epoch': epoch_counter,
                'arch': self.args.arch,
                'cla_acc': cla_acc,
                'con_acc': con_acc,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }, is_best=is_best, filename=os.path.join(self.writer.log_dir, checkpoint_name))  # if it's best, then store it as the best model
            # logging.info('Best classification accuracy:', best_acc)
            logging.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")
        logging.info("Training has finished.")

    def train(self, scaler, train_loader, epoch_counter, n_iter=10000000):
        """Train for one epoch on the training set"""
        batch_time = AverageMeter()
        losses = AverageMeter()
        cla_losses = AverageMeter()
        con_losses = AverageMeter()
        cla_top1 = AverageMeter()
        cla_top5 = AverageMeter()
        con_top1 = AverageMeter()
        con_top5 = AverageMeter()

        # switch to train mode
        self.model.train()

        end = time.time()
        for images, targets in tqdm(train_loader):
            images = torch.cat(images, dim=0)
            images = images.to(self.args.device)
            targets = torch.cat([targets, targets], dim=0)  # <fix>
            targets = targets.to(self.args.device)  # <fix>

            with autocast(enabled=self.args.fp16_precision):
                cla_logits, con_features = self.model(images)

                #### semantic classification ####
                # only use original images (the 2nd half) and the current batch size may be dynamic
                cur_batch_size = images.shape[0] // 2  # current batch size
                cla_logits, cla_targets = cla_logits[cur_batch_size:, :], targets[cur_batch_size:]
                cla_acc1, cla_acc5 = accuracy(cla_logits, cla_targets, topk=(1, 5))
                cla_loss = self.criterion(cla_logits, cla_targets)

                ### contrastive learning ###
                con_logits, con_targets = self.info_nce_loss(con_features, cur_batch_size)
                con_acc1, con_acc5 = accuracy(con_logits, con_targets, topk=(1, 5))
                con_loss = self.criterion(con_logits, con_targets)

                ### multi-task / only classification loss / only contrastive learning loss ###
                loss = self.args.claLoss_weight * con_loss + self.args.claLoss_weight * cla_loss

                ### measure accuracies and losses ###
                cla_top1.update(cla_acc1.item(), cur_batch_size)
                cla_top5.update(cla_acc5.item(), cur_batch_size)
                con_top1.update(con_acc1.item(), cur_batch_size)
                con_top5.update(con_acc5.item(), cur_batch_size)
                cla_losses.update(cla_loss.item())
                con_losses.update(con_loss.item())
                losses.update(loss.item())

            ### compute gradient and do optimizer step ###
            self.optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            ### we do not record the results accrodding to the number of batches
            # if n_iter % self.args.log_every_n_steps == 0:
            #     logging.debug(f"Steps: {n_iter}\tbatch_time: {batch_time.sum}\tLoss: {losses.avg}\tCla_Loss: {cla_losses.avg}\tCon_Loss: {con_losses.avg}" + \
            #         f"\tCla_top1_acc: {cla_top1.avg}\tCon_top1_acc:{con_top1.avg}\tCla_top5_acc: {cla_top5.avg}\tCon_top5_acc: {con_top5.avg}" + \
            #         f"\tLr: {self.scheduler.get_lr()[0]}")

            n_iter += 1

        # add this infos into tensorboard writer in every epoch
        self.writer.add_scalar('train_acc/cla_top1', cla_top1.avg, global_step=epoch_counter)
        self.writer.add_scalar('train_acc/cla_top5', cla_top5.avg, global_step=epoch_counter)
        self.writer.add_scalar('train_acc/con_top1', con_top1.avg, global_step=epoch_counter)
        self.writer.add_scalar('train_acc/con_top5', con_top5.avg, global_step=epoch_counter)
        self.writer.add_scalar('train_loss/cla_loss', cla_losses.avg, global_step=epoch_counter)
        self.writer.add_scalar('train_loss/con_loss', con_losses.avg, global_step=epoch_counter)
        self.writer.add_scalar('train_loss/loss', losses.avg, global_step=epoch_counter)
        self.writer.add_scalar('learning_rate', self.scheduler.get_lr()[0] if self.scheduler is not None else self.args.lr, global_step=epoch_counter)

        # <fix>, print the contrastive and classification loss, Top1 Acc, Cls Acc
        logging.debug(f"Epoch: {epoch_counter}\tbatch_time: {batch_time.sum}\tLoss: {losses.avg}\tCla_Loss: {cla_losses.avg}\tCon_Loss: {con_losses.avg}" +\
                      f"\tCla_top1_acc: {cla_top1.avg}\tCon_top1_acc:{con_top1.avg}\tCla_top5_acc: {cla_top5.avg}\tCon_top5_acc: {con_top5.avg}" +\
                      f"\tLr: {self.scheduler.get_lr()[0] if self.scheduler is not None else self.args.lr}")
        return n_iter

    def test(self, test_loader, threshold = 0.5, n_iter=10000000,):
        """Perform validation on the validation set"""
        batch_time = AverageMeter()
        losses = AverageMeter()
        cla_losses = AverageMeter()
        con_losses = AverageMeter()
        cla_top1 = AverageMeter()
        cla_conf = AverageMeter()
        num_acc = AverageMeter()
        cla_top5 = AverageMeter()
        con_top1 = AverageMeter()
        con_top5 = AverageMeter()

        # switch to evaluate mode
        self.model.eval()

        end = time.time()
        for images, targets in tqdm(test_loader):
            images = torch.cat(images, dim=0)
            images = images.to(self.args.device)
            targets = torch.cat([targets, targets], dim=0)  # <fix>
            targets = targets.to(self.args.device)  # <fix>

            with torch.no_grad():
                with autocast(enabled=self.args.fp16_precision):
                    cla_logits, con_features = self.model(images)

                    #### semantic classification ####
                    # only use original images (the 2nd half) and the current batch size may be dynamic
                    cur_batch_size = images.shape[0] // 2  # current batch size 只用后半段的cla_logits，后半部分是origin image
                    cla_logits, cla_targets = cla_logits[cur_batch_size:, :], targets[cur_batch_size:]
                    cla_acc1, cla_acc5 = accuracy(cla_logits, cla_targets, topk=(1, 5))
                    cla_loss = self.criterion(cla_logits, cla_targets)

                    # count max_mean_confidence
                    softmax= torch.nn.Softmax(dim=1)
                    output = softmax(cla_logits)
                    prob, pred = output.topk(1, 1, True, True)# top1 confidence
                    logits = prob.reshape(-1).float()

                    # when regress do this compute
                    a = logits > threshold
                    b = (logits > threshold).reshape(-1).nonzero().numel()#true/false->0/1->非0个数
                    num = torch.tensor((logits > threshold).reshape(-1).nonzero().numel()).float().cuda()
                    num = num.mul_(100.0 / cur_batch_size)

                    prob = prob.reshape(-1).float().sum(0, keepdim=True)
                    prob = prob.mul_(1.0 / cur_batch_size)
                    

                    ### contrastive learning ###
                    con_logits, con_targets = self.info_nce_loss(con_features, cur_batch_size)
                    con_acc1, con_acc5 = accuracy(con_logits, con_targets, topk=(1, 5))
                    con_loss = self.criterion(con_logits, con_targets)

                    ### multi-task / only classification loss / only contrastive learning loss ###
                    loss = self.args.claLoss_weight * con_loss + self.args.claLoss_weight * cla_loss

                    ### measure accuracies and losses ###
                    cla_top1.update(cla_acc1.item(), cur_batch_size)#86/128
                    cla_top5.update(cla_acc5.item(), cur_batch_size)
                    con_top1.update(con_acc1.item(), cur_batch_size)
                    con_top5.update(con_acc5.item(), cur_batch_size)
                    num_acc.update(num.item(), cur_batch_size)
                    cla_conf.update(prob.item(), cur_batch_size)# 0.61 
                    cla_losses.update(cla_loss.item())
                    con_losses.update(con_loss.item())
                    losses.update(loss.item())

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                # if n_iter % self.args.log_every_n_steps == 0:
                #     logging.debug(f"Steps: {n_iter}\tbatch_time: {batch_time.sum}\tLoss: {losses.avg}\tCla_Loss: {cla_losses.avg}\tCon_Loss: {con_losses.avg}" + \
                #         f"\tCla_top1_acc: {cla_top1.avg}\tCon_top1_acc:{con_top1.avg}\tCla_top5_acc: {cla_top5.avg}\tCon_top5_acc: {con_top5.avg}" + \
                #         f"\tLr: {self.scheduler.get_lr()[0]}")

        # <fix>, print the contrastive and classification loss, Top1 Acc, Cls Acc
        logging.debug(f"Test:\tbatch_time: {batch_time.sum}\tLoss: {losses.avg}\tCla_Loss: {cla_losses.avg}\tCon_Loss: {con_losses.avg}" +\
                      f"\tCla_top1_acc: {cla_top1.avg}\tCon_top1_acc:{con_top1.avg}\tCla_top5_acc: {cla_top5.avg}\tCon_top5_acc: {con_top5.avg}")
        logging.info("Test has finished.")

        return con_top1.avg, cla_top1.avg, cla_conf.avg, num_acc.avg
