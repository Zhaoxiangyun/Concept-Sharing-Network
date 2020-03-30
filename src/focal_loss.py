#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in 
        Focal Loss for Dense Object Detection.
            
            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])
    
        The losses are averaged across observations for each minibatch.
        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5), 
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.
    """
    def __init__(self, class_num, alpha=None, gamma=2, ignore_label=-1, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average
        self.ignore_label = ignore_label
        self.threshold = nn.Threshold(1e-20,1)

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs)
        ids = targets.view(-1, 1)
        mask_vaild = (ids != -1)
        target_selected = torch.masked_select(ids,mask_vaild).view(-1,1)
        if len(target_selected.size()) == 0:
            batch_loss = inputs*0
            if self.size_average:
                loss = batch_loss.mean()
            else:
                loss = batch_loss.sum()
            return loss
        P_selected = torch.masked_select(P, mask_vaild)
        P_selected = P_selected.view(-1,C)
        class_mask = inputs.data.new(P_selected.size()).fill_(0)
        class_mask.scatter_(1, target_selected.data, 1.)
        class_mask = Variable(class_mask)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[target_selected.data.view(-1)]
        probs = (P_selected*class_mask).sum(1).view(-1,1)
        probs = self.threshold(probs)
        log_p = probs.log()
        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p 
        
        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss

        

#if __name__ == "__main__":
#    FL = FocalLoss(class_num=5, gamma=5)
#    CE = nn.CrossEntropyLoss(ignore_index = -1)
#    N = 4
#    C = 5
#    inputs = torch.rand(N, C)
#    targets = torch.LongTensor(N).random_(C)
#    targets[0] = -1
#    targets[1] = -1
#    targets[2] = -1
#    targets[3] = -1
#    inputs_fl = Variable(inputs.clone(), requires_grad=True)
#    targets_fl = Variable(targets.clone())
#    inputs_ce = Variable(inputs.clone(), requires_grad=True)
#    targets_ce = Variable(targets.clone())
#    print('----inputs----')
#    print(inputs)
#    print('---target-----')
#    print(targets)
#    import pdb
#    pdb.set_trace()
#
#    fl_loss = FL(inputs_fl, targets_fl)
#    ce_loss = CE(inputs_ce, targets_ce)
#    print('ce = {}, fl ={}'.format(ce_loss.data[0], fl_loss.data[0]))
#    fl_loss.backward()
#    ce_loss.backward()
#    print(inputs_fl.grad.data)
#    print(inputs_ce.grad.data)
#
#
