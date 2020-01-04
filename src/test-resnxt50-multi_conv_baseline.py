from __future__ import print_function, division
from scipy import misc

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import time
import copy
import os
#import read_lmdb

import torch.nn.functional as F
import sys
#sys.path.insert(0, r'/home/vis/xiangyunzhao/attri-recognition/networks')
import resnext_50_multi_branch

import resnext_50_multi_conv_baseline_new_test
import resnext_50_multi_max_attention
import focal_loss
import caffe_dataset
import pdb
import argparse
from datetime import datetime
# Training the model
# ------------------
#
from sklearn.metrics import average_precision_score



def train_model(model, criterion, criterion2, optimizer, lr_scheduler, dset_loaders, num_epochs, dset_sizes, args):
    since = time.time()

    best_model = model
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        #label_list = [1,6,7,240,241,308,309]
        #label_list = [20,21,35,36,50,51,69,70,90,91,116,117,131,132,163,164,193,194,208,209,274,275,289,290,54,55,56,57,236,237,238,239,240,241,242,243,244,245,246,247,308,309,310,311]
        label_list = [0,1,2,3,4,5,6,7,8,149,150,151,278,279,280,281,282,283,284,285,286,287,288,289,290,291,292,9,10,11,12,13,14,15,16,17,    18,19,20,21,22,23,212,213,214,215,216,308,309,310,311]
        f = open('CUB_stat_train_1.txt','r')
        lines = f.readlines()
        train_valid = []
        for i in label_list:
             word = lines[i]
             word = word.split(' ')
             train_valid.append(int(word[0])+int(word[1]))
        f.close() 
        f = open('CUB_stat_val_1.txt','r')
        lines = f.readlines()
        val_valid = []
        for i in label_list:
             word = lines[i]
             word = word.split(' ')
             val_valid.append(int(word[0])+int(word[1]))
        f.close()
        
           
        # Each epoch has a training and validation phase
        for phase in ['val']:
            for i in range(args.class_num):
            
                vars()['running_corrects_{}'.format(i)] = 0
            
        #    if phase == 'train':
        #        optimizer = lr_scheduler(optimizer, epoch, args.init_lr, args.lr_decay_epoch)
        #        model.train(True)  # Set model to training mode
        #    else:
        #        if (epoch+1)%1 == 0:
            model.train(False)
        #        else:
        #            continue # Set model to evaluate mode
                
            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            iters = 0
            name_index = 0
            labels_gt = np.zeros((0,312))
        
            overall_score = np.zeros((0,args.class_num))
            
            for data in dset_loaders[phase]:
                # get the inputs
                inputs, labels = data
                labels_gt = np.vstack([labels_gt,labels])
                batch_size = inputs.size(0)
                iters += inputs.size()[0]
                # wrap them in Variable
                use_gpu = True
                if use_gpu:
                    inputs, labels = Variable(inputs.cuda()), \
                        Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
               # optimizer.zero_grad()

                # forward
                [outputs, outputs2] = model(inputs)
                _, preds = torch.max(outputs[0].data, 1)

                # focal loss

                #loss1 = criterion_focalloss_class(outputs[0], labels[:,0].contiguous())
                #loss2 = criterion_focalloss_year(outputs[1], labels[:,1].contiguous())
                #loss3 = criterion_focalloss_color(outputs[2], labels[:,2].contiguous())
                #loss4 = criterion_focalloss_type(outputs[3], labels[:,4].contiguous())
                
                # softmax loss
                loss = 0.0
             #   loss1 = criterion(outputs, labels[:,0]) 
             #   attr_list = list(args.selected_attr)
             #   i = 0
                
                plot = False
                for i in range(args.class_num):
                    
                    attri_label = labels[:,label_list[i]]
                    loss = loss + criterion(outputs[i], attri_label) 
                    _, preds = torch.max(outputs[i].data, 1)
                    ri = preds == labels[:,label_list[i]].data
                    gt_label = labels[:,label_list[i]].data
                    if plot:
                          inputs1 = inputs.data
                          inputs1 = inputs1.cpu()
                          inputs1 = inputs1.numpy()
                            
                          images = inputs1.transpose(0,2,3,1)
                          att_path = './baseline_test/' + str(label_list[i])
                          if not os.path.exists(att_path):
                              os.makedirs(att_path)
                              
                          for j in range(inputs1.shape[0]):
                              image = images[j,:,:,:]
                              image = image*100+70
                              if int(ri[j]) == 1:
                                  co = 'correct'
                                  if int(gt_label[j]) == 1:
                                      co = co+'_positive_'
                                  else:
                                      co = co + '_negative_'

                              else:
                                  if int(gt_label[j]) == -1:
                                      co = 'uncertain'
                                  else:    
                                      co = 'wrong'
                                      if int(gt_label[j]) == 1:
                                         co = co+'_positive_'
                                      else:
                                         co = co + '_negative_'

                              #name_index = label_list[i] 
                              im_name = att_path+'/'+str(name_index)+'_' + co + str(j)+'.jpg'
                             # im = Image.fromarray(im)
                             # im = im.resize((224,224), Image.ANTIALIAS)
                              #im = np.array(im.getdata()).reshape(im.size[0],im.size[1],1)
                              im = image
                       #       im = Image.fromarray(im)
                       #       im = im.convert('RGB')

                       #       im.save(im_name)
                              misc.imsave(im_name, im)
                name_index = name_index + 1        

                # backward + optimize only if in training phase

                # statistics
                running_loss += torch.sum(loss.data)
               
                score = 0 
                for j in range(args.class_num):
                    _, preds = torch.max(outputs[j].data, 1)
                    
                    score_i = F.softmax(outputs[j],dim=1) 
                    score_i = score_i.data
                    
                    score_i = score_i.cpu()
                    score_i = score_i.numpy()
                    if j == 0:
                        score = score_i[:,1]
                        score = np.expand_dims(score,1)
                    else:    
                        score = np.hstack([score,np.expand_dims(score_i[:,1],1)]) 
                    vars()['running_corrects_{}'.format(j)] += torch.sum(preds == labels[:,label_list[j]].data)
                print('Epoch: {} Running Loss: {:.4f} Running Acc: {:.4f}'.format(
                    epoch, running_loss/iters, vars()['running_corrects_{}'.format(0)]/iters))
                overall_score = np.vstack([overall_score,score])    
              #  for name, param in model.state_dict().items():
              #          print(name, param.size())
            epoch_loss = running_loss / dset_sizes[phase]
            acc = []

            acc_all = 0
            if phase == 'train':
                valid_set = train_valid
            else:
                valid_set = val_valid

            
            for j in range(args.class_num):

              epoch_acc = vars()['running_corrects_{}'.format(j)] / valid_set[j]
              acc_all = epoch_acc + acc_all
              acc.append(epoch_acc)  
            print(acc)
            acc_ave = acc_all/args.class_num
            print('acc average: ', acc_ave)
            print('{} Epoch Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            # deep copy the model
            #if phase == 'val' and epoch_acc > best_acc:
            if phase == 'train' and epoch_acc > best_acc and epoch%3 == 0:
                best_acc = epoch_acc
                best_model = copy.deepcopy(model)
               
        multi = False
        results = []
        for i in range(args.class_num):
            label_gt_i = labels_gt[:,label_list[i]]
            score = overall_score[label_gt_i > -1]
            label_i = label_gt_i[label_gt_i > -1]
            results.append(average_precision_score(label_i,score[:,i]))
        print('mAP:',results)
        if (epoch+1)%5 == 0:
         if multi:
           torch.save(model.module.state_dict(),"./output/" + args.save_dir +"/resnxt-50_epoch_{}.pth".format(epoch))
         else:
           torch.save(model.state_dict(),"./output/" + args.save_dir +"/resnxt-50_epoch_{}.pth".format(epoch))
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    return best_model

##################################################################parser.add_argument('--lr_decay_epoch', type=int, help='lr_decay_epoch', default=6)####
# Learning rate scheduler

def exp_lr_scheduler(optimizer, epoch, init_lr, lr_decay_epoch):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.1**(epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer

          # car-fg pytorch trainer
def main(args):
          Batch_Size = args.batch_size
          train_path = args.train_path
          val_path = args.val_path
          data_dir = ''



          data_transforms = {
              'train': transforms.Compose([
                  #transforms.ColorJitter(brightness=0.1),
                  transforms.Scale(args.scale),
                  transforms.RandomSizedCrop(int(args.scale*0.875)),
                  transforms.RandomHorizontalFlip(),
                  transforms.ToTensor(),
                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
              ]),
              'val': transforms.Compose([
                  transforms.Scale(args.scale),
                  transforms.CenterCrop(int(args.scale*0.875)),
                  transforms.ToTensor(),
                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
              ]),
          }
          
          train_list = []
          val_list = []
          train_img_list = open(train_path)
          val_img_list = open(train_path)
          for lines in train_img_list:
              train_list.append(lines)
          for lines in val_img_list:
              val_list.append(lines)
          
          criterion_focalloss_class= focal_loss.FocalLoss(ignore_label = -1, gamma=5, class_num=2059)
          criterion_focalloss_year= focal_loss.FocalLoss(ignore_label = -1, gamma=5, class_num=3031)
          criterion_focalloss_color = focal_loss.FocalLoss(ignore_label = -1, gamma=5, class_num=11)
          criterion_focalloss_type = focal_loss.FocalLoss(ignore_label = -1, gamma=5, class_num=46)
          
          criterion = nn.CrossEntropyLoss(ignore_index = -1) 
          dets= dict()
          train_data_loader = caffe_dataset.ImgListLoader(data_dir, train_path," ", data_transforms['train'])
          val_data_loader = caffe_dataset.ImgListLoader(data_dir, val_path," ", data_transforms['val'])
           
          dets['train'] = train_data_loader
          dets['val'] = val_data_loader
          #dset_list = dict()
          
          #dset_loaders = {x: torch.utils.data.DataLoader(lmdb_loader[x], batch_size=Batch_Size,
          #                                               shuffle=True, num_workers=8)
          #                for x in ['train','val']}
          #dset_sizes = {'train': len(train_list),'val':len(val_list)}
          
          dset_loaders = {}
          dset_loaders['train'] = torch.utils.data.DataLoader(dets['train'], batch_size=Batch_Size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
          dset_loaders['val'] = torch.utils.data.DataLoader(dets['val'], batch_size=4, shuffle=False, num_workers=args.num_workers, pin_memory=True)
           
          train_list = []
          val_list = []
          train_img_list = open(train_path)
          val_img_list = open(val_path)
          
          for lines in train_img_list:
               train_list.append(lines)
          for lines in val_img_list:
               val_list.append(lines)
           
          
          dset_sizes = {'train': len(train_list),'val':len(val_list)}
          
          
          
          use_gpu = torch.cuda.is_available()
          
          model_ft = resnext_50_multi_conv_baseline_new_test.resnext50_fg_car(pretrained=True, cropsize = int(args.scale*0.875),  model_dir = args.model_dir, class_num = args.class_num)
          #optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.1, momentum=0.9)
          #optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, model_ft.parameters()), lr=0.1)
          optimizer_ft = []
          if use_gpu:
              model_ft = model_ft.cuda()
            #  model_ft_parallel = torch.nn.DataParallel(model_ft,device_ids=[2,3]).cuda(1)
          
              model_ft_parallel = nn.DataParallel(model_ft, device_ids=[0, 1, 2])
          criterion = nn.CrossEntropyLoss(ignore_index = -1)
          criterion2 = nn.BCEWithLogitsLoss()
          # Observe that all parameters are being optimized
          
          ######################################################################
          # Train and evaluate
          # ^^^^^^^^^^^^^^^^^^
          
          subdir = datetime.strftime(datetime.now(), '%Y%m%d-%H%M%S')          
          args.save_dir = args.save_dir+subdir
          if not os.path.exists("./output/"+args.save_dir):
              os.makedirs("./output/"+args.save_dir)
          if use_gpu:
              model_best = train_model(model_ft, criterion, criterion2, optimizer_ft, exp_lr_scheduler, dset_loaders,
                                       args.epoch, dset_sizes, args)
          else:
              model_best = train_model(model_ft, criterion, criterion2,  optimizer_ft, exp_lr_scheduler, dset_loaders,
                                       args.epoch, dset_sizes, args)
          torch.save(model_best.module.state_dict(),"./output/"+args.save_dir+"/best-train-model.pth".format(epoch))
######################################################################


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, help='batch size', default=120)  
    parser.add_argument('--epoch', type=int, help='epoch', default= 1) 
    parser.add_argument('--save_dir', type=str, help='save_dir', default='recognition')
    parser.add_argument('--lr_decay_epoch', type=int, help='lr_decay_epoch', default=50)
    parser.add_argument('--init_lr', type=float, help='init_lr', default=0.1)
    parser.add_argument('--selected_attr', type=str, help='attribute', default='')
    parser.add_argument('--class_num', type=int, help='attribute number', default=312)
    parser.add_argument('--num_workers', type=int, help='number works', default=4) 
    parser.add_argument('--model_dir', type=str, help='model_dir', default='./models/resnext_50_32x4d.pth')
    parser.add_argument('--train_path', type=str, help='train path', default='train_class.txt')
    parser.add_argument('--val_path', type=str, help='val path', default='val_class.txt')
    parser.add_argument('--multi_gpu', type=bool, help='multi gpu', default='False')
    parser.add_argument('--device_ids', type=str, help='gpu devices', default='1,2,3,4')
    parser.add_argument('--embedding_size', type=int, help='embedding size', default=64)
    parser.add_argument('--scale', type=int, help='image size', default=256)

# parser = argparse.ArgumentParser('--batch-size', type=int, default=120)  
   # parser = argparse.ArgumentParser('--batch-size', type=int, default=120)                  
    return parser.parse_args(argv)
if __name__ == '__main__':
        main(parse_arguments(sys.argv[1:]))
