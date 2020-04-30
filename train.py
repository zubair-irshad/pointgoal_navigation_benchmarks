import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
import torch.utils.data as data
from   torchsummary import summary
from   torchvision.utils import make_grid

from Project import Project
from dataset import Dataset_RNN
from utils import show_dataset,max_seq_length_list,device, show_one_batch, save_checkpoint
from models import CNNEncoder, DecoderRNN

import numpy as np
import os
import torch
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence
from expert import Expert

def pad_collate(batch):
  (xx, yy) = zip(*batch)
  x_lens = [len(x) for x in xx]
  y_lens = [len(y) for y in yy]
  xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
  yy_pad = pad_sequence(yy, batch_first=True, padding_value=-1)
  return xx_pad, yy_pad, x_lens, y_lens


def create_mask(batchsize, max_length, length,device):
    """Given the length create a mask given a padded tensor"""
    tensor_mask = torch.zeros(batchsize, max_length, dtype = torch.bool)
    for idx, row in enumerate(tensor_mask):
        row[:length[idx]] = 1
    tensor_mask.unsqueeze_(-1)
    return tensor_mask.to(device)



def train(log_interval, model,criterion, device, train_loader, optimizer, epoch, batch_size, output_dim, params, writer):
    # set model as training mode

    batch_time = AverageMeter('batch_time')
    data_time = AverageMeter('data_time')
    losses = AverageMeter('loss')
    top1 = AverageMeter('accuracy')
    # progress = ProgressMeter(len(train_loader),
    #                          [batch_time, data_time, losses, top1],
    #                          prefix="Epoch: [{}]".format(epoch))
    N_count = 0   # counting total trained sample in one epoch
    cnn_encoder, rnn_decoder = model
    cnn_encoder.train()
    rnn_decoder.train()
    h = rnn_decoder.init_hidden(batch_size)
    end = time.time()

    for batch_idx, (X, y, x_lengths,_) in enumerate(train_loader):
        
        data_time.update(time.time() - end) # measure data loading time
        h = tuple([each.detach() for each in h]) # Create new variables for hidden state so we do not backprop the entire history   
        X, y = X.to(device), y.to(device) # distribute data to device
        
        N_count += X.size(0)
        #CNN_output and CNN Mask
        encoder_out = cnn_encoder(X)
        embed_mask = create_mask(encoder_out.shape[0],encoder_out.shape[1], x_lengths,device)
        embed_mask = embed_mask.expand_as(encoder_out)
        encoder_out = encoder_out*embed_mask

        #RNN Output
        output,h = rnn_decoder(encoder_out,h, x_lengths)   # output has dim = (batch*seq_length, number of outputs)

        #Creat RNN_mask
        decoder_mask = create_mask(encoder_out.shape[0], encoder_out.shape[1], x_lengths, device)
        decoder_mask = decoder_mask.expand(encoder_out.shape[0], encoder_out.shape[1], output_dim)
        decoder_mask = decoder_mask.view(-1,output_dim)
        output=output*decoder_mask

        # encoder_out = torch.cat((encoder_out, pose),2) # Encode pose_features to the image features

        #Compute Loss & accuracy
        loss = criterion(output, y.view(-1,1).squeeze(1))
        acc = accuracy(output, y.view(-1,1))
        losses.update(loss.item(), X.size(0))
        top1.update(acc, X.size(0))
        
        optimizer.zero_grad()
        loss.backward()
        # nn.utils.clip_grad_norm_(rnn_decoder.parameters(), clip)
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()


        if (batch_idx+1) % log_interval == 0:
            # print(batch_idx+1)
            # print(epoch+1)
            current_iter = (batch_idx+1) + epoch*(len(train_loader))
            print("Epoch: {}/{}...".format(epoch+1, params['epochs']),
                  "Step: [{}/{} ({:.0f}%)]...".format(N_count, len(train_loader.dataset), 100. * (batch_idx + 1) / len(train_loader)),
                  "Time {batch_time.val:.3f} ({batch_time.avg:.3f})".format(batch_time=batch_time),
                  "Train Loss {loss.val:.4f} ({loss.avg:.4f})".format(loss=losses),
                  "Train Accuracy {accuracy.val:.4f} ({accuracy.avg:.4f})".format(accuracy=top1))
            writer.add_scalar('train/loss_train', losses.avg,current_iter)
            writer.add_scalar('train/accuracy', top1.avg,current_iter)

def validate(log_interval, model,criterion, device, val_loader, epoch, batch_size, output_dim, params, writer):

    batch_time = AverageMeter('batch_time')
    losses = AverageMeter('loss')
    top1 = AverageMeter('accuracy')
    N_count = 0   # counting total trained sample in one epoch
    
    # set model as testing mode
    cnn_encoder, rnn_decoder = model
    cnn_encoder.eval()
    rnn_decoder.eval()
    val_h = rnn_decoder.init_hidden(batch_size)

    with torch.no_grad():
        end = time.time()
        for batch_idx, (X, y, x_lengths,_) in enumerate(val_loader):
            val_h = tuple([each.detach() for each in val_h]) # Create new variables for hidden state so we do not backprop the entire history   
            X, y = X.to(device), y.to(device) # distribute data to device
            N_count += X.size(0)
            #CNN_output and CNN Mask
            encoder_out = cnn_encoder(X)
            embed_mask = create_mask(encoder_out.shape[0],encoder_out.shape[1], x_lengths,device)
            embed_mask = embed_mask.expand_as(encoder_out)
            encoder_out = encoder_out*embed_mask

            #RNN Output
            output,val_h = rnn_decoder(encoder_out,val_h, x_lengths)   # output has dim = (batch*seq_length, number of outputs)

            #Creat RNN_mask
            decoder_mask = create_mask(encoder_out.shape[0], encoder_out.shape[1], x_lengths, device)
            decoder_mask = decoder_mask.expand(encoder_out.shape[0], encoder_out.shape[1], output_dim)
            decoder_mask = decoder_mask.view(-1,output_dim)
            output=output*decoder_mask

            # encoder_out = torch.cat((encoder_out, pose),2) # Encode pose_features to the image features

            #Compute Loss & accuracy
            loss = criterion(output, y.view(-1,1).squeeze(1))
            acc = accuracy(output, y.view(-1,1))
            losses.update(loss.item(), X.size(0))
            top1.update(acc, X.size(0))

                    # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()


            if (batch_idx + 1) % log_interval == 0:
                # progress.display(batch_idx)
                current_iter = (batch_idx+1) + epoch*(len(val_loader))
                print(current_iter)
                print("Epoch: {}/{}...".format(epoch+1, params['epochs']),
                      "Step: [{}/{} ({:.0f}%)]...".format(N_count, len(val_loader.dataset), 100. * (batch_idx + 1) / len(val_loader)),
                      "Time {batch_time.val:.3f} ({batch_time.avg:.3f})".format(batch_time=batch_time),
                      "Val Loss {loss.val:.4f} ({loss.avg:.4f})".format(loss=losses),
                      "Val Accuracy {accuracy.val:.4f} ({accuracy.avg:.4f})".format(accuracy=top1))
                writer.add_scalar('val/loss_val', losses.avg,current_iter)
                writer.add_scalar('val/accuracy_val', top1.avg,current_iter)

def main():

    PATH_TO_LOGGING = '/home/mirshad7/habitat_imitation_learning/logger'
    save_model_path = '/home/mirshad7/hierarchical_imitation/learning_module/checkpoint'
    writer = SummaryWriter(PATH_TO_LOGGING)

    # EncoderCNN architecture
    CNN_fc_hidden1   = 256
    CNN_embed_dim    = 150      # latent dim extracted by 2D CNN
    dropout_p_CNN    = 0.3          # dropout probability
    pose_feature_dim = 72

    # DecoderRNN architecture
    RNN_hidden_layers = 3
    RNN_hidden_nodes = 100
    RNN_FC_dim = 50
    output_dim = 6
    dropout_p_RNN = 0.3

    # Detect devices
    img_x=224
    img_y=224
    use_cuda = torch.cuda.is_available()                   # check if GPU exists
    device = torch.device("cuda" if use_cuda else "cpu")   # use CPU or GPU
    params = {
        'lr': 1e-4,
        'batch_size': 15,
        'epochs': 30,
        'model': 'enoder_decoder'
    }

    #Expert Params
    num_scenes             = 72
    num_episodes_per_scene = 10
    min_distance           = 2
    max_distance           = 18
    val_split              = 0.2
    data_path_train        = 'data/datasets/pointnav/gibson/v1/all/training_batch_0.json.gz'
    data_path_val          = 'data/datasets/pointnav/gibson/v1/val/val.json.gz'
    scene_dir              = 'data/scene_datasets/'
    mode                   = "exact_gradient"
    config_path            = "configs/tasks/pointnav_gibson.yaml"

    num_traj_train = num_scenes*num_episodes_per_scene
    num_traj_val   = int(num_traj_train*val_split)
    
    dataloader_params = {'batch_size': params['batch_size'], 'shuffle': True, 'num_workers': 0, 'pin_memory': True} if use_cuda else {}
    log_interval  = 3   # interval for displaying training info
    transform = transforms.Compose([transforms.Resize([img_x, img_y]),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])])

    expert_train                = Expert(data_path_train, scene_dir, mode, config_path, transform)
    images_train, actions_train = expert_train.read_observations_and_actions(num_traj_train,min_distance, max_distance)

    expert_val                  = Expert(data_path_val, scene_dir, mode, config_path, transform)
    images_val, actions_val     = expert_train.read_observations_and_actions(num_traj_val,min_distance, max_distance)


    #Define dataset here
    train_set    = Dataset_RNN(images_train, actions_train)
    val_set      = Dataset_RNN(images_val, actions_val)    
    train_loader = data.DataLoader(train_set, **dataloader_params, collate_fn = pad_collate,drop_last=True)
    val_loader   = data.DataLoader(val_set, **dataloader_params, collate_fn = pad_collate,drop_last=True)


    print("==================================================================================")
    print("                    ...DATA LOADING DONE....                                      ")
    print("                    ...STARTING TRAIN LOOP....                                      ")
    print("==================================================================================")

    # Create model
    cnn_encoder = CNNEncoder(fc_hidden1=CNN_fc_hidden1, CNN_embed_dim=CNN_embed_dim, drop_p=dropout_p_CNN).to(device)
    rnn_decoder = DecoderRNN(embed_dim=CNN_embed_dim, h_RNN_layers=RNN_hidden_layers, 
                             num_hidden=RNN_hidden_nodes, h_FC_dim=RNN_FC_dim, drop_prob=dropout_p_RNN, 
                             num_classes=output_dim).to(device)

    crnn_params = list(cnn_encoder.fc1.parameters()) + list(cnn_encoder.bn1.parameters()) + \
              list(cnn_encoder.fc2.parameters()) + list(rnn_decoder.parameters())

    optimizer = torch.optim.Adam(crnn_params, lr=params['lr'])
    criterion = nn.CrossEntropyLoss(ignore_index=-1)


    #train model
    for epoch in range(params['epochs']):
        train(log_interval, [cnn_encoder, rnn_decoder],criterion, device, train_loader, optimizer, epoch, params['batch_size'], output_dim, params,writer)
        validate(log_interval, [cnn_encoder, rnn_decoder],criterion, device, val_loader, epoch, params['batch_size'],output_dim, params, writer)
        

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
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

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output, target):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        total = target[target!=-1].shape[0]
        _, pred = output.topk(1,1, True)
        pred    = pred[target!=-1]
        target  = target[target!=-1]
        correct = pred.eq(target).sum().cpu().numpy()
        res = correct*(100/total)
        return res

if __name__ == '__main__':
    main()