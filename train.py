# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
from callbacks import CometCallback
from comet_ml import Experiment

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
from data import get_dataloaders
from data import dataset_utils
from utils import show_dataset,max_seq_length_list,device, show_one_batch, save_checkpoint
from models import ResCNNEncoder, DecoderRNN

import numpy as np
import os
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence

def pad_collate(batch):
  (xx, yy, zz) = zip(*batch)
  x_lens = [len(x) for x in xx]
  y_lens = [len(y) for y in yy]
  z_lens = [len(z) for z in zz]

  xx_pad = pad_sequence(xx, batch_first=True, padding_value=0)
  yy_pad = pad_sequence(yy, batch_first=True, padding_value=0)
  zz_pad = pad_sequence(zz, batch_first=True, padding_value=0)

  return xx_pad, yy_pad, zz_pad, x_lens, y_lens, z_lens


def create_mask(batchsize, max_length, length,device):
    """Given the length create a mask given a padded tensor"""
    tensor_mask = torch.zeros(batchsize, max_length, dtype = torch.bool)
    for idx, row in enumerate(tensor_mask):
        row[:length[idx]] = 1

    tensor_mask.unsqueeze_(-1)
    return tensor_mask.to(device)



def train(log_interval, model,criterion, device, train_loader, val_loader, optimizer, epoch, batch_size):
    # set model as training mode
    cnn_encoder, rnn_decoder = model
    cnn_encoder.train()
    rnn_decoder.train()
    val_losses = []
    train_losses = []
    min_val_loss = 1000
    N_count = 0   # counting total trained sample in one epoch

    h = rnn_decoder.init_hidden(batch_size)

    for batch_idx, (X, y, pose, x_lengths,_,_) in enumerate(train_loader):
        h = tuple([each.detach() for each in h])
        # distribute data to device
        X, y, pose = X.to(device), y.to(device), pose.to(device)
        N_count += X.size(0)
        # Create new variables for hidden state so we do not backprop the entire history
        optimizer.zero_grad()
        

        encoder_out = cnn_encoder(X)
        embed_mask = create_mask(encoder_out.shape[0],encoder_out.shape[1], x_lengths, device)
        embed_mask = embed_mask.expand_as(encoder_out)
        encoder_out = encoder_out*embed_mask

        encoder_out = torch.cat((encoder_out, pose),2) # Encode pose_features to the image features

        output,h = rnn_decoder(encoder_out, h, x_lengths)   # output has dim = (batch*seq_length, number of outputs)

        loss = criterion(output, y.view(-1,y.shape[2]).float())
        # # to compute accuracy
        # y_pred = torch.max(output, 1)[1]  # y_pred != output
        # step_score = accuracy_score(y.cpu().data.squeeze().numpy(), y_pred.cpu().data.squeeze().numpy())
        # scores.append(step_score)         # computed on CPU

        loss.backward()
        # nn.utils.clip_grad_norm_(rnn_decoder.parameters(), clip)
        optimizer.step()

        # show information
        if (batch_idx + 1) % log_interval == 0: 
            # Get validation loss
            val_h = rnn_decoder.init_hidden(batch_size)
            cnn_encoder.eval()
            rnn_decoder.eval()
            with torch.no_grad():
                for X, y, pose, x_lengths,_,_ in val_loader:
                    val_h = tuple([each.data for each in val_h])
                    # distribute data to device
                    X, y, pose  = X.to(device), y.to(device), pose.to(device)
                    encoder_out = cnn_encoder(X)
                    embed_mask  = create_mask(encoder_out.shape[0],encoder_out.shape[1], x_lengths,device)
                    embed_mask  = embed_mask.expand_as(encoder_out)
                    encoder_out = encoder_out*embed_mask
                    encoder_out = torch.cat((encoder_out, pose),2) # Encode pose_features to the image features

                    output,val_h = rnn_decoder(encoder_out, val_h, x_lengths)   # output has dim = (batch*seq_length, number of outputs)

                    val_loss = criterion(output, y.view(-1,y.shape[2]).float())
                    val_losses.append(val_loss.item())
            
            cnn_encoder.train()
            rnn_decoder.train()

            print("Epoch: {}/{}...".format(epoch+1, params['epochs']),
                  "Step: [{}/{} ({:.0f}%)]...".format(N_count, len(train_loader.dataset), 100. * (batch_idx + 1) / len(train_loader)),
                  "Loss: {:.6f}...".format(loss.item()),
                  "Val Loss: {:.6f}".format(np.mean(val_losses)))

    
    # is_best = bool(np.mean(val_losses) < min_val_loss)
    # min_val_loss = min(np.mean(val_losses), min_val_loss)
    # ckpt_filename = os.path.join(save_model_path, 'Enocder_Decoder_epoch{}.pth'.format(epoch + 1))
    # save_checkpoint({'cnn_encoder_state_dict': cnn_encoder.state_dict(),
    #                  'rnn_decoder_state_dict': rnn_decoder.state_dict(),
    #                  'optimizer': optimizer.state_dict()}, 
    #                   is_best, ckpt_filename)




    writer.add_scalar('Loss',loss.item(),epoch) # do the same as val_losses
    writer.add_scalar('Mean_Val_Loss',np.mean(val_losses),epoch)

    return train_losses, val_losses

def validation(model, criterion, device, optimizer, test_loader):
    # set model as testing mode
    cnn_encoder, rnn_decoder = model
    cnn_encoder.eval()
    rnn_decoder.eval()

    test_loss = 0
    all_y = []
    all_y_pred = []
    with torch.no_grad():
        for X, y in test_loader:
            # distribute data to device
            X, y = X.to(device), y.to(device)
            batch_size = X.shape[0]
            h = rnn_decoder.init_hidden(batch_size)
            output, h = rnn_decoder(cnn_encoder(X),h)

            loss = criterion(output, y.view(-1,y.shape[2]).float())
            test_loss += loss.item()                 # sum up batch loss
            # y_pred = output.max(1, keepdim=True)[1]  # (y_pred != output) get the index of the max log-probability

            # # collect all y and y_pred in all batches
            # all_y.extend(y)
            # all_y_pred.extend(y_pred)

    test_loss /= len(test_loader.dataset)

    # # compute accuracy
    # all_y = torch.stack(all_y, dim=0)
    # all_y_pred = torch.stack(all_y_pred, dim=0)
    # test_score = accuracy_score(all_y.cpu().data.squeeze().numpy(), all_y_pred.cpu().data.squeeze().numpy())

    # show information
    print('\nTest set ({:d} samples): Average loss: {:.4f}\\n'.format(len(train_loader.dataset), test_loss))

    # save Pytorch models of best record
    # torch.save(cnn_encoder.state_dict(), os.path.join(save_model_path, 'cnn_encoder_epoch{}.pth'.format(epoch + 1)))  # save spatial_encoder
    # torch.save(rnn_decoder.state_dict(), os.path.join(save_model_path, 'rnn_decoder_epoch{}.pth'.format(epoch + 1)))  # save motion_encoder
    # torch.save(optimizer.state_dict(), os.path.join(save_model_path, 'optimizer_epoch{}.pth'.format(epoch + 1)))      # save optimizer
    # print("Epoch {} model saved!".format(epoch + 1))
    return test_loss

if __name__ == '__main__':

    PATH_TO_LOGGING = '/home/mirshad7/hierarchical_imitation/learning_module/logging/GibsontinyandMedium'
    save_model_path = '/home/mirshad7/hierarchical_imitation/learning_module/checkpoint'
    writer = SummaryWriter(PATH_TO_LOGGING)

    # EncoderCNN architecture
    CNN_fc_hidden1, CNN_fc_hidden2,CNN_fc_hidden3 = 1500,1024,512
    CNN_embed_dim = 300      # latent dim extracted by 2D CNN
    img_x, img_y = 224,224 #Specify crop image sizes here 
    dropout_p_CNN = 0.3          # dropout probability
    pose_feature_dim = 72

    # DecoderRNN architecture
    RNN_hidden_layers = 3
    RNN_hidden_nodes = 128
    RNN_FC_dim = 64
    output_dim =2
    dropout_p_RNN = 0.3          # dropout probability

    # Detect devices
    use_cuda = torch.cuda.is_available()                   # check if GPU exists
    device = torch.device("cuda" if use_cuda else "cpu")   # use CPU or GPU

    project = Project()
    # our hyperparameters
    # log_interval =30
    params = {
        'lr': 1e-4,
        'batch_size': 20,
        'epochs': 30,
        'model': 'enoder_decoder'
    }
    dataloader_params = {'batch_size': params['batch_size'], 'shuffle': True, 'num_workers': 0, 'pin_memory': True} if use_cuda else {}
    log_interval  = 3   # interval for displaying training info
   
    
    # logging.info(f'Using device={device} 🚀')

    #Preparing data for datalaoding


    begin_frame, end_frame, skip_frame = 3, 20, 1
    selected_frames = np.arange(begin_frame, end_frame, skip_frame).tolist()

    transform = transforms.Compose([transforms.Resize([img_x, img_y]),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])])

    #Load directory names here
    data_path = project.data_dir
    fnames = os.listdir(data_path)
    all_names = []
    for f in fnames:
        all_names.append(f)


    all_X_list = all_names    
    train_list, test_list = train_test_split(all_X_list,test_size=0.20, random_state=42)
    train_list,val_list = train_test_split(train_list, test_size=0.25, random_state=1)

    train_seq_length, max_tsq_length, min_tsq_length = max_seq_length_list(train_list)
    val_seq_length, max_vsq_length, min_vsq_length = max_seq_length_list(val_list)
    test_seq_length, max_tssq_length, min_tssq_length = max_seq_length_list(test_list)



    train_set, val_set,test_set = dataset_utils.Dataset_RNN(data_path, train_list, train_seq_length, transform=transform), \
                                  dataset_utils.Dataset_RNN(data_path, val_list  , val_seq_length  , transform=transform), \
                                  dataset_utils.Dataset_RNN(data_path, test_list , test_seq_length , transform=transform), \


    train_loader = data.DataLoader(train_set, **dataloader_params, collate_fn = pad_collate, drop_last=True)
    val_loader   = data.DataLoader(val_set, **dataloader_params, collate_fn = pad_collate,drop_last=True)
    test_loader  = data.DataLoader(test_set, **dataloader_params, collate_fn = pad_collate,drop_last=True)

    #SHOW TRAINING LOADER HERE
    # show_one_batch(train_loader,5,5)
    # show_one_batch(valid_loader,5,5)

    # define our comet experiment
    # experiment = Experiment(api_key="QpL4OYD7n2TtWST5joo7FYJlG",
    #                         project_name="hierarhical-imitation", workspace="mirshad7")
    # experiment.log_parameters(params)


    # Create model
    cnn_encoder = ResCNNEncoder(fc_hidden1=CNN_fc_hidden1, fc_hidden2=CNN_fc_hidden2, drop_p=dropout_p_CNN, CNN_embed_dim=CNN_embed_dim).to(device)
    rnn_decoder = DecoderRNN(embed_dim=CNN_embed_dim+pose_feature_dim, h_RNN_layers=RNN_hidden_layers, num_hidden=RNN_hidden_nodes, h_FC_dim=RNN_FC_dim, drop_prob=dropout_p_RNN, output_dim=output_dim).to(device)
    
    crnn_params = list(cnn_encoder.parameters()) + list(rnn_decoder.parameters())
    optimizer = torch.optim.Adam(crnn_params, lr=params['lr'])
    criterion = nn.MSELoss()


    #train model
    epoch_train_losses=[]
    epoch_test_losses=[]
    for epoch in range(params['epochs']):
        t_losses, v_losses     = train(log_interval, [cnn_encoder, rnn_decoder],criterion, device, train_loader, val_loader, optimizer, epoch, params['batch_size'])
        
    # # get the results on the test set
    # logging.info(f'test_acc=({test_acc})')
    # experiment.log_metric('test_acc', test_acc)