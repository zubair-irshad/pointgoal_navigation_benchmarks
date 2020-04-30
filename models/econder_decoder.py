import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

class CNNEncoder(nn.Module):
    def __init__(self, fc_hidden1=256, CNN_embed_dim=150,drop_p=0.3):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(CNNEncoder, self).__init__()
        self.fc_hidden1 = fc_hidden1
        self.drop_p     = drop_p
        self.use_cuda   = torch.cuda.is_available()  
        self.device     = torch.device("cuda" if self.use_cuda else "cpu")   # use CPU or GPU

        resnet = models.resnet18(pretrained=True)
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)

        for param in self.resnet.parameters():
            param.requires_grad = False

        self.fc1 = nn.Linear(resnet.fc.in_features, fc_hidden1)
        self.bn1 = nn.BatchNorm1d(fc_hidden1, momentum=0.01)
        self.fc2 = nn.Linear(fc_hidden1, CNN_embed_dim)
        
    def forward(self, images):
        cnn_embed_seq = []
        test_out=[]
        for t in range(images.size(1)):
            # ResNet CNN
            with torch.no_grad():
                x = self.resnet(images[:, t, :, :, :])  # ResNet
                x = x.view(x.size(0), -1)             # flatten output of conv

            #Add position (states and velocity information here)
            # FC layers
            x = self.bn1(self.fc1(x))
            x = F.relu(x)
            x = F.dropout(x, p=self.drop_p, training=self.training)
            x = self.fc2(x)
            cnn_embed_seq.append(x)
        # swap time and sample dim such that (sample dim, time dim, CNN latent dim)
        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)

        return cnn_embed_seq

class DecoderRNN(nn.Module):
    def __init__(self,embed_dim=300, h_RNN_layers=3, num_hidden=256, h_FC_dim=128, drop_prob=0.3, num_classes=6):
        super(DecoderRNN, self).__init__()

        self.RNN_input_size  = embed_dim
        self.num_hidden      = num_hidden
        self.RNN_layers      = h_RNN_layers
        self.h_FC_dim        = h_FC_dim
        self.drop_p          = drop_prob
        self.num_classes     = num_classes
        self.lstm = nn.LSTM(input_size  = self.RNN_input_size, 
                            hidden_size = self.num_hidden, 
                            num_layers  = self.RNN_layers,batch_first = True)
        self.dropout=nn.Dropout(p=0.3)
        self.fc1 = nn.Linear(self.num_hidden, self.h_FC_dim)
        self.fc2 = nn.Linear(self.h_FC_dim, self.num_classes)
        self.use_cuda = torch.cuda.is_available()  
    
    def forward(self,x, hidden, lengths):
        # self.lstm.flatten_parameters()

        encoder_packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        output_packed,hidden = self.lstm(encoder_packed, hidden)
        lstm_out, _ = pad_packed_sequence(output_packed, batch_first=True)
        lstm_out        = lstm_out.contiguous().view(-1,self.num_hidden)

        #Fully connected layer
        out             = self.fc1(lstm_out)
        out             = F.relu(out)
        out             = F.dropout(out, p=self.drop_p)
        out             = self.fc2(out)

        return out, hidden  

    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data

        if (self.use_cuda):
            hidden = (weight.new(self.RNN_layers, batch_size, self.num_hidden).zero_().cuda(),
                  weight.new(self.RNN_layers, batch_size, self.num_hidden).zero_().cuda())
        else:
            hidden = (weight.new(self.RNN_layers, batch_size, self.num_hidden).zero_(),
                      weight.new(self.RNN_layers, batch_size, self.num_hidden).zero_())

        return hidden
    