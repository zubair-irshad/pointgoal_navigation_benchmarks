import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

class ResCNNEncoder(nn.Module):
    def __init__(self, fc_hidden1=512, fc_hidden2=512,fc_hidden3=512, drop_p=0.3, CNN_embed_dim=300, lengths= None):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(ResCNNEncoder, self).__init__()

        self.fc_hidden1, self.fc_hidden2 = fc_hidden1, fc_hidden2
        self.drop_p = drop_p
        self.multiplier =1
        self.lengths = lengths
        self.use_cuda = torch.cuda.is_available()  
        self.device = torch.device("cuda" if self.use_cuda else "cpu")   # use CPU or GPU

        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        ################################
        #######################
        #Change self.multiplier here to match the output pool size after the last layer. With input image size (224,224), self.multiplier =1

        self.fc1 = nn.Linear(resnet.fc.in_features*self.multiplier, fc_hidden1)
        self.bn1 = nn.BatchNorm1d(fc_hidden1, momentum=0.01)
        self.fc2 = nn.Linear(fc_hidden1, fc_hidden2)
        self.bn2 = nn.BatchNorm1d(fc_hidden2, momentum=0.01)
        self.fc3 = nn.Linear(fc_hidden2, fc_hidden3)
        self.bn3 = nn.BatchNorm1d(fc_hidden3, momentum=0.01)
        self.fc4 = nn.Linear(fc_hidden3, CNN_embed_dim)

    # def create_mask(self, batchsize, max_length, length):
    #     """Given the length create a mask given a padded tensor"""
    #     tensor_mask = torch.zeros(batchsize, max_length, dtype = torch.bool)
    #     for idx, row in enumerate(tensor_mask):
    #         row[:length[idx]] = 1

    #     tensor_mask.unsqueeze_(-1)
    #     return tensor_mask.to(self.device)
        
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
            x = self.bn2(self.fc2(x))
            x = F.relu(x)
            x = F.dropout(x, p=self.drop_p, training=self.training)
            x = self.bn3(self.fc3(x))
            x = F.relu(x)
            x = self.fc4(x)

            cnn_embed_seq.append(x)


        # swap time and sample dim such that (sample dim, time dim, CNN latent dim)
        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)
        
        # embed_mask = self.create_mask(cnn_embed_seq.shape[0],cnn_embed_seq.shape[1], self.lengths)
        # embed_mask = embed_mask.expand_as(cnn_embed_seq)
        # cnn_embed_seq = cnn_embed_seq*embed_mask
        # cnn_embed_seq: shape=(batch, time_step, input_size)

        return cnn_embed_seq

class DecoderRNN(nn.Module):
    def __init__(self,embed_dim=300, h_RNN_layers=3, num_hidden=256, h_FC_dim=128, drop_prob=0.3, output_dim=2):
        super(DecoderRNN, self).__init__()

        self.RNN_input_size = embed_dim
        self.num_hidden     = num_hidden
        self.RNN_layers     = h_RNN_layers
        self.h_FC_dim       = h_FC_dim
        self.drop_p         = drop_prob
        self.output_dim     = output_dim
        self.lstm = nn.LSTM(input_size  = self.RNN_input_size, 
                            hidden_size = self.num_hidden, 
                            num_layers  = self.RNN_layers,batch_first = True)
        self.dropout=nn.Dropout(p=0.3)
        self.fc1 = nn.Linear(self.num_hidden, self.h_FC_dim)
        self.fc2 = nn.Linear(self.h_FC_dim, self.output_dim)
        self.use_cuda = torch.cuda.is_available()  
    
    def forward(self,x,hidden, lengths):
        # self.lstm.flatten_parameters()

        encoder_packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        output_packed,hidden = self.lstm(encoder_packed, hidden)
        lstm_out, _ = pad_packed_sequence(output_packed, batch_first=True)
        lstm_out        = lstm_out.contiguous().view(-1,self.num_hidden)

        #Fully connected layer
        out             = self.dropout(lstm_out)
        out             = self.fc1(out)
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
    



        
# class NetworkNvidia(nn.Module):
#     """NVIDIA model used in the paper."""

#     def __init__(self):
#         """Initialize NVIDIA model.
#         NVIDIA model used
#             Image normalization to avoid saturation and make gradients work better.
#             Convolution: 5x5, filter: 24, strides: 2x2, activation: ELU
#             Convolution: 5x5, filter: 36, strides: 2x2, activation: ELU
#             Convolution: 5x5, filter: 48, strides: 2x2, activation: ELU
#             Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
#             Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
#             Drop out (0.5)
#             Fully connected: neurons: 100, activation: ELU
#             Fully connected: neurons: 50, activation: ELU
#             Fully connected: neurons: 10, activation: ELU
#             Fully connected: neurons: 1 (output)
#         the convolution layers are meant to handle feature engineering.
#         the fully connected layer for predicting the steering angle.
#         the elu activation function is for taking care of vanishing gradient problem.
#         """
#         super(NetworkNvidia, self).__init__()
#         self.conv_layers = nn.Sequential(
#             nn.Conv2d(3, 24, 5, stride=2),
#             nn.ELU(),
#             nn.Conv2d(24, 36, 5, stride=2),
#             nn.ELU(),
#             nn.Conv2d(36, 48, 5, stride=2),
#             nn.ELU(),
#             nn.Conv2d(48, 64, 3),
#             nn.ELU(),
#             nn.Conv2d(64, 64, 3),
#             nn.Dropout(0.5)
#         )
#         self.linear_layers = nn.Sequential(
#             nn.Linear(in_features=64 * 2 * 33, out_features=100),
#             nn.ELU(),
#             nn.Linear(in_features=100, out_features=50),
#             nn.ELU(),
#             nn.Linear(in_features=50, out_features=10),
#             nn.Linear(in_features=10, out_features=1)
#         )

#     def forward(self, input):
#         """Forward pass."""
#         input = input.view(input.size(0), 3, 70, 320)
#         output = self.conv_layers(input)
#         # print(output.shape)
#         output = output.view(output.size(0), -1)
#         output = self.li