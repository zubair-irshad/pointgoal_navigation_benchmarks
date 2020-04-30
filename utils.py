import torch
import os
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid
import pandas as pd
from Project import Project

device = 'cuda' if torch.cuda.is_available() else 'cpu'
plt.interactive(False)


def show(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')
    plt.show()
    

def show_dataset(dataset, n=6):
    imgs = [dataset[i][0] for i in range(n)]
    # print(imgs[0].shape)
    grid = make_grid(imgs)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    plt.tight_layout()
    plt.show()

def show_dl(dl, n=6):
    batch = None
    for batch in dl:
        break
    imgs = batch[0][:n]
    grid = make_grid(imgs)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))
    plt.tight_layout()
    plt.show()

def show_one_batch(dl, rows, cols):
    dataiter = iter(dl)
    images, labels, xl , yl = dataiter.next()
    images,labels = images.to(device), labels.to(device)
    imgs=[]
    for i in range(rows):
        for j in range(cols):
            imgs.append(images[i,5*j,:,:])
            
    print(len(imgs))
    grid = make_grid(imgs,nrow=rows)
    plt.imshow(grid.cpu().numpy().transpose((1, 2, 0)))
    plt.tight_layout()
    plt.savefig('trainloader_grid.png')
    plt.show()

def max_seq_length():
    seq_length =[]
    project = Project()
    data_path = project.data_dir
    fnames = os.listdir(data_path)

    all_names = []
    for f in fnames:
        all_names.append(f)

    for i in range(len(all_names)):
        folder_dir = os.path.join(data_path, all_names[i])
        action_filename = os.path.join(folder_dir,'action.csv')
        df = pd.read_csv(action_filename, header=None)
        seq_length.append(len(df))
        max_sequence_length = np.max(seq_length)
        min_sequence_length = np.min(seq_length)

    return seq_length, max_sequence_length, min_sequence_length

def max_seq_length_list(train_list):
    seq_length =[]
    project = Project()
    data_path = project.data_dir
    for i in range(len(train_list)):
        folder_dir = os.path.join(data_path, train_list[i])
        action_filename = os.path.join(folder_dir,'action.csv')
        df = pd.read_csv(action_filename, header=None)
        seq_length.append(len(df))
        max_sequence_length = np.max(seq_length)
        min_sequence_length = np.min(seq_length)
    return seq_length, max_sequence_length, min_sequence_length

def feat_eng(data):
    data['totl_xyz'] = (data['pos_x']**2 + data['pos_y']**2 + data['pos_z']**2)**0.5
    data['totl_ang_xyz'] = (data['roll']**2 + data['pitch']**2 + data['yaw']**2)**0.5
    
    def mean_change_of_abs_change(x):
        return np.mean(np.diff(np.abs(np.diff(x))))
    
    for col in data.columns:
        data[col + '_mean'] = data[col].mean()
        data[col + '_median'] = data[col].median()
        data[col + '_max'] = data[col].max()
        data[col + '_min'] = data[col].min()
        data[col + '_std'] = data[col].std()
        data[col + '_range'] = data[col + '_max'] - data[col + '_min']
        data[col + '_maxtoMin'] = data[col + '_max'] / data[col + '_min']
        data[col + '_mean_abs_change'] = data[col].diff()
#         data[col + '_mean_change_of_abs_change'] = data[col].apply(mean_change_of_abs_change)
    return data

def feature_engineering_poses(data_frame):
    data_frame[data_frame.columns[-3:]]= data_frame[data_frame.columns[-3:]]*np.pi/180
    data_frame['roll'] =  data_frame['roll'].abs()

    #Basic features
    data_frame = feat_eng(data_frame)
    data_frame = data_frame.fillna(0)
    data_frame = data_frame.values

    return data_frame

def save_checkpoint(state, is_best, filename='/output/checkpoint.pth.tar'):
    """Save checkpoint if a new best is achieved"""
    if is_best:
        print ("=> Saving a new best")
        torch.save(state, filename)  # save checkpoint
    else:
        print ("=> Validation Accuracy did not improve")



