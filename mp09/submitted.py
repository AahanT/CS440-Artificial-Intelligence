import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import pickle 
from models import resnet18


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


"""
1.  Defining and building a PyTorch Dataset
"""
class CIFAR10(Dataset):
    def __init__(self, data_files, transform=None, target_transform=None):
        """
        Initialize the dataset here. Note that transform and target_transform
        correspond to the data transformations for train and test respectively.
        """
        self.transform = transform
        self.target_transform = target_transform
        self.data = []
        self.labels = []
        for i in range(len(data_files)):
            temp = unpickle(data_files[i])
            self.data += temp[b"data"]
            self.labels += temp[b"labels"]

    def __len__(self):
        """
        Return the length of the dataset here.
        """
        return len(self.labels)

    def __getitem__(self, idx):
        """
        Obtain a sample from the dataset. 

        Parameters:
            x:      an integer, used to index into the data.

        Outputs:
            y:      a tuple (image, label), although this is arbitrary so we can use whatever we would like.
        """
        temp = [self.data[idx].reshape(3,32,32), self.labels[idx]]
        temp[0] = torch.Tensor.float(torch.tensor((temp[0])))
        return tuple(temp)

    

def get_preprocess_transform(mode):
    """
    Parameters:
        mode:           "train" or "test" mode to obtain the corresponding transform
    Outputs:
        transform:      a torchvision transforms object e.g. transforms.Compose([...]) etc.
    """

    transform_train = transforms.Compose([transforms.ToTensor()])

    transforms_test = transforms.Compose([transforms.ToTensor()])

    if mode == "train":
        return transform_train
    if mode == "test":
        return transforms_test



def build_dataset(data_files, transform=None):
    """
    Parameters:
        data_files:      a list of strings e.g. "cifar10_batches/data_batch_1" corresponding to the CIFAR10 files to load data
        transform:       the preprocessing transform to be used when loading a dataset sample
    Outputs:
        dataset:      a PyTorch dataset object to be used in training/testing
    """
    
    return CIFAR10(data_files,transform)





"""
2.  Building a PyTorch DataLoader
"""
def build_dataloader(dataset, loader_params):
    """
    Parameters:
        dataset:         a PyTorch dataset to load data
        loader_params:   a dict containing all the parameters for the loader. 
        
    Please ensure that loader_params contains the keys "batch_size" and "shuffle" corresponding to those 
    respective parameters in the PyTorch DataLoader class. 

    Outputs:
        dataloader:      a PyTorch dataloader object to be used in training/testing
    """
    dataloader = DataLoader(dataset, loader_params["batch_size"], loader_params["shuffle"])
    return dataloader


"""
3. (a) Building a neural network class.
"""
class FinetuneNet(torch.nn.Module):
    def __init__(self):
        """
        Initializing the neural network here. We will be performing finetuning
        in this network by follow these steps:
        
        1. Initialize convolutional backbone with pretrained model parameters.
        2. Freeze convolutional backbone.
        3. Initialize linear layer(s). 
        """
        super().__init__()


        self.model = resnet18(pretrained=True)
        self.model.load_state_dict(torch.load('resnet18.pt'))
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.fc = torch.nn.Linear(512,8)


    def forward(self, x):
        """
        Perform a forward pass through the neural net.

        Parameters:
            x:      an (N, input_size) tensor, where N is arbitrary.

        Outputs:
            y:      an (N, output_size) tensor of output from the network
        """

        y = self.model(x)
        return y



"""
3. (b)  Build a model
"""
def build_model(trained=False):
    """
    Parameters:
        trained:         a bool value specifying whether to use a model checkpoint

    Outputs:
        model:           the model to be used for training/testing
    """

    net = FinetuneNet()
    return net


"""
4.  Build a PyTorch optimizer
"""
def build_optimizer(optim_type, model_params, hparams):
    """
    Parameters:
        optim_type:      the optimizer type e.g. "Adam" or "SGD"
        model_params:    the model parameters to be optimized
        hparams:         the hyperparameters (dict type) for usage with learning rate 

    Outputs:
        optimizer:       a PyTorch optimizer object to be used in training
    """
    optimizer = torch.optim.SGD(params=model_params, lr = 0.09)
    return optimizer


"""
5. Training loop for model
"""
def train(train_dataloader, model, loss_fn, optimizer):
    """
    Train the neural network.

    Iterate over all the batches in dataloader:
        1.  The model makes a prediction.
        2.  Calculate the error in the prediction (loss).
        3.  Zero the gradients of the optimizer.
        4.  Perform backpropagation on the loss.
        5.  Step the optimizer.

    Parameters:
        train_dataloader:   a dataloader for the training set and labels
        model:              the model to be trained
        loss_fn:            loss function
        optimizer:          optimizer
    """

    for features, labels in train_dataloader:
        y_pred = model(features)
        loss = loss_fn(y_pred, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


"""
6. Full model training and testing
"""
def run_model():
    """
    Running full model training and testing within this function.

    Outputs:
        model:              trained model
    """
    data_files = ["cifar10_batches/data_batch_1"]
    dataset = build_dataset(data_files, transform=transforms.ToTensor())
    loader_params = {"batch_size": 64, "shuffle": True}
    dataloader = build_dataloader(dataset, loader_params)
    model = build_model()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = build_optimizer("SGD", model.parameters(), {})
    train(dataloader, model, loss_fn, optimizer)
    return model
    
