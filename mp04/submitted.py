import torch
import torch.nn as nn


"""
1.  Build a neural network class.
"""
class NeuralNet(torch.nn.Module):
    def __init__(self):
        """
        Initialize the neural network here.
        """
        super().__init__()
        
        self.cnn1 = torch.nn.Conv2d(3, 100, 35, 35)
        self.hidden = torch.nn.Linear(100, 100)
        self.relu = torch.nn.ReLU()
        self.output = torch.nn.Linear(100, 5)
        self.unflatten = torch.nn.Unflatten(1, (3, 31, 31))
        self.flatten = torch.nn.Flatten(1, 3)
       

    def forward(self, x):
        """
        Perform a forward pass through the neural net.

        Parameters:
            x:      an (N, input_size) tensor, where N is arbitrary.

        Outputs:
            y:      an (N, output_size) tensor of output from the network
        """

        x2 = self.unflatten(x)
        x2 = self.cnn1(x2)
        x2 = self.flatten(x2)
        x_temp = self.hidden(x2)             # input data x flows through the hidden layer
        x_temp = self.relu(x_temp)          # use relu as the activation function for intermediate data x_temp 
        y_pred = self.output(x_temp) 
        return y_pred
        


"""
2. Training my model.
"""
def fit(train_dataloader, test_dataloader, epochs):
    """

    Parameters:
        train_dataloader:   a dataloader for the training set and labels
        test_dataloader:    a dataloader for the testing set and labels
        epochs:             the number of times to iterate over the training set

    Outputs:
        model:              trained model
        loss_fn:            your selected loss function
        optimizer:          your selected optimizer
    """
    
    model = NeuralNet()

    """
    2.1 Create a loss function and an optimizer.

    Selected CrossEntropyLoss as the loss function loss function from PyTorch torch.nn module.
    Select SGD as the optimizer from PyTorch torch.optim module.
    """
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum = 0.9)


    """
    2.2 Train loop
    """
    for epoch in range(epochs):
        print("Epoch #", epoch)
        train(train_dataloader, model, loss_fn, optimizer) 
    return model, loss_fn, optimizer


"""
3. Backward propagation and gradient descent.
"""
def train(train_dataloader, model, loss_fn, optimizer):
    """
    Training the neural network.

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
        prediction = model(features)
        loss = loss_fn(prediction, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

