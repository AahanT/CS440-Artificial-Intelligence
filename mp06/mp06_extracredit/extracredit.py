import torch, random, math, json
import numpy as np
from extracredit_embedding import ChessDataset, initialize_weights

DTYPE=torch.float32
DEVICE=torch.device("cpu")

###########################################################################################
def trainmodel():
    # Well, you might want to create a model a little better than this...
    model = torch.nn.Sequential(
        torch.nn.Conv2d(15, 64, kernel_size=3,stride=1, padding=1),
        torch.nn.ReLU(),
        torch.nn.MaxPool2d(kernel_size=2,stride=2),
        # torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
        # torch.nn.ReLU(),
        # torch.nn.MaxPool2d(kernel_size=2, stride=2),
        torch.nn.Flatten(),
        torch.nn.Linear(64*2*2,128),
        torch.nn.ReLU(),
        torch.nn.Linear(128,1),
        torch.nn.ReLU()

        )

    # ... and if you do, this initialization might not be relevant any more ...
    # model[1].weight.data = initialize_weights()
    # model[1].bias.data = torch.zeros(1)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # ... and you might want to put some code here to train your model:
    trainset = ChessDataset(filename='extracredit_train.txt')
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1000, shuffle=True)
    for epoch in range(2000):
        for x,y in trainloader:
            #pass # Replace this line with some code that actually does the training
            prediction = model(x)
            loss = loss_fn(prediction, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


    # ... after which, you should save it as "model.pkl":
    torch.save(model, 'model.pkl')


###########################################################################################
if __name__=="__main__":
    trainmodel()
    
