#%% 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import sympy as asm
from torch import Tensor
from IPython.display import display, Math
from torch import nn
from typing import List,Dict,Tuple
print(torch.__version__)

#%%

def createDummyData(weight : float, bias : float, split : float = 0.2) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    weight : float = weight
    bias : float = bias

    # Create data
    start : int = 0
    end  : int = 2
    step : float = 0.02

    X : Tensor = torch.arange(start, end, step)
    Y : Tensor = weight*X + bias

    train_split : int =  int((1 - split) * len(X))
    x_train,y_train = X[:train_split], Y[:train_split]
    x_test,y_test=X[train_split:], Y[train_split:]

    print(len(x_train), len(y_train), len(x_test), len(y_test))    
    return x_train, y_train,x_test,y_test

def plotPrediction( train_data : Tensor, train_labels : Tensor,
                    test_data : Tensor, test_labels : Tensor,
                    predictions : Tensor = None):
    
    plt.figure(figsize=(10,8))
    plt.scatter(train_data,train_labels,c="b",s=4, label="Training data")
    plt.scatter(test_data,test_labels,c="g", s=4,label="Testing data")
    if predictions is not None:
        plt.scatter(test_data, predictions, c="r", s=4, label="Prediction data")
    plt.legend(prop={"size":15})
    plt.show()

def plotLoss(y_train_loss : List[np.float32],y_test_loss : List[np.float32], epochs : List[int] ):
    plt.plot(epochs, y_train_loss, label="TrainingLoss")
    plt.plot(epochs, y_test_loss, label="TestingLoss")
    plt.legend()
    plt.show()
# %%
class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(
                                    torch.randn(1, dtype=torch.float),
                                    requires_grad=True ) 
        self.bias = nn.Parameter(torch.randn(1, dtype=torch.float,
                                    requires_grad=True))
    def forward(self, x: Tensor) -> Tensor:
        return self.weights*x + self.bias

    def fit(self,lossFunction : torch.nn,
            optimizer : torch.optim,
            epochs : int ,
            x_train : Tensor,
            y_train : Tensor,
            x_test : Tensor,
            y_test : Tensor,
            
            ):

        train_loss_value : List[Tensor] = []
        test_loss_value : List[Tensor] = []
        epoch_count : List[int] = []

        for epoch in range(epochs):
            epoch = epoch+1
            modelLinearRegression.train() # Put it to training mode
            y_train_pred : Tensor = modelLinearRegression(x_train)
            
            loss : Tensor = lossFunction(y_train_pred, y_train)

            #update gradient
            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            with torch.inference_mode():
                yPred : Tensor = modelLinearRegression(x_test)

                test_loss : Tensor = lossFunction(yPred, y_test)

                if epoch % 10 == 0:
                    epoch_count.append(epoch)
                    train_loss_value.append(loss.detach().numpy())
                    test_loss_value.append(test_loss.detach().numpy())
                    print(f"Epoch: {epoch} | MAE Train Loss: {loss} | MAE Test Loss: {test_loss} ")            
        
        return train_loss_value, test_loss_value, epoch_count
if __name__ =="__main__":
    torch.manual_seed(54)
    x_train,y_train,x_test,y_test = createDummyData(0.9, 0.2)
    plotPrediction(x_train,y_train,x_test,y_test)
    modelLinearRegression : LinearRegression = LinearRegression()
    print(modelLinearRegression.state_dict())
    
    # No training yet, just using some random numbers

    # Predicting
    with torch.inference_mode():
        predictions : Tensor = modelLinearRegression(x_test)

    ## Will lookalike useless
    plotPrediction(x_train,y_train,x_test,y_test, predictions)

    # Check the predictions
    print(f"Number of testing samples: {len(x_test)}") 
    print(f"Number of predictions made: {len(predictions)}")
    print(f"Predicted values:\n{predictions}")
    print(f"Different { torch.sqrt(x_test - predictions)**2}") # High RMSE


    # Now lets do training
    # Define the loss function and optimizer
    learning_rate : float = 0.01
    epochs : int = 1000
    loss : torch.nn = torch.nn.L1Loss()
    optimizer : torch.optim = torch.optim.SGD(params=modelLinearRegression.parameters(),
                                             lr =learning_rate)
    train_losses, testing_losses, epoch_counts = modelLinearRegression.fit(loss, optimizer,1000,x_train,y_train,x_test,y_test)
    #%%
    plotLoss(train_losses, testing_losses, epoch_counts)
