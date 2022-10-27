from pickletools import optimize
import tensorflow as tf
from preprocess import sgf_to_y, vectorisation_img, vectorisation_sgf
from sklearn.model_selection import train_test_split
import numpy as np
import torch
import sklearn.model_selection
import matplotlib.pyplot as plt
from torchinfo import summary

def data_load(nb_sgf,heigh,lenght):
    y=np.concatenate([sgf_to_y('sgf_train/train_sgf'+str(k+1)+'.sgf') for k in range(nb_sgf)])
    x=np.concatenate([vectorisation_img('img_train/train_img_'+str(k+1)+'.png',heigh,lenght) for k in range(nb_sgf)])
    return torch.tensor(x),torch.tensor(y)

class model_img_to_sgf(torch.nn.Module):

    def __init__(self,entry_size):
        super().__init__()
        self.linbranch1 = torch.nn.Linear(entry_size, 40)
        self.linbranch2 = torch.nn.Linear(40, 361)

    def forward(self, x):
        z1 = self.linbranch1(x)
        z1 = torch.nn.ReLU()(z1)
        z2 = self.linbranch2(z1)
        y_pred = torch.nn.ReLU()(z2)
        return y_pred


if __name__ == '__main__':
    # Hyperparameters
    heigh=100
    lenght=100
    m=1000
    test_size=0.2
    epochs=10
    learnig_rate=1e-3
    #Data and model loading
    x,y=data_load(m,heigh,lenght)
    x_train,x_test,y_train,y_test=sklearn.model_selection.train_test_split(x,y,test_size=test_size)
    x_train,y_train,x_test,y_test=x_train.float(),y_train.float(),x_test.float(),y_test.float()
    print(x_train.shape,y_train.shape)
    model=model_img_to_sgf(3*heigh*lenght)
    lossfn=torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learnig_rate)
    
    summary(model, input_size=[(1,3*heigh*lenght)])
    #Training and testing
    training_loss_list=[]
    testing_loss_list=[]
    n_train=x_train.shape[0]
    n_test=x_test.shape[0]
    for ep in range(epochs):
        model.train()
        loss_sum=0
        for i in range(n_train):
            y_pred=model(x_train[i,:])
            loss=lossfn(y_train[i,:],y_pred)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_sum+=loss.item()
        training_loss_list.append(loss_sum/n_train)
        
        loss_sum=0
        model.eval()
        with torch.no_grad():
            for i in range(n_test):
                y_pred=model(x_test[i,:])
                loss=lossfn(y_test[i,:],y_pred)
                loss_sum+=loss.item()
            testing_loss_list.append(loss_sum/n_test)
        print(ep,'done')

    # Loss graph
    plt.close()
    plt.title('Loss graph')
    plt.plot(np.linspace(
        0, epochs, epochs), training_loss_list, 'r', label='Training loss')
    plt.plot(np.linspace(
        0, epochs, epochs), testing_loss_list,  'b', label='Testing loss')
    plt.legend()
    plt.yscale('log')
    plt.show()
