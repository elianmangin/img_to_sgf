import os
import torch
from preprocess import vectorisation_img,sgf_to_y, y_to_sgf
import matplotlib.pyplot as plt
import numpy as np


class train_class:
    
    def __init__(self,img_dir,sgf_dir,heigh,lenght):
        self.img_dir = img_dir
        self.sgf_dir = sgf_dir
        self.heigh =heigh
        self.lenght = lenght
        self.img_names=os.listdir(self.img_dir)
        self.sgf_names=os.listdir(self.sgf_dir)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        x=vectorisation_img("img_train/"+self.img_names[idx],self.heigh,self.lenght)
        y=sgf_to_y("sgf_train/"+self.sgf_names[idx])
        return torch.tensor(x).float(),torch.tensor(y).float()

class test_class :
    
    def __init__(self,img_dir,sgf_dir,heigh,lenght):
        self.img_dir = img_dir
        self.sgf_dir = sgf_dir
        self.heigh =heigh
        self.lenght = lenght
        self.img_names=os.listdir(self.img_dir)
        self.sgf_names=os.listdir(self.sgf_dir)

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        x=vectorisation_img("img_test/"+self.img_names[idx],self.heigh,self.lenght)
        y=sgf_to_y("sgf_test/"+self.sgf_names[idx])
        return torch.tensor(x).float(),torch.tensor(y).float()


class model_img_to_sgf(torch.nn.Module):

    def __init__(self,entry_size):
        super().__init__()
        self.linbranch1 = torch.nn.Linear(entry_size, 50)
        self.linbranch2 = torch.nn.Linear(50,60)
        self.linbranch3 = torch.nn.Linear(60,50)
        self.linbranch4 = torch.nn.Linear(50, 361)

    def forward(self, x):
        z1 = self.linbranch1(x)
        z1 = torch.nn.ReLU()(z1)
        z2 = self.linbranch2(z1)
        z2 = torch.nn.ReLU()(z2)
        z3 = self.linbranch3(z2)
        z3 = torch.nn.ReLU()(z3)
        z4 = self.linbranch4(z3)
        y_pred = torch.nn.ReLU()(z4)
        return y_pred


if __name__ == "__main__":
    heigh=100
    lenght=100
    m=600
    test_size=0.2
    epochs=5
    learnig_rate=5e-3
    batch_size = 8

    training_set=train_class("img_train/","sgf_train/",heigh,lenght)
    validation_set=test_class("img_test/","sgf_test/",heigh,lenght) 
    training_loader = torch.utils.data.DataLoader(training_set, batch_size, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size, shuffle=False)
    
    model=model_img_to_sgf(3*heigh*lenght)
    loss_fn=torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learnig_rate)

    training_loss_list=[]
    testing_loss_list=[]

    for ep in range(epochs):
        print(f"Epoch {ep+1}\n-------------------------------")
        #TRAINING
        size = len(training_loader.dataset)
        running_loss = 0.0
        for batch, (X, y) in enumerate(training_loader):
            pred = model(X)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"TRAINING loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        training_loss_list.append(running_loss/size)
        running_loss = 0.0
        
        #VALIDATION
        running_loss = 0.0
        size = len(validation_loader.dataset)
        for batch, (x, y) in enumerate(validation_loader):
            with torch.no_grad():
                pred = model(x)
                loss = loss_fn(pred, y)
                running_loss += loss.item()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"TESTING loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        testing_loss_list.append(running_loss/size)
        running_loss = 0.0

    # Save model
    torch.save(model.state_dict(), 'model_weights.pth')

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
