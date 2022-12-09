import os
import torch
from preprocess import vectorisation_img,sgf_to_y, y_to_sgf
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch.nn.functional as F

gray_scale_matrix =[0.2989, 0.5870, 0.1140,0]

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
        x=Image.open("img_train/"+self.img_names[idx])
        x.convert("L")
        x=np.array(x)
        x=np.dot(x,gray_scale_matrix)
        x = x/255*2-1

        x=np.resize(x,(1,self.heigh,self.lenght))
        y=sgf_to_y("sgf_train/"+self.sgf_names[idx])
        return torch.tensor(x).float(),torch.tensor(y).squeeze().float()

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
        x=Image.open("img_test/"+self.img_names[idx])
        x.convert("L")
        x=np.array(x)
        x=np.dot(x,gray_scale_matrix)
        x = x/255*2-1
        x=np.resize(x,(1,self.heigh,self.lenght))
        y=sgf_to_y("sgf_test/"+self.sgf_names[idx])
        return torch.tensor(x).float(),torch.tensor(y).squeeze().float()


class model_img_to_sgf(torch.nn.Module):

    def __init__(self,heigh,lengh):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1,4,kernel_size=5,stride =1)
        self.pool=torch.nn.MaxPool2d(5,5)
        self.conv2 = torch.nn.Conv2d(4,8,kernel_size=(5, 5))
        self.linbranch1 = torch.nn.Linear(8*7*7,500)
        self.linbranch2 = torch.nn.Linear(500,400)
        self.linbranch3 = torch.nn.Linear(400, 361)

    def forward(self, x):
        x = (F.relu(self.conv1(x)))
        x = self.pool(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x,1) #ne flatten pas le batch
        x = F.relu(self.linbranch1(x))
        x = F.relu(self.linbranch2(x))
        x = F.relu(self.linbranch3(x))
        return x


if __name__ == "__main__":
    heigh=210
    lenght=210
    test_size=0.2
    epochs=10
    learning_rate=1e-3
    batch_size = 10
    #Passer les images en noir et blanc pour réduire la tailles des entrées sans perdre trop d'info ?

    training_set=train_class("img_train/","sgf_train/",heigh,lenght)
    validation_set=test_class("img_test/","sgf_test/",heigh,lenght) 
    training_loader = torch.utils.data.DataLoader(training_set, batch_size, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation_set, batch_size, shuffle=False)
    
    model=model_img_to_sgf(heigh,lenght)
    loss_fn=torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

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
        training_loss_list.append(running_loss/size*batch_size)
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
        testing_loss_list.append(running_loss/size*batch_size)
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
