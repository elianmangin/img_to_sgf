import torch
from preprocess import round_y, vectorisation_img,y_to_sgf
from PIL import Image
import numpy as np 

import torch.nn.functional as F

class model_img_to_sgf_usage(torch.nn.Module):
    
    def __init__(self,heigh,lengh):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(4,8,kernel_size=5,stride =1)
        self.pool=torch.nn.MaxPool2d(5,5)
        self.conv2 = torch.nn.Conv2d(8,20,kernel_size=(5, 5))
        self.linbranch1 = torch.nn.Linear(20*7*7,500)
        self.linbranch2 = torch.nn.Linear(500,400)
        self.linbranch3 = torch.nn.Linear(400, 361)

    def forward(self, x):
        x = (F.relu(self.conv1(x)))
        x = self.pool(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x) #plus de batch
        x = F.relu(self.linbranch1(x))
        x = F.relu(self.linbranch2(x))
        x = F.relu(self.linbranch3(x))
        return x



heigh=210
lenght=210  
model=model_img_to_sgf_usage(heigh,lenght)
model.load_state_dict(torch.load('model_weights.pth'))
x=Image.open("img_test/train_img_5001.png")
x=np.array(x)
x = x/255*2-1
x=np.resize(x,(4,heigh,lenght))
x = torch.tensor(x).float()
y=model(x)
print(x)
y=round_y(y)
y_to_sgf(y,"result_test_sgf.sgf")


