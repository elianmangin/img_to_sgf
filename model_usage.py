import torch
from preprocess import round_y, vectorisation_img,y_to_sgf
from model_training import model_img_to_sgf


heigh=100
lenght=100   
model=model_img_to_sgf(3*heigh*lenght)
model.load_state_dict(torch.load('model_weights.pth'))
x=torch.tensor(vectorisation_img("img_test/train_img_5001.png",heigh,lenght)).float()
y=model(x)
print(y)
y=round_y(y)
y_to_sgf(y,"result_test_sgf.sgf")


