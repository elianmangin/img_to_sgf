from turtle import back
import numpy as np
import re
from matplotlib import image
import os
from capture_check import fast_capture_pieces
import cv2
from PIL import Image, ImageDraw, ImageFont
import torch

dic_number_to_letter = {0: "a", 1: "b", 2: "c", 3: "d", 4: "e", 5: "f", 6: "g", 7: "h", 8: "i",
                        9: "j", 10: "k", 11: "l", 12: "m", 13: "n", 14: "o", 15: "p", 16: "q", 17: "r", 18: "s"}
dic_letter_to_number = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5, "g": 6, "h": 7, "i": 8,
                        "j": 9, "k": 10, "l": 11, "m": 12, "n": 13, "o": 14, "p": 15, "q": 16, "r": 17, "s": 18}


def sgf_to_y(sgf_path):
    """takes a sgf path and turns it into a (361,1) array corresponding to the intersection of the go board
    Each intersection is either 0 (no stone) 1 (black stone) 0.5 (white stone)
    Doesn't work with real sgfs where stiones are captured, since we will work with sgf without captured stones it doesn't matter"""
    sgf_initial_file = open(sgf_path, 'r')
    # passage en liste pour enlever les \n, les fonctions type replace marchent pas
    l_sgf = sgf_initial_file.read().split('\n')
    sgf = ""
    for k in l_sgf:
        sgf += k
    y = np.zeros((361, 1))
    for k in [i for i in range(len(sgf)) if sgf.startswith(';B[', i)]:
        y[dic_letter_to_number[sgf[k+3]]*19 + dic_letter_to_number[sgf[k+4]]] = 1
    for k in [i for i in range(len(sgf)) if sgf.startswith(';W[', i)]:
        y[dic_letter_to_number[sgf[k+3]]*19 +
            dic_letter_to_number[sgf[k+4]]] = 0.5
    return y.T

def round_y(y):
    """Takes a (1,361) array and rounds it to be a 0.5/1"""
    for i in range(361):
        y[0][i] = round(float(y[0][i]*2))/2
    return y

def y_to_sgf(y,sgf_path):
    """ takes a (1,361) array corresponding to the intersection of the go board
    Each intersection is either 0 (no stone) 1 (black stone) 0.5 (white stone) and turns it into a sgf
    which is stored a sgf_final.sgf"""
    sgf = "(;FF[4]CA[UTF-8]GM[1]SZ[19]"
    y=np.resize(y,(1,361))
    for i in range(361):
        if y[0][i] == 1:
            sgf = sgf + ';B[' + dic_number_to_letter[i//19] + \
                dic_number_to_letter[i % 19]+']'

        if y[0][i] == 0.5:
            sgf = sgf + ';W[' + dic_number_to_letter[i//19] + \
                dic_number_to_letter[i % 19]+']'
    sgf = sgf + ')'
    sgf_final_file = open(sgf_path, 'w')
    sgf_final_file.write(sgf)
    sgf_final_file.close()


def vectorisation_img(img_path,heigh,lenght):
    """Takes an img_path and return a vector of size heigh*lenght*3"""
    img = image.imread(img_path)
    if img.shape[2] == 4: #supprime une éventuelle couche de transparence
        img = np.delete(img, 3, 2)
    img = np.resize(img, (heigh, lenght, 3))
    img = img.ravel()
    x = np.reshape(img, (1,heigh*lenght*3))
    return x
    



def vectorisation_sgf(sgf_folder):
    """Takes the folder containing the sgfs and returns Y
    WARNING: the sgfs ans imgs must be in the same order in their respective folders """
    vector_list = []
    for filename in os.listdir(sgf_folder):
        vector_list.append(sgf_to_y(sgf_folder+filename))
    Y = np.concatenate(vector_list, axis=1)
    return Y.T


def preprocess_sgf(sgf_path):
    """removes all captured stones from an sgf"""
    sgf_initial_file = open(sgf_path, 'r')
    # passage en liste pour enlever les \n, les fonctions type replace marchent pas
    l_sgf = sgf_initial_file.read().split('\n')
    sgf = ""
    for k in l_sgf:
        sgf += k
    if 't' in sgf:
        print('ATTENTION')
        sgf_initial_file.close()
        os.remove(sgf_path)
        return sgf
    black_stones = [(dic_letter_to_number[sgf[i+3]], dic_letter_to_number[sgf[i+4]])
                    for i in range(len(sgf)) if sgf.startswith(';B[', i)]
    white_stones = [(dic_letter_to_number[sgf[i+3]], dic_letter_to_number[sgf[i+4]])
                    for i in range(len(sgf)) if sgf.startswith(';W[', i)]
                    
    black_board=np.zeros((19,19))
    white_board=np.zeros((19,19))
    l_b,l_w=(len(black_stones),len(white_stones))
    for i in range(min(len(black_stones),len(white_stones))):
        black_board[black_stones[i][0],black_stones[i][1]]=1
        black_board, white_board = fast_capture_pieces(black_board,white_board,False,black_stones[i][0],black_stones[i][1])
        white_board[white_stones[i][0],white_stones[i][1]]=1
        black_board, white_board =fast_capture_pieces(black_board,white_board,True,white_stones[i][0],white_stones[i][1])
    if l_b-l_w==1: ## if black plays last cause of a surrender
        black_board[black_stones[-1][0],black_stones[-1][1]]=1
        black_board, white_board = fast_capture_pieces(black_board,white_board,False,black_stones[i][0],black_stones[i][1])
    final_board=black_board+0.5*white_board
    y_to_sgf(final_board.reshape(361,1),sgf_path)
    
def sgf_to_img(sgf_file):
    """Takes an sgf and returns an image of the game (doesn't take captured stones in account)"""
    background = Image.open('image_for_generation/goban_background.PNG') #prendre des screens sur KGS
    white_stone = Image.open('image_for_generation/white_stone.PNG')
    black_stone = Image.open('image_for_generation/black_stone.PNG')
    background = background.resize((520,520))
    bg_w, bg_h = background.width,background.height
    stone_w,stone_h= bg_w//20,bg_h//20
    black_stone = black_stone.resize((stone_w,stone_h))
    white_stone = white_stone.resize((stone_w,stone_h))
    y=sgf_to_y(sgf_file)
    n=0
    y = np.resize(y,361)
    for k in y:
        if k==0.5:
            img_offset = (int((n//19+0.5)*stone_w),int((n%19+0.5)*stone_h),int((n//19+1.5)*stone_w),int((n%19+1.5)*stone_h))
            background.paste(white_stone, img_offset, white_stone)
        if k==1:
            img_offset = (int((n//19+0.5)*stone_w),int((n%19+0.5)*stone_h),int((n//19+1.5)*stone_w),int((n%19+1.5)*stone_h))
            background.paste(black_stone, img_offset, black_stone)
        n+=1
    return background

def generate_random_sgf_and_img(sgf_path,img_path):
    """Generate a random sgf without captured stones and the corresponding image
    and store them in the specified paths"""
    y= torch.rand((1,361))
    y=round_y(y)
    y_to_sgf(y,sgf_path)
    preprocess_sgf(sgf_path)
    img = sgf_to_img(sgf_path)
    img.save(img_path)

if __name__ == "__main__":

    for k in range (10000):
        print(k)
        generate_random_sgf_and_img("sgf_train/random_train_sgf"+str(k)+".sgf","img_train/random_train_img"+str(k)+".png")



    #boucle pour faire sgf_to_img sur tous les sgf non faits
    # k=0
    # for file in os.listdir('sgf_test'):
    #     # im = sgf_to_img('sgf_train/'+file)
    #     # im.save('img_train/train_img_'+file[9:] +'.png',format='png')
    #     # os.rename('img_train/train_img_'+file[9:-4],'img_train/train_img_'+file[9:-4]+'.png')
    #     k+=1
    #     print(k)
    #     img = Image.open('img_test/train_img_'+file[9:-4]+'.png')
    #     img=np.array(img)
    #     img = img/255*2-1
    #     img = Image.fromarray(img)
    #     img.save('img_test/train_img_'+file[9:-4]+'.png',format='png')

#Pistes d'amélioration:
#Ajouter du bruit dans la génération d'image
