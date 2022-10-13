import numpy as np
import re
from matplotlib import image
import os
from capture_check import fast_capture_pieces

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
    return y


def y_to_sgf(y):
    """ takes a (361,1) array corresponding to the intersection of the go board
    Each intersection is either 0 (no stone) 1 (black stone) 0.5 (white stone) and turns it into a sgf
    which is stored a sgf_final.sgf"""
    sgf = "(;FF[4]CA[UTF-8]GM[1]SZ[19]"
    for i in range(361):
        if y[i][0] == 1:
            sgf = sgf + ';B[' + dic_number_to_letter[i//19] + \
                dic_number_to_letter[i % 19]+']'

        if y[i][0] == 0.5:
            sgf = sgf + ';W[' + dic_number_to_letter[i//19] + \
                dic_number_to_letter[i % 19]+']'
    sgf = sgf + ')'
    sgf_final_file = open('sgf_final.sgf', 'w')
    sgf_final_file.write(sgf)
    sgf_final_file.close()


def vectorisation_img(img_folder):
    """Takes the folder containing the images and returns X 
    WARNING: the sgfs ans imgs must be in the same order in their respective folders """
    sizes_h = []
    sizes_l = []
    imgs = []
    for filename in os.listdir(img_folder):
        img = image.imread(img_folder+filename)
        if img.dtype == 'uint8':
            img /= img
        sizes_h.append(img.shape[0])
        sizes_l.append(img.shape[1])
        imgs.append(img)
    m = len(sizes_h)
    # h = int(sum(sizes_h)/m)
    # l = int(sum(sizes_l)/m)
    h = 150
    l = 150
    for i in range(m):
        img = imgs[i]
        if img.shape[2] == 4:
            img = np.delete(img, 3, 2)
        img = np.resize(img, (h, l, 3))
        img = img.ravel()
        imgs[i] = np.reshape(img, (h*l*3, 1))
        # print(max(img))
    X = np.concatenate(imgs, axis=1)
    return X.T


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

    black_stones = [(dic_letter_to_number[sgf[i+3]], dic_letter_to_number[sgf[i+4]])
                    for i in range(len(sgf)) if sgf.startswith(';B[', i)]
    white_stones = [(dic_letter_to_number[sgf[i+3]], dic_letter_to_number[sgf[i+4]])
                    for i in range(len(sgf)) if sgf.startswith(';W[', i)]
    print(len(black_stones),len(white_stones))
    black_board=np.zeros((19,19))
    white_board=np.zeros((19,19))
    for i in range(len(black_stones)):
        black_board[black_stones[i][0],black_stones[i][0]]=1
        fast_capture_pieces(black_board,white_board,False,black_stones[i][0],black_stones[i][0])
        white_board=[white_stones[i][0],white_stones[i][0]]=1
        fast_capture_pieces(black_board,white_board,True,white_stones[i][0],white_stones[i][0])
    final_board=black_board+0.5*white_board
    final_sgf=y_to_sgf(final_board.reshape(361,1))
    sgf_final_file = open('sgf_final.sgf', 'w')
    sgf_final_file.write(final_sgf)
    sgf_final_file.close()
    

if __name__ == "__main__":
    preprocess_sgf("test.sgf")
