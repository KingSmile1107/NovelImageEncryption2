import cv2
import math
import numpy as np
import matplotlib.pyplot as plt

'''
计算图像的信息熵
'''


def _entropy(img):
    print(img)
    img = cv2.imread(img)
    w, h, _ = img.shape
    B, G, R = cv2.split(img)

    def calc_entropy(channel):
        gray, counts = np.unique(channel, return_counts=True)
        entropy = -np.sum((counts / (w * h)) * np.log2(counts / (w * h)))
        return entropy

    R_entropy = calc_entropy(R)
    G_entropy = calc_entropy(G)
    B_entropy = calc_entropy(B)

    return R_entropy, G_entropy, B_entropy


def entropy(raw_img, encrypt_img):
    with open('result.txt', 'a+', encoding='utf8') as f:
        R_entropy, G_entropy, B_entropy = _entropy(raw_img)
        f.write('====原图像信息熵====\n')
        f.write('通道R:\t\t{:.5}\n'.format(R_entropy))
        f.write('通道G:\t\t{:.5}\n'.format(G_entropy))
        f.write('通道B:\t\t{:.5}\n'.format(B_entropy))
        f.write('\n')
        R_entropy, G_entropy, B_entropy = _entropy(encrypt_img)
        f.write('===加密图像信息熵===\n')
        f.write('通道R:\t\t{:.5}\n'.format(R_entropy))
        print('通道R:\t\t{:.5}\n'.format(R_entropy))
        f.write('通道G:\t\t{:.5}\n'.format(G_entropy))
        print('通道G:\t\t{:.5}\n'.format(G_entropy))
        f.write('通道B:\t\t{:.5}\n'.format(B_entropy))
        print('通道B:\t\t{:.5}\n'.format(B_entropy))
        f.write('\n')

        print((R_entropy+G_entropy+B_entropy)/3)


if __name__ == '__main__':
    img = '../imagess/lena.png'
    encrypt_img = '../imagess/lena_jiamied_x1.png'
    entropy(img, encrypt_img)
