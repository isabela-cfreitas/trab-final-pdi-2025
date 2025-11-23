import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

img = Image.open('imagens/pavoes.png')
#img_array = extrai_array(img)
#print(img_array.shape)
img_mono = img.convert("L")
img_arr = extrai_array(img_mono)
print("antes:", img_arr.shape)

img_m = np.stack((img_arr, img_arr, img_arr), axis=2)
print("depois: ", img_m.shape)

def normaliza(arr, max=255):
    return (arr - arr.min()) *max/(arr.max()-arr.min())

def pega_histograma(img):
    h = np.zeros(256)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            h[int(img[i][j])] += 1
    return h


def pega_histogramaRGB(img):
    hr = pega_histograma(img[:,:,0])
    hg = pega_histograma(img[:,:,1])
    hb = pega_histograma(img[:,:,2])
    
    return hr, hg, hb

def equalizacao(img):
    h, w = img.shape[0:2]

    img2 = np.zeros_like(img)

    hist = pega_histograma(img) / (h * w)

    p = 0
    s = np.zeros(256)

    for i in range(len(s)):
        p += hist[i]
        s[i] = np.round(p * 255)
    
    for i in range(h):
       for j in range(w):
          img2[i][j] = s[int(img[i][j])]

    return img2

def equalizacaoRGB(img):
    img2 = np.moveaxis(img, (2, 0, 1), (0, 1, 2))
    img2[0] = equalizacao(img2[0])
    img2[1] = equalizacao(img2[1])
    img2[2] = equalizacao(img2[2])
    img2 = np.moveaxis(img2, (0, 1, 2), (2, 0, 1))

    return img2

def equalizacaoRGB(img):
    res = np.zeros_like(img)
    for c in range(3):
        res[:,:,c] = equalizacao(img[:,:,c])
    return res

def especificacao_direta(img_orig, img_param):
    h,w = img_orig.shape
    h_orig = pega_histograma(normaliza(img_orig))
    h_param = pega_histograma(normaliza(img_param))
    s = np.zeros(256)
    S = np.zeros(256)
    p_orig = 0
    p_param = 0
    for i in range(len(s)):
       p_orig += h_orig[i]
       p_param += h_param[i]
       s[i] = p_orig
       S[i] = p_param

    res = np.zeros_like(img_orig)

    for i in range(h):
       for j in range(w):
           p = s[int(img_orig[i][j])]
           res[i][j] = np.argmin(np.abs(S - p))
    
    return res

def especificacao_direta_RGB(img_orig, img_param):
    img2 = np.moveaxes(img_orig, (2, 0, 1), (0, 1, 2))
    img_param2 = np.moveaxes(img_param, (2, 0, 1), (0, 1, 2))
    
    img2[0] = especificacao_direta(img2[0], img_param2[0])
    img2[1] = especificacao_direta(img2[1], img_param2[1])
    img2[2] = especificacao_direta(img2[2], img_param2[2])
    
    img2 = np.moveaxes(img2, (0, 1, 2), (2, 0, 1))

    return img2

def especificacao_direta_RGB(img_orig, img_param):
    res = np.zeros_like(img_orig)
    for c in range(3):
        res[:,:,c] = especificacao_direta(img_orig[:,:,c], img_param[:,:,c])
    return res

    
def sal_pimenta(img, cor_sal, cor_pimenta, p_sal, p_pimenta):
    img2 = np.copy(img)
    h, w = img.shape[0:2]
    n_sal = p_sal * h*w
    n_pimenta = p_pimenta * h*w
    for i in range(n_sal):
        x = random.randInt(0,h-1)
        y = random.randInt(0,w-1)
        img2[x][y] = cor_sal

    for i in range(n_pimenta):
        x = random.randInt(0,h-1)
        y = random.randInt(0,w-1)
        img2[x][y] = cor_pimenta
    
    return img2

def ruidoGaussiano(img, alfa):
    res = np.copy(img)
    r = np.random.normal(0, 0.15, img.shape[0:2]) * alfa
    for c in range(3):
        ruidos = r * img[:,:,c]
        res[:,:,c] += r
    return res

def hextodec(h):
    if h >= '0' and h <= '9':
        return int(h)
    elif h >= 'A' and h <= 'F':
        return 10 + ord(h) - ord('A')
    elif h >= 'a' and h <= 'f':
        return 10 + ord(h) - ord('a')
    else:
        return 0
        
def decodifica(arq):
    arr = arq2list(arq)
    h, w, c = arr[0].split(' ')
    h = int(h)
    w = int(w)
    c = int(c)
    pixels = arr[1:]
    img = np.zeros((h, w, c))

    i = 0
    for x in range(h):
        for y in range(w):
            for z in range(c):
                img[x][y][c-z-1] = 16 * hextodec(pixels[i][-2*z-2]) + hextodec(pixels[i][-2*z-1])
            i += 1
    
    print(img)
    print("shape:", img.shape, "dtype:", img.dtype)
    if img.shape[2] == 1:
        img = np.reshape(img, (h, w))
    return toPil(img)

import math
def rotaciona(img, angulo):
    arr = extrai_array(img)
    h, w = arr.shape[0:2]
    img2 = np.zeros_like(arr)
    cos = math.cos(angulo*math.pi/180)
    sen = math.sin(angulo*math.pi/180)

    for i in range(h):
        for j in range(w):
            x = int((i-h/2)*cos - (j-w/2)*sen + h/2)
            y = int((i-h/2)*sen + (j-w/2)*cos + w/2)
            if x >= 0 and x < h and y >= 0 and y < w:
                img2[i][j] = arr[x][y]
    
    return toPil(img2)

def conserta_satelite(imgs):
    m = len(imgs)
    h, w, c = imgs[0].shape
    res = np.zeros((h, w, c))
    for img in imgs:
        res += normaliza(img, 255)
    res /= m
    res = equalizacaoRGB(res)
    res /= 255
    return res

def conserta_satelite(imgs):
    m = len(imgs)
    h, w, c = imgs[0].shape
    res = np.median(imgs, axis=0)
    res = equalizacaoRGB(res)
    res /= 255
    return res.astype(np.uint8)


def equalizacao(img):
    h, w = img.shape[0:2]

    img2 = np.zeros_like(img)

    hist = pega_histograma(img) / (h * w)

    p = 0
    s = np.zeros(256)

    for i in range(len(s)):
        p += hist[i]
        s[i] = np.round(p * 255)
    
    for i in range(h):
       for j in range(w):
          img2[i][j] = s[int(img[i][j])]

    return img2

def equalizacaoRGB(img):
    res = np.zeros_like(img)
    for c in range(3):
        res[:,:,c] = equalizacao(img[:,:,c])
    return res