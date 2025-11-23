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

    for b in range(c):
        banda = res[:,:,b]
        res[:,:,b] = normaliza(banda)

    res = res / 255
    res = np.clip(res, 0, 1)
    return res.astype(np.float32)


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

def quantizaNiveis(img, qtd):
    arr = extrai_array(img)
    f = 255/(qtd-1)
    arr = np.round(arr / (f + 0.5)) * f
    return toPil(arr)

def divide_regiao(img, mapa, id, id_, q):
    h, w = img.shape[0:2]
    R = []
    for i in range(h):
        for j in range(w):
            if mapa[i][j] == id:
                R.append((i, j))
    if len(R) < q:
        return False

    ampl = img[:][:][0].max() - img[:][:][0].min()
    channel = 0
    for c in range(1,3):
        aux = img[:][:][c].max() - img[:][:][c].min()
        if (aux > ampl):
            ampl = aux
            channel = c

    mediana = np.median(img[:][:][c])
    for p in R:
        x, y = p
        if img[x][y][channel] > mediana:
            mapa[x][y] = id_
    
    return True

def percurso(img, mapa, o, n, t, q):
    #o é a lista de todas as regioes de todos os níveis
    if n >= len(o): #para cada nível novo cria a lista de regioes do nível
        o.append([])
    
    if n==0 and len(o[0])==0: #no nível 0 só tem uma região, que é a imagem inteira
        o[0] = [0]
    
    for regiao in o[n-1]:
        #o nível atual sempre tem no mínimo o número de regiões do anterior
        if regiao not in o[n]:
            o[n].append(regiao)

        if(len(o[n])>=t):
            return

        id_ = regiao + 2**(n-1) #ids da região do próximo nível criado pela divisão de "regiao"

        if divide_regiao(img, mapa ,regiao, id_, q):#divide a regiao atual
            o[n].append(id_)
            if len(o[n]) >= t:#testa se vai precisar dividir as próximas regiões
                return
        
    if len(o[n]) < t: #fica chamando essa função até atingir o número desejado de sub-regiões
        percurso(img, mapa, o, n+1, t, q)

    return

def median_cut(img, t, q, f):
    arr = extrai_imagem(img)
    h, w = arr.shape[0:2]
    res = np.zeros((arr), dtype=int)
    o = [[0]]
    n=0
    mapa = np.zeros((arr.shape[0:2]))
    percurso(arr, mapa, o, n, t, q)
    ultimo_nivel = o[-1] #último nível(já deve ter o número t de regiões)
    for i in ultimo_nivel:
        R = []
        for j in range(h):
            for k in range(w):
                if mapa[j][k] == i:
                    R.append((j, k))
                    c = f(R) #cor nova baseada nos pixels que estao na sub-região R
        for p in R:
            x, y = p[:2]
            res[x, y] = c
        
    return toPil(res)
        

def median_cut(img, t, q, f):
    arr = extrai_array(img)
    res = np.zeros_like(arr)
    h, w = arr.shape[0:2]
    o = [[0]]
    n=0
    mapa = np.zeros((arr.shape[0:2]))
    percurso(arr, mapa, o, n, t, q)
    ultimo_nivel = o[-1] #último nível(já deve ter o número t de regiões)
    R = {i: [] for i in ultimo_nivel}
    for j in range(h):
        for k in range(w):
            R[mapa[j][k]].append((j, k))
                    
    for i in ultimo_nivel:
        c = f(R[i]) #cor nova baseada nos pixels que estao na sub-região R
        for p in R[i]:
            x, y = p[:2]
            res[x, y] = c

    return toPil(res)

def divide_regiao(img, mapa, id, id_, q):
    h, w = img.shape[0:2]

    # pixels pertencentes à região id
    R = [(i,j) for i in range(h) for j in range(w) if mapa[i,j] == id]

    if len(R) < q:
        return False

    # escolhe canal com maior variação
    channel = 0
    vals = [img[x, y, 0] for (x, y) in R]
    ampl = max(vals) - min(vals)

    for c in range(1,3):
        vals = [img[x, y, c] for (x, y) in R]
        aux = max(vals) - min(vals)
        if aux > ampl:
            ampl = aux
            channel = c

    # mediana correta da região
    mediana = np.median([img[x, y, channel] for (x, y) in R])

    # atribui região nova id'
    for (x, y) in R:
        if img[x, y, channel] > mediana:
            mapa[x, y] = id_

    return True


def percurso(img, mapa, o, n, t, q):

    # cria nível se necessário
    if n >= len(o):
        o.append([])

    # nível 0 contém só região 0
    if n == 0:
        o[0] = [0]

    # regiões do nível anterior
    if n == 0:
        anteriores = o[0]
    else:
        anteriores = o[n-1]

    # processa cada região
    for regiao in anteriores:

        if regiao not in o[n]:
            o[n].append(regiao)

        if len(o[n]) >= t:
            return

        id_ = regiao + 2**n

        if divide_regiao(img, mapa, regiao, id_, q):
            o[n].append(id_)

            if len(o[n]) >= t:
                return

    if len(o[n]) < t:
        percurso(img, mapa, o, n+1, t, q)


def median_cut(img, t, q, f):
    arr = extrai_array(img)
    h, w = arr.shape[0:2]
    res = np.zeros_like(arr)

    mapa = np.zeros((h, w), dtype=int)
    o = [[0]]

    percurso(arr, mapa, o, 0, t, q)

    ultimo_nivel = o[-1]

    for reg_id in ultimo_nivel:

        R = [(j,k) for j in range(h) for k in range(w)
             if mapa[j,k] == reg_id]

        if len(R) == 0:
            continue


        cores = np.array([arr[x, y] for (x, y) in R])  # matriz Nx3

        c = f(cores, axis=0)

        for (x,y) in R:
            res[x,y] = c

    return toPil(res)


"""
Para verificar se as cores estão sendo fragmentadas nas bordas, é possível aplicar um filtro sobel na imagem original e 
sobrepor à gerada pela "median_cut", de forma que fique nítido onde estão as bordas e se elas estão sendo ultrapassadas ou não.
"""

from skimage import color
def aprimora(img, f, alarg=True):
    assert img.mode == 'RGB'
    assert -2 <= f <= 2

    rgb = extrai_array(img) / 255.0 
    hsl = color.rgb2hsl(rgb)
    print(hsl.shape)


def hsv_to_hsl(h, s, v):
    l = v * (1 - s / 2)
    if l == 0 or l == 1:
        s_l = 0.0
    else:
        s_l = (v - l) / min(l, 1 - l)
    return h, s_l, l

def hsv_to_hsl(arr):
    l = v * (1 - s / 2)
    if l == 0 or l == 1:
        s_l = 0.0
    else:
        s_l = (v - l) / min(l, 1 - l)
    return h, s_l, l


def hsv_to_hsl_img(hsv):
    hsv = hsv.astype(float) / 255.0

    h = hsv[..., 0]
    s = hsv[..., 1]
    v = hsv[..., 2]

    l = v * (1 - s / 2)

    denom = np.minimum(l, 1 - l)
    s_l = np.zeros_like(s)

    mask = denom > 1e-9
    s_l[mask] = (v[mask] - l[mask]) / denom[mask]

    h_l = h.copy()

    return np.stack([h_l, s_l, l], axis=-1)

def aprimora(img, f, alarg=True):
    assert img.mode == 'RGB'
    assert -2 <= f <= 2

    hsl = hsv_to_hsl_img(extrai_array(img.convert("HSV")))
    print(hsl.shape)

    print(hsl[0][0][2])

aprimora(img, 1)