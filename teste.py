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

def histograma_CDF(hist):
    acc = 0
    cdf = np.zeros_like(hist)
    for i in range(len(hist)):
        acc += hist[i]
        cdf[i] = acc
    return cdf


def otsu(img, inv=False):
    arr = extrai_array(img.convert("L"))
    h, w = arr.shape[0:2]
    p = pega_histograma(arr) / (h * w)
    cdf = histograma_CDF(p)
    idx_arr = np.arange(256)
    media = np.sum(p * idx_arr)
    th = 0
    variancia_max = 0
    for k in range(1, 255):
        media1 = np.sum((p * idx_arr)[0:k+1]) / cdf[k]
        media2 = np.sum((p * idx_arr)[k+1:256])/(1 - cdf[k])

        variancia = cdf[k] * (media1 - media)**2 + (1 - cdf[k]) * (media2 - media)**2

        if variancia > variancia_max:
            variancia_max = variancia
            th = k
    for x in range(h):
        for y in range(w):
            arr[x][y] = 1 if ((arr[x][y] > th) != inv) else 0
                                
    return toPil(arr)

def cinza_lum(img):
    arr = extrai_array(img)
    arr = arr[...,0] * 0.299 + arr[...,1] * 0.587 + arr[...,2] * 0.114
    return toPil(arr)

def aplica_filtro(img, filtro):
    arr = extrai_array(img)
    h, w = arr.shape[0:2]
    n = filtro.shape[0]
    d = n // 2
    arr2 = np.zeros_like(arr)
    for i in range(d, h-d):
        for j in range(d, w-d):
            arr2[i][j] = np.sum(arr[i-d:i+d+1, j-d:j+d+1] * filtro)
    return toPil(arr2)

def gera_filtro_gaussiano(tam):
    filtro = np.zeros((tam, tam))
    d = tam//2
    for i in range(tam):
        for j in range(tam):
            x, y = i-d, j-d
            filtro[i][j] = (math.e ** (-(x**2 + y**2) / 2)) / (2 * math.pi)
    return filtro

def suaviza(img, tam):
    filtro = gera_filtro_gaussiano(tam)
    return aplica_filtro(img, filtro)

def sobel(img):
    arr = extrai_array(img)
    h, w = arr.shape[0:2]
    Kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    Gx = np.zeros_like(arr)
    Gy = np.zeros_like(arr)
    for i in range(1, h-1):
        for j in range(1, w-1):
            sq = arr[i-1:i+2, j-1:j+2]
            Gx[i, j] = np.sum(sq * Kx)
            Gy[i, j] = np.sum(sq * Ky)
    return np.hypot(Gx, Gy), np.arctan2(Gy, Gx)

def sup_nao_max(G, angulos):
    h, w = G.shape
    res = np.zeros_like(G)
    angulos = ((angulos*180)/math.pi) % 180
    for i in range(1, h-1):
        for j in range(1, w-1):
            q = 255
            r = 255
            
            if (0 <= angulos[i, j] < 22.5) or (157.5 <= angulos[i, j] < 180):
                q = G[i, j+1]
                r = G[i, j-1]

            elif (22.5 <= angulos[i, j] < 67.5):
                q = G[i+1, j-1]
                r = G[i-1, j+1]
            
            elif (67.5 <= angulos[i, j] < 112.5):
                q = G[i+1, j]
                r = G[i-1, j]
            
            elif (112.5 <= angulos[i, j] < 157.5):
                q = G[i-1, j-1]
                r = G[i+1, j+1]
            
            if (G[i, j] >= q) and (G[i, j] >= r):
                res[i, j] = G[i, j]
            else:
                res[i, j] = 0
    return res

def duplo_limiar(Z, alfa, beta):
    h, w = Z.shape[0:2]
    b = beta * Z.max()
    a = alfa * b
    f = np.round(b)+1
    F = 255
    R = np.zeros_like(Z)

    for i in range(h):
        for j in range(w):
            if Z[i][j]>=b:
                R[i][j] = F
            elif a <= Z[i][j]:
                R[i][j] = f
    
    return R, f, F

def histerese(R, f, F):
    h,w = R.shape[0:2]
    for i in range(1, h-1):
        for j in range(1, w-1):
            if R[i][j] == f:
                sem_forte = True
                for x in [i-1, i, i+1]:
                    for y in [j-1, j, j+1]:
                        if R[x][y] >= F:
                            R[i][j] = F
                            sem_forte = False
                if sem_forte:
                    R[i][j] = 0
    return R

def pega_imagem_x(imgs, x):
    return imgs[x, :, :]

def pega_imagem_y(imgs, y):
    return imgs[:, y, :]

def pega_imagem_z(imgs, z):
    return imgs[:, :, z]

def pega_imagens(imgs, x=[], y=[], z=[]):
    ret = []
    for i in x:
        ret.append(toPil(pega_imagem_x(imgs, i)))
    for j in y:
        ret.append(toPil(pega_imagem_y(imgs, j)))
    for k in z:
        ret.append(toPil(pega_imagem_z(imgs, k)))
    return ret

imgs = np.load("dados_medicos/hipercubo_c02.npy")

x = [0, 137, 274, 411, 548]
y = [0, 49, 98, 147, 196, 245]
z = [0, 64, 128, 192, 256, 320]

mostrar_imagens(pega_imagens(imgs, x=x), [f"x={i}" for i in x], ncolunas=len(x))
mostrar_imagens(pega_imagens(imgs, y=y), [f"y={i}" for i in y], ncolunas=len(y))
mostrar_imagens(pega_imagens(imgs, z=z), [f"z={i}" for i in z], ncolunas=len(z))

def pega_histograma_3d(img):
    h = np.zeros(256)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for k in range(img.shape[2]):
                h[int(img[i][j][k])] += 1
    return h

def segmenta(arr_orig, perc=None, th=None):
    # Operador ^: ou exclusivo
    # Garante que somente um dos parâmetros opcionais foi informado
    # Se nenhum deles for informado, lança exceção
    # Se os dois forem informados, também lança exceção
    if not ((perc is None) ^ (th is None)):
        raise ValueError('Apenas um dos parâmetros th ou perc deve ser informados. Não se pode deixar os dois vazios e nem informar ambos.')
    
    #################### COMPLETE COM SEU CÓDIGO #############
    h, w, d = arr_orig.shape[0:3]
    hist = pega_histograma_3d(arr_orig) / (h * w * d)
    cdf = histograma_CDF(hist)
    if perc is None:
        perc = 1 - cdf[th]
    else:
        a, b, c, p = -1, 256, 127, 10000
        while (a < c and c < b):
            aux = 1-cdf[c]
            if (abs(aux - perc) < p):
                th = c
                p = abs(aux - perc)
            if aux < perc:
                b = c
            else:
                a = c
            c = (a + b) // 2
    
    arr = np.zeros_like(arr_orig)
    # for i in range(h):
    #     for j in range(w):
    #         for k in range(d):
    #             c = arr_orig[i][j][k]
    #             arr[i][j][k] = c if c > th else 0
    arr = np.where(arr_orig > th, arr_orig, 0)
    vals = arr[arr > th]
    if vals.size > 1:
        lo = vals.min()
        hi = vals.max()
        arr = np.where(arr > th, (arr - lo) * 255/(hi - lo), 0)
    else:
        arr = np.zeros_like(arr)
    return arr, perc, th