"""
Created on Wed Jan 26 18:04:16 2022
@author: nedir ymamov
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz


def cizgi(offset, length, breadth, draft, pn, wl):
    """  ==>> SUHATLARI <<==  """
    plt.figure(figsize=(10, 2.4))
    pn_new = np.linspace(0, length, 51)
    offset_new = np.zeros((51, offset.shape[1]))
    for i in range(offset.shape[1]):
        f = interp1d(pn, offset[:, i], kind='cubic')
        offset_new[:, i] = np.round(f(pn_new), 3)

    for i in range(offset_new.shape[1]):
        plt.plot(pn_new, offset_new[:, i])
        plt.plot(pn_new, -offset_new[:, i])
    plt.legend(["wl0", "wl0.3", "wl1", "wl2", "wl3", "wl4",
                "wl5", "wl6"], loc="lower center")
    plt.title("Suhatı Eğrileri")
    plt.xlabel("Boyu [m]")
    plt.ylabel("Genişlik [m]")
    plt.savefig('waterline.png')
    plt.show()

    """  ==>> ENKESİTLER <<==  """
    wl_new = np.linspace(0, 1.5*draft, 21)
    offset_new = np.zeros((13, 21))
    for i in range(13):
        f = interp1d(wl, offset[i, :], kind='cubic')
        offset_new[i, :] = np.round(f(wl_new), 3)

    for i in range(13):
        if i <= 6:
            plt.plot(-offset_new[i], wl_new)
        else:
            plt.plot(offset_new[i], wl_new)
    plt.legend(["pn0", "pn0.5", "pn1", "pn2", "pn3",
                "pn4", "pn5", "pn6", "pn7", "pn8", "pn9",
                "pn9.5", "pn10"], loc="upper right")
    plt.plot([-breadth/2 - 0.1, breadth/2 + 3.5],
             [draft, draft], [0, 0], [0, 1.5*draft])
    plt.title("Enkesit Eğrileri")
    plt.xlabel("Genişlik [m]")
    plt.ylabel("Draft [m]")
    plt.savefig('enkesitler.png')
    plt.show()


def offset_expand(offset, length, breadth, draft, pn, wl):
    row, col = 50, 15  # YENİ BOYUTLAR
    wl_new = np.linspace(0, 1.5*draft, col + 1)
    pn_new = np.linspace(0, length, row + 1)
    offset_pre = np.zeros((13, col + 1))
    for i in range(13):
        f = interp1d(wl, offset[i, :], kind='cubic')
        offset_pre[i, :] = np.round(f(wl_new), 3)
    offset_new = np.zeros((row + 1, col + 1))
    for i in range(col + 1):
        f1 = interp1d(pn, offset_pre[:, i], kind='cubic')
        offset_new[:, i] = np.round(f1(pn_new), 3)
    return pn_new, wl_new, offset_new


def alan_moment(offset, wl):
    row, col = offset.shape
    alan = np.zeros((row, col))   # BON-JEAN ALANLARI
    for i in range(row):
        alan[i, 1:] = 2*cumtrapz(offset[i, :], wl)
    alan = np.round(alan, 3)
    moment = np.zeros((row, col))  # BON-JEAN MOMENTLERİ
    for i in range(col):
        moment[:, i] = offset[:, i] * wl[i]
    for i in range(row):
        moment[i, 1:] = 2*cumtrapz(moment[i, :], wl)
    return alan, moment


def hacim_dep(alan, pn):
    col = offset.shape[1]
    hacim = np.zeros(col)  # HACİM HESABI
    for i in range(1, col):
        hacim[i] = np.trapz(alan[:, i], pn)
    hacim = np.round(hacim, 3)

    dep = 1.025 * hacim
    dep = np.round(dep, 3)
    return hacim, dep


def alan_wp(offset, pn):
    col = offset.shape[1]
    Awp = np.zeros(col)  # SU HATTI ALANI
    for i in range(col):
        Awp[i] = np.trapz(offset[:, i], pn)
    Awp = np.round(Awp, 3)
    return Awp


def konum(offset, pn, Awp, alan, hacim):
    col = offset.shape[1]
    """ YÜZME MERKEZİNİN BOYUNA YERİ (KIÇTAN) """
    MxAwp = np.zeros(col)
    for i in range(col):
        MxAwp[i] = 2 * np.trapz(offset[:, i] * pn, pn)
    lcf = MxAwp / Awp - length / 2

    """" HACİM MERKEZİNİZ BOYUNA YERİ (KIÇTAN) """
    lcb = np.zeros(col)
    for i in range(1, col):
        lcb[i] = np.trapz(alan[:, i] * pn, pn) / hacim[i]

    """ HACİM MERKEZİNİN DÜŞEY YERİ """
    kb = np.zeros(col)
    for i in range(1, col):
        kb[i] = np.trapz(moment[:, i], pn) / hacim[i]
    return lcf, lcb, kb


def katsayi(length, offset, alan, hacim, wl):
    # BLOK KATSAYISI
    cb = np.array([0, *hacim[1:] / (length*2*offset[6, 1:] * wl[1:])])

    # ORTA KESİT NARİNLİK KATSAYISI
    cm = np.array([0, *alan[6, 1:] / (2*offset[6, 1:] * wl[1:])])

    # PRİZMATİK KATSAYI
    cp = np.array([0, *cb[1:] / cm[1:]])

    # SU HATTI NARİNLİK KATSAYISI
    cwp = Awp / (length*2 * offset[6, :])
    return cb, cm, cp, cwp


def metasantr(offset, length, Awp, lcf):
    col = offset.shape[1]
    # Icl : ORTA SİMETRİ EKSENİNE GÖRE ATALET MOMENTİ
    Icl = np.zeros(col)
    for i in range(col):
        Icl[i] = (2/3) * np.trapz(offset[:, i]**3, pn)
    # bm : ENİNE METESANTR YARIÇAPI
    bm = np.array([0, *Icl[1:] / hacim[1:]])

    # Im : MASTORİYE GÖRE ATALET MOMENTİ
    Im = np.zeros(col)
    for i in range(col):
        Im[i] = np.trapz(offset[:, i] * pn**2, pn)
    # If : SUHATTI ALANININ YÜZME MERKEZİNDEN
    #      GEÇEN EKSENE GÖRE ATALET MOMENTİ
    If = Im - Awp * (length/2 - lcf)**2
    # bml : BOYUNA METASANTR YARIÇAPI
    bml = np.array([0, *If[1:] / hacim[1:]])
    return bm, bml


def islak_alan(offset, pn, wl):
    row, col = offset.shape
    def arc_length(x, y):
        npts = len(x)
        arc = np.sqrt((x[1] - x[0])**2 + (y[1] - y[0])**2)
        for k in range(1, npts):
            arc = arc + np.sqrt((x[k] - x[k-1])**2 + (y[k] - y[k-1])**2)
        return arc/2
    
    l = np.zeros((row, col))
    for i in range(row):
        for j in range(1, col):
            l[i, j] = round(arc_length(offset[i, j-1:j+1], wl[j-1:j+1]), 3)
    S = np.zeros(col) # ISLAK YÜZEY ALAN EĞRİSİ
    for i in range(1, col):
        S[i] = S[i-1] + 2*np.trapz(l[:, i], pn)
    return S


def bircm_batma(Awp, dep, bml, length):
    TCP = Awp*1.025 / 100  # BİR SATİM BATMA TONAJI
    # BİR SANTİM BATMA TRİM MOMENTİ
    MCT = np.array([0, *dep[1:] * bml[1:] / (100*length)])
    return TCP, MCT


def egriler(length, breadth, draft, alan, moment, hacim,
            dep, Awp, lcf, lcb, kb, cb, cm, cp, cwp, TPC, MCT, S):
    row, col = alan.shape
    
    plt.figure(figsize=(10, 5))
    """ Ağ çizimi """
    for i in range(row):
        plt.plot([pn[i], pn[i]], [0, 1.5*draft], 'k')
    for i in range(col):
        plt.plot([0, length], [wl[i], wl[i]], 'k')

    """  Bon-jean alan eğrileri """
    s = length / 10
    oran = np.max(alan) / s
    for i in range(row-1):
        plt.plot(alan[i] / oran + pn[i], wl, 'g')

    """ Bon-jean moment eğrileri """
    for i in range(row-1):
        plt.plot(alan[i] / (oran + 1.5) + pn[i], wl, 'r--')

    """ Hacim ve Deplasman eğrileri """
    oran = dep[-1] / length
    plt.plot(dep / oran, wl)
    
    plt.plot(hacim / (oran + 1.5), wl, 'r--')

    """ Suhattı Alan eğrileri """
    Awp[2] -= 10
    Awp[4] += 6
    oran = Awp[-1] / (length - 15)
    wl_new = np.linspace(0, 1.5*draft, 25)
    spline = interp1d(wl, Awp, kind = 'quadratic')
    plt.plot(spline(wl_new) / oran, wl_new, 'c--')

    """ LCF, LCB ve KB eğrileri """
    spline = interp1d(wl, lcf, kind = 'cubic')
    plt.plot(spline(wl_new), wl_new, 'r')
    plt.plot(lcb[1:], wl[1:], 'b')
    oran = kb[-1] / (2*s)
    plt.plot(kb/oran, wl, 'b--')

    """ Katsayı eğrileri """
    oran = cb[-1] / (0.5*s)
    plt.plot(cb[1:]/oran, wl[1:], 'b')
    plt.plot(cm[1:]/oran, wl[1:], 'r')
    plt.plot(cp[1:]/oran + s, wl[1:], 'b')
    plt.plot(cwp[1:]/oran + s, wl[1:], 'r')
    
    """ TPC ve MCT eğrileri """
    wl_new = np.linspace(wl[1], 1.5*draft, 50)
    spline = interp1d(wl, TPC, kind = "cubic")
    plt.plot(spline(wl_new) * 3, wl_new, 'r')
    
    oran = MCT[-1]/ (length/3)
    spline = interp1d(wl, MCT, kind = "cubic")
    plt.plot(spline(wl_new) / oran, wl_new)
    
    """ Islak Yüzey Alan eğrisi """
    oran = S[-1] / (length/2)
    plt.plot(length/2 + S/oran, wl)
    
    plt.title("Bon-Jean Eğrileri")
    plt.xlabel("Boyu [m]")
    plt.ylabel("Depth [m]")
    plt.savefig('hidrostatik egriler.png')
    plt.show()


np.set_printoptions(precision = 3)
length = 100
breadth = 15
draft = 6
wl = np.array([0, 0.5, 1, 2, 3, 4, 5, 6]) * draft/4
pn = np.array([0, 0.5, 1, 2, 3, 4, 5, 6,
               7, 8, 9, 9.5, 10]) * length/10
offset = np.loadtxt('s60_cb70.txt', dtype=float)
offset *= breadth/2


cizgi(offset, length, breadth, draft, pn, wl)
pn_new, wl_new, offset_new = offset_expand(
    offset, length, breadth, draft, pn, wl)

alan, moment = alan_moment(offset, wl)
hacim, dep = hacim_dep(alan, pn)
Awp = alan_wp(offset, pn)
lcf, lcb, kb = konum(offset, pn, Awp, alan, hacim)
cb, cm, cp, cwp = katsayi(length, offset, alan, hacim, wl)
bm, bml = metasantr(offset, length, Awp, lcf)
TCP, MCT = bircm_batma(Awp, dep, bml, length)
S = islak_alan(offset, pn, wl)

egriler(length, breadth, draft, alan, moment, hacim,
        dep, Awp, lcf, lcb, kb, cb, cm, cp, cwp, TCP, MCT, S)
