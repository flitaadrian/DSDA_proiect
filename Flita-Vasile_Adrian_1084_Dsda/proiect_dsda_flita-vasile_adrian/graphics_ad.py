import matplotlib.pyplot as plt
import numpy as np
from acp import acp
import seaborn as sb
import pandas as pd


def plot_valori_proprii(alpha):
    plt.figure("Plot valori proprii", figsize=(10, 7))
    plt.title("Plot valori proprii - Distributia variantei")
    plt.xlabel("Componente");
    plt.ylabel("Valori proprii")
    plt.xticks([i for i in range(len(alpha))])
    plt.plot(alpha, 'ro-')
    plt.axhline(1, c='g')
    # plt.show()

def plot_componente(C, k1, k2, etichete, titlu="Componente"):
    plt.figure("Plot " + titlu, figsize=(8, 7))
    plt.title("Plot" + titlu)
    plt.xlabel("Axa " + str(k1))
    plt.ylabel("Axa " + str(k2))
    plt.scatter(C[:, k1], C[:, k2])
    for i in range(len(C)):
        plt.text(C[i, k1], C[i, k2], etichete[i])
    plt.show()


def plot_varianta(model_acp, titlu="Varianta componente"):
    assert isinstance(model_acp, acp)
    fig = plt.figure(figsize=(13, 7))
    assert isinstance(fig, plt.Figure)
    ax = fig.add_subplot(1, 1, 1)
    assert isinstance(ax, plt.Axes)
    ax.set_title(titlu, fontdict={"fontsize": 16, "color": 'b'})
    ax.set_xlabel("Componente", fontdict={"fontsize": 12, "color": 'b'})
    ax.set_ylabel("Varianta", fontdict={"fontsize": 12, "color": 'b'})
    m = len(model_acp.alfa)
    x = np.arange(1, m + 1)
    ax.set_xticks(x)
    ax.plot(x, model_acp.alfa)
    ax.scatter(x, model_acp.alfa, c='r')
    if model_acp.nrcomp_k is not None:
        ax.axhline(1, c='g', label="Kaiser")
    if model_acp.nrcomp_c is not None:
        ax.axhline(model_acp.alfa[model_acp.nrcomp_c - 1], c='m', label="Cattell")
    ax.axhline(model_acp.alfa[model_acp.nrcomp_p - 1], c='c', label="Procent acoperire > 80%")
    ax.legend()



def plot_corelatii(t, var1, var2, titlu="Corelatii variabile-componente", aspect='auto'):
    fig = plt.figure(figsize=(9, 8))
    assert isinstance(fig, plt.Figure)
    ax = fig.add_subplot(1, 1, 1)
    assert isinstance(ax, plt.Axes)
    ax.set_title(titlu, fontdict={"fontsize": 16, "color": "b"})
    ax.set_xlabel(var1, fontdict={"fontsize": 12, "color": "b"})
    ax.set_ylabel(var2, fontdict={"fontsize": 12, "color": "b"})
    ax.set_aspect(aspect)
    u = np.arange(0,2*np.pi,0.01)
    ax.plot(np.cos(u),np.sin(u))
    ax.axhline(0)
    ax.axvline(0)
    ax.scatter(t[var1], t[var2], c="r")
    for i in range(len(t)):
        ax.text(t[var1].iloc[i], t[var2].iloc[i], t.index[i])

def corelograma(t, vmin=-1, vmax=1, titlu="Corelatii variabile-componente"):
    fig = plt.figure(figsize=(9, 8))
    assert isinstance(fig, plt.Figure)
    ax = fig.add_subplot(1, 1, 1)
    assert isinstance(ax, plt.Axes)
    ax.set_title(titlu, fontsize=16, color='b')
    ax_ = sb.heatmap(t, vmin=vmin, vmax=vmax, cmap="RdYlBu", annot=True, ax=ax)
    ax_.set_xticklabels(t.columns, rotation=30, ha="right")

def plot_instante(t, var1, var2, titlu="Plot instante", aspect='auto'):
    fig = plt.figure(figsize=(13, 8))
    assert isinstance(fig, plt.Figure)
    ax = fig.add_subplot(1, 1, 1)
    assert isinstance(ax, plt.Axes)
    ax.set_title(titlu, fontdict={"fontsize": 16, "color": "b"})
    ax.set_xlabel(var1, fontdict={"fontsize": 12, "color": "b"})
    ax.set_ylabel(var2, fontdict={"fontsize": 12, "color": "b"})
    ax.set_aspect(aspect)
    ax.scatter(t[var1], t[var2], c="r")
    for i in range(len(t)):
        ax.text(t[var1].iloc[i], t[var2].iloc[i], t.index[i])


def show():
    plt.show()


def harta(shp, S, camp_legatura, nume_instante, titlu="Harta scoruri"):
    m = np.shape(S)[1]
    t = pd.DataFrame(data={"coduri": nume_instante})
    for i in range(m):
        t["v" + str(i + 1)] = S[:, i]
    shp1 = pd.merge(shp, t, left_on=camp_legatura, right_on="coduri")
    for i in range(m):
        f = plt.figure(titlu + "-" + str(i + 1), figsize=(10, 7))
        ax = f.add_subplot(1, 1, 1)
        ax.set_title(titlu + "-" + str(i + 1))
        shp1.plot("v" + str(i + 1), cmap="Reds", ax=ax, legend=True)
    plt.show()
