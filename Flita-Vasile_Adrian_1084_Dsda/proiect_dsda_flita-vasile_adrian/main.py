from pandas import *
import functions
import numpy as np
from functions import nan_replace
import graphics_ad
from acp import acp

# Citire tabel de date
t = read_csv("U.S. Military Deaths by cause 1980-2010.csv", index_col=0)

variabile = list(t)[1:]

# inlocuire valori nule
if any(t.isna()):
    nan_replace(t)

model_acp = acp(t, variabile)
model_acp.creare_model(std=False, nlib=1)



# tabel varianta
tabel_varianta = model_acp.tabelare_varianta()
tabel_varianta.to_csv("Varianta.csv")

# plot varianță
graphics_ad.plot_varianta(model_acp)

# Componentele Principale
nrcomp = model_acp.nrcomp_p
if model_acp.nrcomp_c is not None:
    if model_acp.nrcomp_c < nrcomp:
        nrcomp = model_acp.nrcomp_c
if model_acp.nrcomp_k is not None:
    if model_acp.nrcomp_k < nrcomp:
        nrcomp = model_acp.nrcomp_k

# Tabel componente
c = model_acp.c
c_t = functions.tabelare(c, t.index,
                         model_acp.etichete_componente,
                         "Componente.csv")
# Plot componente
for i in range(0, nrcomp):
    graphics_ad.plot_instante(c_t, "comp1",
                              model_acp.etichete_componente[i], aspect=1)

# Scoruri
t_scoruri = functions.tabelare(
    model_acp.c / np.sqrt(model_acp.alfa),
    t.index, model_acp.etichete_componente,
    "Scoruri.csv"
)
# Plot scoruri
for i in range(0, nrcomp):
    graphics_ad.plot_instante(t_scoruri, "comp1",
                              model_acp.etichete_componente[i],
                              aspect=1, titlu="Plot scoruri")

# Calcul corelatii componente - variabile

# Salvare corelatii
r_xc = model_acp.r_xc
r_xc_t = functions.tabelare(model_acp.r_xc,
                            model_acp.variabile,
                            model_acp.etichete_componente,
                            "R_xc.csv")
# plot corelații dintre variabilele observate și componente (cercul corelațiilor)
for i in range(0, nrcomp):

    graphics_ad.plot_corelatii(r_xc_t, "comp1", model_acp.etichete_componente[i], aspect=1)
# corelogramă corelații dintre variabilele observate și componente

graphics_ad.corelograma(r_xc_t)

#  Calcul cosinusuri
c2 = c * c
cosin = np.transpose(c2.T / np.sum(c2, axis=1))
cosin_t = functions.tabelare(cosin, t.index,
                             model_acp.etichete_componente, "Cosinusuri.csv")


# Calcul contributii
beta = c2 * 100 / np.sum(c2, axis=0)
beta_t = functions.tabelare(beta, t.index,
                            model_acp.etichete_componente,
                            "Contributii.csv")


# Calcul comunalitati
r_xc2 = r_xc * r_xc
comm = np.cumsum(r_xc2, axis=1)
comm_t = functions.tabelare(comm, variabile, model_acp.etichete_componente, "comunalitati.csv")
graphics_ad.corelograma(comm_t, vmin=0, titlu="Comunalitati")

model_acp.show_grafice()


