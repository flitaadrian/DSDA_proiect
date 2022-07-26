import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt


class acp():
    def __init__(self, t, variabile_observate=None):
        if variabile_observate is None:
            self.variabile = list(t)
        else:
            self.variabile = variabile_observate
        assert isinstance(t, DataFrame)
        self.__x = t[self.variabile].values

    def creare_model(self, std=True, nlib=0):
        x_ = self.x - np.mean(self.x, axis=0)
        if std:
            x_ = x_ / np.std(self.x, axis=0)
        n, m = np.shape(x_)
        r_cov = (1 / (n - nlib)) * x_.T @ x_
        valp, vecp = np.linalg.eig(r_cov)
        k = np.flipud(np.argsort(valp))
        self.__alfa = valp[k]
        self.__a = vecp[:, k]
        self.__c = x_ @ self.__a
        self.etichete_componente = ["comp" + str(i + 1) for i in range(m)]
        # Aplicare criterii de semnificatie pentru componente
        if std:
            self.nrcomp_k = np.where(self.alfa < 1)[0][0]
        else:
            self.nrcomp_k = None
        pondere = np.cumsum(self.alfa / sum(self.alfa))
        self.nrcomp_p = np.where(pondere > 0.8)[0][0] + 1
        eps = self.alfa[:(m - 1)] - self.alfa[1:]
        # print(eps)
        sigma = eps[:(m - 2)] - eps[1:]
        # print(sigma)
        negative = sigma < 0
        if any(negative):
            self.nrcomp_c = np.where(negative)[0][0] + 2
        else:
            self.nrcomp_c = None
        if std:
            self.r_xc = self.a*np.sqrt(self.alfa)
        else:
            self.r_xc = np.corrcoef(self.__x,self.__c,rowvar=False)[:m,m:]

    def tabelare_varianta(self):
        procent_varianta = np.round(self.alfa * 100 / sum(self.alfa), 3)
        tabel_varianta = DataFrame(
            data={
                "Varianta": np.round(self.alfa, 3),
                "Procent varianta": procent_varianta,
                "Varianta cumulata": np.round(np.cumsum(self.alfa), 3),
                "Procent cumulat": np.round(np.cumsum(procent_varianta), 3)
            }, index=self.etichete_componente
        )
        return tabel_varianta

    def show_grafice(self):
        plt.show()

    @property
    def x(self):
        return self.__x

    @property
    def a(self):
        return self.__a

    @property
    def alfa(self):
        return self.__alfa

    @property
    def c(self):
        return self.__c