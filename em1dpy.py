from contextlib import contextmanager
from abc import ABCMeta, abstractmethod

import numpy as np

class Utility():
    @classmethod
    def sample():
        pass

class BaseEm():
    def __init__(x, y, z, hs, current):
        #必ず渡させる引数
        self.x = x
        self.y = y
        self.z = z
        self.hs = hs
        self.current = current
        self.res = res
        self.thickness = thickness

        #デフォルト引数に設定しておく引数(変えたいユーザもいる)
        self.hankel_filter = "kong241" #or anderson801
        self.mu = np.zeros(len(res))
        self.mu[:] = 1.25663706e-6
        self.mu0 = mu[0]
        #指定しておく引数（変えたいユーザはほぼいない）
        self.vmd = 1
        self.mom = 1
        self.epsrn = 8.85418782e-12

        #渡された引数から計算する変数
        self.n_layer = len(res)

class tdem():
    @classmethod
    def buc(self):
        print("be used")

    @classmethod
    def uc(self):
        print("I'm uc")
        tdem.buc()
