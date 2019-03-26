import warnings
from abc import ABCMeta, abstractmethod
import sys

import numpy as np
import scipy.io
from scipy.special import erf, erfc
import math


def explain():
    text = """
    このプログラムの使い方
    1. 時間領域、周波数領域どちらかのクラスをインスタンス化する。
        引数は、
        受信機のx座標：x
        受信機のy座標：y
        受信機のz座標：ｚ
        送信機の高さ：hs # height_source
        層の比抵抗（numpy1次元配列） 例：np.array([100, 100])：res
        層厚(numpy1次元配列)　例：np.array([20])：thickness
        周波数帯域(10^a - 10^b) この、a,bをリストで渡す。 例）[0, 6]：freq_range
        or
        時間幅(10^a - 10^b) この、a,bをリストで渡す。 例）[0, 6]：time_range
        プロット数：plot_number
        hankel変換のフィルター：hankel_filter（kong241 or anderson801）
        を順に渡す。
        デフォルト
        hankel_filter="kong241" or "anderson801"
        を指定することによって、コングのフィルターかアンダーソンのフィルタを選択できる。
        デフォルトはコングのフィルター。

        例）
        import em1dpy as em
        fdem = em.Fdem()

    2. 計算したい関数を実行する。
        vmdの場合
        fdem.vmd()

        circularloopの場合
        fdem.circular_loop(radius)
        radiusはループの半径を指定する。


    """
    print(text)


class TdemUtility():
    @classmethod
    def get_gs_coefficient():
        pass


class BaseEm(metaclass=ABCMeta):
    def __init__(self, x, y, z, hs, current, res, thickness,
                 plot_number, turns=1, hankel_filter="kong241"):
        # 必ず渡させる引数
        self.x = x
        self.y = y
        self.z = z
        self.hs = hs
        self.current = current
        self.res = res  # resistivity
        self.thickness = thickness

        # デフォルト引数に設定しておく引数(変えたいユーザもいる)
        self.turns = turns
        self.hankel_filter = hankel_filter  # or anderson801

        # 指定しておく引数（変えたいユーザはほぼいない）
        self.mu0 = 1.25663706e-6
        self.mu = np.ones(len(res)) * self.mu0
        self.moment = 1
        self.epsrn = 8.85418782e-12

        # 渡された引数から計算する変数
        self.num_layer = len(res) + 1
        self.sigma = 1 / res
        self.r = np.sqrt(x ** 2 + y ** 2)
        if self.r == 0:
            warnings.warn('when r=0, Ex and Ey diverge')
            self.cos_phai = 0
            self.sin_phai = 0
        else:
            self.cos_phai = self.x / self.r
            self.sin_phai = self.y / self.r

        # 第何層目に送受信源があるかを定義
        self.h = np.zeros((1, self.num_layer - 1))
        if z <= 0:  # (1) self.rlayer = 1
            self.rlayer = 1
            for ii in range(2, self.num_layer):  #
                self.h[0, ii - 1] = self.h[0, ii - 2] + thickness[ii - 2]
        else:
            self.rlayer = []
        if self.num_layer == 2 and z > 0:  # (1') self.rlayer = 2
            self.rlayer = 2
        elif (self.num_layer >= 3) and (z > 0):
            for ii in range(2, self.num_layer):  # (2') 2 <= self.rlayer <= self.num_layer-1
                self.h[0, ii - 1] = self.h[0, ii - 2] + thickness[ii - 2]
                if z >= self.h[0, ii - 2] and z <= self.h[0, ii - 1]:
                    self.rlayer = ii
        if self.rlayer == []:  # (3) self.rlayer = self.num_layer
            self.rlayer = ii + 1

        if hs <= 0:  # (1) self.tlayer = 1
            self.tlayer = 1
        else:
            self.tlayer = []
        if self.num_layer == 2 and hs > 0:  # (2) self.tlayer = 2
            self.tlayer = 2
        elif (self.num_layer >= 3) and (hs > 0):
            self.h[0] = 0
            for ii in range(2, self.num_layer):  # (2') 2 <= self.tlayer <= self.num_layer-1
                self.h[0, ii - 1] = self.h[0, ii - 2] + thickness[ii - 2]
                if hs >= self.h[0, ii - 2] and hs <= self.h[0, ii - 1]:
                    self.tlayer = ii
        if self.tlayer == []:  # (3) self.tlayer = self.num_layer
            self.tlayer = ii + 1

        if hankel_filter == "kong241":
                print("hankel filter is kong241")
                matdata = scipy.io.loadmat("hankelset241.mat")
                self.wt0 = matdata["j0"]
                self.wt1 = matdata["j1"]
                self.y_base = matdata["lamdaBase"]
                self.filter_length = len(self.y_base)
        elif hankel_filter == "anderson801":
                print("hankel filter is anderson801")
                matdata = scipy.io.loadmat("anderson_801.mat")
                self.wt0 = matdata["wt0"]
                self.wt1 = matdata["wt1"]
                self.y_base = matdata["yBase"]
                self.filter_length = len(self.y_base)
        else:
            raise Exception(
                "kong241 or anderson801 is only available as filter name")

    def make_kernel(self, transmitter, omega):
        k = np.zeros((1, self.num_layer), dtype=np.complex)
        # k[0,0] = 0 static approximation
        k[0,1:self.num_layer] = (omega ** 2.0 * self.mu * self.epsrn - 1j * omega * self.mu * self.sigma) ** 0.5

        u = np.zeros((self.num_layer, self.filter_length, 1), dtype=complex)
        for ii in range(0,self.num_layer):
            u[ii] = ((self.lamda ** 2 - k[0,ii] ** 2) ** 0.5).reshape((self.filter_length, 1))

        tanhuh = np.zeros((self.num_layer - 1, self.filter_length, 1), dtype=complex)
        for ii in range(1,self.num_layer - 1):
            tanhuh[ii] = np.tanh(u[ii] * self.thickness[ii-1])

        ztilda = np.ones((1,self.num_layer,1),dtype=np.complex)
        ztilda[0,0,0] = 1j * omega * self.mu0
        ztilda[0,1:self.num_layer,0] = 1j * omega * self.mu[0:self.num_layer]

        Y = np.ones((self.num_layer,self.filter_length,1),dtype=np.complex)

        for ii in range (1,self.num_layer+1):
            Y[ii-1] = u[ii-1] / ztilda[0,ii-1,0]

        """compute Reflection coefficient ( (down to up) and (up to down) )"""
        #   reciver(m) > transmitter(n) (down to up)
        if self.tlayer <= self.rlayer:
            Ytilda = np.ones((self.num_layer, self.filter_length, 1), dtype=np.complex)
            Ytilda[self.num_layer - 1] = Y[self.num_layer - 1]  # (1)Ytilda{self.num_layer}

            r_te = np.ones((self.num_layer, self.filter_length, 1), dtype=np.complex)

            for ii in range(self.num_layer - 1, 1, -1):  # (2)Ytilda{self.num_layer-1,self.num_layer,...,2}
                numerator_Y = Ytilda[ii] + Y[ii - 1] * tanhuh[ii - 1]
                denominator_Y = Y[ii - 1] + Ytilda[ii] * tanhuh[ii - 1]
                Ytilda[ii - 1] = Y[ii - 1] * numerator_Y / denominator_Y

                r_te[ii - 1] = (Y[ii - 1] - Ytilda[ii]) / (Y[ii - 1] + Ytilda[ii])

            r_te[0] = (Y[0] - Ytilda[1]) / (Y[0] + Ytilda[1])
            r_te[self.num_layer - 1] = 0

        # reciver(m) < transmitter(n) (up to down)
        if self.tlayer >= self.rlayer:
            Yhat = np.ones((self.num_layer, self.filter_length, 1), dtype=np.complex)
            Zhat = np.ones((self.num_layer, self.filter_length, 1), dtype=np.complex)
            Yhat[0] = Y[0]  # (1)Y{0}

            R_te = np.ones((self.num_layer, self.filter_length, 1), dtype=np.complex)

            for ii in range(2, self.num_layer):
                numerator_Y = Yhat[ii - 2] + Y[ii - 1] * tanhuh[ii - 1]
                denominator_Y = Y[ii - 1] + Yhat[ii - 2] * tanhuh[ii - 1]
                Yhat[ii - 1] = Y[ii - 1] * numerator_Y / denominator_Y  # (2)Yhat{2,3,...,self.num_layer-2,self.num_layer-1}

                R_te[ii - 1] = (Y[ii - 1] - Yhat[ii - 2]) / (Y[ii - 1] + Yhat[ii - 2])

            R_te[self.num_layer - 1] = (Y[self.num_layer - 1] - Yhat[self.num_layer - 2]) / (Y[self.num_layer - 1] + Yhat[self.num_layer - 2])
            R_te[0] = 0

        """compute Upside and Downside reflection coefficient"""
        U_te = np.ones((self.num_layer, self.filter_length, 1), dtype=np.complex)
        U_tm = np.ones((self.num_layer, self.filter_length, 1), dtype=np.complex)
        D_te = np.ones((self.num_layer, self.filter_length, 1), dtype=np.complex)
        D_tm = np.ones((self.num_layer, self.filter_length, 1), dtype=np.complex)

        def krondel(nn, mm):
            if nn == mm:
                return 1
            else:
                return 0

        if self.tlayer < self.rlayer:
            U_te[0] = 0
            U_te[1] = (Y[1] * (1 + r_te[1 - 1]) + Y[1 - 0] * (1 - r_te[1 - 1])) / (2 * Y[1]) * (
                        0 + krondel(1 - 1, self.tlayer - 1) * np.exp(-u[self.tlayer - 1] * (self.h[0, self.tlayer - 1] - self.hs)))

            for jj in range(3, self.rlayer + 1):
                U_te[0] = 0
                U_te[jj - 1] = (Y[jj - 1] * (1 + r_te[jj - 2]) + Y[jj - 2] * (1 - r_te[jj - 2])) / (
                            2 * Y[jj - 1]) * (U_te[jj - 2] * np.exp(
                    -u[jj - 2] * (self.h[0, jj - 2] - self.h[0, jj - 3])) + krondel(jj - 2, self.tlayer - 1) * np.exp(
                    -u[self.tlayer - 1] * (self.h[0, self.tlayer - 1] - self.hs)))

            if self.rlayer == self.num_layer:
                D_te[self.rlayer - 1] = 0
            else:
                D_te[self.rlayer - 1] = U_te[self.rlayer - 1] * np.exp(
                    -u[self.rlayer - 1] * (self.h[0, self.rlayer - 1] - self.h[0, self.rlayer - 2])) * r_te[self.rlayer - 1]
        elif self.tlayer > self.rlayer:
            D_te[self.num_layer - 1] = 0
            D_te[self.num_layer - 2] = (Y[self.num_layer - 2] * (1 + R_te[self.num_layer - 1]) + Y[self.num_layer - 1] * (1 - R_te[self.num_layer - 1])) / (
                        2 * Y[self.num_layer - 2]) * (0 + krondel(self.num_layer, self.tlayer) * np.exp(
                -u[self.tlayer - 1] * (self.hs - self.h[0, self.tlayer - 2])))
            for jj in range(self.num_layer - 2, self.rlayer - 1, -1):
                D_te[jj - 1] = (Y[jj - 1] * (1 + R_te[jj]) + Y[jj] * (1 - R_te[jj])) / (2 * Y[jj - 1]) * (
                            D_te[jj] * np.exp(-u[jj] * (self.h[0, jj] - self.h[0, jj - 1])) + krondel(jj,
                                                                                            self.tlayer - 1) * np.exp(
                        -u[self.tlayer - 1] * (self.hs - self.h[0, self.tlayer - 2])))
            if self.rlayer == 1:
                U_te[self.rlayer - 1] = 0
            else:
                U_te[self.rlayer - 1] = D_te[self.rlayer - 1] * np.exp(
                    u[self.rlayer - 1] * (self.h[0, self.rlayer - 2] - self.h[0, self.rlayer - 1])) * R_te[self.rlayer - 1]
        elif self.tlayer == self.rlayer:
            if self.rlayer == 1:
                U_te[0] = 0
                D_te[self.rlayer - 1] = 1 / (1 - 0) * r_te[self.rlayer - 1] * (
                            0 + np.exp(-u[self.rlayer - 1] * (self.h[0, self.rlayer - 1] - self.hs)))  # 0 is derived from Rte[0] = 0
            elif self.rlayer == self.num_layer:
                U_te[self.rlayer - 1] = 1 / (1 - 0) * R_te[self.rlayer - 1] * (0 + np.exp(
                    u[self.rlayer - 1] * (self.h[0, self.rlayer - 2] - self.hs)))  # 0 is derived from rTE{self.num_layer} = 0
                D_te[self.num_layer - 1] = 0
            else:
                U_te[self.rlayer - 1] = 1 / (1 - R_te[self.rlayer - 1] * r_te[self.rlayer - 1] * np.exp(
                    -2 * u[self.rlayer - 1] * (self.h[0, self.rlayer - 1] - self.h[0, self.rlayer - 2]))) * R_te[self.rlayer - 1] * (
                                               r_te[self.rlayer - 1] * np.exp(
                                           u[self.rlayer - 1] * (self.h[0, self.rlayer - 2] - 2 * self.h[0, self.rlayer - 1] + self.hs)) + np.exp(
                                           u[self.rlayer - 1] * (self.h[0, self.rlayer - 2] - self.hs)))
                D_te[self.rlayer - 1] = 1 / (1 - R_te[self.rlayer - 1] * r_te[self.rlayer - 1] * np.exp(
                    -2 * u[self.rlayer - 1] * (self.h[0, self.rlayer - 1] - self.h[0, self.rlayer - 2]))) * r_te[self.rlayer - 1] * (
                                               R_te[self.rlayer - 1] * np.exp(-u[self.rlayer - 1] * (
                        self.h[0, self.rlayer - 1] - 2 * self.h[0, self.rlayer - 2] + self.hs)) + np.exp(
                                           -u[self.rlayer - 1] * (self.h[0, self.rlayer - 1] - self.hs)))
        """compute Damping coefficient"""
        if self.rlayer == 1:
            e_up = 0
            e_down = np.exp(u[self.rlayer - 1] * (self.z - self.h[0, self.rlayer - 1]))
        elif self.rlayer == self.num_layer:
            e_up = np.exp(-u[self.rlayer - 1] * (self.z - self.h[0, self.rlayer - 2]))
            e_down = 0
        else:
            e_up = np.exp(-u[self.rlayer - 1] * (self.z - self.h[0, self.rlayer - 2]))
            e_down = np.exp(u[self.rlayer - 1] * (self.z - self.h[0, self.rlayer - 1]))


        """kenel function for 1D electromagnetic"""
        kernel_te = U_te[self.rlayer - 1] * e_up + D_te[self.rlayer - 1] * e_down + krondel(self.rlayer, self.tlayer) * np.exp(
            -u[self.tlayer - 1] * np.abs(self.z - self.hs))
        kernel_te_hz = U_te[self.rlayer - 1] * e_up * u[self.rlayer - 1] ** 2 + D_te[self.rlayer - 1] * e_down * u[
            self.rlayer - 1] ** 2 + krondel(self.rlayer, self.tlayer) * np.exp(-u[self.tlayer - 1] * np.abs(self.z - self.hs)) * u[
                          self.tlayer - 1] ** 2
        kernel_te_hr = U_te[self.rlayer - 1] * e_up * u[self.rlayer - 1] - D_te[self.rlayer - 1] * e_down * u[
            self.rlayer - 1] + krondel(self.rlayer, self.tlayer) * np.exp(-u[self.tlayer - 1] * np.abs(self.z - self.hs)) * u[self.tlayer - 1]

        if transmitter == "vmd":
            # p.209 eq. 4.44, 4.45, 4.46
            kernel_e_phai = kernel_te * self.lamda ** 2 / u[self.tlayer - 1]
            kernel_h_r = kernel_te_hr * self.lamda ** 2 / u[self.tlayer - 1]
            kernel_h_z = (kernel_te_hz + k[:, self.rlayer - 1] ** 2 * kernel_te) * self.lamda / u[self.tlayer - 1]
            return {"kernel_e_phai": kernel_e_phai, "kernel_h_r": kernel_h_r, "kernel_h_z": kernel_h_z,"ztilda[0,self.tlayer-1,0]": ztilda[0, self.tlayer - 1, 0]}
        elif transmitter == "circular_loop":
            besk1 = scipy.special.jn(1, self.lamda * self.r)
            besk0 = scipy.special.jn(0, self.lamda * self.r)
            # p.219 eq. 4.86, 4.87, 4.88
            kernel_e_phai = kernel_te * self.lamda * besk1 / u[self.tlayer - 1]
            kernel_h_r = kernel_te * self.lamda * besk1
            kernel_h_z = kernel_te * self.lamda ** 2 * besk0 / u[self.tlayer - 1]
            return {"kernel_e_phai": kernel_e_phai, "kernel_h_r": kernel_h_r, "kernel_h_z": kernel_h_z, "ztilda[0,self.tlayer-1,0]": ztilda[0, self.tlayer - 1, 0]}
        elif transmitter == "coincident_loop":
            besk1rad = scipy.special.jn(1, self.lamda * self.rad)
            # r = rad in besk1
            # p.221 eq. 4.95
            kernel_h_z = kernel_te * self.lamda * besk1rad / u[self.tlayer - 1]
            return {"kernel_h_z": kernel_h_z, "ztilda[0,self.tlayer-1,0]": ztilda[0, self.tlayer - 1, 0]}

    @abstractmethod
    def hankel_calc(self):
        pass

    @abstractmethod
    def repeat_hankel(self):
        pass

    def vmd_base(self, transmitter):
        # 送信源で計算できない座標が入力されたときエラーを出す
        """
        if self.z == 0:
            warnings.warn("送信機vmdでz座標が0のとき、計算精度は保証されていません", stacklevel=2)
        """
        if self.r == 0:
            raise Exception("x, y座標共に0のため、計算結果が発散しました。")

        self.lamda = self.y_base / self.r
        self.moment = self.moment

        ans = self.repeat_hankel(transmitter)

        return ans

    def circular_loop_base(self, rad, transmitter):
        if rad == 0:
            raise Exception("ループの半径を設定してください")

        # 送信源固有のパラメータ設定
        self.rad = rad
        self.lamda = self.y_base / self.rad
        self.moment = self.moment * self.current * self.turns
        self.coincident_loop_moment = self.moment * self.current * self.turns ** 2

        ans = self.repeat_hankel(transmitter)

        return ans

    def coincidet_loop_base(self, rad, transmitter):
        if rad == 0:
            raise Exception("ループの半径を設定してください")

        # 送信源固有のパラメータ設定
        self.rad = rad
        self.lamda = self.y_base / self.rad
        self.moment = self.coincident_loop_moment

        ans = self.repeat_hankel(transmitter)

        return ans


class Fdem(BaseEm):
    def __init__(self, x, y, z, hs, current, res, thickness,
                 freq_range, plot_number, turns = 1, hankel_filter="kong241"):
        # freq_rangeの渡し方 10^x 〜10^y x, yをリストで渡させる
        # コンストラクタの継承
        super().__init__(x, y, z, hs, current, res, thickness, plot_number, turns)
        # 必ず渡される引数
        self.num_freq = plot_number
        # 渡される引数から計算する変数
        self.freq = np.logspace(freq_range[0], freq_range[1], plot_number)

    def hankel_calc(self, transmitter, index_freq):
        ans = {}
        omega = 2 * np.pi * self.freq[index_freq - 1]
        kernel = self.make_kernel(transmitter, omega)

        if transmitter == "vmd":
            e_phai = np.dot(self.wt1.T, kernel["kernel_e_phai"]) / self.r  # / self.r derive from digital filter convolution
            h_r = np.dot(self.wt1.T, kernel["kernel_h_r"]) / self.r
            h_z = np.dot(self.wt0.T, kernel["kernel_h_z"]) / self.r
            ans["e_x"] = (-1 * kernel["ztilda[0,self.tlayer-1,0]"] * - self.sin_phai * e_phai)/(4 * np.pi)
            ans["e_y"] = (-1 * kernel["ztilda[0,self.tlayer-1,0]"] * self.cos_phai * e_phai)/(4 * np.pi)
            ans["e_z"] = 0
            ans["h_x"] = (self.cos_phai * h_r)/(4 * np.pi)
            ans["h_y"] = (self.sin_phai * h_r)/(4 * np.pi)
            ans["h_z"] = (h_z)/(4 * np.pi)
        elif transmitter == "circular_loop":
            e_phai = np.dot(self.wt1.T, kernel["kernel_e_phai"]) / self.rad
            h_r = np.dot(self.wt1.T, kernel["kernel_h_r"]) / self.rad
            h_z = np.dot(self.wt1.T, kernel["kernel_h_z"]) / self.rad
            ans["e_x"] = (-1 * kernel["ztilda[0,self.tlayer-1,0]"] * self.rad * - self.sin_phai * e_phai) / 2
            ans["e_y"] = (-1 * kernel["ztilda[0,self.tlayer-1,0]"] * self.rad * self.cos_phai * e_phai) / 2
            ans["e_z"] = 0
            ans["h_x"] = (self.rad * self.cos_phai * h_r) / 2
            ans["h_y"] = (self.rad * self.sin_phai * h_r) / 2
            ans["h_z"] = (self.rad * h_z) / 2
        elif transmitter == "coincident_loop":
            h_z_co = np.dot(self.wt1.T, kernel["kernel_h_z"]) / self.rad
            ans["e_x"] = 0
            ans["e_y"] = 0
            ans["e_z"] = 0
            ans["h_x"] = 0
            ans["h_y"] = 0
            ans["h_z"] = (1 * np.pi * self.rad ** 2 * h_z_co)
        return ans

    def repeat_hankel(self, transmitter):
        ans = np.zeros((self.num_freq, 6), dtype=complex)
        for index_freq in range(1, self.num_freq + 1):
            em_field = self.hankel_calc(transmitter, index_freq)
            # 電場の計算
            ans[index_freq - 1, 0] = em_field["e_x"]
            ans[index_freq - 1, 1] = em_field["e_y"]
            ans[index_freq - 1, 2] = em_field["e_z"]
            # 磁場の計算
            ans[index_freq - 1, 3] = em_field["h_x"]
            ans[index_freq - 1, 4] = em_field["h_y"]
            ans[index_freq - 1, 5] = em_field["h_z"]

            ans = self.moment * ans

        return {"freq": self.freq
                , "e_x": ans[:, 0],"e_y": ans[:, 1],"e_z": ans[:, 2]
                , "h_x": ans[:, 3], "h_y": ans[:, 4], "h_z": ans[:, 5]
                }

    def vmd(self):
        transmitter = sys._getframe().f_code.co_name
        ans = self.vmd_base(transmitter)
        return ans

    def circular_loop(self, rad):
        transmitter = sys._getframe().f_code.co_name
        ans = self.circular_loop_base(rad, transmitter)
        return ans

    def coincident_loop (self, rad):
        transmitter = sys._getframe().f_code.co_name
        ans = self.coincident_loop_base(rad, transmitter)
        return ans


    def vmd_analytical(self):
        # 角速度・波数の算出
        omega = 2 * np.pi * self.freq
        k1d = (omega ** 2 * self.mu0 * self.epsrn
               - 1j * omega * self.mu0 * self.sigma[0]) ** 0.5

        # 解析解の計算
        numerator = - self.vmd_moment * (3 - (3 + 3j * k1d * self.r
                                         - k1d ** 2 * self.r ** 2)
                                         * np.exp(-1j * k1d * self.r))
        denominator = 2 * np.pi * self.sigma[0] * self.r ** 4
        e_phai = numerator / denominator

        numerator = self.vmd_moment\
                    * (9 - (9 + 9j * k1d * self.r
                       - 4 * k1d ** 2 * self.r ** 2
                       - 1j * k1d ** 3 * self.r ** 3)
                       * np.exp(-1j * k1d * self.r))
        denominator = 2 * np.pi * k1d ** 2 * self.r ** 5
        h_z = numerator / denominator

        return {"freq": self.freq, "e_phai": e_phai, "h_z": h_z}

    def circular_loop_analytical(self, rad):
        omega = 2 * np.pi * self.freq
        k1d = (omega ** 2 * self.mu0 * self.epsrn
               - 1j * omega * self.mu0 * self.sigma[0]) ** 0.5
        h_z = -self.circular_loop_moment / k1d ** 2 / rad ** 3\
              * (3 - (3 + 3j * k1d * rad - k1d**2 * rad ** 2)
              * np.exp(-1j * k1d * rad))

        return {"freq": self.freq, "h_z": h_z}


###
class Tdem(BaseEm):
    def __init__(self, x, y, z, hs, current, res, thickness,
                 wave_form, time_range, plot_number, turns=1, hankel_filter="kong241"):
        # freq_rangeの渡し方 10^x 〜10^y x, yをリストで渡させる
        # コンストラクタの継承
        super().__init__(x, y, z, hs, current, res, thickness, plot_number, turns)
        # 必ず渡される引数
        self.plot_number = plot_number
        wave_form_list = ["impulse", "step_on", "step_off"]
        if wave_form not in wave_form_list:
            error_content = """Please set valid wave form. 'impulse' or 'step_on' or 'step_off' is only available"
                            """
            raise Exception(error_content)
        else:
            self.wave_form = wave_form

        # 渡される引数から計算する変数
        self.times = np.logspace(time_range[0], time_range[1], plot_number)
        # GS法のフィルター数
        self.gs_filter_length = 12

    def get_gs_coefficient(self, gs_filter_length):
        L = gs_filter_length
        v = np.zeros(L)
        nn2 = L / 2
        for n in range(1, L + 1):
            z = 0.0
            for k in range(math.floor((n + 1) / 2), int(min(n, nn2) + 1)):
                numerator = (k ** nn2) * math.factorial(2 * k)
                denominator = (math.factorial(nn2 - k) * math.factorial(k)
                               * math.factorial(k - 1) * math.factorial(n - k)
                               * math.factorial(2 * k - n))
                z = z + numerator / denominator

            v[n - 1] = (-1) ** (n + nn2) * z
        return v

    def hankel_calc(self, transmitter, time):
        ans = {}
        sum_ = 0.0
        ln2_on_t = np.log(2.0) / time
        # laplace transform parameters
        # if dbdt = 1, divide answer omega(equal s in lanlace domain)
        if self.wave_form == "impulse":
            dbdt = 0
        elif (self.wave_form == "step_on") or (self.wave_form == "step_off"):
            dbdt = 1

        for n in range(1, self.gs_filter_length + 1):
            p = n * ln2_on_t
            omega = p / 1j
            kernel = self.make_kernel(transmitter, omega)
            sum_ = sum_ + self.gs_coefficient[n - 1] * kernel["kernel_h_z"] / (omega ** dbdt)

        if transmitter == "vmd":
            h_z = np.dot(self.wt0.T, sum_)
            ans["h_z"] = ln2_on_t * h_z / (4 * np.pi * self.r)
        elif transmitter == "circular_loop":
            h_z = np.dot(self.wt1.T, sum_)
            ans["h_z"] = ln2_on_t * self.rad * h_z / (self.rad * 2)

        return ans

    def repeat_hankel(self, transmitter):
        ans = np.zeros(self.plot_number, dtype=complex)
        self.gs_coefficient = self.get_gs_coefficient(self.gs_filter_length)
        for time_index, time in enumerate(self.times):
            hankel_result = self.hankel_calc(transmitter, time)
            ans[time_index] = hankel_result["h_z"]

        return {"h_z": ans,  "time": self.times}

    def vmd(self):
        transmitter = sys._getframe().f_code.co_name
        ans = self.vmd_base(transmitter)
        h_z0 = - self.vmd_moment / (4 * np.pi * self.r ** 3)
        if self.wave_form == "impulse":
            pass
        elif self.wave_form == "step_on":
            ans["h_z"] = np.imag(ans["h_z"])
        elif self.wave_form == "step_off":
            ans["h_z"] = h_z0 - np.imag(ans["h_z"])

        return ans

    def circular_loop(self, rad):
        transmitter = sys._getframe().f_code.co_name
        ans = self.circular_loop_base(rad, transmitter)
        h_z0 = self.current / (2 * self.rad)
        if self.wave_form == "impulse":
            ans["h_z"] = abs(ans["h_z"])
        elif self.wave_form == "step_on":
            ans["h_z"] = np.imag(ans["h_z"])
        elif self.wave_form == "step_off":
            ans["h_z"] = h_z0 - abs(np.imag(ans["h_z"]))

        return ans

    def vmd_analytical(self):
        theta = np.sqrt(self.mu0 / (4 * (self.times) * self.res[0]))
        if self.wave_form == "impulse":
            h_z = (self.vmd_moment / (2 * np.pi * self.mu0 * self.sigma[0] * self.r ** 5))\
                   * (9 * erf(theta * self.r) - (2 * theta * self.r / (np.pi ** 0.5))\
                   * (9 + 6 * (theta ** 2) * (self.r ** 2) + 4 * (theta ** 4) * (self.r ** 4))\
                   * np.exp(-(theta ** 2) * (self.r ** 2)))
        elif self.wave_form == "step_on":
            h_z = self.vmd_moment / (4 * np.pi * self.r ** 3)\
                  * (9 / ( 2 * theta ** 2 * self.r ** 2) * erf(theta * self.r) + erfc(theta * self.r)\
                  -1 / np.sqrt(np.pi) * (9 / theta / self.r + 4 * theta * self.r) * np.exp(-1 * theta ** 2 * self.r ** 2))
        elif self.wave_form == "step_off":
            h_z = -self.vmd_moment / (4 * np.pi * self.r ** 3)\
                  * (9 / ( 2 * theta ** 2 * self.r ** 2) * erf(theta * self.r) - erf(theta * self.r)\
                  -1 / np.sqrt(np.pi) * (9 / theta / self.r + 4 * theta * self.r)\
                  * np.exp(-1 * theta ** 2 * self.r ** 2))
            pass

        return {"h_z": h_z, "times": self.times}

    def circular_loop_analytical(self, rad):
        theta = np.sqrt(self.mu0 / (4 * (self.times) * self.res[0]))
        if self.wave_form == "impulse":
            h_z = -self.current / (self.mu0 * self.sigma[0] * self.rad ** 3)\
                  * (3 * erf(theta * self.rad) - (2 / np.pi**(1/2)) * theta * self.rad
                     * (3 + 2 * theta ** 2 * rad ** 2) * np.exp(-theta ** 2 * rad ** 2))
        elif self.wave_form == "step_on":
            print("analytical ans doesn't exist")
            h_z = np.zeros(len(self.times))
        elif self.wave_form == "step_off" or "step_on":
            h_z = self.current / (2 * self.rad)\
                  * ((3 / (np.sqrt(np.pi) * theta * self.rad)) * np.exp(-theta ** 2 * self.rad ** 2)
                     + (1 - 3 / (2 * theta ** 2 * self.rad ** 2)) * erf(theta * self.rad))

        return {"h_z": h_z, "times": self.times}


import em1dpy as em
fdem = em.Fdem(10,10,-10,100,1,np.array([100,100,100]),np.array([20,20]),[0, 6],100, 1,"kong241")
a = fdem.circular_loop(100)
print(a)

