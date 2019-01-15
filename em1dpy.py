import warnings
from abc import ABCMeta, abstractmethod
import sys

import numpy as np
import scipy.io
from scipy.special import erf


def explain():
    text = """
    このプログラムの使い方
    1. 時間領域、周波数領域どちらかのクラスをインスタンス化する。
        引数は、
        受信機のx座標：x
        受信機のy座標：y
        受信機のz座標：ｚ
        送信機の高さ：height_source
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
    def __init__(self, x, y, z, height_source, current, res, thickness, plot_number, hankel_filter="kong241"):
        # 必ず渡させる引数
        self.x = x
        self.y = y
        self.z = z
        self.height_source = - height_source
        self.current = current
        self.res = res  # resistivity
        self.thickness = thickness

        # デフォルト引数に設定しておく引数(変えたいユーザもいる)
        self.hankel_filter = hankel_filter  # or anderson801

        # 指定しておく引数（変えたいユーザはほぼいない）
        self.mu0 = 1.25663706e-6
        self.mu = np.ones(len(res)) * self.mu0
        self.vmd_moment = 1
        self.moment = 1
        self.epsrn = 8.85418782e-12

        # 渡された引数から計算する変数
        self.num_layer = len(res)
        self.sigma = 1 / res
        self.r = np.sqrt(x ** 2 + y ** 2)
        self.cos_phai = self.x / self.r
        self.sin_phai = self.y / self.r

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
        k = (omega ** 2.0 * self.mu * self.epsrn
             - 1j * omega * self.mu * self.sigma) ** 0.5
        u0 = self.lamda
        u = np.zeros((self.num_layer, self.filter_length, 1), dtype=complex)

        for ii in range(self.num_layer):
            u[ii] = ((self.lamda ** 2 - k[ii] ** 2) ** 0.5)\
                    .reshape((self.filter_length, 1))

        tanhuh = np.zeros((self.num_layer - 1, self.filter_length, 1), dtype=complex)

        for ii in range(self.num_layer - 1):
            tanhuh[ii] = np.tanh(u[ii] * self.thickness[ii])

        z_hat0 = 1j * omega * self.mu0
        y_0 = u0 / z_hat0
        z_hat = 1j * omega * self.mu
        y_ = np.zeros((self.num_layer, self.filter_length, 1), dtype=complex)
        y_hat = np.zeros((self.num_layer, self.filter_length, 1), dtype=complex)

        for ii in range(self.num_layer):
            y_[ii] = u[ii] / z_hat[ii]

        y_hat[self.num_layer - 1, :, 0] = y_[self.num_layer - 1, :, 0]

        if self.num_layer >= 2:
            numerator = y_hat[self.num_layer - 1, :, 0] \
                        + y_[self.num_layer - 2, :, 0] \
                        * tanhuh[self.num_layer - 2, :, 0]
            denominator = y_[self.num_layer - 2, :, 0] \
                          + y_hat[self.num_layer - 1, :, 0] \
                          * tanhuh[self.num_layer - 2, :, 0]
            y_hat[self.num_layer - 2, :, 0] = y_[self.num_layer - 2, :, 0] \
                                              * numerator / denominator

            if self.num_layer >= 3:
                for ii in range(self.num_layer - 2, 0, -1):
                    numerator = y_hat[ii, :, 0] \
                                + y_[ii - 1, :, 0] * tanhuh[ii - 1, :, 0]
                    denominator = y_[ii - 1, :, 0] \
                                  + y_hat[ii, :, 0] * tanhuh[ii, :, 0]
                    y_hat[ii - 1, :, 0] = y_[ii - 1, :, 0]\
                                          * numerator / denominator

        elif self.num_layer == 1:
            # 1層のとき、特に処理なし
            pass

        gamma_te = ((y_0 - y_hat[0, :, 0].reshape(self.filter_length, 1))
                    / (y_0 + y_hat[0, :, 0].reshape(self.filter_length, 1)))
        e_up = np.exp(-u0 * (self.z + self.height_source))
        e_down = np.exp(u0 * (self.z - self.height_source))

        if transmitter == "vmd":
            kernel_e_phai = (e_up + gamma_te * e_down) * self.lamda ** 2 / u0
            kernel_h_r = (e_up - gamma_te * e_down) * self.lamda ** 2
            kernel_h_z = kernel_e_phai * self.lamda
            return {"kernel_e_phai": kernel_e_phai, "kernel_h_r": kernel_h_r,
                    "kernel_h_z": kernel_h_z, "z_hat0": z_hat0}
        elif transmitter == "circular_loop" or "coincident_loop":
            pass

    @abstractmethod
    def hankel_calc(self):
        pass

    @abstractmethod
    def repeat_hankel(self):
        pass

    def vmd(self):
        # 送信源で計算できない座標が入力されたときエラーを出す
        """
        if self.z == 0:
            warnings.warn("送信機vmdでz座標が0のとき、計算精度は保証されていません", stacklevel=2)
        """
        if self.r == 0:
            raise Exception("x, y座標共に0のため、計算結果が発散しました。")

        # 送信源固有のパラメータ設定
        transmitter = sys._getframe().f_code.co_name

        self.lamda = self.y_base / self.r
        self.moment = self.vmd_moment

        ans = self.repeat_hankel(transmitter)

        return ans


class Fdem(BaseEm):
    def __init__(self, x, y, z, height_source, current, res, thickness, freq_range, plot_number, hankel_filter="kong241"):
        # freq_rangeの渡し方 10^x 〜10^y x, yをリストで渡させる
        # コンストラクタの継承
        super().__init__(x, y, z, height_source, current, res, thickness, plot_number)
        # 必ず渡される引数
        self.num_freq = plot_number
        # 渡される引数から計算する変数
        self.freq = np.logspace(freq_range[0], freq_range[1], plot_number)

    def hankel_calc(self, transmitter, index_freq):
        ans = {}
        omega = 2 * np.pi * self.freq[index_freq - 1]
        kernel = self.make_kernel(transmitter, omega)
        e_phai = np.dot(self.wt1.T, kernel["kernel_e_phai"])
        h_r = np.dot(self.wt1.T, kernel["kernel_h_r"])

        if transmitter == "vmd":
            h_z = np.dot(self.wt0.T, kernel["kernel_h_z"])
            ans["e_phai"] = (-1 * kernel["z_hat0"] * e_phai) / (4 * np.pi * self.r)
            ans["h_r"] = h_r / (4 * np.pi * self.r)
            ans["h_z"] = h_z / (4 * np.pi * self.r)

        elif transmitter == "circular_loop" or "coincident_loop":
            pass

        return ans

    def repeat_hankel(self, transmitter):
        e_phai = np.zeros(self.num_freq, dtype=complex)
        h_r = np.zeros(self.num_freq, dtype=complex)
        ans = np.zeros((self.num_freq, 6), dtype=complex)

        for index_freq in range(1, self.num_freq + 1):
            em_field = self.hankel_calc(transmitter, index_freq)
            e_phai[index_freq - 1] = em_field["e_phai"].reshape(1)
            h_r[index_freq - 1] = em_field["h_r"]
            # 電場の計算
            ans[index_freq - 1, 0] = - self.sin_phai * e_phai[index_freq - 1]  # Ex
            ans[index_freq - 1, 1] = self.cos_phai * e_phai[index_freq - 1]  # Ey
            ans[index_freq - 1, 2] = 0  # Ez
            # 磁場の計算
            ans[index_freq - 1, 3] = self.cos_phai * h_r[index_freq - 1]  # Hx
            ans[index_freq - 1, 4] = self.sin_phai * h_r[index_freq - 1]  # Hy
            ans[index_freq - 1, 5] = em_field["h_z"]  # Hz

            ans = self.moment * ans

        return {"freq": self.freq
                , "e_phai": e_phai
                , "h_x": ans[:, 3], "h_y": ans[:, 4], "h_z": ans[:, 5]
                }

    def loop(self):
        pass

    def coincident_loop():
        pass

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

class Tdem():
    @classmethod
    def buc(self):
        print("be used")

    @classmethod
    def uc(self):
        print("I'm uc")
        Tdem.buc()
