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
    def __init__(self, x, y, z, height_source, current, res, thickness,
                 plot_number, turns=1, hankel_filter="kong241"):
        # 必ず渡させる引数
        self.x = x
        self.y = y
        self.z = z
        self.height_source = - height_source
        self.current = current
        self.res = res  # resistivity
        self.thickness = thickness

        # デフォルト引数に設定しておく引数(変えたいユーザもいる)
        self.turns = turns
        self.hankel_filter = hankel_filter  # or anderson801

        # 指定しておく引数（変えたいユーザはほぼいない）
        self.mu0 = 1.25663706e-6
        self.mu = np.ones(len(res)) * self.mu0
        self.vmd_moment = 1
        self.moment = 1
        self.circular_loop_moment = self.moment * current
        self.coincident_loop_moment = self.moment * turns
        self.epsrn = 8.85418782e-12

        # 渡された引数から計算する変数
        self.num_layer = len(res)
        self.sigma = 1 / res
        self.r = np.sqrt(x ** 2 + y ** 2)
        if self.r == 0:
            warnings.warn('when r=0, Ex and Ey diverge')
            self.cos_phai = 0
            self.sin_phai = 0
        else:
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
            besk1 = scipy.special.jn(1, self.lamda * self.r)
            besk0 = scipy.special.jn(0, self.lamda * self.r)
            besk1rad = scipy.special.jn(1, self.lamda * self.rad)
            # p.219 eq. 4.86, 4.87, 4.88
            kernel_e_phai = (e_up + gamma_te * e_down) * self.lamda * besk1 / u0
            kernel_h_r = (e_up - gamma_te * e_down) * self.lamda * besk1
            kernel_h_z = (e_up + gamma_te * e_down) * (self.lamda ** 2) * besk0 / u0
            kernel_h_z_co = (e_up + gamma_te * e_down) * self.lamda * besk1rad / u0
            return {"kernel_e_phai": kernel_e_phai, "kernel_h_r": kernel_h_r,
                    "kernel_h_z": kernel_h_z, "kernel_h_z_co": kernel_h_z_co, "z_hat0": z_hat0}

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
        self.moment = self.vmd_moment

        ans = self.repeat_hankel(transmitter)

        return ans

    def circular_loop_base(self, rad, transmitter):
        if rad == 0:
            raise Exception("ループの半径を設定してください")

        # 送信源固有のパラメータ設定
        self.rad = rad
        self.lamda = self.y_base / self.rad
        self.moment = self.circular_loop_moment

        ans = self.repeat_hankel(transmitter)

        return ans


class Fdem(BaseEm):
    def __init__(self, x, y, z, height_source, current, res, thickness,
                 freq_range, plot_number, turns = 1, hankel_filter="kong241"):
        # freq_rangeの渡し方 10^x 〜10^y x, yをリストで渡させる
        # コンストラクタの継承
        super().__init__(x, y, z, height_source, current, res, thickness, plot_number, turns)
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

        elif transmitter == "circular_loop":
            h_z = np.dot(self.wt1.T, kernel["kernel_h_z"])
            h_z_co = np.dot(self.wt1.T, kernel["kernel_h_z_co"])
            ans["e_phai"] = (-1 * kernel["z_hat0"] * self.rad * e_phai)\
                             / (2 * self.rad)
            ans["h_r"] = (self.rad * h_r) / (2 * self.rad)
            ans["h_z"] = (self.rad * h_z) / (2 * self.rad)
            ans["hz_co"] = (1 * np.pi * (self.rad ** 2)
                            * h_z_co) / self.rad

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

    def vmd(self):
        transmitter = sys._getframe().f_code.co_name
        ans = self.vmd_base(transmitter)
        return ans

    def circular_loop(self, rad):
        transmitter = sys._getframe().f_code.co_name
        ans = self.circular_loop_base(rad, transmitter)
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


class Tdem(BaseEm):
    def __init__(self, x, y, z, height_source, current, res, thickness,
                 wave_form, time_range, plot_number, turns=1, hankel_filter="kong241"):
        # freq_rangeの渡し方 10^x 〜10^y x, yをリストで渡させる
        # コンストラクタの継承
        super().__init__(x, y, z, height_source, current, res, thickness, plot_number, turns)
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
