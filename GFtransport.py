import numpy as np
from TightBindingHamiltonian import *
import time
from multiprocessing import Pool
from tqdm import tqdm
from GFtransportlib import *
class GFtransport:
    # ---------------------参数设置----------------------#
    YITA = 0.00001
    UCONSTANT = 2
    K0T = 0.005
    Leftu = 0
    Rightu = 0
    HubbardConvergeLimit = 1e-10
    HubbardMaxSteps = 1000
    SelfEngConvergeMaxSteps = 1000
    SelfEngConvergeLimit = 1e-10
    EnergyIntegralPoints = 5000
    EnergyIntegralUplimit = 2
    EnergyIntegralDownlimit = -2
    # 费米函数Pade极点和Pade系数
    Zp = [
        3.14159265358979, 9.42477796076938, 15.7079632679490, 21.9911485751286, 28.2743338823081,
        34.5575191894877, 40.8407044966673, 47.1238898038469, 53.4070751110264, 59.6902604182060, 
        65.9734457253857, 72.2566310325653, 78.5398163397448, 84.8230016469244, 91.1061869541042,
        97.3893722612836, 103.672557568463, 109.955742875643, 116.238928182822, 122.522113490002,
        128.805298797182, 135.088484104361, 141.371669411541, 147.654854718720, 153.938040025931,
        160.221225337789, 166.504411064022, 172.787618736671, 179.071504241591, 185.367056788274,
        191.765840917865, 198.604853405223, 206.408041821521, 215.571490277720, 226.351733940274,
        239.014955600449, 253.914927652947, 271.536176382141, 292.546298521369, 317.878333420979,
        348.866868683407, 387.481622295028, 436.748895347483, 501.562785736091, 590.381915505246,
        719.188910973252, 922.237961322803, 1288.59691621174, 2144.85826724958, 6430.38305655137
    ]

    Rp = [
        -1.00000000000006, -1.00000000000006, -1.00000000000006, -1.00000000000005, -1.00000000000006,
        -1.00000000000004, -1.00000000000006, -1.00000000000007, -1.00000000000008, -1.00000000000003,
        -1.00000000000003, -1.00000000000008, -1.00000000000007, -1.00000000000003, -1.00000000000004,
        -1.00000000000010, -1.00000000000010, -1.00000000000006, -1.00000000000008, -1.00000000000004,
        -1.00000000000011, -1.00000000000007, -1.00000000000001, -1.00000000000013, -1.00000000002635,
        -1.00000000356494, -1.00000028617572, -1.00001350894231, -1.00036640449500, -1.00541429156939,
        -1.04034520308096, -1.15176329223453, -1.34239856930299, -1.58075028465733, -1.85759664889990,
        -2.18251588967103, -2.57301253702303, -3.05350330706287, -3.65863003883404, -4.43961938737623,
        -5.47553549617149, -6.89420558997177, -8.91379740829603, -11.9322664503426, -16.7399689357642,
        -25.0957592727172, -41.5993339168980, -81.6993117416476, -227.242614478673, -2046.52127705660
    ]
    # ---------------------类变量----------------------#

    # 电子密度初始值
    ElectrodeLElecDensityUpAvgInit = None
    ElectrodeLElecDensityDownAvgInit = None
    ElectrodeRElecDensityUpAvgInit = None
    ElectrodeRElecDensityDownAvgInit = None
    ElecDensityUpAvgInit = None
    ElecDensityDownAvgInit = None
    # 哈密顿量
    Hc = None
    Hl0 = None
    Hr0 = None
    Hl1 = None
    Hr1 = None
    Vlc = None
    Vrc = None
    # 经过Hubbard自洽的哈密顿量和电子密度
    HcUp = None
    HcDown = None
    Hl0Up = None
    Hl0Down = None
    Hr0Up = None
    Hr0Down = None
    ElecDensityUpAvg = None
    ElecDensityDownAvg = None
    ElectrodeLElecDensityUpAvg = None
    ElectrodeLElecDensityDownAvg = None
    ElectrodeRElecDensityUpAvg = None
    ElectrodeRElecDensityDownAvg = None

    


    ######################Hubbard迭代#######################
    # ------------------------参数--------------------------#
    # Hc 器件自相关
    # ElecDensityUpAvg 器件上自旋电子平均密度
    # ElecDensityDownAvg 器件下自旋电子平均密度
    # Hl0 左电极自相关
    # ElectrodeLElecDensityUpAvg 左电极上自旋电子平均密度
    # ElectrodeLElecDensityDownAvg 左电极下自旋电子平均密度
    # Hr0 右电极自相关
    # ElectrodeRElecDensityUpAvg 右电极上自旋电子平均密度
    # ElectrodeRElecDensityDownAvg 右电极下自旋电子平均密度
    # Hl1 左电极互相关
    # Hr1 右电极互相关
    # Vlc左电极和器件互相关
    # Vlc右电极和器件互相关
    # ------------------------返回--------------------------#
    # ElecDensityUpAvg 上自旋电子平均密度
    # ElecDensityDownAvg 上自旋电子平均密度


    def hubbardSelfConsist(self):
        print("***Hubbard SCF Start***")
        print("----------------------------------------------------------")
        print("Step\tElecDensityUpAvgDiffAvg\t\tElecDensityDownAvgDiffAvg\t\tTime\t\teUp\t\teDown")
        # 单元原子数
        Hc = self.Hc
        Hl0 = self.Hl0
        Hr0 = self.Hr0
        Hl1 = self.Hl1
        Hr1 = self.Hr1
        Vlc = self.Vlc
        Vrc = self.Vrc
        M = Hc.shape[0]

        # 构建新的哈密顿矩阵
        ElecDensityUpAvg = np.zeros((M, M, self.HubbardMaxSteps + 1), dtype=float)
        ElecDensityDownAvg = np.zeros((M, M, self.HubbardMaxSteps + 1), dtype=float)
        ElecDensityUpAvg[:, :, 0] = self.ElecDensityUpAvgInit
        ElecDensityDownAvg[:, :, 0] = self.ElecDensityDownAvgInit
        ElectrodeLElecDensityUpAvg = self.ElectrodeLElecDensityUpAvgInit
        ElectrodeLElecDensityDownAvg = self.ElectrodeLElecDensityDownAvgInit
        ElectrodeRElecDensityUpAvg = self.ElectrodeRElecDensityUpAvgInit
        ElectrodeRElecDensityDownAvg = self.ElectrodeRElecDensityDownAvgInit
        

        # 迭代计算上下自旋电子密度
        for n in range(1, self.HubbardMaxSteps):
            # 计时
            StartTime = time.time()

            # 构建新的哈密顿矩阵
            identity_M = np.identity(M)
            HcUp = Hc + self.UCONSTANT * (ElecDensityDownAvg[:, :, n - 1] - 0.5 * identity_M)
            HcDown = Hc + self.UCONSTANT * (ElecDensityUpAvg[:, :, n - 1] - 0.5 * identity_M)
            Hl0Up = Hl0 + self.UCONSTANT * (ElectrodeLElecDensityDownAvg - 0.5 * np.identity(Hl0.shape[0]))
            Hl0Down = Hl0 + self.UCONSTANT * (ElectrodeLElecDensityUpAvg - 0.5 * np.identity(Hl0.shape[0]))
            Hr0Up = Hr0 + self.UCONSTANT * (ElectrodeRElecDensityDownAvg - 0.5 * np.identity(Hr0.shape[0]))
            Hr0Down = Hr0 + self.UCONSTANT * (ElectrodeRElecDensityUpAvg - 0.5 * np.identity(Hr0.shape[0]))

            ElecDensityUpAvg[:, :, n] = np.eye(M) * 0.5
            ElecDensityDownAvg[:, :, n] = np.eye(M) * 0.5


            # Pade展开费米函数求能量积分

            for ii in range(0, 50):
                u = min(self.Leftu, self.Rightu)
                ee = u + (1j * self.Zp[ii] * self.K0T)
                SigmaLUp = calElectrodeSelfEng(ee, HcUp, Hl0Up, Hl1, Vlc, self.YITA, self.SelfEngConvergeMaxSteps, self.SelfEngConvergeLimit).squeeze()
                SigmaLDown = calElectrodeSelfEng(ee, HcDown, Hl0Down, Hl1, Vlc, self.YITA, self.SelfEngConvergeMaxSteps, self.SelfEngConvergeLimit).squeeze()
                SigmaRUp = calElectrodeSelfEng(ee, HcUp, Hr0Up, Hr1, Vrc, self.YITA, self.SelfEngConvergeMaxSteps, self.SelfEngConvergeLimit).squeeze()
                SigmaRDown = calElectrodeSelfEng(ee, HcDown, Hr0Down, Hr1, Vrc, self.YITA, self.SelfEngConvergeMaxSteps, self.SelfEngConvergeLimit).squeeze()
                GRcUp = np.linalg.inv(ee * identity_M - HcUp - SigmaLUp - SigmaRUp)
                GRcDown = np.linalg.inv(ee * identity_M - HcDown - SigmaLDown - SigmaRDown)
                ElecDensityUpAvg[:, :, n] -= 2 * self.K0T * self.Rp[ii] * GRcUp.real
                ElecDensityDownAvg[:, :, n] -= 2 * self.K0T * self.Rp[ii] * GRcDown.real
            if not np.isclose(self.Leftu, self.Rightu):
                deltae = 2 * np.abs(self.Leftu - self.Rightu) / self.EnergyIntegralPoints
                energy_range = np.linspace(-np.abs(self.Leftu - self.Rightu), np.abs(self.Leftu - self.Rightu), self.EnergyIntegralPoints)
                args_list = (energy_range, deltae, self.Leftu, self.Rightu, HcUp, HcDown, Hl0Up, Hl0Down, Hl1, Vlc, Hr0Up, Hr0Down, Hr1, Vrc, self.YITA, self.K0T, self.SelfEngConvergeMaxSteps, self.SelfEngConvergeLimit)
                ElecDensityUp, ElecDensityDown = compute_elec_density(args_list)
                ElecDensityUpAvg[:, :, n] += ElecDensityUp
                ElecDensityDownAvg[:, :, n] += ElecDensityDown
            # 记录中心器件和电极电子密度
            ElecDensityUpAvg[:, :, n] = np.diag(np.diag(ElecDensityUpAvg[:, :, n]))
            ElecDensityDownAvg[:, :, n] = np.diag(np.diag(ElecDensityDownAvg[:, :, n]))

            ElectrodeLElecDensityUpAvg = ElecDensityUpAvg[:Hl0.shape[0], :Hl0.shape[0], n]
            ElectrodeLElecDensityDownAvg = ElecDensityDownAvg[:Hl0.shape[0], :Hl0.shape[0], n]

            ElectrodeRElecDensityUpAvg = ElecDensityUpAvg[-Hr0.shape[0]:, - Hr0.shape[0]:, n]
            ElectrodeRElecDensityDownAvg = ElecDensityDownAvg[-Hr0.shape[0]:, - Hr0.shape[0]:, n]
            # 计算总能量
            # totalEUp, totalEDown = self.calTotalEnergy()
            # 记录电子密度变化，判断是否收敛 
            ElecDensityUpAvgDiffAvg = np.sum(np.abs(ElecDensityUpAvg[:, :, n] - ElecDensityUpAvg[:, :, n - 1]))
            ElecDensityDownAvgDiffAvg = np.sum(np.abs(ElecDensityDownAvg[:, :, n] - ElecDensityDownAvg[:, :, n - 1]))
            TimeCost = time.time() - StartTime
            print(
                "{}\t{:.12f}\t\t\t{:.12f}\t\t\t\t{:.3f}s".format(
                    n, ElecDensityUpAvgDiffAvg, ElecDensityDownAvgDiffAvg, TimeCost
                )
            )
            # 收敛判据
            if (ElecDensityUpAvgDiffAvg < self.HubbardConvergeLimit and ElecDensityDownAvgDiffAvg < self.HubbardConvergeLimit):

                print("----------------------------------------------------------")
                print("\t\n***SCF Converged within {} Steps!***\t\n".format(n))
                break
        if (ElecDensityUpAvgDiffAvg > self.HubbardConvergeLimit or ElecDensityDownAvgDiffAvg > self.HubbardConvergeLimit):
            print("----------------------------------------------------------")
            print("-------------------------ERROR----------------------------")
            print("\t\n***SCF Not Converged within {} Steps!***\t\n".format(n))
        # 保存当前电子密度数据，Device
        self.ElecDensityUpAvg = ElecDensityUpAvg[:, :, n]
        self.ElecDensityDownAvg = ElecDensityDownAvg[:, :, n]
        # 保存当前电子密度数据，电极
        self.ElectrodeLElecDensityUpAvg = ElecDensityUpAvg[Hl0.shape[0]: 2 * Hl0.shape[0], Hl0.shape[0]: 2 * Hl0.shape[0], n]
        self.ElectrodeLElecDensityDownAvg = ElecDensityDownAvg[Hl0.shape[0]: 2 * Hl0.shape[0], Hl0.shape[0]: 2 * Hl0.shape[0], n]

        self.ElectrodeRElecDensityUpAvg = ElecDensityUpAvg[-2 * Hr0.shape[0] : - Hr0.shape[0], -2 * Hr0.shape[0] : - Hr0.shape[0], n]
        self.ElectrodeRElecDensityDownAvg = ElecDensityDownAvg[-2 * Hr0.shape[0] : - Hr0.shape[0], -2 * Hr0.shape[0] : - Hr0.shape[0], n]
        self.ElectrodeLElecDensityUpAvg = ElecDensityUpAvg[:Hl0.shape[0], :Hl0.shape[0], n]
        self.ElectrodeLElecDensityDownAvg = ElecDensityDownAvg[:Hl0.shape[0], :Hl0.shape[0], n]

        self.ElectrodeRElecDensityUpAvg = ElecDensityUpAvg[-Hr0.shape[0]:, - Hr0.shape[0]:, n]
        self.ElectrodeRElecDensityDownAvg = ElecDensityDownAvg[-Hr0.shape[0]:, - Hr0.shape[0]:, n]           
        # 保存上下自旋哈密顿量矩阵
        
        self.HcUp = Hc + self.UCONSTANT * (self.ElecDensityDownAvg - (np.identity(M) * 0.5))
        self.HcDown = Hc + self.UCONSTANT * (self.ElecDensityUpAvg - (np.identity(M) * 0.5))
        self.Hl0Up = Hl0 + self.UCONSTANT * (self.ElectrodeLElecDensityDownAvg - (np.identity(Hl0.shape[0]) * 0.5))
        self.Hl0Down = Hl0 + self.UCONSTANT * (self.ElectrodeLElecDensityUpAvg - (np.identity(Hl0.shape[0]) * 0.5))
        self.Hr0Up = Hr0 + self.UCONSTANT * (self.ElectrodeRElecDensityDownAvg - (np.identity(Hr0.shape[0]) * 0.5))
        self.Hr0Down = Hr0 + self.UCONSTANT * (self.ElectrodeRElecDensityUpAvg - (np.identity(Hr0.shape[0]) * 0.5))
        return np.diag(ElecDensityUpAvg[:, :, n].diagonal(0)), np.diag(ElecDensityDownAvg[:, :, n].diagonal(0))


    ######################透射率计算#######################
    # ------------------------参数--------------------------#
    # e 计算的能量点
    # Hc 器件自相关
    # H0 电极自相关
    # H1 电极互相关
    # Vlc左电极和器件互相关
    # Vlc右电极和器件互相关
    # ElecDensityUpAvg 器件上自旋电子平均密度
    # ElecDensityDownAvg 器件下自旋电子平均密度
    # ElectrodeLElecDensityUpAvg 左电极上自旋电子平均密度
    # ElectrodeLElecDensityDownAvg 左电极下自旋电子平均密度
    # ElectrodeRElecDensityUpAvg 右电极上自旋电子平均密度
    # ElectrodeRElecDensityDownAvg 右电极下自旋电子平均密度
    # ------------------------返回--------------------------#
    # TUp 上自旋电子透射率
    # TDown 下自旋电子透射率
    def calTransmission(self, e):
        Hc = self.Hc
        Hl0 = self.Hl0
        Hr0 = self.Hr0
        Hl1 = self.Hl1
        Hr1 = self.Hr1
        Vlc = self.Vlc
        Vrc = self.Vrc
        ElecDensityUpAvg = self.ElecDensityUpAvg
        ElecDensityDownAvg = self.ElecDensityDownAvg
        TUp = []
        TDown = []
        M = Hc.shape[0]
        print("***Calculation of Transmission Start***")
        # 构建哈密顿量
        
        HcUp = Hc + self.UCONSTANT * (ElecDensityDownAvg - (np.identity(M) * 0.5))
        HcDown = Hc + self.UCONSTANT * (ElecDensityUpAvg - (np.identity(M) * 0.5))
        Hl0Up = Hl0
        Hl0Down = Hl0
        Hr0Up = Hr0
        Hr0Down = Hr0

        e = tqdm(e)
        for ee in e:
            e.set_description("Calculating Transmission Point")

            # 计算上自旋电子左右电极自能
            SigmaLUp = calElectrodeSelfEng(ee, HcUp, Hl0Up, Hl1, Vlc, self.YITA, self.SelfEngConvergeMaxSteps, self.SelfEngConvergeLimit).squeeze()
            SigmaRUp = calElectrodeSelfEng(ee, HcUp, Hr0Up, Hr1, Vrc, self.YITA, self.SelfEngConvergeMaxSteps, self.SelfEngConvergeLimit).squeeze()

            # 计算下自旋电子左右电极自能
            SigmaLDown = calElectrodeSelfEng(ee, HcDown, Hl0Down, Hl1, Vlc, self.YITA, self.SelfEngConvergeMaxSteps, self.SelfEngConvergeLimit).squeeze()
            SigmaRDown = calElectrodeSelfEng(ee, HcDown, Hr0Down, Hr1, Vrc, self.YITA, self.SelfEngConvergeMaxSteps, self.SelfEngConvergeLimit).squeeze()

            # 线宽函数
            GammaLUp = 1j * (SigmaLUp - np.conjugate(SigmaLUp))
            GammaRUp = 1j * (SigmaRUp - np.conjugate(SigmaRUp))
            GammaLDown = 1j * (SigmaLDown - np.conjugate(SigmaLDown))
            GammaRDown = 1j * (SigmaRDown - np.conjugate(SigmaRDown))
            # 计算器件的格林函数
            identity_matrix = (ee + 1j * self.YITA) * np.identity(M)
            GRcUp = np.linalg.inv(np.ascontiguousarray(identity_matrix - HcUp - SigmaLUp - SigmaRUp))
            GRcDown = np.linalg.inv(np.ascontiguousarray(identity_matrix - HcDown - SigmaLDown - SigmaRDown))
            # 计算透射率
            tUp = np.trace(
                np.dot(
                    np.dot(np.dot(GRcUp, GammaRUp), np.transpose(GRcUp).conjugate()),
                    GammaLUp,
                )
            )
            tDown = np.trace(
                np.dot(
                    np.dot(np.dot(GRcDown, GammaRDown), np.transpose(GRcDown).conjugate()),
                    GammaLDown,
                )
            )

            TUp.append(tUp.real)
            TDown.append(tDown.real)
        print("***Calculation of Transmission Finished***")
        return TUp, TDown

    ###################计算体系局域电流####################
    # 同时会保存局域电流为DFTB的形式
    # ------------------------参数--------------------------#
    # e 能量
    # H 总哈密顿矩阵
    # Hv
    # HvHD
    # ------------------------返回--------------------------#
    # LocalCurrentUpL 从电极L到R上自旋电流
    # LocalCurrentUpR 从电极R到L上自旋电流
    # LocalCurrentDownL 从电极L到R下自旋电流
    # LocalCurrentDownR 从电极R到L下自旋电流
    def calLocalCurrent(self, e, H, Hv, HvHD):
        
        print("***Calculation of Local Current Start***")

        # 构建哈密顿量
        HUp = H + Hv + HvHD
        HDown = H - Hv + HvHD

        HcUp = HUp[40:-40, 40:-40]
        HcDown = HDown[40:-40, 40:-40]
        Hl0Up = HUp[20:40, 20:40]
        Hl0Down = HDown[20:40, 20:40]
        Hr0Up = HUp[-40:-20, -40:-20]
        Hr0Down = HDown[-40:-20, -40:-20]
        Hl1Up = HUp[:20, 20:40]
        Hl1Down = HDown[:20, 20:40]
        Hr1Up = HUp[-20:, -40:-20]
        Hr1Down = HDown[-20:, -40:-20]
        VlcUp = HUp[20:40, 40:200]
        VlcDown = HDown[20:40, 40:200]
        VrcUp = HUp[200:220, 40:200]
        VrcDown = HDown[200:220, 40:200]

        M = HcUp.shape[0]
        
        # 计算上自旋电子左右电极自能
        SigmaLUpRetarded = calElectrodeSelfEng(e, HcUp, Hl0Up, Hl1Up, VlcUp, self.YITA, self.SelfEngConvergeMaxSteps, self.SelfEngConvergeLimit).squeeze()
        SigmaRUpRetarded = calElectrodeSelfEng(e, HcUp, Hr0Up, Hr1Up, VrcUp, self.YITA, self.SelfEngConvergeMaxSteps, self.SelfEngConvergeLimit).squeeze()
        SigmaLUpAdvanced = calElectrodeSelfEng(e, HcUp, Hl0Up, Hl1Up, VlcUp, -self.YITA, self.SelfEngConvergeMaxSteps, self.SelfEngConvergeLimit).squeeze()
        SigmaRUpAdvanced = calElectrodeSelfEng(e, HcUp, Hr0Up, Hr1Up, VrcUp, -self.YITA, self.SelfEngConvergeMaxSteps, self.SelfEngConvergeLimit).squeeze()


        # 计算下自旋电子左右电极自能
        SigmaLDownRetarded = calElectrodeSelfEng(e, HcDown, Hl0Down, Hl1Down, VlcDown, self.YITA, self.SelfEngConvergeMaxSteps, self.SelfEngConvergeLimit).squeeze()
        SigmaRDownRetarded = calElectrodeSelfEng(e, HcDown, Hr0Down, Hr1Down, VrcDown, self.YITA, self.SelfEngConvergeMaxSteps, self.SelfEngConvergeLimit).squeeze()
        SigmaLDownAdvanced = calElectrodeSelfEng(e, HcDown, Hl0Down, Hl1Down, VlcDown, -self.YITA, self.SelfEngConvergeMaxSteps, self.SelfEngConvergeLimit).squeeze()
        SigmaRDownAdvanced = calElectrodeSelfEng(e, HcDown, Hr0Down, Hr1Down, VrcDown, -self.YITA, self.SelfEngConvergeMaxSteps, self.SelfEngConvergeLimit).squeeze()

        # 线宽函数
        
        GammaLUp = 1j * (SigmaLUpRetarded - SigmaLUpAdvanced)
        GammaRUp = 1j * (SigmaRUpRetarded - SigmaRUpAdvanced)
        GammaLDown = 1j * (SigmaLDownRetarded - SigmaLDownAdvanced)
        GammaRDown = 1j * (SigmaRDownRetarded - SigmaRDownAdvanced)

        # 计算器件的格林函数 GR(A)cUp R-Retarded A-Advanced
        identity_matrix = (e + 1j * self.YITA) * np.identity(M)
        GRcUp = np.linalg.inv(identity_matrix - HcUp - SigmaLUpRetarded - SigmaRUpRetarded)
        GRcDown = np.linalg.inv(identity_matrix - HcDown - SigmaLDownRetarded - SigmaRDownRetarded)
        GAcUp = np.linalg.inv(identity_matrix - HcUp - SigmaLUpAdvanced - SigmaRUpAdvanced)
        GAcDown = np.linalg.inv(identity_matrix - HcDown - SigmaLDownAdvanced - SigmaRDownAdvanced)

        # 计算格林函数G^< GLess 
        # GLessUp(Down)L(R) = GRcUp(Down) * GammaL(R)Up(Down) * GAcUp(Down) 
        # L(R)指电子来源，例如L指从L电极到R电极
        GLessUpL = np.dot(np.dot(GRcUp, GammaLUp), GAcUp)
        GLessDownL = np.dot(np.dot(GRcDown, GammaLDown), GAcDown)
        GLessUpR = np.dot(np.dot(GRcUp, GammaRUp), GAcUp)
        GLessDownR = np.dot(np.dot(GRcDown, GammaRDown), GAcDown)

        # # 计算电子流密度
        # RhoUpL = np.diagonal(GLessUpL).real / np.pi
        # RhoDownL = np.diagonal(GLessDownL).real / np.pi

        # # 计算局域态密度
        # DOSUp = - np.diagonal(GRcUp).imag / np.pi
        # DOSDown = - np.diagonal(GRcDown).imag / np.pi

        def calculate(Hc, GLess):
            LocalCurrent = np.zeros(Hc.shape, dtype=complex)
            for i in range(Hc.shape[0]):
                for j in range(Hc.shape[1]):
                    LocalCurrent[i, j] = -1j * (Hc[j, i] * GLess[i, j] - GLess[j, i] * Hc[i, j])
            return LocalCurrent.real

        # 计算局域电流
        LocalCurrentUpL = calculate(HcUp, GLessUpL)
        LocalCurrentUpR = calculate(HcUp, GLessUpR)
        LocalCurrentDownL = calculate(HcDown, GLessDownL)
        LocalCurrentDownR = calculate(HcDown, GLessDownR)

        def saveLocalCurrent(path, LocalCurrent):

            with open(path, 'w') as f:
                LocalCurrentIndex = np.argsort(abs(LocalCurrent), axis = 1)[:,::-1]
                for i in range(HcUp.shape[0]):
                    f.write(str(i + 1) + "  ")
                    for j in range(HcUp.shape[1]):
                        if np.abs(LocalCurrent[i, LocalCurrentIndex[i, j]]) < 1e-50:
                            break
                        f.write(str(LocalCurrentIndex[i, j] + 1) + "  ")
                        f.write(str(LocalCurrent[i, LocalCurrentIndex[i, j]]) + "  ")
                    f.write("\n")
        
        saveLocalCurrent('.\LocalCurrent\lcurrentUp.txt', LocalCurrentUpL)
        saveLocalCurrent('.\LocalCurrent\lcurrentDown.txt', LocalCurrentDownL)
        saveLocalCurrent('.\LocalCurrent\\rcurrentUp.txt', LocalCurrentUpR)
        saveLocalCurrent('.\LocalCurrent\\rcurrentDown.txt', LocalCurrentDownR)

        return LocalCurrentUpL, LocalCurrentUpR, LocalCurrentDownL, LocalCurrentDownR

    # 多线程任务
    def calculUp(self, i):
            interval = (self.EnergyIntegralUplimit - self.EnergyIntegralDownlimit) / self.EnergyIntegralPoints
            result = None
            point = i * interval - (self.EnergyIntegralUplimit - self.EnergyIntegralDownlimit) * 0.5
            if point / self.K0T < -100:
                result = interval * np.linalg.inv(point * np.identity(self.HcUp.shape[0]) - self.HcUp - calElectrodeSelfEngTotal(point, self.HcUp, self.Hl0Up, self.Hr0Up, self.Hl1, self.Hr1, self.Vlc, self.Vrc, self.YITA, self.SelfEngConvergeMaxSteps, self.SelfEngConvergeLimit).squeeze())
            elif point / self.K0T > 100:
                result = np.zeros((self.HcUp.shape[0], self.HcUp.shape[0]))
            else:
                GRc = np.linalg.inv(point * np.identity(self.HcUp.shape[0]) - self.HcUp - calElectrodeSelfEngTotal(point, self.HcUp, self.Hl0Up, self.Hr0Up, self.Hl1, self.Hr1, self.Vlc, self.Vrc, self.YITA, self.SelfEngConvergeMaxSteps, self.SelfEngConvergeLimit).squeeze())
                coeff = 1 / (np.exp(point/ self.K0T) + 1)
                result = interval * GRc * coeff
            return result
    def calculDown(self, i):
            interval = (self.EnergyIntegralUplimit - self.EnergyIntegralDownlimit) / self.EnergyIntegralPoints
            result = None
            point = i * interval - (self.EnergyIntegralUplimit - self.EnergyIntegralDownlimit) * 0.5
            if point / self.K0T < -100:
                result = interval * np.linalg.inv(point * np.identity(self.HcDown.shape[0]) - self.HcDown - calElectrodeSelfEngTotal(point, self.HcDown, self.Hl0Down, self.Hr0Down, self.Hl1, self.Hr1, self.Vlc, self.Vrc, self.YITA, self.SelfEngConvergeMaxSteps, self.SelfEngConvergeLimit).squeeze())
            elif point / self.K0T > 100:
                result = np.zeros((self.HcDown.shape[0], self.HcDown.shape[0]))
            else:
                GRc = np.linalg.inv(point * np.identity(self.HcDown.shape[0]) - self.HcDown - calElectrodeSelfEngTotal(point, self.HcDown, self.Hl0Down, self.Hr0Down, self.Hl1, self.Hr1, self.Vlc, self.Vrc, self.YITA, self.SelfEngConvergeMaxSteps, self.SelfEngConvergeLimit).squeeze())
                coeff = 1 / (np.exp(point/ self.K0T) + 1)
                result = interval * GRc * coeff
            return result
    def calTotalEnergy(self):
        

        integralRange = range(self.EnergyIntegralPoints)
        with Pool() as p:
            results = p.map(self.calculUp, integralRange)
        eTotalUp = np.stack(results, axis=-1)

        with Pool() as p:
            results = p.map(self.calculDown, integralRange)
        eTotalDown = np.stack(results, axis=-1)
        return np.imag(np.sum(eTotalUp, axis = 2).trace()), np.imag(np.sum(eTotalDown, axis = 2).trace())
    
    
    ###################初始化####################
    # 初始化，读取哈密顿量矩阵
    def __init__(self, configPath):
        
        # 设置一些常数
        # yita
        self.YITA = 0.0001
        # Hubbard系数
        self.UCONSTANT = 2
        # 玻尔兹曼常数*温度
        self.K0T = 0.005
        # 化学势
        self.Leftu = 0
        self.Rightu = 0
        # Hubbard自洽收敛精度
        self.HubbardConvergeLimit = 1e-3
        # Hubbard自洽最大步数
        self.HubbardMaxSteps = 1000
        # 电极自能计算收敛精度
        self.SelfEngConvergeLimit = 1e-10
        # 电极自能计算最大步数
        self.SelfEngConvergeMaxSteps = 5000
        # 能量积分计算精度
        self.EnergyIntegralPoints = 5000
        # 能量积分计算上限
        self.EnergyIntegralUplimit = 2
        # 能量积分计算下限
        self.EnergyIntegralDownlimit = -2
        
        # 从文件读取Hamiltonian矩阵
        Hl0, Hr0, Hl1, Hr1, Vlc, Vrc, Hc, \
        ElectrodeLElecDensityUpAvgInit, ElectrodeLElecDensityDownAvgInit,\
        ElectrodeRElecDensityUpAvgInit,ElectrodeRElecDensityDownAvgInit,\
        ElecDensityUpAvgInit, ElecDensityDownAvgInit = LoadHamiltonianTxt(configPath)

        self.Hl0 = Hl0
        self.Hr0 = Hr0
        self.Hl1 = Hl1
        self.Hr1 = Hr1
        self.Vlc = Vlc
        self.Vrc = Vrc
        self.Hc = Hc
        self.ElectrodeLElecDensityUpAvgInit = ElectrodeLElecDensityUpAvgInit
        self.ElectrodeLElecDensityDownAvgInit = ElectrodeLElecDensityDownAvgInit
        self.ElectrodeRElecDensityUpAvgInit = ElectrodeRElecDensityUpAvgInit
        self.ElectrodeRElecDensityDownAvgInit = ElectrodeRElecDensityDownAvgInit
        self.ElecDensityUpAvgInit = ElecDensityUpAvgInit
        self.ElecDensityDownAvgInit = ElecDensityDownAvgInit
    