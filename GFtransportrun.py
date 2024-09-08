from GFtransport import *
from matplotlib import pyplot as plt
import matplotlib
if __name__ == '__main__':
     plt.rc('font',family='Times New Roman')
     matplotlib.rcParams['font.family'] = 'Times New Roman'
     matplotlib.rcParams['mathtext.default'] = 'regular'
#     GF = GFtransport()
#     nx = 12 
#     ny = 5 
#     t = 1.0
#     coordinatesX, coordinatesY, H0, Hv, HvHD = haldaneGrapheneHamitonlian(nx, ny, t)
#     LocalCurrentUpL, LocalCurrentUpR, LocalCurrentDownL, LocalCurrentDownR = GF.calLocalCurrent(0.01, H0, Hv, HvHD)
     GF = GFtransport('.\Hamiltonian\\4C_11C_4C.txt')
     # 设置一些常数
     # yita
     GF.YITA = 0.01
     # Hubbard系数 单位eV
     GF.UCONSTANT = 0.8
     # 玻尔兹曼常数*温度 50K 单位eV
     GF.K0T = 0.005
     # 化学势
     GF.Leftu = 0
     GF.Rightu = 0
     # Hubbard自洽收敛精度
     GF.HubbardConvergeLimit = 1e-8
     # Hubbard自洽最大步数
     GF.HubbardMaxSteps = 500
     # 电极自能计算收敛精度
     GF.SelfEngConvergeLimit = 1e-8
     # 电极自能计算最大步数
     GF.SelfEngConvergeMaxSteps = 200
     # 能量积分计算精度
     GF.EnergyIntegralPoints = 3000
     # 能量积分计算上限
     GF.EnergyIntegralUplimit = 2
     # 能量积分计算下限
     GF.EnergyIntegralDownlimit = -2
     Polarization = []
     SpinFiltering = []
     Transmission = []
     vars = np.linspace(0, 0.3, 31)    # 器件Hubbard电子自洽
     # for i in vars:
     #      print("Current site energy is {}".format(-i))
     #      # GF.Hc[0, 0] = -i
     #      # GF.Hc[-1, -1] = -i
     #      # GF.Vlc[-1, 0] = -i
     #      # GF.Vrc[0, -1] = -i
     #      GF.Leftu = i
     #      GF.hubbardSelfConsist()
     #      # -------------------------计算透射率---------------------------#

     #      # 计算透射率
     #      Steps = 1
     #      e = np.linspace(0, 0, Steps)
     #      TUp, TDown = GF.calTransmission(e)
     #      TUp = np.array(TUp)
     #      TDown = np.array(TDown)
     #      SpinFiltering.append(np.sum(TUp - TDown) / np.sum(TUp + TDown))
     #      Polarization.append(np.sum(np.diag(np.abs(GF.ElecDensityUpAvg - GF.ElecDensityDownAvg))))
     #      Transmission.append(TUp + TDown)
     #      print(SpinFiltering[-1], Polarization[-1], Transmission[-1][0])

          # # -------------------------透射率绘图---------------------------#
          # fig = plt.figure(figsize=(12, 6), dpi=100)
          # p1 = fig.add_subplot(211)
          # p1.plot(e, TUp, "b")
          # p1.plot(e, TDown, "r", linestyle='--')
          # plt.title('Transmission')
          # # -------------------------极化绘图---------------------------#
          # p2 = fig.add_subplot(212)
          # p2.plot(range(GF.ElecDensityUpAvg.shape[0]), np.diag(GF.ElecDensityUpAvg), "b")
          # p2.plot(range(GF.ElecDensityDownAvg.shape[0]), np.diag(GF.ElecDensityDownAvg), "r", linestyle='--')
          # plt.ylim((0.4, 0.6))
          # plt.title('Polarisation')
          # plt.show()
     # np.save("data.npy", {"Transmission":Transmission, "SpinFiltering":SpinFiltering, "Polarization":Polarization})
     
     data = np.load("./data.npy", allow_pickle=True)
     Transmission = data.item()["Transmission"]
     SpinFiltering = data.item()["SpinFiltering"]
     Polarization = data.item()["Polarization"]
     # 创建主图和子图
     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), gridspec_kw={'height_ratios': [3, 1]})

     # 第一个子图 - Spin Filtering Ratio 和 Spin Polarization
     ax1.plot(vars, SpinFiltering, color='black', marker='+', markersize=10, markeredgewidth=1, label='Spin Filtering')
     ax1.set_ylabel('Spin Filtering Ratio', fontsize=26, labelpad=10)  # 增加labelpad确保标题显示全
     ax1.grid(True, linestyle='--', alpha=0.7)
     ax1.spines['top'].set_color('black')
     ax1.spines['right'].set_color('black')
     ax1.spines['bottom'].set_color('black')
     ax1.spines['left'].set_color('black')
     ax1.tick_params(which='both', direction='in', length=4, width=1, colors='black', labelsize=22)
     ax1.set_yticks(np.linspace(0, 0.10, 6))
     ax1.set_ylim(-0.005, 0.11)

     # 隐藏第一个图的x轴刻度数字
     ax1.tick_params(axis='x', labelbottom=False)

     # twin axes for Spin Polarization
     ax3 = ax1.twinx()
     ax3.plot(vars, Polarization, color='#c94737', marker='+', markersize=10, markeredgewidth=1, label='Spin Polarization')
     ax3.set_ylabel('Spin Polarization', fontsize=26, color='#c94737', labelpad=10)  # 增加labelpad确保标题显示全
     ax3.tick_params(axis='y', colors='#c94737', length=4, width=1, labelsize=22)
     ax3.set_yticks(np.linspace(0, 1.5, 6))
     ax3.set_ylim(-0.075, 1.65)

     # 第二个子图 - Transmission
     ax2.plot(vars, Transmission, color='black', marker='+', markersize=10, markeredgewidth=1, label='Transmission')
     ax2.set_ylabel(r'$T_{total}$', fontsize=26, labelpad=23)  # 增加labelpad确保标题显示全
     ax2.tick_params(which='both', direction='in', length=4, width=1, colors='black', labelsize=22)
     ax2.grid(True, linestyle='--', alpha=0.7)
     ax2.set_yticks(np.linspace(0, 1.6, 3))
     ax2.set_ylim(-0.08, 1.76)

     # 在第二幅图下方添加x轴标题
     ax2.set_xlabel("Bias (V)", fontsize=26)
     # 设置子图之间的间距
     plt.subplots_adjust(hspace=0.1, right=0.85, left = 0.15)  # 减少图间距

     # 显示图形
     plt.show()