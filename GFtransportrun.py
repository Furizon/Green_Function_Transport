from GFtransport import *
from matplotlib import pyplot as plt
if __name__ == '__main__':
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
    GF.HubbardMaxSteps = 2000
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
    couple = np.linspace(0, 0.3, 20)    # 器件Hubbard电子自洽
    for i in couple:
        print("Current site energy is {}".format(-i))
        # GF.Hc[0, 0] = -i
        # GF.Hc[-1, -1] = -i
     #    GF.Vlc[-1, 0] = -i
     #    GF.Vrc[0, -1] = -i
        GF.Leftu = i
        GF.hubbardSelfConsist()
        # -------------------------计算透射率---------------------------#
        
        # 计算透射率
        Steps = 1
        e = np.linspace(0, 0, Steps)
        TUp, TDown = GF.calTransmission(e)
        TUp = np.array(TUp)
        TDown = np.array(TDown)
        SpinFiltering.append(np.sum(TUp - TDown) / np.sum(TUp + TDown))
        Polarization.append(np.sum(np.diag(np.abs(GF.ElecDensityUpAvg - GF.ElecDensityDownAvg))))
        
     #    # -------------------------透射率绘图---------------------------#
     #    fig = plt.figure(figsize=(12, 6), dpi=100)
     #    p1 = fig.add_subplot(211)
     #    p1.plot(e, TUp, "b")
     #    p1.plot(e, TDown, "r", linestyle='--')
     #    plt.title('Transmission')
     #    # -------------------------极化绘图---------------------------#
     #    p2 = fig.add_subplot(212)
     #    p2.plot(range(GF.ElecDensityUpAvg.shape[0]), np.diag(GF.ElecDensityUpAvg), "b")
     #    p2.plot(range(GF.ElecDensityDownAvg.shape[0]), np.diag(GF.ElecDensityDownAvg), "r", linestyle='--')
     #    plt.ylim((0.4, 0.6))
     #    plt.title('Polarisation')
     #    plt.show()
        
    plt.figure(figsize=(8, 6))
    ax = plt.axes(polar=False)
    plt.plot(couple, SpinFiltering, color='black', marker='+'
         ,markeredgecolor='black',markersize='15',markeredgewidth=2)
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.xlabel("Bias(V)", fontsize=26)
    ax.set_ylabel('Spin Filtering Ratio', labelpad=15, rotation=90, fontsize=26)
    ax.spines['top'].set_color('black')    # 设置顶部边框线颜色为灰色
    ax.spines['right'].set_color('black')  # 设置右侧边框线颜色为灰色
    ax.spines['bottom'].set_color('black') # 设置底部边框线颜色为黑色
    ax.spines['left'].set_color('black')  # 设置左侧边框线颜色为黑色
    ax.tick_params(which='both', direction='in', length=4, width=1, colors='black')
    
    ax2 = ax.twinx()
    ax2.set_ylabel('Spin Polarization', labelpad=15, rotation=90, fontsize=26, color='#c94737', )
    ax2.yaxis.get_offset_text().set_size(22)
    ax2.plot(couple, Polarization, color='#c94737', marker='+'
         ,markeredgecolor='#c94737',markersize='10',markeredgewidth=1)
    ax2.tick_params(length=4, width=1, colors='#c94737', labelsize=22)
    # ax2.set_ylim(-0.0005, 0.015)
    # plt.xticks([0, 0.4, 0.8, 1.2, 1.6, 2.0, 2.4, 2.8, 3.0], fontsize=22)
    # plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0], fontsize=22)
    
    # plt.xlim(0, 3.1)
    # plt.ylim(-0.05, 1.0)
    # plt.legend(loc='best', fontsize=26, framealpha=0)
    plt.tight_layout()
    plt.show()
    