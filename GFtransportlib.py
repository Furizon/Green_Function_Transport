import numpy as np
from numba import njit, prange, set_num_threads
set_num_threads(4)
###################迭代求计算电极自能####################
# ------------------------参数--------------------------#
# ee 能量, 可以为数组
# Hc 器件自相关
# H0 电极自相关
# H1 电极互相关
# Vlc左电极和器件互相关
# Vlc右电极和器件互相关
# ------------------------返回--------------------------#
# SigmaL 左电极自能
# SigmaR 右电极自能
@njit(fastmath = True, nogil = True)
def calElectrodeSelfEng(e, Hc, H0, H1, Vxc, yita, SelfEngConvergeMaxSteps, SelfEngConvergeLimit):
    # 电极单元原子数
    N = H0.shape[0]
    # 器件单元原子数
    M = Hc.shape[0]
    # 重新定义变量
    e = np.complex128(e)
    yita = np.complex128(yita)
    alphas = np.zeros((N, N, SelfEngConvergeMaxSteps), dtype=np.complex128)
    alpha = np.zeros((N, N, SelfEngConvergeMaxSteps), dtype=np.complex128)
    beta = np.zeros((N, N, SelfEngConvergeMaxSteps), dtype=np.complex128)
    gamma = np.zeros((N, N, SelfEngConvergeMaxSteps), dtype=np.complex128)

    alphas[:, :, 1] = H0
    alpha[:, :, 1] = H0
    beta[:, :, 1] = np.transpose(H1).conjugate()
    gamma[:, :, 1] = H1
    SigmaX = np.zeros((M, M), dtype = np.complex128)

    # 迭代计算系数
    for n in range(2, SelfEngConvergeMaxSteps):
        identity_matrix = (e + 1j * yita) * np.identity(N)
        inv_matrix = np.linalg.inv(identity_matrix - alpha[:, :, n - 1])

        # 确保矩阵是连续的
        beta_cont = np.ascontiguousarray(beta[:, :, n - 1])
        inv_matrix_cont = np.ascontiguousarray(inv_matrix)
        gamma_cont = np.ascontiguousarray(gamma[:, :, n - 1])
        alphas_prev_cont = np.ascontiguousarray(alphas[:, :, n - 1])
        alpha_prev_cont = np.ascontiguousarray(alpha[:, :, n - 1])

        alphas[:, :, n] = alphas_prev_cont + np.dot(np.dot(beta_cont, inv_matrix_cont), gamma_cont)
        alpha[:, :, n] = alpha_prev_cont + np.dot(np.dot(gamma_cont, inv_matrix_cont), beta_cont) + np.dot(np.dot(beta_cont, inv_matrix_cont), gamma_cont)
        beta[:, :, n] = np.dot(np.dot(beta_cont, inv_matrix_cont), beta_cont)
        gamma[:, :, n] = np.dot(np.dot(gamma_cont, inv_matrix_cont), gamma_cont)

        # 判断收敛，若收敛则提前退出
        if np.sum(np.abs(alphas[:, :, n] - alphas_prev_cont)) < SelfEngConvergeLimit:
            break

    # 计算表面格林函数
    surfaceG = np.linalg.inv(identity_matrix - np.ascontiguousarray(alphas[:, :, n]))

    # 计算自能，确保所有矩阵类型一致，并转换为连续数组
    Vxc = np.ascontiguousarray(Vxc.astype(np.complex128))
    surfaceG = np.ascontiguousarray(surfaceG)
    SigmaX = np.dot(np.dot(np.transpose(Vxc).conjugate(), surfaceG), Vxc)
    return SigmaX


###################迭代求计算电极自能####################
# ------------------------参数--------------------------#
# ee 能量, 可以为数组
# Hc 器件自相关
# Hl0 左电极自相关
# Hr0 右电极自相关
# Hl1 左电极互相关
# Hr1 右电极互相关
# Vlc左电极和器件互相关
# Vrc右电极和器件互相关
# ------------------------返回--------------------------#
# 左右电极自能之和
@njit(fastmath = True, nogil = True)
def calElectrodeSelfEngTotal(e, Hc, Hl0, Hr0, Hl1, Hr1, Vlc, Vrc, yita, SelfEngConvergeMaxSteps, SelfEngConvergeLimit):
    SigmaL = calElectrodeSelfEng(e, Hc, Hl0, Hl1, Vlc, yita, SelfEngConvergeMaxSteps, SelfEngConvergeLimit)
    SigmaR = calElectrodeSelfEng(e, Hc, Hr0, Hr1, Vrc, yita, SelfEngConvergeMaxSteps, SelfEngConvergeLimit)
    return SigmaL + SigmaR

@njit(fastmath = True, nogil = True)
def fermi(e, u, K0T):
    return 1 / (1 + np.exp((e - u) / K0T))

@njit(fastmath = True, nogil = True)
def compute_elec_density_pint(args, e):
    energy_range, deltae, Leftu, Rightu, HcUp, HcDown, Hl0Up, Hl0Down, Hl1, Vlc, Hr0Up, Hr0Down, Hr1, Vrc, YITA, K0T, SelfEngConvergeMaxSteps, SelfEngConvergeLimit = args
    M = HcUp.shape[0]

    SigmaLUpRetarded = calElectrodeSelfEng(e, HcUp, Hl0Up, Hl1, Vlc, YITA, SelfEngConvergeMaxSteps, SelfEngConvergeLimit)
    SigmaRUpRetarded = calElectrodeSelfEng(e, HcUp, Hr0Up, Hr1, Vrc, YITA, SelfEngConvergeMaxSteps, SelfEngConvergeLimit)
    SigmaLUpAdvanced = calElectrodeSelfEng(e, HcUp, Hl0Up, Hl1, Vlc, -YITA, SelfEngConvergeMaxSteps, SelfEngConvergeLimit)
    SigmaRUpAdvanced = calElectrodeSelfEng(e, HcUp, Hr0Up, Hr1, Vrc, -YITA, SelfEngConvergeMaxSteps, SelfEngConvergeLimit)

    SigmaLDownRetarded = calElectrodeSelfEng(e, HcDown, Hl0Down, Hl1, Vlc, YITA, SelfEngConvergeMaxSteps, SelfEngConvergeLimit)
    SigmaRDownRetarded = calElectrodeSelfEng(e, HcDown, Hr0Down, Hr1, Vrc, YITA, SelfEngConvergeMaxSteps, SelfEngConvergeLimit)
    SigmaLDownAdvanced = calElectrodeSelfEng(e, HcDown, Hl0Down, Hl1, Vlc, -YITA, SelfEngConvergeMaxSteps, SelfEngConvergeLimit)
    SigmaRDownAdvanced = calElectrodeSelfEng(e, HcDown, Hr0Down, Hr1, Vrc, -YITA, SelfEngConvergeMaxSteps, SelfEngConvergeLimit)

    GammaLUp = 1j * (SigmaLUpRetarded - SigmaLUpAdvanced)
    GammaRUp = 1j * (SigmaRUpRetarded - SigmaRUpAdvanced)
    GammaLDown = 1j * (SigmaLDownRetarded - SigmaLDownAdvanced)
    GammaRDown = 1j * (SigmaRDownRetarded - SigmaRDownAdvanced)

    GRcUp = np.linalg.inv((e + 1j * YITA) * np.identity(M) - HcUp - SigmaLUpRetarded - SigmaRUpRetarded)
    GRcDown = np.linalg.inv((e + 1j * YITA) * np.identity(M) - HcDown - SigmaLDownRetarded - SigmaRDownRetarded)
    GAcUp = np.linalg.inv((e - 1j * YITA) * np.identity(M) - HcUp - SigmaLUpAdvanced - SigmaRUpAdvanced)
    GAcDown = np.linalg.inv((e - 1j * YITA) * np.identity(M) - HcDown - SigmaLDownAdvanced - SigmaRDownAdvanced)

    if Rightu > Leftu:
        ElecDensityUp = (np.dot(np.dot(GRcUp, GammaRUp), GAcUp) * (fermi(e, Rightu, K0T) - fermi(e, Leftu, K0T)) * deltae).real / (2 * np.pi)
        ElecDensityDown = (np.dot(np.dot(GRcDown, GammaRDown), GAcDown) * (fermi(e, Rightu, K0T) - fermi(e, Leftu, K0T)) * deltae).real / (2 * np.pi)
    else:
        ElecDensityUp = (np.dot(np.dot(GRcUp, GammaLUp), GAcUp) * (fermi(e, Leftu, K0T) - fermi(e, Rightu, K0T)) * deltae).real / (2 * np.pi)
        ElecDensityDown = (np.dot(np.dot(GRcDown, GammaLDown), GAcDown) * (fermi(e, Leftu, K0T) - fermi(e, Rightu, K0T)) * deltae).real / (2 * np.pi)
    return ElecDensityUp, ElecDensityDown

# 计算能量积分
@njit(fastmath=True, parallel=True, nogil=True, cache = True)
def compute_elec_density(args):
    energy_range, deltae, Leftu, Rightu, HcUp, HcDown, Hl0Up, Hl0Down, Hl1, Vlc, Hr0Up, Hr0Down, Hr1, Vrc, YITA, K0T, SelfEngConvergeMaxSteps, SelfEngConvergeLimit = args
    M = HcUp.shape[0]
    
    # 创建局部结果存储
    ElecDensityUp_local = np.zeros((M, M, len(energy_range)))
    ElecDensityDown_local = np.zeros((M, M, len(energy_range)))

    # 使用 prange 并行处理能量积分
    for i in prange(len(energy_range)):
        e = energy_range[i]
        ElecDensityUp_local[:, :, i], ElecDensityDown_local[:, :, i] = compute_elec_density_pint(args, e)
    # 累加结果
    ElecDensityUp = np.sum(ElecDensityUp_local, axis=2)
    ElecDensityDown = np.sum(ElecDensityDown_local, axis=2)

    return ElecDensityUp, ElecDensityDown
