import numpy as np
import matplotlib.pyplot as plt

###################添加缓冲层####################
# ------------------------参数--------------------------#
# Hc 器件自相关
# Hl0 左电极自相关
# Hr0 右电极自相关
# Hl1 左电极互相关
# Hr1 右电极互相关
# Vlc 左电极和器件互相关
# Vlc 右电极和器件互相关
# ------------------------返回--------------------------#
# Hc 器件自相关
# Vlc 左电极和器件互相关
# Vlc 右电极和器件互相关
def addBuffer(Hl0, Hr0, Hl1, Hr1, Vlc, Vrc, Hc, \
            ElectrodeLElecDensityUpAvg, ElectrodeLElecDensityDownAvg, \
            ElectrodeRElecDensityUpAvg, ElectrodeRElecDensityDownAvg, \
            DeviceElecDensityUpAvg, DeviceElecDensityDownAvg):
    # 增加Buffer
    Hl0length = Hl0.shape[0]
    Hr0length = Hr0.shape[0]
    Hclength = Hc.shape[0]
    HcBuffered = np.zeros((Hl0length + Hr0length + Hclength, Hl0length + Hr0length + Hclength))

    HcBuffered[:Hl0length, :Hl0length] = Hl0
    HcBuffered[:Hl0length, Hl0length : Hl0length + Hclength] = Vlc
    HcBuffered[Hl0length : Hl0length + Hclength, :Hl0length] = np.transpose(Vlc)

    HcBuffered[Hl0length : Hl0length + Hclength, Hl0length : Hl0length + Hclength] = Hc

    HcBuffered[Hl0length + Hclength :, Hl0length + Hclength :] = Hr0
    HcBuffered[Hl0length + Hclength :, Hl0length : Hl0length + Hclength] = Vrc
    HcBuffered[Hl0length : Hl0length + Hclength, Hl0length + Hclength :] = np.transpose(Vrc)

    VlcBuffered = np.zeros((Hl0length, Hl0length + Hclength + Hr0length))
    VrcBuffered = np.zeros((Hl0length, Hl0length + Hclength + Hr0length))
    VlcBuffered[:, :Hl0length] = Hl1
    VrcBuffered[:, Hl0length + Hclength :] = Hr1

    DeviceElecDensityUpAvgBuffered = np.zeros((Hl0length + Hr0length + Hclength, Hl0length + Hr0length + Hclength))
    DeviceElecDensityDownAvgBuffered = np.zeros((Hl0length + Hr0length + Hclength, Hl0length + Hr0length + Hclength))

    DeviceElecDensityUpAvgBuffered[:Hl0length, :Hl0length] = ElectrodeLElecDensityUpAvg
    DeviceElecDensityUpAvgBuffered[Hl0length : Hl0length + Hclength, Hl0length : Hl0length + Hclength] = DeviceElecDensityUpAvg
    DeviceElecDensityUpAvgBuffered[Hl0length + Hclength :, Hl0length + Hclength :] = ElectrodeRElecDensityUpAvg

    DeviceElecDensityDownAvgBuffered[:Hl0length, :Hl0length] = ElectrodeLElecDensityDownAvg
    DeviceElecDensityDownAvgBuffered[Hl0length : Hl0length + Hclength, Hl0length : Hl0length + Hclength] = DeviceElecDensityDownAvg
    DeviceElecDensityDownAvgBuffered[Hl0length + Hclength :, Hl0length + Hclength :] = ElectrodeRElecDensityDownAvg

    return Hl0, Hr0, Hl1, Hr1, VlcBuffered, VrcBuffered, HcBuffered, \
            ElectrodeLElecDensityUpAvg, ElectrodeLElecDensityDownAvg, \
            ElectrodeRElecDensityUpAvg, ElectrodeRElecDensityDownAvg, \
            DeviceElecDensityUpAvgBuffered, DeviceElecDensityDownAvgBuffered


def removeBuffer(Hl0, Hr0, Hl1, Hr1, VlcBuffered, VrcBuffered, HcBuffered, \
            DeviceElecDensityUpAvgBuffered, DeviceElecDensityDownAvgBuffered):
    # 去除Buffer
    Hl0length = Hl0.shape[0]
    Hr0length = Hr0.shape[0]
    Hclength = HcBuffered.shape[0] - Hl0length - Hr0length

    Hc = HcBuffered[Hl0length : Hl0length + Hclength, Hl0length : Hl0length + Hclength]

    Vlc = HcBuffered[:Hl0length, Hl0length : Hl0length + Hclength]
    Vrc = HcBuffered[Hl0length + Hclength :, Hl0length : Hl0length + Hclength]

    DeviceElecDensityUpAvg = DeviceElecDensityUpAvgBuffered[Hl0length : Hl0length + Hclength, Hl0length : Hl0length + Hclength]

    DeviceElecDensityDownAvg = DeviceElecDensityDownAvgBuffered[Hl0length : Hl0length + Hclength, Hl0length : Hl0length + Hclength]



    return Hl0, Hr0, Hl1, Hr1, Vlc, Vrc, Hc, \
            DeviceElecDensityUpAvg, DeviceElecDensityDownAvg


###################读取原子坐标文件转换成字典####################
# ------------------------参数--------------------------#
# FilePath 文件地址
# ------------------------返回--------------------------#
# StructureProperties {
#   'NumOfAtoms':原子数量, 
#   'UnicellVectors':电极方向矢量, 
#   'TypeOfLeads':电极类型, 
#   'LeftElectrodeAtomsPositions':左电极原子坐标,
#   'RightElectrodeAtomsPositions':右电极原子坐标, 
#   'DeviceAtomsPositions':器件原子坐标
#}
def readFileAndConvert(FilePath):
    FileHandler = open(FilePath, "r")
    ListOfLines = FileHandler.readlines()
    FileHandler.close()
    TypeOfLeads = ListOfLines[3]
    
    # 读取原胞基矢和电极长度
    UnicellVectors = np.zeros((5, 3))
    for i in range(5, 10):
        UnicellVectors[i - 5, :] = np.array(ListOfLines[i].split(), dtype=float)

    NumOfAtoms = np.array(ListOfLines[11].split(), dtype=int)

    # 读取原子坐标
    AtomsPosition = []
    for i in range(13, 13 + NumOfAtoms[0]):
        
        tup = (ListOfLines[i].split()[0], np.array([eval(j) for j in ListOfLines[i].split()[1:]]))
        AtomsPosition.append(tup)
    
    # 按照左电极、器件、右电极把原子分开
    LeftElectrodeAtomsPositions = []
    RightElectrodeAtomsPositions = []
    DeviceAtomsPositions = []
    for i, Atom in enumerate(AtomsPosition):
        if Atom[1][2] < 0:
            LeftElectrodeAtomsPositions.append(Atom)
        elif Atom[1][2] > UnicellVectors[2, 2]:
            RightElectrodeAtomsPositions.append(Atom)
        else:
            DeviceAtomsPositions.append(Atom)


    StructureProperties = {'NumOfAtoms':NumOfAtoms, 'UnicellVectors':UnicellVectors, 'TypeOfLeads':TypeOfLeads, 'LeftElectrodeAtomsPositions':LeftElectrodeAtomsPositions,
                           'RightElectrodeAtomsPositions':RightElectrodeAtomsPositions, 'DeviceAtomsPositions':DeviceAtomsPositions}
    return StructureProperties


###################根据距离和原子类型计算跳跃能####################
# ------------------------参数--------------------------#
# Atom1:原子类型
# Atom2:原子类型
# Distance:距离
# ------------------------返回--------------------------#
# hopping大小
def HoppingEvaluate(Atom1, Atom2, Distance):

    return -1
    return 2.7 * np.exp(- 0.5 * Distance + 0.5)


###################根据原子坐标生成并返回哈密顿矩阵####################
# ------------------------参数--------------------------#
# Path: 原子坐标文件路径
# ------------------------返回--------------------------#
# Hc 器件自相关
# Hl0 左电极自相关
# Hr0 右电极自相关
# Hl1 左电极互相关
# Hr1 右电极互相关
# Vlc左电极和器件互相关
# Vlc右电极和器件互相关
# Vlc右电极和器件互相关
def GenerateHamiltonianMatrixFromFile(Path):
    # 生成Hamiltonian矩阵
    StructureProperties = readFileAndConvert(Path)
    Hl0 = GenerateHamiltonianMatrix(
        StructureProperties.get("LeftElectrodeAtomsPositions"),
        StructureProperties.get("LeftElectrodeAtomsPositions"),
    )
    Hr0 = GenerateHamiltonianMatrix(
        StructureProperties.get("RightElectrodeAtomsPositions"),
        StructureProperties.get("RightElectrodeAtomsPositions"),
    )
    Vlc = GenerateHamiltonianMatrix(
        StructureProperties.get("LeftElectrodeAtomsPositions"),
        StructureProperties.get("DeviceAtomsPositions"),
    )
    Vrc = GenerateHamiltonianMatrix(
        StructureProperties.get("RightElectrodeAtomsPositions"),
        StructureProperties.get("DeviceAtomsPositions"),
    )
    Hc = GenerateHamiltonianMatrix(
        StructureProperties.get("DeviceAtomsPositions"),
        StructureProperties.get("DeviceAtomsPositions"),
    )

    Hl1 = GenerateHamiltonianMatrix(
        StructureProperties.get("LeftElectrodeAtomsPositions"),
        StructureProperties.get("LeftElectrodeAtomsPositions"),
        True,
        -StructureProperties.get("UnicellVectors")[4, :],
    )
    Hr1 = GenerateHamiltonianMatrix(
        StructureProperties.get("RightElectrodeAtomsPositions"),
        StructureProperties.get("RightElectrodeAtomsPositions"),
        True,
        StructureProperties.get("UnicellVectors")[4, :],
    )
    return Hl0, Hr0, Hl1, Hr1, Vlc, Vrc, Hc

###################根据原子坐标集合生成并返回哈密顿矩阵####################
# ------------------------参数--------------------------#
# AtomsPositions1:1号区域原子信息
# AtomsPositions2:2号区域原子信息
# Shift: 是否要偏移，一般用于，计算电极互相关矩阵
# ShiftVector: 偏移向量
# ------------------------返回--------------------------#
# Hamiltonian: 哈密顿矩阵
def GenerateHamiltonianMatrix(AtomsPositions1, AtomsPositions2, Shift = False, ShiftVector = None):
    
    if Shift:
        NewAtomsPositions1 = []
        for Atom1 in AtomsPositions1:

            NewAtom1Tup = (Atom1[0], Atom1[1] + ShiftVector)
            NewAtomsPositions1.append(NewAtom1Tup)
        AtomsPositions1 = NewAtomsPositions1
    
    Hamiltonian = np.zeros((len(AtomsPositions1), len(AtomsPositions2)))
    
    # 计算任意两个位点之间的距离并存入Hamiltonian矩阵
    for i, Atom1 in enumerate(AtomsPositions1):
        for j, Atom2 in enumerate(AtomsPositions2):
            if np.linalg.norm(Atom1[1] - Atom2[1]) > 1e-2:
                Hamiltonian[i, j] = np.linalg.norm(Atom1[1] - Atom2[1])
    # 确认最近邻、次近邻距离
    if Hamiltonian.any() > 1e-2:
        NearestHopping = Hamiltonian[Hamiltonian > 1e-2].min()
    else:
        NearestHopping = 0
    if Hamiltonian.any() > NearestHopping + 1e-2:
        NextNearestHopping = Hamiltonian[Hamiltonian > NearestHopping + 1e-2].min()
    else:
        NextNearestHopping = NearestHopping
    #ThirdNearestHopping = Hamiltonian[Hamiltonian > NextNearestHopping + 1e-2].min()
   
    # 统一距离为近邻和次近邻
    Zero = np.frompyfunc(lambda x:0 if x < 1e-2 else x, 1, 1)
    Nearest = np.frompyfunc(lambda x:NearestHopping if x > 1e-2 and x < NearestHopping + 1e-2 else x, 1, 1)
    NextNearest = np.frompyfunc(lambda x:NextNearestHopping if x > NearestHopping + 1e-2 and x < NextNearestHopping + 1e-2 else x, 1, 1)
    Ignore = np.frompyfunc(lambda x:0 if x > NearestHopping + 1e-2 else x, 1, 1)
    Hamiltonian = Ignore(NextNearest(Nearest(Zero(Hamiltonian))))

    # 将距离转换为实际的Hopping大小
    for i, Atom1 in enumerate(AtomsPositions1):
        for j, Atom2 in enumerate(AtomsPositions2):
            if np.linalg.norm(Atom1[1] - Atom2[1]) > 1e-2 and Hamiltonian[i, j] > 1e-2:
                Hamiltonian[i, j] = HoppingEvaluate(Atom1, Atom2, Hamiltonian[i, j])

    return np.array(Hamiltonian, dtype=float)


###################保存体系信息到txt里####################
# ------------------------参数--------------------------#

# ------------------------返回--------------------------#

def SaveHamiltonianTxt(path, Hl0, Hr0, Hl1, Hr1, Vlc, Vrc, Hc, 
                       ElectrodeLElecDensityUpAvgInit, ElectrodeLElecDensityDownAvgInit,
                       ElectrodeRElecDensityUpAvgInit, ElectrodeRElecDensityDownAvgInit,
                       DeviceElecDensityUpAvgInit, DeviceElecDensityDownAvgInit,
                       Description):

    # 保存到txt中
    with open(path, 'w') as f:
        f.write(Description)
        f.write("\n") 
        f.write("&Hl0\n&\n") 
        np.savetxt(f, Hl0)
        f.write("&Hr0\n&\n") 
        np.savetxt(f, Hr0)
        f.write("&Hl1\n&\n") 
        np.savetxt(f, Hl1)
        f.write("&Hr1\n&\n") 
        np.savetxt(f, Hr1)
        f.write("&Vlc\n&\n") 
        np.savetxt(f, Vlc)
        f.write("&Vrc\n&\n") 
        np.savetxt(f, Vrc)
        f.write("&Hc\n&\n") 
        np.savetxt(f, Hc)
        f.write("&ElectrodeLElecDensityUpAvgInit\n&\n") 
        np.savetxt(f, ElectrodeLElecDensityUpAvgInit)
        f.write("&ElectrodeLElecDensityDownAvgInit\n&\n") 
        np.savetxt(f, ElectrodeLElecDensityDownAvgInit)
        f.write("&ElectrodeRElecDensityUpAvgInit\n&\n") 
        np.savetxt(f, ElectrodeRElecDensityUpAvgInit)
        f.write("&ElectrodeRElecDensityDownAvgInit\n&\n") 
        np.savetxt(f, ElectrodeRElecDensityDownAvgInit)
        f.write("&DeviceElecDensityUpAvgInit\n&\n") 
        np.savetxt(f, DeviceElecDensityUpAvgInit)
        f.write("&DeviceElecDensityDownAvgInit\n&\n") 
        np.savetxt(f, DeviceElecDensityDownAvgInit)
        f.write("&")


###################从txt读取体系信息####################
# ------------------------参数--------------------------#

# ------------------------返回--------------------------#

def LoadHamiltonianTxt(path):
    with open(path, 'r') as f:
        Description = f.readline()
        print(Description)
        next(f)
        data = f.read()


    matrices = []
    for matrix_str in data.split('&')[1::2]:
        # 将矩阵字符串转换为numpy数组
        matrix = np.loadtxt(matrix_str.split('\n')[0:-1], dtype=float)
        matrices.append(matrix)

    Hl0 =  matrices[0]
    Hr0 = matrices[1]
    Hl1 = matrices[2]
    Hr1 = matrices[3]
    Vlc = matrices[4]
    Vrc = matrices[5]
    Hc = matrices[6]
    ElectrodeLElecDensityUpAvgInit = matrices[7]
    ElectrodeLElecDensityDownAvgInit = matrices[8]
    ElectrodeRElecDensityUpAvgInit = matrices[9]
    ElectrodeRElecDensityDownAvgInit = matrices[10]
    DeviceElecDensityUpAvgInit = matrices[11]
    DeviceElecDensityDownAvgInit = matrices[12]

    np.set_printoptions(threshold = np.inf, linewidth = np.inf)

    print("------------------------------Hl0------------------------------")
    print(Hl0)
    print("------------------------------Hr0------------------------------")
    print(Hr0)
    print("------------------------------Hl1------------------------------")
    print(Hl1)
    print("------------------------------Hr1------------------------------")
    print(Hr1)
    print("------------------------------Vlc------------------------------")
    print(Vlc)
    print("------------------------------Vrc------------------------------")
    print(Vrc)
    print("------------------------------Hc------------------------------")
    print(Hc)
    print("------------------------------ElectrodeLElecDensityUpAvgInit------------------------------")
    print(ElectrodeLElecDensityUpAvgInit)
    print("------------------------------ElectrodeLElecDensityDownAvgInit------------------------------")
    print(ElectrodeLElecDensityDownAvgInit)
    print("------------------------------ElectrodeRElecDensityUpAvgInit------------------------------")
    print(ElectrodeRElecDensityUpAvgInit)
    print("------------------------------ElectrodeRElecDensityDownAvgInit------------------------------")
    print(ElectrodeRElecDensityDownAvgInit)
    print("------------------------------DeviceElecDensityUpAvgInit------------------------------")
    print(DeviceElecDensityUpAvgInit)
    print("------------------------------DeviceElecDensityDownAvgInit------------------------------")
    print(DeviceElecDensityDownAvgInit)

    return Hl0, Hr0, Hl1, Hr1, Vlc, Vrc, Hc, \
            ElectrodeLElecDensityUpAvgInit, ElectrodeLElecDensityDownAvgInit, \
            ElectrodeRElecDensityUpAvgInit, ElectrodeRElecDensityDownAvgInit, \
            DeviceElecDensityUpAvgInit, DeviceElecDensityDownAvgInit


###################保存体系信息到npz里####################
# ------------------------参数--------------------------#

# ------------------------返回--------------------------#

def SaveHamiltonianNumpyZ(path, Hl0, Hr0, Hl1, Hr1, Vlc, Vrc, Hc, 
                       ElectrodeLElecDensityUpAvgInit, ElectrodeLElecDensityDownAvgInit,
                       ElectrodeRElecDensityUpAvgInit, ElectrodeRElecDensityDownAvgInit,
                       DeviceElecDensityUpAvgInit, DeviceElecDensityDownAvgInit,
                       Description):
    np.savez(path, Hl0=Hl0, Hr0=Hr0, Hl1=Hl1, Hr1=Hr1, Vlc=Vlc, Vrc=Vrc, Hc=Hc, 
             ElectrodeLElecDensityUpAvgInit = ElectrodeLElecDensityUpAvgInit, 
             ElectrodeLElecDensityDownAvgInit = ElectrodeLElecDensityDownAvgInit,
             ElectrodeRElecDensityUpAvgInit = ElectrodeRElecDensityUpAvgInit,
             ElectrodeRElecDensityDownAvgInit = ElectrodeRElecDensityDownAvgInit,
             DeviceElecDensityUpAvgInit = DeviceElecDensityUpAvgInit,
             DeviceElecDensityDownAvgInit = DeviceElecDensityDownAvgInit,
             Description=Description)

###################从Npz文件读取体系信息####################
# ------------------------参数--------------------------#

# ------------------------返回--------------------------#

def LoadHamiltonianNumpyZ(path):
    data = np.load(path)
    Description = data["Description"]
    Hl0 = data["Hl0"]
    Hr0 = data["Hr0"]
    Hl1 = data["Hl1"]
    Hr1 = data["Hr1"]
    Vlc = data["Vlc"]
    Vrc = data["Vrc"]
    Hc = data["Hc"]

    ElectrodeLElecDensityUpAvgInit = data["ElectrodeLElecDensityUpAvgInit"]
    ElectrodeLElecDensityDownAvgInit = data["ElectrodeLElecDensityDownAvgInit"]
    ElectrodeRElecDensityUpAvgInit = data["ElectrodeRElecDensityUpAvgInit"]
    ElectrodeRElecDensityDownAvgInit = data["ElectrodeRElecDensityDownAvgInit"]
    # 中心器件初始电荷极化分布
    DeviceElecDensityUpAvgInit = data["DeviceElecDensityUpAvgInit"]
    DeviceElecDensityDownAvgInit = data["DeviceElecDensityDownAvgInit"]
    np.set_printoptions(threshold = np.inf, linewidth = np.inf)
    print("------------------------------Description------------------------------")
    print(Description)
    print("------------------------------Hl0------------------------------")
    print(Hl0)
    print("------------------------------Hr0------------------------------")
    print(Hr0)
    print("------------------------------Hl1------------------------------")
    print(Hl1)
    print("------------------------------Hr1------------------------------")
    print(Hr1)
    print("------------------------------Vlc------------------------------")
    print(Vlc)
    print("------------------------------Vrc------------------------------")
    print(Vrc)
    print("------------------------------Hc------------------------------")
    print(Hc)
    print("------------------------------ElectrodeLElecDensityUpAvgInit------------------------------")
    print(ElectrodeLElecDensityUpAvgInit)
    print("------------------------------ElectrodeLElecDensityDownAvgInit------------------------------")
    print(ElectrodeLElecDensityDownAvgInit)
    print("------------------------------ElectrodeRElecDensityUpAvgInit------------------------------")
    print(ElectrodeRElecDensityUpAvgInit)
    print("------------------------------ElectrodeRElecDensityDownAvgInit------------------------------")
    print(ElectrodeRElecDensityDownAvgInit)
    print("------------------------------DeviceElecDensityUpAvgInit------------------------------")
    print(DeviceElecDensityUpAvgInit)
    print("------------------------------DeviceElecDensityDownAvgInit------------------------------")
    print(DeviceElecDensityDownAvgInit)
    return Hl0, Hr0, Hl1, Hr1, Vlc, Vrc, Hc, \
            ElectrodeLElecDensityUpAvgInit, ElectrodeLElecDensityDownAvgInit, \
            ElectrodeRElecDensityUpAvgInit, ElectrodeRElecDensityDownAvgInit, \
            DeviceElecDensityUpAvgInit, DeviceElecDensityDownAvgInit

###################生成一维碳原子链哈密顿矩阵####################
# ------------------------参数--------------------------#

# ------------------------返回--------------------------#
def GenerateCACHamiltonianMatrix():
    # 生成CAC, Hamiltonian矩阵

    H = -np.eye(26, k = -1) - np.eye(26, k = 1)
    H[7, 8] = -0.3
    H[8, 7] = -0.3
    H[17, 18] = -0.3
    H[18, 17] = -0.3
    Hl0 = H[:4, :4]
    print(Hl0)
    Hr0 = H[-4:, -4:]
    print(Hr0)
    Hl1 = H[:4, 4:8]
    print(Hl1)
    Hr1 = H[22:, 18:22]
    print(Hr1)
    Vlc = H[4:8, 8:18]
    print(Vlc)
    Vrc = H[18:22, 8:18]
    print(Vrc)
    Hc = H[8:18, 8:18]
    print(Hc)
    # 电极初始电荷极化分布
    ElectrodeLElecDensityUpAvgInit = 0.5 * np.identity(Hl0.shape[0])
    ElectrodeLElecDensityDownAvgInit = np.identity(Hl0.shape[0]) - ElectrodeLElecDensityUpAvgInit

    ElectrodeRElecDensityUpAvgInit = 0.5 * np.identity(Hl0.shape[0])
    ElectrodeRElecDensityDownAvgInit = np.identity(Hr0.shape[0]) - ElectrodeRElecDensityUpAvgInit

    # 中心器件初始电荷极化分布
    DeviceElecDensityUpAvgInit = 0.5 * np.identity(Hc.shape[0])
    DeviceElecDensityDownAvgInit = np.identity(Hc.shape[0]) - DeviceElecDensityUpAvgInit
    
    return Hl0, Hr0, Hl1, Hr1, Vlc, Vrc, Hc, \
            ElectrodeLElecDensityUpAvgInit, ElectrodeLElecDensityDownAvgInit, \
            ElectrodeRElecDensityUpAvgInit, ElectrodeRElecDensityDownAvgInit, \
            DeviceElecDensityUpAvgInit, DeviceElecDensityDownAvgInit


###################生成石墨烯纳米带哈密顿矩阵####################
# ------------------------参数--------------------------#

# ------------------------返回--------------------------#
def GenerateGrapheneHamiltonianMatrix():
    # 生成石墨烯带Hamiltonian矩阵

    Hl0 = np.array([[0, -1, 0, 0],
                    [-1, 0, -1, 0],
                    [0, -1, 0, -1],
                    [0, 0, -1, 0]])

    Hr0 = np.array([[0, -1, 0, 0],
                    [-1, 0, -1, 0],
                    [0, -1, 0, -1],
                    [0, 0, -1, 0]])

    # 电极之间跳跃能取-1,
    Hl1 = np.array([[0, -1, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, -1, 0]])

    Hr1 = np.array([[0, 0, 0, 0],
                    [-1, 0, 0, 0],
                    [0, 0, 0, -1],
                    [0, 0, 0, 0]])

    #电极与器件耦合
    Vlc = np.array([[ 0., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]])

    Vrc = np.array([[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]])

    # 中心器件的哈密顿矩阵
    Hc = np.array([[ 0., -1.,  0.,  0.,  0., -1.,  0.,  0.,  0.,  0.,  0.,  0.],
       [-1.,  0., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0., -1.,  0., -1.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0., -1.,  0.,  0.,  0., -1.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0., -1.,  0.,  0.,  0., -1.,  0.,  0.],
       [-1.,  0.,  0.,  0., -1.,  0., -1.,  0.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0., -1.,  0., -1.,  0., -1.,  0.,  0.,  0.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0., -1.,  0.,  0.,  0., -1.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1.,  0.,  0.],
       [ 0.,  0.,  0.,  0., -1.,  0.,  0.,  0., -1.,  0., -1.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0., -1.,  0., -1.,  0., -1.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., -1.,  0.]])
    
    return Hl0, Hr0, Hl1, Hr1, Vlc, Vrc, Hc

def haldaneGrapheneHamitonlian(nx, ny, t):

    ldSO=0.1*t;       # 自旋轨道耦合
    ldHD=-0.02*t;      # 光场项

    # 原胞
    x0 = np.array([np.sqrt(3) / 2, 0, 0, np.sqrt(3) / 2])
    y0 = np.array([-0.5, 0, 1, 1.5])

    # 扩胞成nx*ny的纳米带
    coordinatesX = np.array([])
    coordinatesY = np.array([])
    for i in range(nx):
        for j in range(ny):
            coordinatesX = np.concatenate((coordinatesX, x0 + np.sqrt(3) * i))
            coordinatesY = np.concatenate((coordinatesY, y0 + 3 * j))

    # 生成基本哈密顿量矩阵
    totalAtomNum = nx * ny * 4
    H0 = np.zeros((totalAtomNum, totalAtomNum))
    Hv = np.zeros((totalAtomNum, totalAtomNum), dtype=complex)
    HvHD = np.zeros((totalAtomNum, totalAtomNum), dtype=complex)
    distances = np.zeros((totalAtomNum, totalAtomNum))
    for i in range(totalAtomNum):
        for j in range(totalAtomNum):
            # 寻找最近邻和次近邻
            distances[i, j] = np.sqrt((coordinatesX[i] - coordinatesX[j]) ** 2 + (coordinatesY[i] - coordinatesY[j]) ** 2)

            if distances[i, j] > 0.1 and distances[i, j] < 1.1:
                H0[i, j] = t
            elif distances[i, j] < 2.1:
                Hv[i, j] = ldSO / (3 * np.sqrt(3))
                HvHD[i, j] = ldHD / (3 * np.sqrt(3))

    # 自旋轨道耦合和Haldane项
    for i in range(totalAtomNum):
        for j in range(totalAtomNum):
            vCoefficient = 0
            if distances[i, j] > 1.1 and distances[i, j] < 2.1:
                for k in range(totalAtomNum):
                    # 寻找一个格点k, i和j都是k的最近邻
                    if distances[i, k] > 0 and distances[i, k] < 1.1 and distances[j, k] > 0 and distances[j, k] < 1.1:
                        # 判断相对k来说，j是在i的顺时针方向还是逆时针方向，扩展一维做叉乘
                        crossResult = np.cross([coordinatesX[j] - coordinatesX[k], coordinatesY[j] - coordinatesY[k] , 0], 
                                 [coordinatesX[i] - coordinatesX[k], coordinatesY[i] - coordinatesY[k] , 0])
                        if crossResult[2] > 0:
                            vCoefficient = 1
                        else:
                            vCoefficient = -1
            Hv[i, j] = Hv[i, j] * vCoefficient * 1j
            HvHD[i, j] = HvHD[i, j] * vCoefficient * 1j

    return coordinatesX, coordinatesY, H0, Hv, HvHD

def saveAtomCoordinates(coordinatesX, coordinatesY, coordinatesZ, path):

    with open(path, 'w') as f:
        f.write('% The number of probes\n')
        f.write('2\n')
        f.write('% The type of the integer\n')
        f.write('-3	3\n')
        f.write('% Uni-cell vector\n')
        f.write('27.10000000	0.00000000	0.00000000\n0.00000000	20.00000000	0.00000000\n0.00000000	0.00000000	6.14880466\n0.00000000	0.00000000	2.459500000\n00000000	0.00000000	2.45950000\n')
        f.write('%Total number of device_structure\n')
        f.write('')
        f.write('%Atom site\n')
        for i in range(coordinatesX.shape[0]):


            
            f.write('C\t')
            f.write(str(coordinatesX[i]))
            f.write('\t')
            f.write(str(coordinatesY[i]))
            f.write('\t')
            f.write(str(coordinatesZ[i]))
            f.write('\n')


if __name__ == '__main__':
    nx = 12 
    ny = 5 
    t = 1.0


    coordinatesX, coordinatesY, H0, Hv, HvHD = haldaneGrapheneHamitonlian(nx, ny, t)
    saveAtomCoordinates(coordinatesX[40:200], coordinatesY[40:200], np.zeros(160), 'D:\Documents\PythonQT\Hamiltonian\position.txt')
    
    SaveHamiltonianTxt('D:\Documents\PythonQT\Hamiltonian\\4C_10C_4C.txt', *GenerateCACHamiltonianMatrix(), "CAC")
    