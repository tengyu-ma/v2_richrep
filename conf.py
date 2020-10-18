import os
import socket

from pathlib import Path


V2Config = [
    '1_1_128_128_2|128',
    '2_2_64_64_2|128',
    '4_4_32_32_2|128',
    '8_8_16_16_2|128',
    '16_16_8_8_2|128',
    '32_32_4_4_2|128',
    '64_64_2_2_2|128',
    '128_128_1_1_2|128',
]

ModelNet40Categories = [
    'airplane', 'bathtub', 'bed', 'bench', 'bookshelf',
    'bottle', 'bowl', 'car', 'chair', 'cone',
    'cup', 'curtain', 'desk', 'door', 'dresser',
    'flower_pot', 'glass_box', 'guitar', 'keyboard', 'lamp',
    'laptop', 'mantel', 'monitor', 'night_stand', 'person',
    'piano', 'plant', 'radio', 'range_hood', 'sink',
    'sofa', 'stairs', 'stool', 'table', 'tent',
    'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox'
]

ProjDir = Path(__file__).parent
InitFile = os.path.join(ProjDir, 'exps/cache/init_file.pickle')
InitDf = os.path.join(ProjDir, 'exps/cache/init_df.csv')

HostName = socket.gethostname()
V2DataDirs = {
    'VUSE-103978002': '/home/mat/Data/v2/ModelNet/ModelNet40/DSCDSC/SOFT_C16384',  # Lab Titan X, 10.20.141.250
    'ENG-AIVASLAB1': '/home/mat/Data/v2/ModelNet/ModelNet40/DSCDSC/SOFT_C16384',  # My Lab 1060, 10.20.141.40
    'VUSE-10397': '/home/mat/Data/v2/ModelNet/ModelNet40/DSCDSC/SOFT_C16384',  # Ryan Lab 1060, 10.20.141.186
    'tengyu-ubuntu': '/media/tengyu/DataU/Data/ModelNet/ModelNet40/DSCDSC/SOFT_C16384',  # Home
    'vampire': '/data/aivas_lab/Data/V2/SOFT_C16384',  # ACCRE
}

V2LogDirs = {
    'VUSE-103978002': '/home/mat/Log/V2Exp',  # Lab Titan X, 10.20.141.250
    'ENG-AIVASLAB1': '/home/mat/Log/V2Exp',  # My Lab 1060, 10.20.141.40
    'VUSE-10397': '/home/mat/Log/V2Exp',  # Ryan Lab 1060, 10.20.141.186
    'tengyu-ubuntu': '/media/tengyu/DataU/Log/V2Exp',  # Home
    'vampire': '/data/aivas_lab/Log/V2Exp',  # ACCRE
}

V2DataDir = V2DataDirs[HostName]
V2LogDir = V2LogDirs[HostName]

V2DataMeanStdCacheFile = os.path.join(ProjDir, 'exps/cache/mean_std_cache.pickle')
