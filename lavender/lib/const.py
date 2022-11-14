#/usr/bin/env python3
"""Commonly used Constants 


"""
import sys
import numpy as np
CWD=sys.path[0]



# Physical Constants
G=9.8 # m/s2
R_EARTH=6371000
T0=273.15 # K
DIS2LAT=180/(np.pi*R_EARTH)        #Distance to Latitude
SCALE_VEL=5.0 # m/s

# Calculate Mesh
TEST_Z={'layers':np.arange(0, 3000, 200).astype(np.float32),
        'magic_idz':np.array([0]).astype(np.int32),
        'sep_z':np.array([200]).astype(np.float32)}

DENSE_Z=np.concatenate((
    np.arange(0, 300, 20), 
    np.arange(300, 1000, 50),
    np.arange(1000, 2000, 100),
    np.arange(2000,6000, 500),
    np.arange(6000, 10000, 1000), 
    np.arange(10000, 20000, 2000))).astype(np.float)

# Render
## font
BIGFONT=22
MIDFONT=18
SMFONT=14
## fig
FIG_WIDTH=10
FIG_HEIGHT=10
FRM_MARGIN=[0.0, 0.0, 1.0, 1.0]

# Miscellaneous
YMDHM='%Y%m%d%H%M'
YMDHMS='%Y%m%d%H%M%S'


