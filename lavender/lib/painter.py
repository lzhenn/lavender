#!/usr/bin/env python3
'''
Module Painter
'''

import sys

import matplotlib.pyplot as plt
import os
from . import io

print_prefix='lib.painter>>'
CWD=sys.path[0]

# Constants
BIGFONT=22
MIDFONT=18
SMFONT=14


def render3d(cfg):
    '''
    render 3d plot of the dumped partical file 
    '''
    file_hdler=io.FileHandler(cfg)
    for ts in file_hdler.file_list:
        icode=file_hdler.load_frame(ts)
        if icode==0:
            break
