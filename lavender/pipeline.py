#!/usr/bin/env python3
'''
Date: Nov 03, 2022
Lavender is a Lagrangian dispersion model to calculate the 
dispersion of atmosphreic tracers using WRF output files.

This is the main script to drive the model

History:
Nov 03, 2022 --- Kick off! 
Nov 06, 2022 --- FLEXPART output support for rendering
Nov 21, 2022 --- Advection done
Dec 09, 2022 --- Preliminary Hanna turbulence   

L_Zealot
'''
import sys, os
import logging, logging.config
from shutil import copyfile
from .core import driver
from .lib import cfgparser, utils, painter

# path to the top-level handler
CWD=sys.path[0]

# path to this module
MWD=os.path.split(os.path.realpath(__file__))[0]

def waterfall():
    '''
    Waterfall rundown!
    '''
    #if not(os.path.exists(CWD+'/config.case.ini')):
    copyfile(
        os.path.join(MWD,'conf','config.case.ini'), 
        os.path.join(CWD,'config.case.ini'))

    #print('Template config file created, please modify it and run again!')
    cfg=cfgparser.read_cfg(os.path.join(CWD,'config.case.ini'))
    
    # logging manager
    logging.config.fileConfig(os.path.join(MWD,'conf','config.logging.ini'))
    
    utils.write_log('Config validation test...')
    #cfgparser.cfg_valid_test(cfg)

    if cfg['RUNTIME'].getboolean('run_kernel'):
        utils.write_log('Build driver...')
        drv=driver.Driver(cfg)
        
        # driver drives!
        drv.drive()

    
    if cfg['postprocess'].getboolean('visualize'):
        utils.write_log('Post 3D rendering...')
        painter.render3d(cfg,MWD)
    