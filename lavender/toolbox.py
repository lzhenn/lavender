#!/usr/bin/env python3
'''
Date: Dec 15, 2022

Toolbox for external call to implement data conversion

History:
Dec 15, 2022 --- Kick off! 

L_Zealot
'''
import sys, os
import logging, logging.config
from .lib import cfgparser, utils, io, const

# path to the top-level handler
CWD=sys.path[0]

# path to this module
MWD=os.path.split(os.path.realpath(__file__))[0]

def flxpt2ply(outpath):
    '''
    convert flexpart data to ply format
    '''
    #if not(os.path.exists(CWD+'/config.case.ini')):
    #print('Template config file created, please modify it and run again!')
    cfg=cfgparser.read_cfg(os.path.join(CWD,'config.case.ini'))
    
    # logging manager
    logging.config.fileConfig(os.path.join(MWD,'conf','config.logging.ini'))
    

    utils.write_log('Convert flexpart data to ply format...')
    fh=io.OutHandler(cfg)

    # get global idx
    for tf, fn in zip(fh.tfrms, fh.file_list):
        # init particle array in current frame
        prtarray=fh.load_frame(fn)
        if prtarray is None:
            continue
        outfn=os.path.join(outpath, 'ptcl_'+tf.strftime(const.YMDHM)+'.ply') 
        io.outply(prtarray, outfn) 