#/usr/bin/env python3
"""specific module for IO"""
import datetime, struct, os
import numpy as np
import pandas as pd
from scipy.io import FortranFile, FortranEOFError

from ..core import particals
from . import utils, const, cfgparser

print_prefix='lib.io>>'

class FileHandler(object):

    '''
    Construct File Handler for dumped partical data 
    in unformatted fortran binary file
    
    Methods
    -----------
    __init__:   initialize FLEXPHandler and loading data
    write_frame: write flexpart format partical dump file
    '''
    
    def __init__(self, cfg):
        '''
        Initialize FileHandler with config and loading data
        '''
        utils.write_log(print_prefix+'construct FileHandler')
        
        self.strt_time=datetime.datetime.strptime(
            cfg['INPUT']['init_t'], const.YMDHM)
        self.end_time=datetime.datetime.strptime(
            cfg['INPUT']['end_t'], const.YMDHM)

        self.fmt=cfg['OUTPUT']['file_fmt']
        self.path=cfg['OUTPUT']['file_root']
        self.file_wildcard=cfg['OUTPUT']['file_wildcard']
        self.frq=cfg['OUTPUT']['output_frq']
        specs=cfgparser.cfg_get_varlist(cfg, 'EMISSION', 'specs')
        self.nspec=len(specs)

        self.__construct_file_list()

        
    def write_frame(self):
        '''
        write flexpart format partical dump file
        '''
        pass

    def load_frame(self, fn):
        '''
        load data according to single file 
        '''
        fn_full=os.path.join(self.path, fn)
        
        if not(os.path.exists(fn_full)):
            utils.write_log(
                print_prefix+fn_full+' not exist, skip...', lvl=30)
            return -1

        utils.write_log(print_prefix+'loading file: '+fn_full)
        
        if self.fmt=='flexpart':
            self.__load_flexpart(fn_full)
        elif self.fmt=='nc':
            self.__load_nc(fn_full)
        else:
            utils.throw_error(
                print_prefix+'unknown file format: '+self.fmt)
            return -1
        return 0
    def __load_flexpart(self, fn):
        '''
        load flexpart format partical dump file
        '''

        rec_slab_temp=(
            'xlon', 'xlat', 'ztra1', 
            'itramem', 'topo', 'pvi', 'qvi', 'rhoi',
            'hmixi', 'tri', 'tti')
        dump_file = FortranFile(fn, 'r')        
        
        # header
        rec=dump_file.read_record('12c')
        (itime, nparts, iomode_xycoord)=struct.unpack('III', rec)
        
        # construct partical array
        prtarray=particals.Particals(nparts, self.nspec)

        # id, rec_slab, and species
        len_slab_flag=str(
            (1+len(rec_slab_temp)+self.nspec)*4)+'c'
        decode_fmt='IfffIfffffff'+'f'*self.nspec
        for prtid in range(nparts):    
            try: 
                rec=dump_file.read_record(len_slab_flag)
            except FortranEOFError:
                break
            decode_rec=struct.unpack(decode_fmt, rec)
            for itmid, itm in enumerate(rec_slab_temp):
                getattr(prtarray, itm)[prtid]=decode_rec[itmid+1]

        # slab
        slab=__gen_flxpt_temp()

    def __construct_file_list(self):
        '''
        construct file list according to file wildcard
        '''
        try:
            time_frms= pd.date_range(
                start=self.strt_time, end=self.end_time, freq=self.frq)
        except:
            utils.throw_error(
                print_prefix+'''cannot generate time frames,
                check init_t, end_t, and output_frq in config file''')
        
        self.file_list=[]
        for ts in time_frms:
            self.file_list.append(
                utils.parse_tswildcard(ts, self.file_wildcard))
    

    