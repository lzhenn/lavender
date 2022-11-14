#/usr/bin/env python3
"""specific module for IO"""
import datetime, struct, os, gc
import pandas as pd
from scipy.io import FortranFile, FortranEOFError

from ..core import particals
from . import utils, const, cfgparser

import netCDF4 as nc4
import wrf 

print_prefix='lib.io>>'

class FileHandler():
    """ file handler class """
    def __init__(self, strt_time_str, end_time_str):
        self.strt_time=datetime.datetime.strptime(
            strt_time_str, const.YMDHM)
        
        self.end_time=datetime.datetime.strptime(
            end_time_str, const.YMDHM)
    
    def construct_file_list(self):
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
        self.tfrms=time_frms 
        self.file_list=[]
        for ts in time_frms:
            self.file_list.append(
                utils.parse_tswildcard(ts, self.file_wildcard))


class InHandler(FileHandler):

    def __init__(self, cfg):
        '''
        Initialize Infile handler with config and loading data
        '''
        utils.write_log(print_prefix+'construct InFileHandler')
        FileHandler.__init__(
            self, cfg['INPUT']['init_t'], cfg['INPUT']['end_t'])

        self.wrf_root=utils.parse_tswildcard(
            self.strt_time, cfg['INPUT']['wrfout_path'])
        self.file_wildcard=cfg['INPUT']['wrfout_wildcard']
        self.frq=cfg['INPUT']['feed_frq']
 
        self.construct_file_list()
        self.__check_data_exist()
        self.load_meta()

    def __check_data_exist(self):
        '''
        check wrfout availablility
        '''
        for fn in self.file_list:
            fn_full=os.path.join(self.wrf_root, fn)
            if not os.path.exists(fn_full):
                utils.throw_error(
                    print_prefix+'''cannot find file %s,
                    check output and wrfout_wildcard in config file'''%fn)
    
    def load_meta(self):
        '''
        load wrfout meta data 
        '''
        fn=self.file_list[0]
        fn_full=os.path.join(self.wrf_root, fn)
        wrf_hdl=nc4.Dataset(fn_full)
        
        # collect global attr
        self.dx=wrf_hdl.DX
        abz3d=wrf.getvar(wrf_hdl,'z') # model layer elevation above sea level
        abz3d_stag=wrf.getvar(wrf_hdl,'zstag') # model layer elevation above sea level
        ter=wrf.getvar(wrf_hdl,'ter') # terrain height  

        # model layer elevation above terrain
        #self.z=self.z-ter.mean(['south_north','west_east'])
        self.z=(abz3d-ter).mean(['south_north','west_east'])
        self.z_stag=(abz3d_stag-ter).mean(['south_north','west_east'])

        # get index of z for near surface layer and free atm 
        (self.n_sn, self.n_we)=ter.shape # on mass grid
        
        # lats lons on mass and staggered grids
        self.XLAT=wrf.getvar(wrf_hdl,'XLAT')
        self.XLONG=wrf.getvar(wrf_hdl,'XLONG')
        self.XLAT_U=wrf.getvar(wrf_hdl,'XLAT_U')
        self.XLONG_U=wrf.getvar(wrf_hdl,'XLONG_U')
        self.XLAT_V=wrf.getvar(wrf_hdl,'XLAT_V')
        self.XLONG_V=wrf.getvar(wrf_hdl,'XLONG_V')
        wrf_hdl.close() 
        gc.collect()

    def load_frame(self, frm):
        '''
        load wrfout data 
        '''
        utils.write_log(print_prefix+'load %s' % self.file_list[frm])
        fn=self.file_list[frm]
        fn_full=os.path.join(self.wrf_root, fn)
        wrf_hdl=nc4.Dataset(fn_full)
 
        # template UVW
        self.U = wrf.getvar(wrf_hdl, 'U')
        self.V = wrf.getvar(wrf_hdl, 'V')
        self.W = wrf.getvar(wrf_hdl, 'W')
        self.T = wrf.getvar(wrf_hdl, 'T')
        self.p = wrf.getvar(wrf_hdl, 'pres') # Full Model Pressure in Pa
        # template UV10 and T2
        self.U10 = wrf.getvar(wrf_hdl, 'U10')
        self.V10 = wrf.getvar(wrf_hdl, 'V10')
        self.T2 = wrf.getvar(wrf_hdl, 'T2')
        wrf_hdl.close() 
        gc.collect()

       


class OutHandler(FileHandler):

    '''
    Construct Out File Handler for dumped partical data 
    in unformatted fortran binary file
    
    Methods
    -----------
    __init__:   initialize FLEXPHandler and loading data
    write_frame: write flexpart format partical dump file
    '''
    
    def __init__(self, cfg):
        '''
        Initialize OutHandler with config and loading data
        '''
        FileHandler.__init__(
            self, cfg['INPUT']['init_t'], cfg['INPUT']['end_t'])

        utils.write_log(print_prefix+'construct OutFileHandler')
        self.fmt=cfg['OUTPUT']['file_fmt']
        self.path=cfg['OUTPUT']['file_root']
        self.file_wildcard=cfg['OUTPUT']['file_wildcard']
        self.frq=cfg['OUTPUT']['output_frq']
        specs=cfgparser.cfg_get_varlist(cfg, 'EMISSION', 'specs')
        self.nspec=len(specs)

        self.construct_file_list()

        
    def write_frame(self):
        '''
        write partical dump file
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
            return None 

        utils.write_log(print_prefix+'loading file: '+fn_full)
        
        if self.fmt=='flexpart':
            prtarray=self.__load_flexpart(fn_full)
        elif self.fmt=='nc':
            prtarray=self.__load_nc(fn_full)
        else:
            utils.throw_error(
                print_prefix+'unknown file format: '+self.fmt)
        utils.write_log(
            print_prefix+'%s loaded successfully!' % fn_full)
        return prtarray
    
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
        prtarray=particals.Particals(nptcls=nparts, nspec=self.nspec)

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
        
        return prtarray

   