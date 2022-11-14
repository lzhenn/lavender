#!/usr/bin/env python3
'''
Date: Nov 03, 2022
Lavender is a Lagrangian dispersion model to calculate the 
dispersion of atmosphreic tracers using WRF output files.

This is the main script to drive the model

History:
Nov 03, 2022 --- Kick off! 
Nov 06, 2022 --- FLEXPART output support for rendering

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
    copyfile(MWD+'/conf/config.case.ini', CWD+'/config.case.ini')
    #print('Template config file created, please modify it and run again!')
    cfg=cfgparser.read_cfg(CWD+'/config.case.ini')
    
    # logging manager
    logging.config.fileConfig(MWD+'/conf/config.logging.ini')
    
    utils.write_log('Config validation test...')
    #cfgparser.cfg_valid_test(cfg)

    utils.write_log('Build driver...')
    drv=driver.Driver(cfg)
    
    # driver drives!
    drv.drive()

    exit()
    
    if cfg['postprocess'].getboolean('visualize'):
        utils.write_log('Post 3D rendering...')
        painter.render3d(cfg,MWD)
    

    # lock the tasks Apr 1 2021
    cfg_hdl['CORE']['ntasks']='1'

    utils.write_log('Read Input Observations...')
    obv_df=pd.read_csv(cfg_hdl['INPUT']['input_root']+cfg_hdl['INPUT']['input_obv'],header=0,
            names=['yyyymmddhhMM','lat','lon','height','wind_speed','wind_dir','temp','rh','pres','attr1','attr2'])
    # make sure the list is sorted by datetime and long enough
    obv_df=obv_df.sort_values(by='yyyymmddhhMM') 
    utils.write_log('Input Quality Control...')
    lib.obv_constructor.obv_examiner(obv_df, cfg_hdl)
    
    utils.write_log('Read Wind Profile Exponents...')
    wind_prof_df=pd.read_csv('./db/power_coef_wind.csv')
    time_mgr.toc('INPUT MODULE')

    utils.write_log('Construct WRFOUT Handler...')
    fields_hdl=lib.preprocess_wrfinp.wrf_mesh(cfg_hdl)
    
    utils.write_log('Construct Observation Satation Objs...')
    obv_lst=[] 
    for row in obv_df.itertuples():
        obv_lst.append(lib.obv_constructor.obv(row, wind_prof_df, cfg_hdl, fields_hdl))
    
    # get area mean pvalue 
    fields_hdl.get_area_pvalue([obv.prof_pvalue for obv in obv_lst])
    # setup Ekman layer and geostrophic wind in obv wind profile
    lib.obv_constructor.set_upper_wind(fields_hdl, obv_lst)

    utils.write_log('Construct Model Clocks and Interpolating Estimators.....')
    clock_cfg=lib.model_clock.clock_cfg_parser(cfg_hdl)
    ntasks=clock_cfg['nclock']
    clock_lst=[]
    estimator_lst=[]
    
    for i in range(0, ntasks):
        clock_lst.append(lib.model_clock.model_clock(clock_cfg, i))
        estimator_lst.append(core.aeolus.aeolus(fields_hdl, cfg_hdl))

    time_mgr.toc('CONSTRUCTION MODULE')

    if ntasks == 1:
        utils.write_log('Aeolus Interpolating Estimator Casting...')
        clock=clock_lst[0]
        estimator=estimator_lst[0]
        while not(clock.done):
           
            estimator.cast(obv_lst, fields_hdl, clock)
            time_mgr.toc('CAST MODULE')

            utils.write_log('Output Diagnostic UVW Fields...')
            core.aeolus.output_fields(cfg_hdl, estimator, clock)
            time_mgr.toc('OUTPUT MODULE')
            
            clock.advance()
    else:
        utils.write_log('Multiprocessing initiated. Master process %s.' % os.getpid())
        # let's do the multiprocessing magic!
        # start process pool
        results=[]
        process_pool = Pool(processes=ntasks)
        for itsk in range(ntasks):  
            results.append(process_pool.apply_async(run_mtsk,args=(itsk, obv_lst, clock_lst[itsk], 
                estimator_lst[itsk],fields_hdl,cfg_hdl,)))

        process_pool.close()
        process_pool.join()
        time_mgr.toc('MULTI CAST MODULE')

    time_mgr.dump()




def run_mtsk(itsk, obv_lst, clock, estimator, fields_hdl, cfg_hdl):
    """
    Aeolus cast function for multiple processors
    """
    utils.write_log('TASK[%02d]: Aeolus Interpolating Estimator Casting...' % itsk)
    while not(clock.done):
           
        estimator.cast(obv_lst, fields_hdl, clock)

        utils.write_log('TASK[%02d]: Output Diagnostic UVW Fields...' % itsk )
        core.aeolus.output_fields(cfg_hdl, estimator, clock)
        
        clock.advance()
    utils.write_log('TASK[%02d]: Aeolus Subprocessing Finished!' % itsk)
    
    return 0

if __name__=='__main__':
    main_run()
