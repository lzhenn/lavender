#!/usr/bin/env python3
'''
Module Painter


History:
Nov 06, 2022 --- render3d serial version completed!
Nov 06, 2022 --- FLEXPART output support for rendering

'''

import os
import numpy as np
import xarray as xr 
from multiprocessing import Pool
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import cmaps

from . import io, utils, const, mtask_tool 

print_prefix='lib.painter>>'

   

def render3d(cfg,MWD):
    '''
    render 3d plot of the dumped partical file 
    '''
    utils.write_log(print_prefix+'render 3d plot...')
    # vis domain bdy    
    latN, latS=\
        float(cfg['OUTPUT']['latN']), float(cfg['OUTPUT']['latS'])
    lonW, lonE=\
        float(cfg['OUTPUT']['lonW']), float(cfg['OUTPUT']['lonE'])
    
    # get topo data
    topo=__get_topo_data(cfg,MWD)
    topo=topo.sel(y=slice(latS,latN), x=slice(lonW,lonE))
    topo_lat,topo_lon,topo_z=topo['y'], topo['x'], topo['z']
    ny=topo_lat.shape[0]
    
    # subsampling
    max_ny=100
    if ny>max_ny:
        zoom_r=int(np.floor(ny/max_ny))+1
        topo_lat=topo_lat[::zoom_r]
        topo_lon=topo_lon[::zoom_r]
        topo_z=topo_z[::zoom_r,::zoom_r] 
    topo_lon, topo_lat = np.meshgrid(topo_lon.values, topo_lat.values)
    
    dx = np.diff(topo_lon[1,1:3])
    dy = np.diff(topo_lat[1:3,1])
    # get partical dump data
    file_hdler=io.OutHandler(cfg)
    
    # mtsks_lst[iproc(strt_subidx,end_subidx)]
    mtsks_lst=mtask_tool.alloc_mtasks_lst(
        int(cfg['postprocess']['ntasks']), file_hdler.file_list)
    ntasks=len(mtsks_lst) 
    
    # MULTIPROCESSING: start process pool
    process_pool = Pool(processes=ntasks)
    results=[]

    # open tasks ID 0 to ntasks-1
    for itsk in range(0, ntasks):  
        # start and end file index for each task
        result=process_pool.apply_async(
            __mtsk_render3d, 
            args=(
                itsk, mtsks_lst, file_hdler, 
                topo_lat, topo_lon, topo_z, dx, dy))
        results.append(result)
    print(results[0].get())
    process_pool.close()
    process_pool.join() 


def __mtsk_render3d(
        itsk, seg_list, fh, topo_lat, topo_lon, topo_z, dx, dy):
    
    len_seg=len(seg_list[itsk])    
    for subid, ts in enumerate(seg_list[itsk]):
        # get global idx
        gidx=fh.file_list.index(ts)

        # init particle array in current frame
        prtarray=fh.load_frame(ts)
        
        if prtarray is None:
            continue
        
        utils.write_log(
            '%sTASK[%04d]: subtask (%04d/%04d) rendering %s with nptcl=%d' % (
                print_prefix, itsk, subid+1, len_seg, ts, prtarray.nptcls))
        
        plon,plat,pz=prtarray.xlon, prtarray.xlat, prtarray.ztra1
       
        fig = plt.figure(
            figsize=[const.FIG_WIDTH, const.FIG_HEIGHT],dpi=150)

        ax = fig.add_axes(const.FRM_MARGIN, projection='3d')
   
        res = len(topo_lon[0,:])
        # camera position in xyz
        xyz = np.array(__sph2cart(*__sphview(ax)), ndmin=3).T   
        # "distance" of bars from camera
        zo = np.multiply(
            [topo_lon, topo_lat, np.zeros_like(topo_z)], xyz).sum(0)  
        bars = np.empty(topo_lon.shape, dtype=object)

        ncmap = ListedColormap(cmaps.GMT_globe(range(0,209,2))) 
        #ncmap = ListedColormap(["red","blue","green"])
        cnlevels = np.concatenate(
            (np.arange(-3150,0,50),np.arange(0,6150,150)))


        for i,(x,y,dz,o) in enumerate(__ravzip(
            topo_lon, topo_lat,topo_z, zo)):
            for nll in range(0,len(cnlevels),1):
                if nll==0 and dz<cnlevels[nll]:
                    color0 = ncmap([nll])
                    break
                elif (nll>0 and nll<(len(cnlevels)-1)) and (
                    dz>=cnlevels[nll-1] and dz<cnlevels[nll]):
                    color0 = ncmap([nll])
                    break
                elif nll==(len(cnlevels)-1) and dz>=cnlevels[nll]:
                    color0 = ncmap([nll+1])

            j, k = divmod(i, res)
            bars[j, k] = pl = ax.bar3d(x, y, 0, dx, dy, dz, color0)
            pl._sort_zpos = o
        
        ax.scatter(
            plon, plat,pz, marker='.', color='white',
            zorder=999999, s=1, alpha=0.1)   
        
        ax.set_facecolor('k')
        ax.set_zscale('log')
        #ax.set_xlim(lonW,lonE)
        #ax.set_ylim(latS,latN)
        ax.set_zlim(0, 10000)
        ax.view_init(elev=53.0-gidx/6.0, azim=-75+gidx/6.0)
        ax.grid(False)
        plt.axis('off')
 
        plt.title(
            'AirTracers %s' % fh.tfrms[gidx].strftime(const.YMDHM),
            fontsize=const.MIDFONT)
                
        figpath=os.path.join(
            const.CWD, 
            utils.parse_tswildcard(fh.tfrms[gidx], fh.fig_wildcard))
        plt.savefig(figpath, dpi=150)
        plt.close('all')
        utils.write_log(
            '%sTASK[%04d]: subtask (%04d/%04d) render3D done!' % (
                print_prefix, itsk, subid+1, len_seg))
        # break for test
        # break
    return 0
def __sph2cart(r, theta, phi):
    '''spherical to cartesian transformation.'''
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z

def __sphview(ax):
    '''returns the camera position for 3D axes in spherical coordinates'''
    r = np.square(
        np.max([ax.get_xlim(), ax.get_ylim()], 1)).sum()
    theta, phi = np.radians((90-ax.elev, ax.azim))
    return r, theta, phi

def __ravzip(*itr):
    '''flatten and zip arrays'''
    return zip(*map(np.ravel, itr))

 
    
        
def __get_topo_data(cfg,MWD):
    '''
    get topo data from topo file
    '''
    topo_file=os.path.join(MWD,'db/etopo.nc')
    if not(os.path.exists(topo_file)):
        utils.write_log(
            print_prefix+topo_file+' not exist, skip...', lvl=30)
        return None
    else:
        utils.write_log(print_prefix+'load topo data from '+topo_file)
        topo=xr.open_dataset(topo_file)
        return topo
    exit()