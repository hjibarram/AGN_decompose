#!/usr/bin/env python
import glob, os,sys,timeit
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import pixel_to_skycoord
import AGNdecomp.tools.tools as tol
from AGNdecomp.tools.mcmc import evaluate_2dPSF
import warnings
warnings.filterwarnings("ignore")


def prof_ana(cube,cubeE,hdr,sig=2,prior_config='priors_prop.yml',mod_ind=0,verbose=False,singlepsf=False,psamp=10,vas='',dir_o='',name='spectra',str_p=False,local=False,moffat=False,ncpu=10,sp=0):
    Inpvalues, Infvalues, Supvalues, Namevalues, Labelvalues, model_name=tol.get_priorsvalues(prior_config,verbose=verbose,mod_ind=mod_ind)
    if dir_o != '':
        tol.sycall('mkdir -p '+dir_o)
    nz,nx,ny=cube.shape
    et_c=0
    th_c=0
    tpt=''
    if str_p:
        try:
            if singlepsf:
                p_px=tol.get_somoth_val(name,dir=dir_o,sigma=5,sp=psamp,val=5,out_p=True,deg=5,tp=tpt+vas)
                p_py=tol.get_somoth_val(name,dir=dir_o,sigma=5,sp=psamp,val=6,out_p=True,deg=5,tp=tpt+vas)
                p_ds=tol.get_somoth_val(name,dir=dir_o,sigma=5,sp=psamp,val=7,out_p=True,deg=5,tp=tpt+vas)
                p_bts=tol.get_somoth_val(name,dir=dir_o,sigma=5,sp=psamp,val=8,out_p=True,deg=5,tp=tpt+vas)  
                p_eli=tol.get_somoth_val(name,dir=dir_o,sigma=5,sp=psamp,val=15,out_p=True,deg=5,tp=tpt+vas,mask_val=[1.0,0.05])
                p_tht=tol.get_somoth_val(name,dir=dir_o,sigma=5,sp=psamp,val=16,out_p=True,deg=5,tp=tpt+vas,mask_val=[180.0,20])
            else:
                p_ds=tol.get_somoth_val(name,dir=dir_o,sigma=5,sp=psamp,val=7,out_p=True,deg=5,tp=tpt+vas)
                p_px=tol.get_somoth_val(name,dir=dir_o,sigma=5,sp=psamp,val=5,out_p=True,deg=10,tp=tpt)
                p_py=tol.get_somoth_val(name,dir=dir_o,sigma=5,sp=psamp,val=6,out_p=True,deg=10,tp=tpt)
                p_bs=tol.get_somoth_val(name,dir=dir_o,sigma=5,sp=psamp,val=11,out_p=True,deg=5,tp=tpt)
                p_Re=tol.get_somoth_val(name,dir=dir_o,sigma=5,sp=psamp,val=12,out_p=True,deg=5,tp=tpt)
                p_ns=tol.get_somoth_val(name,dir=dir_o,sigma=5,sp=psamp,val=13,out_p=True,deg=5,tp=tpt)
                p_eli=tol.get_somoth_val(name,dir=dir_o,sigma=5,sp=psamp,val=15,out_p=True,deg=5,tp=tpt+vas,mask_val=[1.0,0.05])
                p_tht=tol.get_somoth_val(name,dir=dir_o,sigma=5,sp=psamp,val=16,out_p=True,deg=5,tp=tpt+vas,mask_val=[180.0,10])
            str_p=True
        except:
            str_p=False      
    try:
        dx=np.sqrt((hdr['CD1_1'])**2.0+(hdr['CD1_2'])**2.0)*3600.0
        dy=np.sqrt((hdr['CD2_1'])**2.0+(hdr['CD2_2'])**2.0)*3600.0
    except:
        try:
            dx=hdr['CD1_1']*3600.0
            dy=hdr['CD2_2']*3600.0
        except:
            dx=hdr['CDELT1']*3600.
            dy=hdr['CDELT2']*3600.
    dpix=(np.abs(dx)+np.abs(dy))/2.0    
    crpix=hdr["CRPIX3"]
    try:
        cdelt=hdr["CD3_3"]
    except:
        cdelt=hdr["CDELT3"]
    crval=hdr["CRVAL3"]
    wave_f=crval+cdelt*(np.arange(nz)+1-crpix)
    wcs = WCS(hdr)
    wcs=wcs.celestial
    if sp > 0:
        nz_t=int(nz/sp)
        spt='_sp'+str(int(sp))
    else:
        nz_t=nz
        spt=''    

    if local == False:
        if moffat:
            ft=open(dir_o+name+'_moffat'+spt+vas+'.csv','w')
            ft.write('WAVE , FLUX , FLUXN , RA , DEC , pix_X , pix_Y , alpha , beta , PSF ,Io , bs , Re , ns , At , e , th0\n')
        else:
            ft=open(dir_o+name+'_gaussian'+spt+vas+'.csv','w')
            ft.write('WAVE , FLUX , FLUXN , RA , DEC , pix_X , pix_Y , Sigma , PSF \n')   
        for i in range(0, nz_t):
            if sp > 0:
                i0=int(i*sp)
                i1=int((i+1)*sp)
                if i1 > nz:
                    i1=nz
                if i0 > nz:
                    i0=int(nz-sp)    
                map1=np.nanmean(cube[i0:i1,:,:],axis=0)
                map1e=np.nanmean(cubeE[i0:i1,:,:],axis=0)
                wave_1=np.nanmean(wave_f[i0:i1])
            else:
                map1=cube[i,:,:]
                map1e=cubeE[i,:,:]
                wave_1=wave_f[i]  
            At=0
            bt=25
            ds_m=1
            pi_x=0
            pi_y=0
            bs_c=0
            Re_c=0
            ns_c=0
            et_c=0
            th_c=0      
            if str_p:
                ds_m=p_ds(wave_1)
                pi_x=p_px(wave_1)
                pi_y=p_py(wave_1)
                if singlepsf:
                    bt=p_bts(wave_1)
                    et_c=p_eli(wave_1)
                    th_c=p_tht(wave_1)  
                else:
                    bs_c=p_bs(wave_1) 
                    Re_c=p_Re(wave_1)/dpix
                    ns_c=p_ns(wave_1)
                    et_c=p_eli(wave_1)
                    th_c=p_tht(wave_1) 
            valsI={}
            valsI['At']=At
            valsI['alpha']=ds_m
            valsI['beta']=bt
            valsI['xo']=pi_x
            valsI['yo']=pi_y
            valsI['bs']=bs_c
            valsI['Re']=Re_c
            valsI['ns']=ns_c
            valsI['ellip']=et_c
            valsI['theta']=th_c        
            valsI['dxo']=0
            valsI['dyo']=0    
            if moffat:
                dx_m1,dy_m1,ds_m1,db_m1,psf1,Ft,FtF,Io_m,bn_m,Re_m,ns_m,At0,e0_m,th0_m=evaluate_2dPSF(map1,map1e,name=name+spt,Labelvalues=Labelvalues,Namevalues=Namevalues,Inpvalues=Inpvalues,Infvalues=Infvalues,Supvalues=Supvalues,sig=sig,singlepsf=singlepsf,moffat=moffat,ncpu=ncpu,valsI=valsI) 
            else:
                dx_m1,dy_m1,ds_m1,psf1,Ft,FtF=evaluate_2dPSF(map1,map1e,name=name+spt,model=False,sig=sig,mc=mc,ncpu=ncpu)
            sky1=pixel_to_skycoord(dx_m1,dy_m1,wcs)
            val1=sky1.to_string('hmsdms')
            if verbose:
                if moffat:
                    if singlepsf:
                        print("wave=",wave_1,'FLUX=',FtF,'FLUXN=',Ft,'RADEC=',val1,"x_1=",dx_m1,"y_1=",dy_m1,"alpha_1=",ds_m1*dpix,"beta_1=",db_m1,"psf_1=",psf1*dpix,"At0=",At0,"e=",e0_m,"th0=",th0_m)
                    else:
                        print("wave=",wave_1,'FLUX=',FtF,'FLUXN=',Ft,'RADEC=',val1,"x_1=",dx_m1,"y_1=",dy_m1,"alpha_1=",ds_m1*dpix,"beta_1=",db_m1,"psf_1=",psf1*dpix,"Io=",Io_m,"bn=",bn_m,"Re=",Re_m,"ns=",ns_m,"At0=",At0,"e=",e0_m,"th0=",th0_m)
                else:
                    print("wave=",wave_1,'FLUX=',FtF,'FLUXN=',Ft,'RADEC=',val1,"x_1=",dx_m1,"y_1=",dy_m1,"sigma_1=",ds_m1*dpix,"psf_1=",psf1*dpix)
            if moffat:
                ft.write(str(wave_1)+' , '+str(FtF)+' , '+str(Ft)+' , '+val1.replace('s -','s , -').replace('s +','s , +')+' , '+str(dx_m1)+' , '+str(dy_m1)+' , '+str(ds_m1)+' , '+str(db_m1)+' , '+str(psf1*dpix)+' , '+str(Io_m)+' , '+str(bn_m)+' , '+str(Re_m*dpix)+' , '+str(ns_m)+' , '+str(At0)+' , '+str(e0_m)+' , '+str(th0_m)+' \n')
            else:
                ft.write(str(wave_1)+' , '+str(FtF)+' , '+str(Ft)+' , '+val1.replace('s -','s , -').replace('s +','s , +')+' , '+str(dx_m1)+' , '+str(dy_m1)+' , '+str(ds_m1)+' , '+str(psf1*dpix)+' \n')
        ft.close()
    else:
        if sp > 0:
            ntw=np.where((wave_f > 4850) & (wave_f < 5150))[0]
            map1=np.nanmean(cube[ntw,:,:],axis=0)
            map1e=np.nanmean(cubeE[ntw,:,:],axis=0)
            wave_1=np.nanmean(wave_f[ntw])
        else:
            map1=np.nanmean(cube,axis=0)
            map1e=np.nanmean(cubeE,axis=0)
        At=0
        bt=25
        ds_m=1
        pi_x=0
        pi_y=0
        bs_c=0
        Re_c=0
        ns_c=0
        et_c=0
        th_c=0       
        if str_p:
            pi_x=p_px(wave_1)
            pi_y=p_py(wave_1)
            ds_m=p_ds(wave_1)
            if singlepsf: 
                bt=p_bts(wave_1)
                et_c=p_eli(wave_1)
                th_c=p_tht(wave_1)   
            else:          
                bs_c=p_bs(wave_1)  
                Re_c=p_Re(wave_1)
                ns_c=p_ns(wave_1)
                et_c=p_eli(wave_1)
                th_c=p_tht(wave_1) 
        valsI={}
        valsI['At']=At
        valsI['alpha']=ds_m
        valsI['beta']=bt
        valsI['xo']=pi_x
        valsI['yo']=pi_y
        valsI['bs']=bs_c
        valsI['Re']=Re_c
        valsI['ns']=ns_c
        valsI['ellip']=et_c
        valsI['theta']=th_c        
        valsI['dxo']=0
        valsI['dyo']=0   
        if moffat:
            dx_m1,dy_m1,ds_m1,db_m1,psf1,Ft,FtF,Io_m,bn_m,Re_m,ns_m,At0,e0_m,th0_m=evaluate_2dPSF(map1,map1e,name=name+spt,Labelvalues=Labelvalues,Namevalues=Namevalues,Inpvalues=Inpvalues,Infvalues=Infvalues,Supvalues=Supvalues,sig=sig,plot_f=True,singlepsf=singlepsf,moffat=moffat,ncpu=ncpu,valsI=valsI)
        else:
            dx_m1,dy_m1,ds_m1,psf1,Ft,FtF=evaluate_2dPSF(map1,map1e,name=name+spt,sig=sig,plot_f=True,ncpu=ncpu,valsT=valsT)
        sky1=pixel_to_skycoord(dx_m1,dy_m1,wcs)
        val1=sky1.to_string('hmsdms')
        if moffat:
            if singlepsf:
                print('FLUX=',FtF,'FLUXN=',Ft,'RADEC=',val1,"x_1=",dx_m1,"y_1=",dy_m1,"alpha_1=",ds_m1*dpix,"beta_1=",db_m1,"psf_1=",psf1*dpix,"At0=",At0,"e=",e0_m,"th0=",th0_m)
            else:
                print('FLUX=',FtF,'FLUXN=',Ft,'RADEC=',val1,"x_1=",dx_m1,"y_1=",dy_m1,"alpha_1=",ds_m1*dpix,"beta_1=",db_m1,"psf_1=",psf1*dpix,"Io=",Io_m,"bn=",bn_m,"Re=",Re_m,"ns=",ns_m,"At0=",At0,"e=",e0_m,"th0=",th0_m)
        else:
            print('FLUX=',FtF,'FLUXN=',Ft,'RADEC=',val1,"x_1=",dx_m1,"y_1=",dy_m1,"sigma_1=",ds_m1*dpix,"psf_1=",psf1*dpix)