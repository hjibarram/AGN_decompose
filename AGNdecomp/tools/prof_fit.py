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

def prof_ana(cube,cubeE,hdr,sig=2,prior_config='priors_prop.yml',wavew1=4850,wavew2=5150,mod_ind=0,mod_ind0=0,verbose=False,singlepsf=False,psamp=10,tp='',dir_o='',name='spectra',str_p=False,local=False,moffat=False,ncpu=10,sp=0):
    Inpvalues, Infvalues, Supvalues, Namevalues, Labelvalues, model_name=tol.get_priorsvalues(prior_config,verbose=verbose,mod_ind=mod_ind)
    if dir_o != '':
        tol.sycall('mkdir -p '+dir_o)
    nz,nx,ny=cube.shape
    if str_p:
        try:
        if true:
            Namevalues0=tol.get_priorsvalues(prior_config,verbose=verbose,mod_ind=mod_ind0,onlynames=True)
            p_val=[]
            for i in range(0, len(Namevalues0)):
                p_vals.extend([tol.get_somoth_val(name,dir=dir_o,sigma=5,sp=psamp,val=i+5,out_p=True,deg=5,tp=tp)])
            '''
            if singlepsf:
                p_px=tol.get_somoth_val(name,dir=dir_o,sigma=5,sp=psamp,val=5,out_p=True,deg=5,tp=tp)
                p_py=tol.get_somoth_val(name,dir=dir_o,sigma=5,sp=psamp,val=6,out_p=True,deg=5,tp=tp)
                p_ds=tol.get_somoth_val(name,dir=dir_o,sigma=5,sp=psamp,val=7,out_p=True,deg=5,tp=tp)
                p_bts=tol.get_somoth_val(name,dir=dir_o,sigma=5,sp=psamp,val=8,out_p=True,deg=5,tp=tp)
                p_At=tol.get_somoth_val(name,dir=dir_o,sigma=5,sp=psamp,val=14,out_p=True,deg=5,tp=tp)  
                p_eli=tol.get_somoth_val(name,dir=dir_o,sigma=5,sp=psamp,val=15,out_p=True,deg=5,tp=tp,mask_val=[1.0,0.05])
                p_tht=tol.get_somoth_val(name,dir=dir_o,sigma=5,sp=psamp,val=16,out_p=True,deg=5,tp=tp,mask_val=[180.0,20])
            #else:
                p_px=tol.get_somoth_val(name,dir=dir_o,sigma=5,sp=psamp,val=5,out_p=True,deg=5,tp=tp)
                p_py=tol.get_somoth_val(name,dir=dir_o,sigma=5,sp=psamp,val=6,out_p=True,deg=5,tp=tp)
                p_ds=tol.get_somoth_val(name,dir=dir_o,sigma=5,sp=psamp,val=7,out_p=True,deg=5,tp=tp)
                p_bts=tol.get_somoth_val(name,dir=dir_o,sigma=5,sp=psamp,val=8,out_p=True,deg=5,tp=tp)  
                p_is=tol.get_somoth_val(name,dir=dir_o,sigma=5,sp=psamp,val=10,out_p=True,deg=5,tp=tp)
                p_bs=tol.get_somoth_val(name,dir=dir_o,sigma=5,sp=psamp,val=11,out_p=True,deg=5,tp=tp)
                p_Re=tol.get_somoth_val(name,dir=dir_o,sigma=5,sp=psamp,val=12,out_p=True,deg=5,tp=tp)
                p_ns=tol.get_somoth_val(name,dir=dir_o,sigma=5,sp=psamp,val=13,out_p=True,deg=5,tp=tp)
                p_At=tol.get_somoth_val(name,dir=dir_o,sigma=5,sp=psamp,val=14,out_p=True,deg=5,tp=tp)
                p_eli=tol.get_somoth_val(name,dir=dir_o,sigma=5,sp=psamp,val=15,out_p=True,deg=5,tp=tp,mask_val=[1.0,0.05])
                p_tht=tol.get_somoth_val(name,dir=dir_o,sigma=5,sp=psamp,val=16,out_p=True,deg=5,tp=tp,mask_val=[180.0,10])'''
        #    str_p=True
        #except:
        #    str_p=False
        else:
            Namevalues0=Namevalues      
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
        valt=Namevalues
        head_vals=''
        for namev in Namevalues:
            head_vals=head_vals+' , '+namev
        ft=open(dir_o+name+'_'+model_name+spt+tp+'.csv','w')
        ft.write('wave , flux , fluxN , ra , dec , psf'+head_vals+'\n') 
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
            Io_c=0
            bs_c=0
            Re_c=0
            ns_c=0
            et_c=0
            th_c=0      
            if str_p:
                ds_m=p_ds(wave_1)
                pi_x=p_px(wave_1)
                pi_y=p_py(wave_1)
                bt=p_bts(wave_1)
                At=p_At(wave_1)
                et_c=p_eli(wave_1)
                th_c=p_tht(wave_1)
                if singlepsf == False:
                    Io_c=p_is(wave_1)
                    bs_c=p_bs(wave_1) 
                    Re_c=p_Re(wave_1)/dpix
                    ns_c=p_ns(wave_1) 
            valsI={}
            valsI['At']=At
            valsI['alpha']=ds_m
            valsI['beta']=bt
            valsI['xo']=pi_x
            valsI['yo']=pi_y
            valsI['Io']=Io_c
            valsI['bs']=bs_c
            valsI['Re']=Re_c
            valsI['ns']=ns_c
            valsI['ellip']=et_c
            valsI['theta']=th_c        
            valsI['dxo']=0
            valsI['dyo']=0    
            if moffat:
                pars_max,psf1,Ft,FtF=evaluate_2dPSF(map1,map1e,name=name+spt,Labelvalues=Labelvalues,Namevalues=Namevalues,Inpvalues=Inpvalues,Infvalues=Infvalues,Supvalues=Supvalues,sig=sig,singlepsf=singlepsf,moffat=moffat,ncpu=ncpu,valsI=valsI) 
            else:
                pars_max,psf1,Ft,FtF=evaluate_2dPSF(map1,map1e,name=name+spt,Labelvalues=Labelvalues,Namevalues=Namevalues,Inpvalues=Inpvalues,Infvalues=Infvalues,Supvalues=Supvalues,sig=sig,ncpu=ncpu)
            sky1=pixel_to_skycoord(pars_max['xo'],pars_max['yo'],wcs)
            val1=sky1.to_string('hmsdms')
            if verbose:
                linet='wave='+str(wave_1)+' FLUX='+str(FtF)+' FLUXN='+str(Ft)+' RADEC='+str(val1)+' PSF='+str(psf1*dpix)+' '
                linev=''
                for val in Namevalues:
                    linev=linev+' '+val+'='+str(pars_max[val])
                print(linet+linev)
            linet=str(wave_1)+' , '+str(FtF)+' , '+str(Ft)+' , '+val1.replace('s -','s , -').replace('s +','s , +')+' , '+str(psf1*dpix)
            linev=''
            for val in Namevalues:
                linev=linev+' , '+str(pars_max[val])
            ft.write(linet+linev+' \n')
        ft.close()
    else:
        if sp > 0:
            ntw=np.where((wave_f > wavew1) & (wave_f < wavew2))[0]
            map1=np.nanmean(cube[ntw,:,:],axis=0)
            map1e=np.nanmean(cubeE[ntw,:,:],axis=0)
            wave_1=np.nanmean(wave_f[ntw])
        else:
            map1=np.nanmean(cube,axis=0)
            map1e=np.nanmean(cubeE,axis=0)
        #At=0
        #bt=25
        #ds_m=1
        #pi_x=0
        #pi_y=0
        #Io_c=0
        #bs_c=0
        #Re_c=0
        #ns_c=0
        #et_c=0
        #th_c=0
        valsI={}
        for i in range(0, len(Namevalues0)):
            if str_p:
                p_val=p_vals[i]
                val_t=p_val(wave_1)
                valsI[Namevalues0[i]]=val_t
            else:
                valsI[Namevalues0[i]]=Inpvalues[i]
        valsI={}
        #valsI['At']=At
        #valsI['alpha']=ds_m
        #valsI['beta']=bt
        #valsI['xo']=pi_x
        #valsI['yo']=pi_y
        #valsI['Io']=Io_c
        #valsI['bs']=bs_c
        #valsI['Re']=Re_c
        #valsI['ns']=ns_c
        #valsI['ellip']=et_c
        #valsI['theta']=th_c        
        valsI['dxo']=0
        valsI['dyo']=0   
        if moffat:
            pars_max,psf1,Ft,FtF=evaluate_2dPSF(map1,map1e,name=name+spt,Labelvalues=Labelvalues,Namevalues=Namevalues,Inpvalues=Inpvalues,Infvalues=Infvalues,Supvalues=Supvalues,sig=sig,plot_f=True,singlepsf=singlepsf,moffat=moffat,ncpu=ncpu,valsI=valsI)
        else:
            pars_max,psf1,Ft,FtF=evaluate_2dPSF(map1,map1e,name=name+spt,Labelvalues=Labelvalues,Namevalues=Namevalues,Inpvalues=Inpvalues,Infvalues=Infvalues,Supvalues=Supvalues,sig=sig,plot_f=True,ncpu=ncpu,valsT=valsT)
        sky1=pixel_to_skycoord(pars_max['xo'],pars_max['yo'],wcs)
        val1=sky1.to_string('hmsdms')
        linet='wave='+str(wave_1)+' FLUX='+str(FtF)+' FLUXN='+str(Ft)+' RADEC='+str(val1)+' PSF='+str(psf1*dpix)
        linev=''
        for val in Namevalues:
            linev=linev+' '+val+'='+str(pars_max[val])
        print(linet+linev)