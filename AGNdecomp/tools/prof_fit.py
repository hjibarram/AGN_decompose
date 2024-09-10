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


def prof_ana(cube,cubeE,hdr,sig=2,verbose=False,beta=True,fcenter=False,ellip=False,re_int=False,singlepsf=False,psamp=10,ds_i=0,ro_i=0,Lt_i=0,vas='',dir_o='',name='spectra',trip=False,ring=False,psft=False,str_p=False,local=False,moffat=False,mc=False,ncpu=10,sp=0,coef=[-9.50013525e-21,8.18487432e-16,-1.88951248e-11,1.87198198e-07,-8.58072070e-04,3.90811581e+00],bt=0,psf_t=False):
    if dir_o != '':
        tol.sycall('mkdir -p '+dir_o)
    nz,nx,ny=cube.shape
    et_c=0
    th_c=0
    if trip:
        ring=False
    if ring:
        pia_x=tol.get_somoth_val(name.replace('_res',''),dir=dir_o,sigma=0,sp=1,val=5,out_p=False,deg=10)
        pia_y=tol.get_somoth_val(name.replace('_res',''),dir=dir_o,sigma=0,sp=1,val=6,out_p=False,deg=10)
        tpt='_ring'
    elif trip:
        tpt='_trip'
    else:
        tpt=''
    #p_ds=get_somoth_val(name,dir=dir_o,sigma=5,sp=10,val=7,out_p=True,deg=5,tp=tpt+vas)    
    if str_p:
        #
        try:
        #if True:
            if singlepsf:
                p_px=tol.get_somoth_val(name,dir=dir_o,sigma=5,sp=psamp,val=5,out_p=True,deg=5,tp=tpt+vas)
                p_py=tol.get_somoth_val(name,dir=dir_o,sigma=5,sp=psamp,val=6,out_p=True,deg=5,tp=tpt+vas)
                p_ds=tol.get_somoth_val(name,dir=dir_o,sigma=5,sp=psamp,val=7,out_p=True,deg=5,tp=tpt+vas)
                if beta:
                    p_bts=tol.get_somoth_val(name,dir=dir_o,sigma=5,sp=psamp,val=8,out_p=True,deg=5,tp=tpt+vas)  
                if ellip:
                    p_eli=tol.get_somoth_val(name,dir=dir_o,sigma=5,sp=psamp,val=15,out_p=True,deg=5,tp=tpt+vas,mask_val=[1.0,0.05])
                    p_tht=tol.get_somoth_val(name,dir=dir_o,sigma=5,sp=psamp,val=16,out_p=True,deg=5,tp=tpt+vas,mask_val=[180.0,20])
            else:
                if psft == False:
                    p_ds=tol.get_somoth_val(name,dir=dir_o,sigma=5,sp=psamp,val=7,out_p=True,deg=5,tp=tpt+vas)
                p_px=tol.get_somoth_val(name,dir=dir_o,sigma=5,sp=psamp,val=5,out_p=True,deg=10,tp=tpt)
                p_py=tol.get_somoth_val(name,dir=dir_o,sigma=5,sp=psamp,val=6,out_p=True,deg=10,tp=tpt)
                p_bs=tol.get_somoth_val(name,dir=dir_o,sigma=5,sp=psamp,val=11,out_p=True,deg=5,tp=tpt)
                p_Re=tol.get_somoth_val(name,dir=dir_o,sigma=5,sp=psamp,val=12,out_p=True,deg=5,tp=tpt)
                p_ns=tol.get_somoth_val(name,dir=dir_o,sigma=5,sp=psamp,val=13,out_p=True,deg=5,tp=tpt)
                if ellip:
                    p_eli=tol.get_somoth_val(name,dir=dir_o,sigma=5,sp=psamp,val=15,out_p=True,deg=5,tp=tpt+vas,mask_val=[1.0,0.05])
                    p_tht=tol.get_somoth_val(name,dir=dir_o,sigma=5,sp=psamp,val=16,out_p=True,deg=5,tp=tpt+vas,mask_val=[180.0,10])
            if ring:
                p_ds=tol.get_somoth_val(name,dir=dir_o,sigma=5,sp=psamp,val=7,out_p=True,deg=5,tp='_ring')
                p_ro=tol.get_somoth_val(name,dir=dir_o,sigma=5,sp=psamp,val=8,out_p=True,deg=5,tp='_ring')
            if trip:
                p_ls=tol.get_somoth_val(name,dir=dir_o,sigma=5,sp=psamp,val=16,out_p=True,deg=15,tp='_trip',convt=True)#sigma=20
            str_p=True
        except:
            str_p=False
    #print(str_p)        
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
    #print(dpix)
    crpix=hdr["CRPIX3"]
    try:
        cdelt=hdr["CD3_3"]
    except:
        cdelt=hdr["CDELT3"]
    crval=hdr["CRVAL3"]
    wave_f=crval+cdelt*(np.arange(nz)+1-crpix)
    pt = np.poly1d(coef)
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
            if ring:
                ft=open(dir_o+name+'_moffat'+spt+vas+'_ring.csv','w')
                ft.write('WAVE , FLUX , FLUXN , RA , DEC , pix_X , pix_Y , sigma , r0 , PSF ,Io , bs , Re , ns , At , e , th0\n')
            if trip:
                ft=open(dir_o+name+'_moffat'+spt+vas+'_trip.csv','w')
                ft.write('WAVE , FLUX , FLUXN , RA , DEC , pix_X , pix_Y , alpha , beta , PSF ,Io , bs , Re , ns , At , e , Lt\n')                
            else:
                ft=open(dir_o+name+'_moffat'+spt+vas+'.csv','w')
                ft.write('WAVE , FLUX , FLUXN , RA , DEC , pix_X , pix_Y , alpha , beta , PSF ,Io , bs , Re , ns , At , e , th0\n')
        else:
            ft=open(dir_o+name+'_gaussian'+spt+vas+'.csv','w')
            ft.write('WAVE , FLUX , FLUXN , RA , DEC , pix_X , pix_Y , Sigma , PSF \n')
        et=0    
        for i in range(0, nz_t):
            if sp > 0:
                i0=int(i*sp)
                i1=int((i+1)*sp)
                if i1 > nz:
                    i1=nz
                if i0 > nz:
                    i0=int(nz-sp)    
                map1=np.nanmean(cube[i0:i1,:,:],axis=0)#mean
                map1e=np.nanmean(cubeE[i0:i1,:,:],axis=0)
                wave_1=np.nanmean(wave_f[i0:i1])
                if ring:
                    pi_x=np.nanmean(pia_x[i0:i1])
                    pi_y=np.nanmean(pia_y[i0:i1])
            else:
                map1=cube[i,:,:]
                map1e=cubeE[i,:,:]
                wave_1=wave_f[i]
                if ring:
                    pi_x=pi_x[i]
                    pi_y=pi_y[i]
            if psf_t:
                if ring:
                    if ds_i == 0:
                        ds_i=0.5 
                        ro_i=2.0
                    psf_coef=0
                else:
                    #pt=get_somoth_val(name,dir=dir_o,sigma=5,sp=10,val=7,out_p=True,deg=5)   
                    if psft == False:#singlepsf and 
                        psf_coef=p_ds(wave_1)
                    else: 
                        psf_coef=pt(wave_1)/dpix
                    ro_i=0
                    ds_i=0
            else:
                if ring:
                    if i == 0:
                        p_ds=tol.get_somoth_val(name,dir=dir_o,sigma=5,sp=10,val=7,out_p=False,deg=5,tp='_ring')
                        p_ro=tol.get_somoth_val(name,dir=dir_o,sigma=5,sp=10,val=8,out_p=False,deg=5,tp='_ring')
                    ds_i=p_ds(wave_1)
                    ro_i=p_ro(wave_1)  
                    psf_coef=0
                else:
                    if psft == False:#singlepsf and 
                        psf_coef=p_ds(wave_1)
                    else:
                        pt=tol.get_somoth_val(name,dir=dir_o,sigma=5,sp=10,val=9,out_p=True,deg=5)    
                        psf_coef=pt(wave_1)/dpix
                    ro_i=0
                    ds_i=0
            if str_p:
                if not ring:
                    pi_x=p_px(wave_1)
                    pi_y=p_py(wave_1)
                    Lt_c = 0.0
                    if trip:
                        Lt_c=p_ls(wave_1)
                        if Lt_c < 0.0:
                            Lt_c = 0.0
                else:
                    ds_i=p_ds(wave_1)
                    ro_i=p_ro(wave_1)
                if singlepsf:
                    bs_c=3.0
                    ns_c=0
                    if beta:
                        bt=p_bts(wave_1)
                        Re_c=0.
                    else:
                        Re_c=100
                    if ellip:
                        et_c=p_eli(wave_1)
                        th_c=p_tht(wave_1)#147 
                        ellip=False  
                        et=1
                    else:
                        if et==1:
                            et_c=p_eli(wave_1)
                            th_c=p_tht(wave_1)#147 
                        else:
                            et_c=0
                            th_c=0    
                else:
                    bs_c=p_bs(wave_1)    
                #bs_c=p_bs(wave_1)
                    Re_c=p_Re(wave_1)
                    ns_c=p_ns(wave_1)
                    if ellip:
                        et_c=p_eli(wave_1)
                        th_c=p_tht(wave_1)#147 
                        ellip=False
                        et=1
                    else: 
                        if et==1:
                            et_c=p_eli(wave_1)
                            th_c=p_tht(wave_1)#147 
                        else:
                            et_c=0
                            th_c=0    
            else:
                if not ring:
                    pi_x=0
                    pi_y=0
                bs_c=0
                Re_c=0
                ns_c=0  
                Lt_c=0
                et_c=0
                th_c=0   
            if pi_x == 0 and pi_y == 0:
                fcenter=False    
            #print(pi_y,pi_x)               
            if moffat:
                dx_m1,dy_m1,ds_m1,db_m1,psf1,Ft,FtF,Io_m,bn_m,Re_m,ns_m,At0,e0_m,th0_m=evaluate_2dPSF(map1,map1e,name=name+spt,model=False,ring=ring,trip=trip,sig=sig,fcenter=fcenter,beta=beta,ellip=ellip,re_int=re_int,singlepsf=singlepsf,moffat=moffat,mc=mc,ncpu=ncpu,db_m=bt,psf_coef=psf_coef,pi_x=pi_x,pi_y=pi_y,bs_c=bs_c,Re_c=Re_c,ns_c=ns_c,psft=psft,ro_i=ro_i,ds_i=ds_i,Lt_c=Lt_c,e_m=et_c,tht_m=th_c)#,ds_m=72.2)#comentar parametros db_m,ds_m 
            else:
                dx_m1,dy_m1,ds_m1,psf1,Ft,FtF=evaluate_2dPSF(map1,map1e,name=name+spt,model=False,sig=sig,mc=mc,ncpu=ncpu) 
            sky1=pixel_to_skycoord(dx_m1,dy_m1,wcs)
            val1=sky1.to_string('hmsdms')
            if verbose:
                if moffat:
                    if ring:
                        print("wave=",wave_1,'FLUX=',FtF,'FLUXN=',Ft,'RADEC=',val1,"x_1=",dx_m1,"y_1=",dy_m1,"sigma_1=",ds_m1*dpix,"r0_1=",db_m1,"psf_1=",psf1*dpix,"Io=",Io_m,"bn=",bn_m,"Re=",Re_m,"ns=",ns_m,"At0=",At0,"e=",e0_m,"th0=",th0_m)
                    elif trip:
                        print("wave=",wave_1,'FLUX=',FtF,'FLUXN=',Ft,'RADEC=',val1,"x_1=",dx_m1,"y_1=",dy_m1,"alpha_1=",ds_m1*dpix,"beta_1=",db_m1,"psf_1=",psf1*dpix,"Io=",Io_m,"bn=",bn_m,"Re=",Re_m,"ns=",ns_m,"At0=",At0,"e=",e0_m,"Lt=",th0_m)
                    elif singlepsf:
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
        et=0
        if sp > 0:
            map1=np.nanmean(cube[0:sp,:,:],axis=0)
            map1e=np.nanmean(cubeE[0:sp,:,:],axis=0)
            wave_1=np.nanmean(wave_f[0:sp])
            map1=cube[214,:,:]
            map1e=cubeE[214,:,:]
            wave_1=wave_f[214]
            #map1=cube[920,:,:]
            #map1e=cubeE[920,:,:]
            #wave_1=wave_f[920]
#            map1=cube[2147,:,:]#
#            map1e=cubeE[2147,:,:]#2372
#            wave_1=wave_f[2147]
        #    map1=cube[2447,:,:]#
        #    map1e=cubeE[2447,:,:]#2372
        #    wave_1=wave_f[2447]
#            map1=cube[2316,:,:]#
#            map1e=cubeE[2316,:,:]#2372
#            wave_1=wave_f[2316]   
            #map1=cube[2095,:,:]#       # Este
            #map1e=cubeE[2095,:,:]#2372 # Este
            #wave_1=wave_f[2095]        # Este
#            map1=cube[2053,:,:]#
#            map1e=cubeE[2053,:,:]#2372
#            wave_1=wave_f[2053]               
            map1=cube[2267,:,:]#       # Este
            map1e=cubeE[2267,:,:]#2372 # Este
            wave_1=wave_f[2267]        # Este
            map1=cube[2119,:,:]#       # Este
            map1e=cubeE[2119,:,:]#2372 # Este
            wave_1=wave_f[2119]        # Este      
#            map1=cube[2085,:,:]#       # Este
#            map1e=cubeE[2085,:,:]#2372 # Este
#            wave_1=wave_f[2085]        # Este                        
#            map1=cube[2479,:,:]#
#            map1e=cubeE[2479,:,:]#2372
#            wave_1=wave_f[2479]              
            #map1=np.nanmean(cube[2147:2157,:,:],axis=0)
            #map1e=np.nanmean(cubeE[2147:2157,:,:],axis=0)
            #wave_1=np.nanmean(wave_f[2147:2157])
            if ring:
                pi_x=np.nanmean(pia_x[0:sp])
                pi_y=np.nanmean(pia_y[0:sp])
                pi_x=pia_x[214]#
                pi_y=pia_y[214]
#                pi_x=pia_x[920]
#                pi_y=pia_y[920]
#                pi_x=pia_x[2147]
#                pi_y=pia_y[2147]
        #        pi_x=pia_x[2447]
        #        pi_y=pia_y[2447]     
#                pi_x=pia_x[2316]
#                pi_y=pia_y[2316]      
                #pi_x=pia_x[2095] # Este
                #pi_y=pia_y[2095] # Este     
#                pi_x=pia_x[2053]
#                pi_y=pia_y[2053]
                pi_x=pia_x[2267] #2123 Este
                pi_y=pia_y[2267] # Este
                pi_x=pia_x[2119] #2123 Este
                pi_y=pia_y[2119] # Este                
#                pi_x=pia_x[2479]
#                pi_y=pia_y[2479]                                     
            print(wave_1)
            if psf_t:
                if ring:
                    psf_coef=0
                    if ds_i == 0:
                        ds_i=0.5
                        ro_i=2.0
                else: 
                    if psft == False:#singlepsf and 
                        psf_coef=p_ds(wave_1)
                    else:   
                        psf_coef=pt(wave_1)/dpix
                    #print(psf_coef,psft)
                    ds_i=0
                    ro_i=0
            else:
                if ring:
                    print(name)
                    pt=tol.get_somoth_val(name,dir=dir_o,sigma=5,sp=10,val=7,out_p=False,deg=5,tp='_ring')
                    ds_i=pt(wave_1)
                    pt=tol.get_somoth_val(name,dir=dir_o,sigma=5,sp=10,val=8,out_p=False,deg=5,tp='_ring')
                    ro_i=pt(wave_1)  
                    psf_coef=0
                else:
                    if psft == False:#singlepsf and 
                        psf_coef=p_ds(wave_1)
                    else:
                        pt=tol.get_somoth_val(name,dir=dir_o,sigma=5,sp=10,val=9,out_p=True,deg=5)    
                        psf_coef=pt(wave_1)/dpix
                    ds_i=0
                    ro_i=0
        else:
            map1=np.nanmean(cube,axis=0)
            map1e=np.nanmean(cubeE,axis=0)
            ds_i=0
            ro_i=0
            psf_coef=0
            if ring:
                pi_x=np.nanmean(pia_x)
                pi_y=np.nanmean(pia_y)
        if str_p:
            if not ring:
                pi_x=p_px(wave_1)
                pi_y=p_py(wave_1)
                Lt_c = 0.0
                if trip:
                    Lt_c=p_ls(wave_1)
                    if Lt_c < 0.0:
                        Lt_c = 0.0
            if singlepsf: 
                bs_c=3.0 
                ns_c=0
                if beta:
                    bt=p_bts(wave_1)
                    Re_c=0.0
                else:
                    Re_c=100.
                if ellip:
                    et_c=p_eli(wave_1)
                    th_c=p_tht(wave_1)#147   
                    ellip=False
                    et=1
                else:
                    if et==1:
                        et_c=p_eli(wave_1)
                        th_c=p_tht(wave_1)#147 
                    else:
                        et_c=0
                        th_c=0   
            else:          
                bs_c=p_bs(wave_1)            
                #bs_c=p_bs(wave_1)
                Re_c=p_Re(wave_1)
                ns_c=p_ns(wave_1)
                if ellip:
                    et_c=p_eli(wave_1)
                    th_c=p_tht(wave_1)#147 
                    ellip=False
                    et=1
                else: 
                    if et==1:
                        et_c=p_eli(wave_1)
                        th_c=p_tht(wave_1)#147 
                    else:
                        et_c=0
                        th_c=0   
                #print(Re_c)
        else:
            if not ring:
                pi_x=0
                pi_y=0
            bs_c=0
            Re_c=0
            ns_c=0
            Lt_c=0 
            et_c=0
            th_c=0    
        if pi_x == 0 and pi_y == 0:
            fcenter=False                                          
        if moffat:
            dx_m1,dy_m1,ds_m1,db_m1,psf1,Ft,FtF,mod,Io_m,bn_m,Re_m,ns_m,At0,e0_m,th0_m=evaluate_2dPSF(map1,map1e,name=name+spt,sig=sig,plot_f=True,trip=trip,ring=ring,fcenter=fcenter,beta=beta,ellip=ellip,singlepsf=singlepsf,moffat=moffat,mc=mc,ncpu=ncpu,db_m=bt,psf_coef=psf_coef,pi_x=pi_x,pi_y=pi_y,bs_c=bs_c,Re_c=Re_c,ns_c=ns_c,psft=psft,ro_i=ro_i,ds_i=ds_i,Lt_c=Lt_c,e_m=et_c,tht_m=th_c)
        else:
            dx_m1,dy_m1,ds_m1,psf1,Ft,FtF,mod=evaluate_2dPSF(map1,map1e,name=name+spt,sig=sig,plot_f=True,mc=mc,ncpu=ncpu)
        sky1=pixel_to_skycoord(dx_m1,dy_m1,wcs)
        val1=sky1.to_string('hmsdms')
        if moffat:
            if ring:
                print('FLUX=',FtF,'FLUXN=',Ft,'RADEC=',val1,"x_1=",dx_m1,"y_1=",dy_m1,"sigma_1=",ds_m1*dpix,"r0_1=",db_m1,"psf_1=",psf1*dpix,"Io=",Io_m,"bn=",bn_m,"Re=",Re_m,"ns=",ns_m,"At0=",At0,"e=",e0_m,"th0=",th0_m)
            elif trip:
                print('FLUX=',FtF,'FLUXN=',Ft,'RADEC=',val1,"x_1=",dx_m1,"y_1=",dy_m1,"alpha_1=",ds_m1*dpix,"beta_1=",db_m1,"psf_1=",psf1*dpix,"Io=",Io_m,"bn=",bn_m,"Re=",Re_m,"ns=",ns_m,"At0=",At0,"e=",e0_m,"Lt=",th0_m)
            elif singlepsf:
                print('FLUX=',FtF,'FLUXN=',Ft,'RADEC=',val1,"x_1=",dx_m1,"y_1=",dy_m1,"alpha_1=",ds_m1*dpix,"beta_1=",db_m1,"psf_1=",psf1*dpix,"At0=",At0,"e=",e0_m,"th0=",th0_m)
            else:
                print('FLUX=',FtF,'FLUXN=',Ft,'RADEC=',val1,"x_1=",dx_m1,"y_1=",dy_m1,"alpha_1=",ds_m1*dpix,"beta_1=",db_m1,"psf_1=",psf1*dpix,"Io=",Io_m,"bn=",bn_m,"Re=",Re_m,"ns=",ns_m,"At0=",At0,"e=",e0_m,"th0=",th0_m)
        else:
            print('FLUX=',FtF,'FLUXN=',Ft,'RADEC=',val1,"x_1=",dx_m1,"y_1=",dy_m1,"sigma_1=",ds_m1*dpix,"psf_1=",psf1*dpix)

