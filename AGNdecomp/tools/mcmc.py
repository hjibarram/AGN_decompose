#!/usr/bin/env python
import glob, os,sys,timeit
import matplotlib
import numpy as np
import emcee
from astropy.convolution import convolve, convolve_fft, Gaussian2DKernel
import matplotlib.pyplot as plt
import AGNdecomp.tools.tools as tol
from AGNdecomp.tools.priors import lnprob_Dmoffat3,lnprob_Dmoffat2,lnprob_Dmoffat0,lnprob_Dmoffat
from AGNdecomp.tools.priors import lnprob_ring3,lnprob_ring2,lnprob_ring0,lnprob_ring
from AGNdecomp.tools.priors import lnprob_moffat3_s,lnprob_moffat2_s,lnprob_moffat0_s,lnprob_moffat_s
from AGNdecomp.tools.priors import lnprob_moffat3,lnprob_moffat2,lnprob_moffat0,lnprob_moffat
from AGNdecomp.tools.priors import lnprob_gaussian


def mcmc(p0,nwalkers,niter,ndim,lnprob,data,verbose=False,multi=True,tim=False,ncpu=10):
    if tim:
        import time
    if multi:
        #import os
        #os.environ["OMP_NUM_THREADS"] = "1"
        from multiprocessing import Pool
        from multiprocessing import cpu_count
        ncput=cpu_count()
        if ncpu > ncput:
            ncpu=ncput
        if ncpu == 0:
            ncpu=None
        with Pool(ncpu) as pool:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=data,pool=pool)
            if tim:
                start = time.time()
            if verbose:
                print("Running burn-in...")
            p0, _, _ = sampler.run_mcmc(p0, 1000)#1000
            sampler.reset()
            if verbose:
                print("Running production...")
            pos, prob, state = sampler.run_mcmc(p0, niter)
            if tim:
                end = time.time()
                multi_time = end - start
                print("Multiprocessing took {0:.1f} seconds".format(multi_time))
    else:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=data)
        if tim:
            start = time.time()
        if verbose:
            print("Running burn-in...")
        p0, _, _ = sampler.run_mcmc(p0, 1000)
        sampler.reset()
        if verbose:
            print("Running production...")
        pos, prob, state = sampler.run_mcmc(p0, niter)
        if tim:
            end = time.time()
            serial_time = end - start
            print("Serial took {0:.1f} seconds".format(serial_time))
    return sampler, pos, prob, state


def evaluate_2dPSF(pf_map,pf_mapE,name='test',Namevalues=[],Inpvalues=[],Infvalues=[],Supvalues=[],model=True,autocent=True,fcenter=False,sig=2,plot_f=False,beta=True,re_int=False,ellip=False,singlepsf=False,trip=False,ring=False,moffat=False,mc=False,ncpu=10,psft=False,valsT={},ds_i=0,db_m=0,ro_i=0,psf_coef=0,e_m=0.0,tht_m=0.0,pi_x=0,pi_y=0,bs_c=0,Re_c=0,ns_c=0,Lt_c=4.4,dyo=0,dxo=0):
    if len(valsT):
        if 'ds_i' in valsT:
            ds_i=valsT['ds_i']
        if 'db_m' in valsT:
            db_m=valsT['db_m']
        if 'ro_i' in valsT:
            ro_i=valsT['ro_i']
        if 'psf_coef' in valsT:
            psf_coef=valsT['psf_coef']
        if 'e_m' in valsT:
            e_m=valsT['e_m']
        if 'tht_m' in valsT:
            tht_m=valsT['tht_m']
        if 'pi_x' in valsT:
            pi_x=valsT['pi_x']
        if 'pi_y' in valsT:
            pi_y=valsT['pi_y']
        if 'bs_c' in valsT:
            bs_c=valsT['bs_c']
        if 'Re_c' in valsT:
            Re_c=valsT['Re_c']
        if 'ns_c' in valsT:
            ns_c=valsT['ns_c']
        if 'Lt_c' in valsT:
            Lt_c=valsT['Lt_c']
        if 'dxo' in valsT:
            dxo=valsT['dxo']
        if 'dyo' in valsT:
            dyo=valsT['dyo']    
    nx,ny=pf_map.shape
    if trip:
        ring=False
    if db_m > 0 and psf_coef > 0:
        if moffat:
            ds_m=psf_coef/(2.0*np.sqrt(2.0**(1./db_m)-1))
        else:
            ds_m=2.5/(2.0*np.sqrt(2.0*np.log(2.0)))
    else:
        db_m=569.29
        ds_m=72.2
    if psft == False:#singlepsf and 
        ds_m=psf_coef   
    if ring:
        if ds_i > 0:
            ds_m = ds_i
        else:
            ds_m = 1.0
        if ro_i > 0:
            ro_m = ro_i
        else:
            ro_m = 2.0
    if autocent:
        if sig == 0:
            pf_map_c=pf_map
        else:
            try:
                PSF=Gaussian2DKernel(stddev=sig)
            except:
                PSF=Gaussian2DKernel(x_stddev=sig,y_stddev=sig)
            pf_map_c=convolve(pf_map, PSF)
        min_in=np.unravel_index(np.nanargmax(pf_map_c), (nx,ny))
    else:
        min_in=[dxo,dyo]
    At=np.nanmax(pf_map)
    x_t=np.arange(ny)-min_in[1]
    y_t=np.arange(nx)-min_in[0]
    x_t=np.array([x_t]*nx)
    y_t=np.array([y_t]*ny).T
    if not mc:
        n_ds=100
        n_be=15
        n_dx=50
        n_dy=50
        ds_t=np.arange(n_ds)/np.float(n_ds)*(4.0-1.0)+1.0
        be_t=10**(np.arange(n_be)/float(n_be)*(np.log10(10.0)-np.log10(0.5))+np.log10(0.5))
        dx=np.arange(n_dx)/np.float(n_dx)*10-5.0
        dy=np.arange(n_dy)/np.float(n_dy)*10-5.0
        if moffat:
            chi2_m=np.zeros([n_ds,n_be,n_dx,n_dy])
        else:
            chi2_m=np.zeros([n_ds,n_dx,n_dy])
        if moffat:
            for t in range(0, n_be):
                for j in range(0, n_ds):
                    for k in range(0, n_dx):
                        for w in range(0, n_dy):
                            spec_t=At*(1.0+(((x_t-dx[k])/ds_t[j])**2.0+((y_t-dy[w])/ds_t[j])**2.0))**(-be_t[t])
                            chi2_m[j,t,k,w]=np.nansum((pf_map-spec_t)**2.0)
        else:
            for j in range(0, n_ds):
                for k in range(0, n_dx):
                    for w in range(0, n_dy):
                        spec_t=np.exp(-0.5*((((x_t-dx[k])/ds_t[j])**2.0)+((y_t-dy[w])/ds_t[j])**2.0))*At
                        chi2_m[j,k,w]=np.nansum((pf_map-spec_t)**2.0)
        if moffat:
            min_int=np.unravel_index(np.nanargmin(chi2_m), (n_ds,n_be,n_dx,n_dy))
            ds_m=ds_t[min_int[0]]
            db_m=be_t[min_int[1]]
            dx_m=dx[min_int[2]]
            dy_m=dy[min_int[3]]  
        else:              
            min_int=np.unravel_index(np.nanargmin(chi2_m), (n_ds,n_dx,n_dy))
            ds_m=ds_t[min_int[0]]
            dx_m=dx[min_int[1]]
            dy_m=dy[min_int[2]]
        At0=At
    else:
        valsI={}
        keysI={}
        keysI['ellip']=ellip
        keysI['alpha']=True#alpha
        keysI['beta']=beta
        keysI['fcenter']=fcenter
        keysI['re_int']=re_int
        valsI['db_m']=db_m
        valsI['e_m']=e_m
        valsI['tht_m']=tht_m
        valsI['dx']=pi_x-min_in[1]
        valsI['dy']=pi_y-min_in[0]
        valsI['At1']=At
        valsI['al_m']=5.0
        valsI['Re_c']=Re_c
        valsI['bn']=bs_c
        valsI['ns']=ns_c 
        if Re_c > 0:
            if psft:
                if ring:
                    data = (pf_map, pf_mapE, x_t, y_t, At, bs_c, ns_c, pi_x-min_in[1], pi_y-min_in[0])
                if trip:
                    data = (pf_map, pf_mapE, x_t, y_t, db_m, At, bs_c, ns_c, Lt_c)
                elif singlepsf:
                    data=(pf_map, pf_mapE, x_t, y_t, valsI, keysI, Infvalues, Supvalues, Inpvalues)
                    #data = (pf_map, pf_mapE, x_t, y_t, db_m, At, pi_x-min_in[1], pi_y-min_in[0], e_m, tht_m, beta, ellip)    
                else: 
                    data=(pf_map, pf_mapE, x_t, y_t, valsI, keysI, Infvalues, Supvalues, Inpvalues)
                    #data = (pf_map, pf_mapE, x_t, y_t, db_m, At, bs_c, ns_c, e_m, tht_m, ellip, Re_c, re_int, pi_x-min_in[1], pi_y-min_in[0], fcenter)
            else:
                if ring:
                    data = (pf_map, pf_mapE, x_t, y_t, ds_m, ro_m, At, bs_c, ns_c, pi_x-min_in[1], pi_y-min_in[0])
                if trip:
                    data = (pf_map, pf_mapE, x_t, y_t, ds_m, db_m, At, bs_c, ns_c, Lt_c)
                elif singlepsf:
                    data = (pf_map, pf_mapE, x_t, y_t, ds_m, db_m, At, pi_x-min_in[1], pi_y-min_in[0], e_m, tht_m, ellip)
                else:
                    data = (pf_map, pf_mapE, x_t, y_t, ds_m, db_m, At, bs_c, ns_c, e_m, tht_m, ellip, Re_c, re_int, pi_x-min_in[1], pi_y-min_in[0], fcenter)
        else:
            if psft:
                if ring:
                    data = (pf_map, pf_mapE, x_t, y_t, At, pi_x-min_in[1], pi_y-min_in[0])
                elif singlepsf:
                    data = (pf_map, pf_mapE, x_t, y_t, db_m, At, e_m, tht_m, beta, ellip)
                else:
                    data = (pf_map, pf_mapE, x_t, y_t, db_m, At, e_m, tht_m, ellip)
            else:
                if ring:
                    data = (pf_map, pf_mapE, x_t, y_t, At, ds_m, ro_m, pi_x-min_in[1], pi_y-min_in[0])
                elif singlepsf:
                    data = (pf_map, pf_mapE, x_t, y_t, ds_m, db_m, At, e_m, tht_m, ellip) 
                else:
                    data = (pf_map, pf_mapE, x_t, y_t, ds_m, db_m, At)
        #theta, spec, specE, x_t, y_t, 
        #db_m, dx, dy, e_m, tht_m, al_m, bn, ns, Lt_m, ds_m, ro_m, Re_c
        #ring, beta, ellip, aplha, fcenter, re_int                       
        nwalkers=240
        niter=1024
        if moffat:
            if Re_c > 0:
                if psft:
                    if ring:
                        initial = np.array([At*0.9, At*0.1, 3, 0.5, 2.0])
                    elif singlepsf:
                        if beta:
                            if ellip:    
                                initial = np.array([At*0.9, 14.8, 36.8, 0.0, 0])
                            else:
                                initial = np.array([At*0.9, 14.8, 36.8])
                        else:
                            if ellip:
                                initial = np.array([*Inpvalues])#At*0.9, 14.8, 0.0, 0])
                            else:
                                initial = np.array([*Inpvalues])#At*0.9, 14.8]) 
                    else:
                        if ellip:
                            if fcenter:
                                if re_int:
                                    initial = np.array([At*0.9, At*0.1, 14.8, 0.0, 0.0])
                                else:
                                    initial = np.array([At*0.9, At*0.1, 3, 14.8, 0.0, 0.0])
                            else:
                                if re_int:
                                    initial = np.array([At*0.9, 0.2, 0.0, At*0.1, 14.8, 0.0, 0.0])
                                else:
                                    initial = np.array([At*0.9, 0.2, 0.0, At*0.1, 3, 14.8, 0.0, 0.0])
                        else:
                            if fcenter:
                                if re_int:
                                    initial = np.array([At*0.9, At*0.1, 14.8])
                                else:
                                    initial = np.array([At*0.9, At*0.1, 3, 14.8])
                            else:
                                if re_int:
                                    initial = np.array([At*0.9, 0.2, 0.0, At*0.1, 14.8])
                                else:
                                    initial = np.array([At*0.9, 0.2, 0.0, At*0.1, 3, 14.8])
                else:
                    if ring:
                        initial = np.array([At*0.9, At*0.1, 3])  
                    elif singlepsf:   
                        initial = np.array([At*0.9])      
                    else:
                        if ellip:
                            if fcenter:
                                if re_int:
                                    initial = np.array([At*0.9, At*0.1, 0.0, 0.0])
                                else:
                                    initial = np.array([At*0.9, At*0.1, 3, 0.0, 0.0])
                            else:
                                if re_int:
                                    initial = np.array([At*0.9, 0.2, 0.0, At*0.1, 0.0, 0.0])
                                else:
                                    initial = np.array([At*0.9, 0.2, 0.0, At*0.1, 3, 0.0, 0.0])
                        else:
                            if fcenter:
                                if re_int:
                                    initial = np.array([At*0.9, At*0.1])
                                else:
                                    initial = np.array([At*0.9, At*0.1, 3])
                            else:
                                if re_int:
                                    initial = np.array([At*0.9, 0.2, 0.0, At*0.1])
                                else:
                                    initial = np.array([At*0.9, 0.2, 0.0, At*0.1, 3])
            else:
                if psft:
                    if ring:
                        initial = np.array([At*0.9, At*0.1, 0.5, 3, 1.0, 0.5, 2.0])#, 0.0, 0.0])
                    elif trip:
                        initial = np.array([At*0.9, 0.2, 0.0, At*0.1, 0.5, 3, 1.0, 14.8, 4.4])
                    elif singlepsf:
                        if beta:    
                            if ellip:
                                initial = np.array([At*0.9, 0.2, 0.0, 14.8, 36.8, 0.0, 0])
                            else:
                                initial = np.array([At*0.9, 0.2, 0.0, 14.8, 36.8])
                        else:
                            if ellip:
                                initial = np.array([At*0.9, 0.2, 0.0, 14.8, 0.0, 0])
                            else:
                                initial = np.array([At*0.9, 0.2, 0.0, 14.8])        
                    else:
                        if ellip:
                            initial = np.array([At*0.9, 0.2, 0.0, At*0.1, 0.5, 3, 1.0, 14.8, 0.0, 0.0])
                        else:
                            initial = np.array([At*0.9, 0.2, 0.0, At*0.1, 0.5, 3, 1.0, 7.8])
                else:
                    if ring:
                        initial = np.array([At*0.9, At*0.1, 0.5, 3, 1.0])#, 0.0, 0.0])
                    elif trip:
                        initial = np.array([At*0.9, 0.2, 0.0, At*0.1, 0.5, 3, 1.0, 4.4])
                    elif singlepsf:    
                        initial = np.array([At*0.9, 0.2, 0.0])    
                    else:
                        initial = np.array([At*0.9, 0.2, 0.0, At*0.1, 0.5, 3, 1.0])
        else:
            initial = np.array([At, 0.2, 0.0, 1.75])
        
        ndim = len(initial)
        p0 = [np.array(initial) + 1e-5 * np.random.randn(ndim) for i in range(nwalkers)]
        if moffat:
            if plot_f:
                tim=True
            else:
                tim=False
            tim=True    
            if Re_c > 0:
                if psft:
                    if ring:
                        sampler, pos, prob, state = mcmc(p0,nwalkers,niter,ndim,lnprob_ring3,data,tim=tim,ncpu=ncpu)
                    elif trip:
                        sampler, pos, prob, state = mcmc(p0,nwalkers,niter,ndim,lnprob_Dmoffat3,data,tim=tim,ncpu=ncpu)
                    elif singlepsf:
                        sampler, pos, prob, state = mcmc(p0,nwalkers,niter,ndim,lnprob_moffat3_s,data,tim=tim,ncpu=ncpu)        
                    else:
                        sampler, pos, prob, state = mcmc(p0,nwalkers,niter,ndim,lnprob_moffat3,data,tim=tim,ncpu=ncpu)
                else:
                    if ring:
                        sampler, pos, prob, state = mcmc(p0,nwalkers,niter,ndim,lnprob_ring2,data,tim=tim,ncpu=ncpu)
                    elif trip:
                        sampler, pos, prob, state = mcmc(p0,nwalkers,niter,ndim,lnprob_Dmoffat2,data,tim=tim,ncpu=ncpu)
                    elif singlepsf:
                        sampler, pos, prob, state = mcmc(p0,nwalkers,niter,ndim,lnprob_moffat2_s,data,tim=tim,ncpu=ncpu)     
                    else:
                        sampler, pos, prob, state = mcmc(p0,nwalkers,niter,ndim,lnprob_moffat2,data,tim=tim,ncpu=ncpu)
            else:
                if psft:
                    if ring:
                        sampler, pos, prob, state = mcmc(p0,nwalkers,niter,ndim,lnprob_ring0,data,tim=tim,ncpu=ncpu)
                    elif trip:
                        sampler, pos, prob, state = mcmc(p0,nwalkers,niter,ndim,lnprob_Dmoffat0,data,tim=tim,ncpu=ncpu)
                    elif singlepsf:    
                        sampler, pos, prob, state = mcmc(p0,nwalkers,niter,ndim,lnprob_moffat0_s,data,tim=tim,ncpu=ncpu)    
                    else:
                        sampler, pos, prob, state = mcmc(p0,nwalkers,niter,ndim,lnprob_moffat0,data,tim=tim,ncpu=ncpu)
                else:
                    if ring:
                        sampler, pos, prob, state = mcmc(p0,nwalkers,niter,ndim,lnprob_ring,data,tim=tim,ncpu=ncpu)
                    elif trip:
                        sampler, pos, prob, state = mcmc(p0,nwalkers,niter,ndim,lnprob_Dmoffat,data,tim=tim,ncpu=ncpu)
                    elif singlepsf:
                        sampler, pos, prob, state = mcmc(p0,nwalkers,niter,ndim,lnprob_moffat_s,data,tim=tim,ncpu=ncpu)    
                    else:    
                        sampler, pos, prob, state = mcmc(p0,nwalkers,niter,ndim,lnprob_moffat,data,tim=tim,ncpu=ncpu)
        else:
            sampler, pos, prob, state = mcmc(p0,nwalkers,niter,ndim,lnprob_gaussian,data,ncpu=ncpu)
            
        samples = sampler.flatchain
        theta_max  = samples[np.argmax(sampler.flatlnprobability)]
        if moffat:
            if Re_c > 0:
                #At0,Io_m=theta_max
                if psft:
                    if ring:
                        At0,Io_m,Re_m,ds_m,r0_m=theta_max
                        dx_m=pi_x-min_in[1]
                        dy_m=pi_y-min_in[0]
                    elif trip:
                        At0,dx_m,dy_m,Io_m,Re_m,ds_m=theta_max
                        Lt_m=Lt_c
                    elif singlepsf:
                        if beta:
                            if ellip:
                                At0,ds_m,db_m,e_m,tht_m=theta_max
                            else:    
                                At0,ds_m,db_m=theta_max
                        else:
                            if ellip:
                                At0,ds_m,e_m,tht_m=theta_max
                            else:
                                At0,ds_m=theta_max
                        dx_m=pi_x-min_in[1]
                        dy_m=pi_y-min_in[0]    
                    else:
                        if ellip:
                            if fcenter:
                                if re_int:
                                    At0,Io_m,ds_m,e_m,tht_m=theta_max
                                    Re_m=Re_c
                                else:    
                                    At0,Io_m,Re_m,ds_m,e_m,tht_m=theta_max
                                dx_m=pi_x-min_in[1]
                                dy_m=pi_y-min_in[0]  
                            else:
                                if re_int:
                                    At0,dx_m,dy_m,Io_m,ds_m,e_m,tht_m=theta_max
                                    Re_m=Re_c
                                else:
                                    At0,dx_m,dy_m,Io_m,Re_m,ds_m,e_m,tht_m=theta_max
                        else:
                            if fcenter:
                                if re_int:
                                    At0,Io_m,ds_m=theta_max
                                    Re_m=Re_c
                                else:    
                                    At0,Io_m,Re_m,ds_m=theta_max
                                dx_m=pi_x-min_in[1]
                                dy_m=pi_y-min_in[0]  
                            else:
                                if re_int:
                                    At0,dx_m,dy_m,Io_m,ds_m=theta_max
                                    Re_m=Re_c
                                else:
                                    At0,dx_m,dy_m,Io_m,Re_m,ds_m=theta_max
                else:
                    if ring:
                        At0,Io_m,Re_m=theta_max
                        r0_m=ro_m
                        dx_m=pi_x-min_in[1]
                        dy_m=pi_y-min_in[0]
                    elif trip:
                        At0,dx_m,dy_m,Io_m,Re_m=theta_max
                        Lt_m=Lt_c
                    elif singlepsf:
                        At0=theta_max    
                        At0=At0[0]
                        dx_m=pi_x-min_in[1]
                        dy_m=pi_y-min_in[0]    
                    else:
                        if ellip:
                            if fcenter:
                                if re_int:
                                    At0,Io_m,e_m,tht_m=theta_max
                                    Re_m=Re_c
                                else:
                                    At0,Io_m,Re_m,e_m,tht_m=theta_max
                                dx_m=pi_x-min_in[1]
                                dy_m=pi_y-min_in[0] 
                            else:
                                if re_int:
                                    At0,dx_m,dy_m,Io_m,e_m,tht_m=theta_max
                                    Re_m=Re_c
                                else:
                                    At0,dx_m,dy_m,Io_m,Re_m,e_m,tht_m=theta_max
                        else:
                            if fcenter:
                                if re_int:
                                    At0,Io_m=theta_max
                                    Re_m=Re_c
                                else:
                                    At0,Io_m,Re_m=theta_max
                                dx_m=pi_x-min_in[1]
                                dy_m=pi_y-min_in[0] 
                            else:
                                if re_int:
                                    At0,dx_m,dy_m,Io_m=theta_max
                                    Re_m=Re_c
                                else:
                                    At0,dx_m,dy_m,Io_m,Re_m=theta_max
                #dx_m=pi_x-min_in[1]
                #dy_m=pi_y-min_in[0]
                bn_m=bs_c
                #Re_m=Re_c
                ns_m=ns_c
            else:
                if psft:
                    if ring:
                        At0,Io_m,bn_m,Re_m,ns_m,ds_m,r0_m=theta_max
                        dx_m=pi_x-min_in[1]
                        dy_m=pi_y-min_in[0]
                    elif trip:
                        At0,dx_m,dy_m,Io_m,bn_m,Re_m,ns_m,ds_m,Lt_m=theta_max
                    elif singlepsf:
                        if beta:
                            if ellip:
                                At0,dx_m,dy_m,ds_m,db_m,e_m,tht_m=theta_max
                            else:
                                At0,dx_m,dy_m,ds_m,db_m=theta_max    
                        else:
                            if ellip:
                                At0,dx_m,dy_m,ds_m,e_m,tht_m=theta_max
                            else:
                                At0,dx_m,dy_m,ds_m=theta_max     
                    else:
                        if ellip:
                            At0,dx_m,dy_m,Io_m,bn_m,Re_m,ns_m,ds_m,e_m,tht_m=theta_max
                        else:
                            At0,dx_m,dy_m,Io_m,bn_m,Re_m,ns_m,ds_m=theta_max
                else:
                    if ring:
                        At0,Io_m,bn_m,Re_m,ns_m=theta_max
                        r0_m=ro_m
                        dx_m=pi_x-min_in[1]
                        dy_m=pi_y-min_in[0]
                    elif trip:
                        At0,dx_m,dy_m,Io_m,bn_m,Re_m,ns_m,Lt_m=theta_max
                    elif singlepsf:   
                        At0,dx_m,dy_m=theta_max     
                    else:
                        At0,dx_m,dy_m,Io_m,bn_m,Re_m,ns_m=theta_max#,e_m,th0_m=theta_max
            #tp=np.percentile(samples[:, 0], [16, 50, 84])
            #At0=tp[1]
            #tp=np.percentile(samples[:, 1], [16, 50, 84])
            #dx_m=tp[1]
            #tp=np.percentile(samples[:, 2], [16, 50, 84])
            #dy_m=tp[1]
            #tp=np.percentile(samples[:, 3], [16, 50, 84])
            #Io_m=tp[1]
            #tp=np.percentile(samples[:, 3], [16, 50, 84])
            #bn_m=tp[1]
            #tp=np.percentile(samples[:, 3], [16, 50, 84])
            #Re_m=tp[1]
            #tp=np.percentile(samples[:, 3], [16, 50, 84])
            #ns_m=tp[1]
        else:
            At0,dx_m,dy_m,ds_m=theta_max
        #print(At0)    
               
        
    
    if plot_f:
        import matplotlib.pyplot as plt
        cm=plt.cm.get_cmap('jet')
        lev=np.sqrt(np.arange(0.0,10.0,1.5)+0.008)/np.sqrt(10.008)*np.amax(pf_map)
        fig, ax = plt.subplots(figsize=(6.8*1.1,5.5*1.2))
        ict=plt.imshow(np.log10(pf_map),cmap=cm) 
        cbar=plt.colorbar(ict)
        ics=plt.contour(pf_map,lev,colors='k',linewidths=1)            
        cbar.set_label(r"Relative Density")
        fig.tight_layout()
        plt.show()
    
    ft_num=np.nansum(pf_map)
    dt=1.0#.7/.5
    ds_m=ds_m*dt
    if moffat:
        if ring:
            ft_fit=At0*2*np.pi*ds_m*(ds_m*np.exp(-0.5*(r0_m/ds_m)**2.0)+r0_m*np.sqrt(2.*np.pi)/2.0*(1+erf(r0_m/ds_m/np.sqrt(2.0))))
        elif trip:
            ft_fit=3*np.pi*ds_m**2.0*At0/(db_m-1.0)
        else:
            ft_fit=np.pi*ds_m**2.0*At0/(db_m-1.0)
    else:
        ft_fit=2*np.pi*ds_m**2.0*At0
    if model:
        if moffat:
            if ring:
                theta_t=At0,dx_m,dy_m,ds_m,r0_m
                spec_t=mod.ring_model_s(theta_t, x_t, y_t)
            elif trip:
                theta_t=At0,dx_m,dy_m,ds_m,db_m,Lt_m
                spec_t=mod.Dmoffat_model_s(theta_t, x_t, y_t)
            else:
                if singlepsf:
                    #print(ds_m,e_m,tht_m)
                    if ellip:
                        r1=np.sqrt((x_t-dx_m)**2.0+(y_t-dy_m)**2.0)
                        rt=tol.radi_ellip(x_t-dx_m,y_t-dy_m,e_m,tht_m)
                        #print(np.nanmean(rt/r1))
                        spec_t=At0*(1.0+((rt/ds_m)**2.0))**(-db_m)
                    else:
                        spec_t=At0*(1.0+(((x_t-dx_m)/ds_m)**2.0+((y_t-dy_m)/ds_m)**2.0))**(-db_m)
                else:
                    if ellip:
                        rt=tol.radi_ellip(x_t-dx_m,y_t-dy_m,e_m,tht_m)
                        spec_t=At0*(1.0+((rt/ds_m)**2.0))**(-db_m)
                    else:
                        spec_t=At0*(1.0+(((x_t-dx_m)/ds_m)**2.0+((y_t-dy_m)/ds_m)**2.0))**(-db_m)
            r1=np.sqrt((x_t-dx_m)**2.0+(y_t-dy_m)**2.0)
            #tht=np.arctan2(y_t-dy_m,x_t-dx_m)
            bt=r1#ellipse2(tht,r1,e_m,th0_m)
            if singlepsf:
                spec_hst=0.0
            else:
                spec_hst=Io_m*np.exp(-bn_m*((bt/Re_m)**(1./ns_m)-1))
            #spec_hst=Io_m*np.exp(-bn_m*((bt/Re_m)**(1./ns_m)-1))
        else:
            spec_t=np.exp(-0.5*((((x_t-dx_m)/ds_m)**2.0)+((y_t-dy_m)/ds_m)**2.0))*At0
        if plot_f:    
            fig, ax = plt.subplots(figsize=(6.8*1.1,5.5*1.2))
            ict=plt.imshow(np.log10(spec_t),cmap=cm) 
            cbar=plt.colorbar(ict)
            ics=plt.contour(spec_t,lev,colors='k',linewidths=1)
            ics=plt.contour(pf_map,lev,colors='red',linewidths=1)            
            cbar.set_label(r"Relative Density")
            fig.tight_layout()
            plt.show()
            
            fig, ax = plt.subplots(figsize=(6.8*1.1,5.5*1.2))
            ict=plt.imshow(np.log10(pf_map-spec_t),cmap=cm) #np.log10
            cbar=plt.colorbar(ict)
            ics=plt.contour((pf_map-spec_t),lev,colors='k',linewidths=1)
            cbar.set_label(r"Relative Density")
            fig.tight_layout()
            plt.show()
            
            fig, ax = plt.subplots(figsize=(6.8*1.1,5.5*1.2))
            ict=plt.imshow(np.log10(pf_map-spec_t-spec_hst),cmap=cm) 
            cbar=plt.colorbar(ict)
            ics=plt.contour((pf_map-spec_t-spec_hst),lev,colors='k',linewidths=1)
            cbar.set_label(r"Relative Density")
            fig.tight_layout()
            plt.show()
            
            if mc:
                if moffat:
                    if Re_c > 0:
                        if psft:
                            if ring:
                                labels = ['At','Io','Re','sigma','r0']
                            elif singlepsf:
                                if beta:
                                    if ellip:
                                        labels = [r'$A_t$',r'$\alpha$',r'$\beta$',r'$e$',r'$\theta$']
                                    else:
                                        labels = [r'$A_t$',r'$\alpha$',r'$\beta$']
                                else:
                                    if ellip:
                                        labels = [r'$A_t$',r'$\alpha$',r'$e$',r'$\theta$']
                                    else:
                                        labels = [r'$A_t$',r'$\alpha$']    
                            else:
                                if ellip:
                                    if fcenter:
                                        if re_int:
                                            labels = [r'$A_t$',r'$I_o$',r'$\alpha$',r'$e$',r'$\theta$']
                                        else:
                                            labels = [r'$A_t$',r'$I_o$',r'$R_e$',r'$\alpha$',r'$e$',r'$\theta$']
                                    else:
                                        if re_int:
                                            labels = [r'$A_t$',r'$x_0$',r'$y_0$',r'$I_o$',r'$\alpha$',r'$e$',r'$\theta$']
                                        else:
                                            labels = [r'$A_t$',r'$x_0$',r'$y_0$',r'$I_o$',r'$R_e$',r'$\alpha$',r'$e$',r'$\theta$']
                                else:
                                    if fcenter:
                                        if re_int:
                                            labels = [r'$A_t$',r'$I_o$',r'$\alpha$']
                                        else:
                                            labels = [r'$A_t$',r'$I_o$',r'$R_e$',r'$\alpha$']
                                    else:
                                        if re_int:
                                            labels = [r'$A_t$',r'$x_0$',r'$y_0$',r'$I_o$',r'$\alpha$']
                                        else:
                                            labels = [r'$A_t$',r'$x_0$',r'$y_0$',r'$I_o$',r'$R_e$',r'$\alpha$']
                        else:
                            if ring:
                                labels = ['At','Io','Re']
                            if singlepsf:
                                labels = [r'$A_t$']    
                            else:
                                if ellip:
                                    if fcenter:
                                        if re_int:
                                            labels = [r'$A_t$',r'$I_o$',r'$e$',r'$\theta$']
                                        else:
                                            labels = [r'$A_t$',r'$I_o$',r'$R_e$',r'$e$',r'$\theta$']
                                    else:
                                        if re_int:
                                            labels = [r'$A_t$',r'$x_0$',r'$y_0$',r'$I_o$',r'$e$',r'$\theta$']
                                        else:
                                            labels = [r'$A_t$',r'$x_0$',r'$y_0$',r'$I_o$',r'$R_e$',r'$e$',r'$\theta$']
                                else:
                                    if fcenter:
                                        if re_int:
                                            labels = [r'$A_t$',r'$I_o$']
                                        else:
                                            labels = [r'$A_t$',r'$I_o$',r'$R_e$']
                                    else:
                                        if re_int:
                                            labels = [r'$A_t$',r'$x_0$',r'$y_0$',r'$I_o$']
                                        else:
                                            labels = [r'$A_t$',r'$x_0$',r'$y_0$',r'$I_o$',r'$R_e$']
                    else:
                        if psft:
                            if ring:
                                labels = ['At','Io','bn','Re','ns','sigma','r0']
                            elif trip:
                                labels = ['At','x1','x2','Io','bn','Re','ns','alpha','Lt']
                            elif singlepsf:
                                if beta:
                                    if ellip:
                                        labels = [r'$A_t$','r$x_0$',r'$y_0$',r'$\alpha$',r'$\beta$',r'$e$',r'$\theta$']
                                    else:
                                        labels = [r'$A_t$','r$x_0$',r'$y_0$',r'$\alpha$',r'$\beta$']
                                else:
                                    if ellip:
                                        labels = [r'$A_t$',r'$x_0$',r'$y_0$',r'$\alpha$',r'$e$',r'$\theta$']
                                    else:
                                        labels = [r'$A_t$',r'$x_0$',r'$y_0$',r'$\alpha$']    
                            else:
                                if ellip:
                                    labels = [r'$A_t$',r'$x_0$',r'$y_0$',r'$I_o$',r'$b_n$',r'$R_e$',r'$n_s$',r'$\alpha$',r'$e$',r'$\theta$']
                                else:
                                    labels = [r'$A_t$',r'$x_0$',r'$y_0$',r'$I_o$',r'$b_n$',r'$R_e$',r'$n_s$',r'$\alpha$']
                        else:
                            if ring:
                                labels = ['At','Io','bn','Re','ns']
                            elif trip:
                                labels = ['At','x1','x2','Io','bn','Re','ns','Lt']
                            elif singlepsf:
                                labels = [r'$A_t$',r'$x_0$',r'$y_0$']    
                            else:
                                labels = [r'$A_t$',r'$x_0$',r'$y_0$',r'$I_o$',r'$b_n$',r'$R_e$',r'$n_s$']
                    #At0,dx_m,dy_m,Io_m,bn_m,Re_m,ns_m
                else:
                    labels = ['At','x1','x2','sigma']
                import corner  
                #fig = figsize=(6.8*1.1,6.8*1.1) 
                fig = corner.corner(samples,show_titles=True,labels=labels,plot_datapoints=True,quantiles=[0.16, 0.5, 0.84],title_kwargs={"fontsize": 16},label_kwargs={"fontsize": 16})
                #fig.set_size_inches(6.8, 6.8)
                fig.set_size_inches(15.8*len(labels)/8.0, 15.8*len(labels)/8.0)
                
                #fig.tight_layout()
                fig.savefig('corners_NAME.pdf'.replace('NAME',name))
                #plt.show()
            
        dx_m=dx_m+min_in[1]#0
        dy_m=dy_m+min_in[0]#1
        if moffat:
            if ring:
                psf=np.nan
            else:
                psf=ds_m*2.0*np.sqrt(2.0**(1./db_m)-1)
        else:
            psf=ds_m*2.0*np.sqrt(2.0*np.log10(2.0))
        #print(ft_num/ft_fit)
        if moffat:
            if ring:
                return dx_m,dy_m,ds_m,r0_m,psf,ft_num,ft_fit,spec_t,Io_m,bn_m,Re_m,ns_m,At0,e_m,tht_m
            elif trip:
                return dx_m,dy_m,ds_m,db_m,psf,ft_num,ft_fit,spec_t,Io_m,bn_m,Re_m,ns_m,At0,e_m,Lt_m
            elif singlepsf:
                return dx_m,dy_m,ds_m,db_m,psf,ft_num,ft_fit,spec_t,0,0,0,0,At0,e_m,tht_m
            else:
                return dx_m,dy_m,ds_m,db_m,psf,ft_num,ft_fit,spec_t,Io_m,bn_m,Re_m,ns_m,At0,e_m,tht_m
        else:
            return dx_m,dy_m,ds_m,psf,ft_num,ft_fit,spec_t
    else:
            
        dx_m=dx_m+min_in[1]
        dy_m=dy_m+min_in[0]
        if moffat:
            if ring:
                psf=np.nan
            else:
                psf=ds_m*2.0*np.sqrt(2.0**(1./db_m)-1)
        else:
            psf=ds_m*2.0*np.sqrt(2.0*np.log10(2.0))
        if moffat:
            if ring:
                return dx_m,dy_m,ds_m,r0_m,psf,ft_num,ft_fit,Io_m,bn_m,Re_m,ns_m,At0,e_m,tht_m
            elif trip:
                return dx_m,dy_m,ds_m,db_m,psf,ft_num,ft_fit,Io_m,bn_m,Re_m,ns_m,At0,e_m,Lt_m
            elif singlepsf:
                return dx_m,dy_m,ds_m,db_m,psf,ft_num,ft_fit,0,0,0,0,At0,e_m,tht_m
            else:
                return dx_m,dy_m,ds_m,db_m,psf,ft_num,ft_fit,Io_m,bn_m,Re_m,ns_m,At0,e_m,tht_m
        else:    
            return dx_m,dy_m,ds_m,psf,ft_num,ft_fit