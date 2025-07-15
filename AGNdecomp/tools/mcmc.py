#!/usr/bin/env python
import glob, os,sys,timeit
import matplotlib
import numpy as np
import emcee
from astropy.convolution import convolve, convolve_fft, Gaussian2DKernel
import AGNdecomp.tools.tools as tol
import AGNdecomp.tools.models as mod
from AGNdecomp.tools.priors import lnprob_moffat0
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


def evaluate_2dPSF(pf_map,pf_mapE,name='test',Labelvalues=[],Namevalues=[],Inpvalues=[],Infvalues=[],Supvalues=[],path_out='',savefig=True,autocent=True,sig=2,plot_f=False,singlepsf=False,moffat=False,ncpu=10,valsI={}):
    dxo=valsI['dxo']
    dyo=valsI['dyo']    
    nx,ny=pf_map.shape
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
    valsI['At']=At
    valsI['xo']=valsI['xo']-min_in[1]
    valsI['yo']=valsI['yo']-min_in[0]
    print("Input values: ",valsI)
    if singlepsf:
        host=False  
    else: 
        host=True
    data = (pf_map, pf_mapE, x_t, y_t, valsI, Infvalues, Supvalues, Namevalues, host)
    nwalkers=240
    niter=1024
    initial = np.array([*Inpvalues])
    ndim = len(initial)
    p0 = [np.array(initial) + 1e-5 * np.random.randn(ndim) for i in range(nwalkers)]
    if moffat:
        if plot_f:
            tim=True
        else:
            tim=False
        sampler, pos, prob, state = mcmc(p0,nwalkers,niter,ndim,lnprob_moffat0,data,tim=tim,ncpu=ncpu)
    else:
        sampler, pos, prob, state = mcmc(p0,nwalkers,niter,ndim,lnprob_gaussian,data,ncpu=ncpu) 
    samples = sampler.flatchain
    theta_max  = samples[np.argmax(sampler.flatlnprobability)]
    pars_max = {}
    keys=list(valsI.keys())
    for key in keys:
        pars_max[key]=valsI[key]
    for i in range(0, len(Namevalues)):
        pars_max[Namevalues[i]]=theta_max[i]
    ft_num=np.nansum(pf_map)
    if moffat:
        ft_fit=np.pi*pars_max['alpha']**2.0*pars_max['At']/(pars_max['beta']-1.0)
    else:
        ft_fit=2*np.pi*pars_max['sigma']**2.0*pars_max['At']
    if plot_f:
        if moffat:
            spec_t=mod.moffat_modelF(pars_max, x_t=x_t, y_t=y_t, host=host)
            if singlepsf:
                spec_hst=0.0
            else:
                spec_hst=mod.moffat_modelF(pars_max, x_t=x_t, y_t=y_t, agn=False)
        else:
            spec_t=mod.gaussian_modelF(pars_max, x_t=x_t, y_t=y_t)
  
        tol.plot_models_maps(pf_map,spec_t,spec_hst,samples,name=name,path_out=path_out,savefig=savefig,Labelvalues=Labelvalues)
                            
    dx_m=pars_max['xo']+min_in[1]
    dy_m=pars_max['yo']+min_in[0]
    if moffat:
        psf=pars_max['alpha']*2.0*np.sqrt(2.0**(1./pars_max['beta'])-1)
    else:
        psf=pars_max['sigma']*2.0*np.sqrt(2.0*np.log10(2.0))
    if moffat:
        if singlepsf:
            return dx_m,dy_m,pars_max['alpha'],pars_max['beta'],psf,ft_num,ft_fit,0,0,0,0,pars_max['At'],pars_max['ellip'],pars_max['theta']
        else:
            return dx_m,dy_m,pars_max['alpha'],pars_max['beta'],psf,ft_num,ft_fit,pars_max['Io'],pars_max['bn'],pars_max['Re'],pars_max['ns'],pars_max['At'],pars_max['ellip'],pars_max['theta']
    else:
        return dx_m,dy_m,ds_m,psf,ft_num,ft_fit