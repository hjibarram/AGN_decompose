#!/usr/bin/env python
import numpy as np
import emcee
from astropy.convolution import convolve, Gaussian2DKernel
import AGNdecomp.tools.tools as tol
import AGNdecomp.tools.models as mod
from AGNdecomp.tools.priors import lnprob_multmodel

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
            p0, _, _ = sampler.run_mcmc(p0, 1000)
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


def evaluate_2dPSF(pf_map,pf_mapE,name='test',Model_name='moffat',Labelvalues=[],Namevalues=[],Inpvalues=[],Infvalues=[],Supvalues=[],path_out='',savefig=True,autocent=True,sig=2,plot_f=False,ncpu=10,valsI={}):
    if plot_f:
        tim=True
    else:
        tim=False
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
        min_in=[valsI['dyo'],valsI['dxo']]
    x_t=np.arange(ny)-min_in[1]
    y_t=np.arange(nx)-min_in[0]
    x_t=np.array([x_t]*nx)
    y_t=np.array([y_t]*ny).T
    valsI['xo']=valsI['xo']-min_in[1]
    valsI['yo']=valsI['yo']-min_in[0]
    #print("Input values: ",valsI)
    data = (pf_map, pf_mapE, x_t, y_t, valsI, Infvalues, Supvalues, Namevalues, Model_name)
    nwalkers=240
    niter=1024
    initial = np.array([*Inpvalues])
    ndim = len(initial)
    p0 = [np.array(initial) + 1e-5 * np.random.randn(ndim) for i in range(nwalkers)]
    sampler, pos, prob, state = mcmc(p0,nwalkers,niter,ndim,lnprob_multmodel,data,tim=tim,ncpu=ncpu)
    samples = sampler.flatchain
    theta_max  = samples[np.argmax(sampler.flatlnprobability)]
    pars_max = {}
    keys=list(valsI.keys())
    for key in keys:
        pars_max[key]=valsI[key]
    for i in range(0, len(Namevalues)):
        pars_max[Namevalues[i]]=theta_max[i]
    ft_num=np.nansum(pf_map)
    if Model_name=='moffat':
        ft_fit=np.pi*pars_max['alpha']**2.0*pars_max['At']/(pars_max['beta']-1.0)
    else:
        ft_fit=2*np.pi*pars_max['sigma']**2.0*pars_max['At']
    if plot_f:
        if Model_name=='moffat':
            spec_t=mod.moffat_modelF(pars_max, x_t=x_t, y_t=y_t, host=False)
            spec_hst=mod.moffat_modelF(pars_max, x_t=x_t, y_t=y_t, agn=False)
        else:
            spec_t=mod.gaussian_modelF(pars_max, x_t=x_t, y_t=y_t)
            spec_hst=spec_t*0
        tol.plot_models_maps(pf_map,spec_t,spec_hst,samples,name=name,path_out=path_out,savefig=savefig,Labelvalues=Labelvalues)
    pars_max['xo']=pars_max['xo']+min_in[1]
    pars_max['yo']=pars_max['yo']+min_in[0]
    if Model_name=='moffat':
        psf=pars_max['alpha']*2.0*np.sqrt(2.0**(1./pars_max['beta'])-1)
    else:
        psf=pars_max['sigma']*2.0*np.sqrt(2.0*np.log10(2.0))
    return pars_max,psf,ft_num,ft_fit