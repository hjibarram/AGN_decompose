#!/usr/bin/env python
import glob, os,sys,timeit
import matplotlib
import numpy as np
import emcee
from astropy.convolution import convolve, convolve_fft, Gaussian2DKernel
import matplotlib.pyplot as plt
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


def evaluate_2dPSF(pf_map,pf_mapE,name='test',Labelvalues=[],Namevalues=[],Inpvalues=[],Infvalues=[],Supvalues=[],savefig=True,model=True,autocent=True,fcenter=False,sig=2,plot_f=False,beta=True,re_int=False,ellip=False,singlepsf=False,trip=False,ring=False,moffat=False,mc=False,ncpu=10,psft=False,valsT={},ds_i=0,db_m=0,ro_i=0,psf_coef=0,e_m=0.0,tht_m=0.0,pi_x=0,pi_y=0,bs_c=0,Re_c=0,ns_c=0,Lt_c=4.4,dyo=0,dxo=0):
    if len(valsT) > 0:
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
    if psft == False:
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
        valsI['be_m']=db_m
        valsI['e_t']=e_m
        valsI['th_t']=tht_m
        valsI['xo']=pi_x-min_in[1]
        valsI['yo']=pi_y-min_in[0]
        valsI['At']=At
        valsI['al_m']=5.0
        valsI['Re']=Re_c
        valsI['bn']=bs_c
        valsI['ns']=ns_c 
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
            tim=True
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
        if savefig:
            fig.savefig('Original_NAME.pdf'.replace('NAME',name))
        else:
            plt.show()
    
    ft_num=np.nansum(pf_map)
    if moffat:
        ft_fit=np.pi*pars_max['ds_m']**2.0*pars_max['At']/(pars_max['be_m']-1.0)
    else:
        ft_fit=2*np.pi*pars_max['ds_m']**2.0*pars_max['At']
    if model:
        if moffat:
            spec_t=mod.moffat_modelF(pars_max, x_t=x_t, y_t=y_t, host=host)
            if singlepsf:
                spec_hst=0.0
            else:
                spec_hst=mod.moffat_modelF(pars_max, x_t=x_t, y_t=y_t, agn=False)
        else:
            spec_t=mod.gaussian_modelF(pars_max, x_t=x_t, y_t=y_t)
        if plot_f:    
            fig, ax = plt.subplots(figsize=(6.8*1.1,5.5*1.2))
            ict=plt.imshow(np.log10(spec_t),cmap=cm) 
            cbar=plt.colorbar(ict)
            ics=plt.contour(spec_t,lev,colors='k',linewidths=1)
            ics=plt.contour(pf_map,lev,colors='red',linewidths=1)            
            cbar.set_label(r"Relative Density")
            fig.tight_layout()
            if savefig:
                fig.savefig('Model_NAME.pdf'.replace('NAME',name))
            else:
                plt.show()
            
            fig, ax = plt.subplots(figsize=(6.8*1.1,5.5*1.2))
            ict=plt.imshow(np.log10(pf_map-spec_t),cmap=cm) #np.log10
            cbar=plt.colorbar(ict)
            ics=plt.contour((pf_map-spec_t),lev,colors='k',linewidths=1)
            cbar.set_label(r"Relative Density")
            fig.tight_layout()
            if savefig:
                fig.savefig('Residual1_NAME.pdf'.replace('NAME',name))
            else:
                plt.show()
            
            fig, ax = plt.subplots(figsize=(6.8*1.1,5.5*1.2))
            ict=plt.imshow(np.log10(pf_map-spec_t-spec_hst),cmap=cm) 
            cbar=plt.colorbar(ict)
            ics=plt.contour((pf_map-spec_t-spec_hst),lev,colors='k',linewidths=1)
            cbar.set_label(r"Relative Density")
            fig.tight_layout()
            if savefig:
                fig.savefig('Residual2_NAME.pdf'.replace('NAME',name))
            else:
                plt.show()
            
            if mc:
                labels = [*Labelvalues]
                import corner  
                fig = corner.corner(samples,show_titles=True,labels=labels,plot_datapoints=True,quantiles=[0.16, 0.5, 0.84],title_kwargs={"fontsize": 16},label_kwargs={"fontsize": 16})
                fig.set_size_inches(15.8*len(labels)/8.0, 15.8*len(labels)/8.0)
                fig.savefig('corners_NAME.pdf'.replace('NAME',name))
                            
        dx_m=pars_max['xo']+min_in[1]
        dy_m=pars_max['yo']+min_in[0]
        if moffat:
            psf=pars_max['ds_m']*2.0*np.sqrt(2.0**(1./pars_max['be_m'])-1)
        else:
            psf=pars_max['ds_m']*2.0*np.sqrt(2.0*np.log10(2.0))
        if moffat:
            if singlepsf:
                return dx_m,dy_m,pars_max['ds_m'],pars_max['be_m'],psf,ft_num,ft_fit,spec_t,0,0,0,0,At0,e_m,tht_m
            else:
                return dx_m,dy_m,pars_max['ds_m'],pars_max['be_m'],psf,ft_num,ft_fit,spec_t,pars_max['Io'],pars_max['bn'],pars_max['Re'],pars_max['ns'],pars_max['At'],pars_max['e_t'],pars_max['th_t']
        else:
            return dx_m,dy_m,ds_m,psf,ft_num,ft_fit,spec_t
    else:
        dx_m=pars_max['xo']+min_in[1]
        dy_m=pars_max['yo']+min_in[0]
        if moffat:
            psf=pars_max['ds_m']*2.0*np.sqrt(2.0**(1./pars_max['be_m'])-1)
        else:
            psf=pars_max['ds_m']*2.0*np.sqrt(2.0*np.log10(2.0))
        if moffat:
            if singlepsf:
                return dx_m,dy_m,pars_max['ds_m'],pars_max['be_m'],psf,ft_num,ft_fit,0,0,0,0,pars_max['At'],pars_max['e_t'],pars_max['th_t']
            else:
                return dx_m,dy_m,pars_max['ds_m'],pars_max['be_m'],psf,ft_num,ft_fit,pars_max['Io'],pars_max['bn'],pars_max['Re'],pars_max['ns'],pars_max['At'],pars_max['e_t'],pars_max['th_t']
        else:    
            return dx_m,dy_m,ds_m,psf,ft_num,ft_fit