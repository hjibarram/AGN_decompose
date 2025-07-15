#!/usr/bin/env python
import glob, os,sys,timeit
import numpy as np
import AGNdecomp.tools.models as mod

def lnlike_Multmodel():
    model=mod.Multmodel(theta, x_t=x_t, y_t=y_t, ds_t=ds_m, be_t=db_m, bn=bn, ns=ns, Lt=Lt_m)
    LnLike = -0.5*np.nansum(((spec-model)/specE)**2.0)
    return LnLike
                   
def lnlike_moffat(theta, spec, specE, x_t, y_t, valsI, Namevalues):
    model=mod.moffat_model(theta, valsI, Namevalues, x_t=x_t, y_t=y_t)
    LnLike = -0.5*np.nansum(((spec-model)/specE)**2.0)
    return LnLike
                    
def lnlike_gaussian(theta, spec, specE, x_t, y_t):
    model=mod.gaussian_model(theta, x_t=x_t, y_t=y_t)
    LnLike = -0.5*np.nansum(((spec-model)/specE)**2.0)
    return LnLike   

def lnprior_mod(theta, Infvalues, Supvalues):
    boolf=True 
    f_param=theta
    for i in range(0, len(f_param)):
        bool1=(f_param[i] <= Supvalues[i])
        bool2=(f_param[i] >= Infvalues[i])
        boolf=(bool1 & bool2) & boolf
    if boolf:
        return 0.0
    else:
        return -np.inf    

def lnprob_moffat(theta, spec, specE, x_t, y_t, valsI, Infvalues, Supvalues, Namevalues):
    lp = lnprior_mod(theta, Infvalues, Supvalues)
    if not np.isfinite(lp):
        return -np.inf
    else:
        return lp + lnlike_moffat(theta, spec, specE, x_t, y_t, valsI, Namevalues)

def lnprob_gaussian(theta, spec, specE, x_t, y_t, valsI, Infvalues, Supvalues, Namevalues):
    lp = lnprior_mod(theta, Infvalues, Supvalues)
    if not np.isfinite(lp):
        return -np.inf
    else:
        return lp + lnlike_gaussian(theta, spec, specE, x_t, y_t, valsI, Namevalues)  