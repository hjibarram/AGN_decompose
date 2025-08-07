#!/usr/bin/env python
import numpy as np
import AGNdecomp.tools.models as mod
                   
def lnlike_multmodel(theta, spec, specE, x_t, y_t, valsI, Namevalues, Namemodel, Usermods, datapsf):
    model=mod.multi_model(theta, valsI, Namevalues, Namemodel, Usermods, x_t=x_t, y_t=y_t, datapsf=datapsf)
    LnLike = -0.5*np.nansum(((spec-model)/specE)**2.0)
    return LnLike

def lnprior_multmodel(theta, Infvalues, Supvalues):
    boolf=True 
    for i in range(0, len(theta)):
        bool1=(theta[i] <= Supvalues[i])
        bool2=(theta[i] >= Infvalues[i])
        boolf=(bool1 & bool2) & boolf
    if boolf:
        return 0.0
    else:
        return -np.inf    

def lnprob_multmodel(theta, spec, specE, x_t, y_t, valsI, Infvalues, Supvalues, Namevalues, Namemodel, Usermods, datapsf):
    lp = lnprior_multmodel(theta, Infvalues, Supvalues)
    if not np.isfinite(lp):
        return -np.inf
    else:
        return lp + lnlike_multmodel(theta, spec, specE, x_t, y_t, valsI, Namevalues, Namemodel, Usermods, datapsf)