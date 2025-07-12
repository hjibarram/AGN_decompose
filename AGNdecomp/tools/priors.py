#!/usr/bin/env python
import glob, os,sys,timeit
import numpy as np
import AGNdecomp.tools.models as mod

def lnlike_Multmodel():
    model=mod.Multmodel(theta, x_t=x_t, y_t=y_t, ds_t=ds_m, be_t=db_m, bn=bn, ns=ns, Lt=Lt_m)
    LnLike = -0.5*np.nansum(((spec-model)/specE)**2.0)
    return LnLike

def lnlike_Dmoffat3(theta, spec, specE, x_t, y_t, db_m, bn, ns, Lt_m):
    model=mod.Dmoffat_model3(theta, x_t=x_t, y_t=y_t, be_t=db_m, bn=bn, ns=ns, Lt=Lt_m)
    LnLike = -0.5*np.nansum(((spec-model)/specE)**2.0)
    return LnLike

def lnlike_Dmoffat2(theta, spec, specE, x_t, y_t, ds_m, db_m, bn, ns, Lt_m):
    model=mod.Dmoffat_model2(theta, x_t=x_t, y_t=y_t, ds_t=ds_m, be_t=db_m, bn=bn, ns=ns, Lt=Lt_m)
    LnLike = -0.5*np.nansum(((spec-model)/specE)**2.0)
    return LnLike
                    
def lnlike_Dmoffat0(theta, spec, specE, x_t, y_t, db_m):
    model=mod.Dmoffat_model0(theta, x_t=x_t, y_t=y_t, be_t=db_m)
    LnLike = -0.5*np.nansum(((spec-model)/specE)**2.0)
    return LnLike
                   
def lnlike_Dmoffat(theta, spec, specE, x_t, y_t, ds_m, db_m):
    model=mod.Dmoffat_model(theta, x_t=x_t, y_t=y_t, ds_t=ds_m, be_t=db_m)
    LnLike = -0.5*np.nansum(((spec-model)/specE)**2.0)
    return LnLike
                 
def lnlike_ring3(theta, spec, specE, x_t, y_t, bn, ns, dx, dy):
    model=mod.ring_model_residual3(theta, x_t=x_t, y_t=y_t, bn=bn, ns=ns, dx=dx, dy=dy)
    LnLike = -0.5*np.nansum(((spec-model)/specE)**2.0)#/np.float(len(theta))
    return LnLike
                 
def lnlike_ring2(theta, spec, specE, x_t, y_t, ds_m, ro_m, bn, ns, dx, dy):
    model=mod.ring_model_residual2(theta, x_t=x_t, y_t=y_t, ds_t=ds_m, r0=ro_m, bn=bn, ns=ns, dx=dx, dy=dy)
    LnLike = -0.5*np.nansum(((spec-model)/specE)**2.0)#/np.float(len(theta))
    return LnLike
                 
def lnlike_ring0(theta, spec, specE, x_t, y_t, dx, dy):
    model=mod.ring_model_residual0(theta, x_t=x_t, y_t=y_t, dx=dx, dy=dy)
    LnLike = -0.5*np.nansum(((spec-model)/specE)**2.0)#/np.float(len(theta))
    return LnLike
                
def lnlike_ring(theta, spec, specE, x_t, y_t, ds_m, ro_m, dx, dy):
    model=mod.ring_model_residual(theta, x_t=x_t, y_t=y_t, ds_t=ds_m, r0=ro_m, dx=dx, dy=dy)
    LnLike = -0.5*np.nansum(((spec-model)/specE)**2.0)#/np.float(len(theta))
    return LnLike
    
def lnlike_moffat3_s(theta, spec, specE, x_t, y_t, valsI, keysI, Namevalues):
    model=mod.moffat_model3_s(theta, valsI, keysI, Namevalues, x_t=x_t, y_t=y_t)
    LnLike = -0.5*np.nansum(((spec-model)/specE)**2.0)
    return LnLike
                   
def lnlike_moffat3(theta, spec, specE, x_t, y_t, valsI, keysI, Namevalues):
    model=mod.moffat_model3(theta, valsI, keysI, Namevalues, x_t=x_t, y_t=y_t)
    LnLike = -0.5*np.nansum(((spec-model)/specE)**2.0)
    return LnLike
                     
def lnlike_moffat2_s(theta, spec, specE, x_t, y_t, ds_m, db_m, dx, dy, e_m, tht_m, ellip):
    model=mod.moffat_model2_s(theta, x_t=x_t, y_t=y_t, ds_t=ds_m, be_t=db_m, dx=dx, dy=dy, e_m=e_m, tht_m=tht_m, ellip=ellip)
    LnLike = -0.5*np.nansum(((spec-model)/specE)**2.0)#/np.float(len(theta))
    return LnLike
                     
def lnlike_moffat2(theta, spec, specE, x_t, y_t, ds_m, db_m, bn, ns, e_m, tht_m, ellip, dxt, dyt, fcenter, re_int, Re_c):
    model=mod.moffat_model2(theta, x_t=x_t, y_t=y_t, ds_t=ds_m, be_t=db_m, bn=bn, ns=ns, e_m=e_m, tht_m=tht_m, ellip=ellip, dxt=dxt, dyt=dyt, fcenter=fcenter, re_int=re_int, Re_c=Re_c)
    LnLike = -0.5*np.nansum(((spec-model)/specE)**2.0)#/np.float(len(theta))
    return LnLike
                     
def lnlike_moffat0_s(theta, spec, specE, x_t, y_t, db_m, e_m, tht_m, beta, ellip):
    model=mod.moffat_model0_s(theta, x_t=x_t, y_t=y_t, db_m=db_m, e_m=e_m, tht_m=tht_m, beta=beta, ellip=ellip)
    LnLike = -0.5*np.nansum(((spec-model)/specE)**2.0)#/np.float(len(theta))
    return LnLike
                   
def lnlike_moffat0(theta, spec, specE, x_t, y_t, valsI, keysI, Namevalues):#db_m, e_m, tht_m, ellip):
    model=mod.moffat_model0(theta, valsI, keysI, Namevalues, x_t=x_t, y_t=y_t)#, be_t=db_m, e_m=e_m, tht_m=tht_m, ellip=ellip)
    LnLike = -0.5*np.nansum(((spec-model)/specE)**2.0)#/np.float(len(theta))
    return LnLike
                    
def lnlike_moffat_s(theta, spec, specE, x_t, y_t, ds_m, db_m, e_m, tht_m, ellip):
    model=mod.moffat_model_st(theta, x_t=x_t, y_t=y_t, ds_t=ds_m, be_t=db_m, e_m=e_m, tht_m=tht_m, ellip=ellip)
    LnLike = -0.5*np.nansum(((spec-model)/specE)**2.0)#/np.float(len(theta))
    return LnLike
                  
def lnlike_moffat(theta, spec, specE, x_t, y_t, ds_m, db_m):
    model=mod.moffat_model(theta, x_t=x_t, y_t=y_t, ds_t=ds_m, be_t=db_m)
    LnLike = -0.5*np.nansum(((spec-model)/specE)**2.0)#/np.float(len(theta))
    return LnLike
                    
def lnlike_gaussian(theta, spec, specE, x_t, y_t):
    model=mod.gaussian_model(theta, x_t=x_t, y_t=y_t)
    LnLike = -0.5*np.nansum(((spec-model)/specE)**2.0)#/np.float(len(theta))
    return LnLike   

#theta, spec, specE, x_t, y_t, db_m, dx, dy, e_m, tht_m, al_m, bn, ns, Lt_m, ds_m, ro_m, ring, beta, ellip, aplha, fcenter, re_int, Re_c
def lnprior_Dmofft3(theta, At1=20):
    At,dx,dy,Io,Re,ds_t=theta
    if ((ds_t >= 2) and (ds_t <= 22)) and ((At >= 0) and (At < At1)) and ((dx >= -5) and (dx <= 5)) and ((dy >= -5) and (dy <= 5)) and (Io >= 0) and ((Re >= 0.5) and (Re <= 50.0)):#10.0# and ((e >= 0.0) and (e <= 10.0)) and ((th0 >= 0) and (th0 <= 2.0*np.pi)):                 
        return 0.0
    else:
        return -np.inf

def lnprior_Dmofft2(theta, At1=20):
    At,dx,dy,Io,Re=theta
    if ((At >= 0) and (At < At1)) and ((dx >= -5) and (dx <= 5)) and ((dy >= -5) and (dy <= 5)) and (Io >= 0) and ((Re >= 0.5) and (Re <= 50.0)):#10.0# and ((e >= 0.0) and (e <= 10.0)) and ((th0 >= 0) and (th0 <= 2.0*np.pi)):                 
        return 0.0
    else:
        return -np.inf

def lnprior_Dmofft0(theta, At1=20):
    At,dx,dy,Io,bn,Re,ns,ds_t,Lt=theta
    if ((Lt >= 0.0) and (Lt <= 5.5)) and ((ds_t >= 2) and (ds_t <= 22)) and ((At >= 0) and (At < (At1))) and ((dx >= -5) and (dx <= 5)) and ((dy >= -5) and (dy <= 5)) and (Io >= 0) and ((bn >= 0.5) and (bn <= 50.0)) and ((Re >= 0.5) and (Re <= 50.0)) and ((ns >= 0.5) and (ns <= 2.0)):#10.0# and ((e >= 0.0) and (e <= 10.0)) and ((th0 >= 0) and (th0 <= 2.0*np.pi)):                 
        return 0.0
    else:
        return -np.inf

def lnprior_Dmofft(theta, At1=20):
    At,dx,dy,Io,bn,Re,ns,Lt=theta
    if ((Lt >= 0.0) and (Lt <= 5.5)) and ((At >= 0) and (At < (At1))) and ((dx >= -5) and (dx <= 5)) and ((dy >= -5) and (dy <= 5)) and (Io >= 0) and ((bn >= 0.5) and (bn <= 50.0)) and ((Re >= 0.5) and (Re <= 50.0)) and ((ns >= 0.5) and (ns <= 2.0)):#10.0# and ((e >= 0.0) and (e <= 10.0)) and ((th0 >= 0) and (th0 <= 2.0*np.pi)):                 
        return 0.0
    else:
        return -np.inf


def lnprior_ring3(theta, At1=20):
    At,Io,Re,ds_t,r0=theta
    if ((r0 >= 1.8) and (r0 <= 2.2)) and ((ds_t >= 0.4) and (ds_t <= 0.6)) and ((At >= 0) and (At < At1)) and (Io >= 0) and ((Re >= 0.5) and (Re <= 50.0)):#10.0# and ((e >= 0.0) and (e <= 10.0)) and ((th0 >= 0) and (th0 <= 2.0*np.pi)):                 
        return 0.0
    else:
        return -np.inf

def lnprior_ring2(theta, At1=20):
    At,Io,Re=theta
    if ((At >= 0) and (At < At1)) and (Io >= 0) and ((Re >= 0.5) and (Re <= 50.0)):#10.0# and ((e >= 0.0) and (e <= 10.0)) and ((th0 >= 0) and (th0 <= 2.0*np.pi)):                 
        return 0.0
    else:
        return -np.inf

def lnprior_ring0(theta, At1=20):
    At,Io,bn,Re,ns,ds_t,r0=theta
    if ((r0 >= 1.8) and (r0 <= 3.2)) and ((ds_t >= 0.4) and (ds_t <= 1.6)) and ((At >= 0) and (At < At1)) and (Io >= 0) and ((bn >= 0.5) and (bn <= 50.0)) and ((Re >= 0.5) and (Re <= 50.0)) and ((ns >= 0.5) and (ns <= 2.0)):#10.0# and ((e >= 0.0) and (e <= 10.0)) and ((th0 >= 0) and (th0 <= 2.0*np.pi)):    
        return 0.0
    else:
        return -np.inf

def lnprior_ring(theta, At1=20):
    At,Io,bn,Re,ns=theta
    if ((At >= 0) and (At < At1)) and (Io >= 0) and ((bn >= 0.5) and (bn <= 50.0)) and ((Re >= 0.5) and (Re <= 50.0)) and ((ns >= 0.5) and (ns <= 2.0)):#10.0# and ((e >= 0.0) and (e <= 10.0)) and ((th0 >= 0) and (th0 <= 2.0*np.pi)):    
        return 0.0
    else:
        return -np.inf

def lnprior_mofft3_s(theta, Infvalues, Supvalues, valsI, keysI):#At1=20, beta=True, ellip=False, alpha=True):
    boolf=True 
    At1=valsI['At1']
    ellip=keysI['ellip']
    re_int=keysI['re_int']
    fcenter=keysI['fcenter']
    if beta:
        if ellip:
            At,ds_t,db_t,e_t,tht_t=theta
            boolf=((ds_t >= 5)) & ((db_t >= 0.5)) & ((At >= 0) & (At < At1)) & ((e_t >= 0) & (e_t < 1)) & ((tht_t >= 0.0) & (tht_t < 180.0)) & boolf
        else:
            At,ds_t,db_t=theta
            boolf=((ds_t >= 5)) & ((db_t >= 0.5)) & ((At >= 0) & (At < At1)) & boolf
    else:
        if ellip:
            At,ds_t,e_t,tht_t=theta
            boolf=((ds_t >= 5)) & ((At >= 0) & (At < At1)) & ((e_t >= 0) & (e_t < 1)) & ((tht_t >= 0.0) & (tht_t < 180.)) & boolf
        else:
            if alpha:
                At,ds_t=theta
                boolf=((ds_t >= 5)) & ((At >= 0)) & boolf
            else:
                At=theta
                boolf=((At >= 0))
    f_param=theta
    for i in range(0, len(f_parm)):
        bool1=(f_parm[i] <= Supvalues[i])
        bool2=(f_parm[i] >= Infvalues[i])
        boolf=(bool1 & bool2) & boolf

    if boolf:
        return 0.0
    else:
        return -np.inf 

def lnprior_mofft3(theta, Infvalues, Supvalues, valsI, keysI):#At1=20, ellip=False, fcenter=False, re_int=False):
    boolf=True 
    At1=valsI['At1']
    ellip=keysI['ellip']
    re_int=keysI['re_int']
    fcenter=keysI['fcenter']
    '''
    if ellip:
        if fcenter:
            if re_int:
                At,Io,ds_t,e,th0=theta
                boolf=((ds_t >= 10) & (ds_t <= 22)) & ((At >= 0) & (At < At1)) & (Io >= 0) & ((e >= 0.0) & (e <= 10.0)) & ((th0 >= 0) & (th0 < 180)) & boolf
            else:
                At,Io,Re,ds_t,e,th0=theta
                boolf=((ds_t >= 10) & (ds_t <= 22)) & ((At >= 0) & (At < At1)) & (Io >= 0) & ((Re >= 0.5) & (Re <= 50.0)) & ((e >= 0.0) & (e <= 10.0)) & ((th0 >= 0) & (th0 < 180)) & boolf
        else:
            if re_int:
                At,dx,dy,Io,ds_t,e,th0=theta
                boolf=((ds_t >= 10) & (ds_t <= 22)) & ((At >= 0) & (At < At1)) & ((dx >= -5) & (dx <= 5)) & ((dy >= -5) & (dy <= 5)) & (Io >= 0) & ((e >= 0.0) & (e <= 10.0)) & ((th0 >= 0) & (th0 < 180)) & boolf
            else:
                At,dx,dy,Io,Re,ds_t,e,th0=theta
                boolf=((ds_t >= 10) & (ds_t <= 22)) & ((At >= 0) & (At < At1)) & ((dx >= -5) & (dx <= 5)) & ((dy >= -5) & (dy <= 5)) & (Io >= 0) & ((Re >= 0.5) & (Re <= 50.0)) & ((e >= 0.0) & (e <= 10.0)) & ((th0 >= 0) & (th0 < 180)) & boolf       
    else:
        if fcenter:
            if re_int:
                At,Io,ds_t=theta
                boolf=((ds_t >= 10) & (ds_t <= 22)) & ((At >= 0) & (At < At1)) & (Io >= 0) & boolf
            else:
                At,Io,Re,ds_t=theta
                boolf=((ds_t >= 10) & (ds_t <= 22)) & ((At >= 0) & (At < At1)) & (Io >= 0) & ((Re >= 0.5) & (Re <= 50.0)) & boolf
        else:
            if re_int:
                At,dx,dy,Io,ds_t=theta
                boolf=((ds_t >= 10) & (ds_t <= 22)) & ((At >= 0) & (At < At1)) & ((dx >= -5) & (dx <= 5)) & ((dy >= -5) & (dy <= 5)) & (Io >= 0) & boolf
            else:
                At,dx,dy,Io,Re,ds_t=theta
                boolf= ((ds_t >= 10) & (ds_t <= 22)) & ((At >= 0) & (At < At1)) & ((dx >= -5) & (dx <= 5)) & ((dy >= -5) & (dy <= 5)) & (Io >= 0) & ((Re >= 0.5) & (Re <= 50.0)) & boolf                
    '''
    f_param=theta
    for i in range(0, len(f_parm)):
        bool1=(f_parm[i] <= Supvalues[i])
        bool2=(f_parm[i] >= Infvalues[i])
        boolf=(bool1 & bool2) & boolf

    if boolf:
        return 0.0
    else:
        return -np.inf            
                                     
    
def lnprior_mofft2_s(theta, At1=20):
    At=theta#,e,th0
    #and (At < At1)
    if ((At >= 0) and (At < At1)):# and (Io >= 0):# and ((Re >= 0.5) and (Re <= 50.0)):#10.0# and ((e >= 0.0) and (e <= 10.0)) and ((th0 >= 0) and (th0 <= 2.0*np.pi)):                 
        return 0.0
    else:
        return -np.inf    

def lnprior_mofft2(theta, At1=20, ellip=False, fcenter=False, re_int=False):
    if ellip:
        if fcenter:
            if re_int:
                At,Io,e,th0=theta
                if ((At >= 0) and (At < At1)) and (Io >= 0) and ((e >= 0.0) and (e <= 10.0)) and ((th0 >= 0) and (th0 <= 180)):                 
                    return 0.0
                else:
                    return -np.inf
            else:
                At,Io,Re,e,th0=theta
                if ((At >= 0) and (At < At1)) and (Io >= 0) and ((Re >= 0.5) and (Re <= 50.0)) and ((e >= 0.0) and (e <= 10.0)) and ((th0 >= 0) and (th0 <= 180)):                 
                    return 0.0
                else:
                    return -np.inf
        else:
            if re_int:
                At,dx,dy,Io,e,th0=theta
                if ((At >= 0) and (At < At1)) and ((dx >= -5) and (dx <= 5)) and ((dy >= -5) and (dy <= 5)) and (Io >= 0) and ((e >= 0.0) and (e <= 10.0)) and ((th0 >= 0) and (th0 <= 180)):                 
                    return 0.0
                else:
                    return -np.inf
            else:
                At,dx,dy,Io,Re,e,th0=theta
                if ((At >= 0) and (At < At1)) and ((dx >= -5) and (dx <= 5)) and ((dy >= -5) and (dy <= 5)) and (Io >= 0) and ((Re >= 0.5) and (Re <= 50.0)) and ((e >= 0.0) and (e <= 10.0)) and ((th0 >= 0) and (th0 <= 180)):                 
                    return 0.0
                else:
                    return -np.inf
    else:
        if fcenter:
            if re_int:
                At,Io=theta
                if ((At >= 0) and (At < At1)) and (Io >= 0):                 
                    return 0.0
                else:
                    return -np.inf
            else:
                At,Io,Re=theta
                if ((At >= 0) and (At < At1)) and (Io >= 0) and ((Re >= 0.5) and (Re <= 50.0)):                 
                    return 0.0
                else:
                    return -np.inf
        else:
            if re_int:
                At,dx,dy,Io=theta
                if ((At >= 0) and (At < At1)) and ((dx >= -5) and (dx <= 5)) and ((dy >= -5) and (dy <= 5)) and (Io >= 0):               
                    return 0.0
                else:
                    return -np.inf
            else:
                At,dx,dy,Io,Re=theta
                if ((At >= 0) and (At < At1)) and ((dx >= -5) and (dx <= 5)) and ((dy >= -5) and (dy <= 5)) and (Io >= 0) and ((Re >= 0.5) and (Re <= 50.0)):               
                    return 0.0
                else:
                    return -np.inf
    
def lnprior_mofft0_s(theta,  Infvalues, Supvalues, valsI, keysI):#At1=20, beta=True, ellip=False):
    boolf=True 
    At1=valsI['At1']
    ellip=keysI['ellip']
    '''
    if beta:
        if ellip:
            At,dx,dy,ds_t,db_t,e_t,tht_t=theta
            if ((ds_t >= 5)) and ((db_t >= 0.5)) and ((At >= 0) and (At < (At1))) and ((dx >= -5) and (dx <= 5)) and ((dy >= -5) and (dy <= 5)) and ((e_t >= 0) and (e_t < 1)) and ((tht_t >= 0.0) and (tht_t < 180.)):# and (Io >= 0) and ((bn >= 0.5) and (bn <= 50.0)) and ((Re >= 0.5) and (Re <= 50.0)) and ((ns >= 0.5) and (ns <= 5.0)):#10.0# and ((e >= 0.0) and (e <= 10.0)) and ((th0 >= 0) and (th0 <= 2.0*np.pi)):                 
                return 0.0
            else:
                return -np.inf
        else:
            At,dx,dy,ds_t,db_t=theta
            if ((ds_t >= 5)) and ((db_t >= 0.5)) and ((At >= 0) and (At < (At1))) and ((dx >= -5) and (dx <= 5)) and ((dy >= -5) and (dy <= 5)):# and (Io >= 0) and ((bn >= 0.5) and (bn <= 50.0)) and ((Re >= 0.5) and (Re <= 50.0)) and ((ns >= 0.5) and (ns <= 5.0)):#10.0# and ((e >= 0.0) and (e <= 10.0)) and ((th0 >= 0) and (th0 <= 2.0*np.pi)):                 
                return 0.0
            else:
                return -np.inf
    else:
        if ellip:
            At,dx,dy,ds_t,e_t,tht_t=theta
            if ((ds_t >= 5)) and ((At >= 0) and (At < (At1))) and ((dx >= -5) and (dx <= 5)) and ((dy >= -5) and (dy <= 5)) and ((e_t >= 0) and (e_t < 1)) and ((tht_t >= 0.0) and (tht_t < 180.)):# and (Io >= 0) and ((bn >= 0.5) and (bn <= 50.0)) and ((Re >= 0.5) and (Re <= 50.0)) and ((ns >= 0.5) and (ns <= 5.0)):#10.0# and ((e >= 0.0) and (e <= 10.0)) and ((th0 >= 0) and (th0 <= 2.0*np.pi)):                 
                return 0.0
            else:
                return -np.inf    
        else:
            At,dx,dy,ds_t=theta
            if ((ds_t >= 5)) and ((At >= 0)) and ((dx >= -5) and (dx <= 5)) and ((dy >= -5) and (dy <= 5)):# and (Io >= 0) and ((bn >= 0.5) and (bn <= 50.0)) and ((Re >= 0.5) and (Re <= 50.0)) and ((ns >= 0.5) and (ns <= 5.0)):#10.0# and ((e >= 0.0) and (e <= 10.0)) and ((th0 >= 0) and (th0 <= 2.0*np.pi)):                 
                return 0.0
            else:
                return -np.inf    
    '''      
    f_param=theta
    for i in range(0, len(f_parm)):
        bool1=(f_parm[i] <= Supvalues[i])
        bool2=(f_parm[i] >= Infvalues[i])
        boolf=(bool1 & bool2) & boolf

    if boolf:
        return 0.0
    else:
        return -np.inf          

def lnprior_mofft0(theta, Infvalues, Supvalues, valsI, keysI):#At1=20, ellip=False):
    boolf=True 
    #At1=valsI['At1']
    #ellip=keysI['ellip']
    '''
    if ellip:
        At,dx,dy,Io,bn,Re,ns,ds_t,e,th0=theta
        if ((ds_t >= 10) and (ds_t <= 22)) and ((At >= 0) and (At < (At1))) and ((dx >= -5) and (dx <= 5)) and ((dy >= -5) and (dy <= 5)) and (Io >= 0) and ((bn >= 0.5) and (bn <= 50.0)) and ((Re >= 0.5) and (Re <= 50.0)) and ((ns >= 0.5) and (ns <= 2.0)) and ((e >= 0.0) and (e <= 10.0)) and ((th0 >= 0) and (th0 <= 180.0)):                 
            return 0.0
        else:
            return -np.inf
    else:
        At,dx,dy,Io,bn,Re,ns,ds_t=theta
        if ((ds_t >= 2) and (ds_t <= 8.5)) and ((At >= 0) and (At < (At1))) and ((dx >= -5) and (dx <= 5)) and ((dy >= -5) and (dy <= 5)) and (Io >= 0) and ((bn >= 0.5) and (bn <= 50.0)) and ((Re >= 0.5) and (Re <= 50.0)) and ((ns >= 0.5) and (ns <= 2.0)):#10.0# and ((e >= 0.0) and (e <= 10.0)) and ((th0 >= 0) and (th0 <= 2.0*np.pi)):                 
            return 0.0
        else:
            return -np.inf
    '''
    f_param=theta
    for i in range(0, len(f_parm)):
        bool1=(f_parm[i] <= Supvalues[i])
        bool2=(f_parm[i] >= Infvalues[i])
        boolf=(bool1 & bool2) & boolf

    if boolf:
        return 0.0
    else:
        return -np.inf    

def lnprior_mofft_s(theta, At1=20):
    At,dx,dy=theta#,e,th0
    #Ao=Io*np.exp(bn)
    #if (At >= 0) and ((dx >= -5) and (dx <= 5)) and ((dy >= -5) and (dy <= 5)) and ((ds_t >= 1) and (ds_t <= 10)) and ((be_t >= 0.5) and (be_t <= 10)):
    if ((At >= 0) and (At < (At1))) and ((dx >= -5) and (dx <= 5)) and ((dy >= -5) and (dy <= 5)):# and (Io >= 0) and ((bn >= 0.5) and (bn <= 50.0)) and ((Re >= 0.5) and (Re <= 50.0)) and ((ns >= 0.5) and (ns <= 5.0)):#10.0# and ((e >= 0.0) and (e <= 10.0)) and ((th0 >= 0) and (th0 <= 2.0*np.pi)):                 
        return 0.0
    else:
        return -np.inf

def lnprior_mofft(theta, At1=20):
    At,dx,dy,Io,bn,Re,ns=theta#,e,th0
    #Ao=Io*np.exp(bn)
    #if (At >= 0) and ((dx >= -5) and (dx <= 5)) and ((dy >= -5) and (dy <= 5)) and ((ds_t >= 1) and (ds_t <= 10)) and ((be_t >= 0.5) and (be_t <= 10)):
    if ((At >= 0) and (At < (At1))) and ((dx >= -5) and (dx <= 5)) and ((dy >= -5) and (dy <= 5)) and (Io >= 0) and ((bn >= 0.5) and (bn <= 50.0)) and ((Re >= 0.5) and (Re <= 50.0)) and ((ns >= 0.5) and (ns <= 2.0)):#10.0# and ((e >= 0.0) and (e <= 10.0)) and ((th0 >= 0) and (th0 <= 2.0*np.pi)):                 
        return 0.0
    else:
        return -np.inf

def lnprior_gaussian(theta):
    At,dx,dy,ds_t=theta
    if (At >= 0) and ((dx >= -5) and (dx <= 5)) and ((dy >= -5) and (dy <= 5)) and ((ds_t >= 1) and (ds_t <= 10)):             
        return 0.0
    else:
        return -np.inf    
    
def lnprob_Dmoffat3(theta, spec, specE, x_t, y_t, db_m, At, bn, ns, Lt_m):
    lp = lnprior_Dmofft3(theta, At1=At)
    if not np.isfinite(lp):
        return -np.inf
    else:
        return lp + lnlike_Dmoffat3(theta, spec, specE, x_t, y_t, db_m, bn, ns, Lt_m) 

def lnprob_Dmoffat2(theta, spec, specE, x_t, y_t, ds_m, db_m, At, bn, ns, Lt_m):
    lp = lnprior_Dmofft2(theta, At1=At)
    if not np.isfinite(lp):
        return -np.inf
    else:
        return lp + lnlike_Dmoffat2(theta, spec, specE, x_t, y_t, ds_m, db_m, bn, ns, Lt_m) 

def lnprob_Dmoffat0(theta, spec, specE, x_t, y_t, db_m, At):
    lp = lnprior_Dmofft0(theta, At1=At)
    if not np.isfinite(lp):
        return -np.inf
    else:
        return lp + lnlike_Dmoffat0(theta, spec, specE, x_t, y_t, db_m) 
    
def lnprob_Dmoffat(theta, spec, specE, x_t, y_t, ds_m, db_m, At):
    lp = lnprior_Dmofft(theta, At1=At)
    if not np.isfinite(lp):
        return -np.inf
    else:
        return lp + lnlike_Dmoffat(theta, spec, specE, x_t, y_t, ds_m, db_m)     
    

def lnprob_ring3(theta, spec, specE, x_t, y_t, At, bn, ns, dx, dy):
    lp = lnprior_ring3(theta, At1=At)
    if not np.isfinite(lp):
        return -np.inf
    else:
        return lp + lnlike_ring3(theta, spec, specE, x_t, y_t, bn, ns, dx, dy)
    
def lnprob_ring2(theta, spec, specE, x_t, y_t, ds_m, ro_m, At, bn, ns, dx, dy):
    lp = lnprior_ring2(theta, At1=At)
    if not np.isfinite(lp):
        return -np.inf
    else:
        return lp + lnlike_ring2(theta, spec, specE, x_t, y_t, ds_m, ro_m, bn, ns, dx, dy)

def lnprob_ring0(theta, spec, specE, x_t, y_t, At, dx, dy):
    lp = lnprior_ring0(theta, At1=At)
    if not np.isfinite(lp):
        return -np.inf
    else:
        return lp + lnlike_ring0(theta, spec, specE, x_t, y_t, dx, dy)
    
def lnprob_ring(theta, spec, specE, x_t, y_t, At, ds_m, ro_m, dx, dy):
    lp = lnprior_ring(theta, At1=At)
    if not np.isfinite(lp):
        return -np.inf
    else:
        return lp + lnlike_ring(theta, spec, specE, x_t, y_t, ds_m, ro_m, dx, dy)    

def lnprob_moffat3_s(theta, spec, specE, x_t, y_t, valsI, keysI, Infvalues, Supvalues, Namevalues):
    lp = lnprior_mofft3_s(theta, Infvalues, Supvalues, valsI, keysI)
    if not np.isfinite(lp):
        return -np.inf
    else:
        return lp + lnlike_moffat3_s(theta, spec, specE, x_t, y_t, valsI, keysI)

def lnprob_moffat3(theta, spec, specE, x_t, y_t, valsI, keysI, Infvalues, Supvalues, Namevalues):
    lp = lnprior_mofft3(theta, Infvalues, Supvalues, valsI, keysI)
    if not np.isfinite(lp):
        return -np.inf
    else:
        return lp + lnlike_moffat3(theta, spec, specE, x_t, y_t, valsI, keysI, Namevalues)

def lnprob_moffat2_s(theta, spec, specE, x_t, y_t, ds_m, db_m, At, dx, dy, e_t, tht_t, ellip):
    lp = lnprior_mofft2_s(theta, At1=At)
    if not np.isfinite(lp):
        return -np.inf
    else:
        return lp + lnlike_moffat2_s(theta, spec, specE, x_t, y_t, ds_m, db_m, dx, dy, e_t, tht_t, ellip) 

def lnprob_moffat2(theta, spec, specE, x_t, y_t, ds_m, db_m, At, bn, ns, e_t, tht_t, ellip, Re_c, re_int, dxt, dyt, fcenter):
    lp = lnprior_mofft2(theta, At1=At, ellip=ellip, fcenter=fcenter, re_int=re_int) 
    if not np.isfinite(lp):
        return -np.inf
    else:
        return lp + lnlike_moffat2(theta, spec, specE, x_t, y_t, ds_m, db_m, bn, ns, e_t, tht_t, ellip, dxt, dyt, fcenter, re_int, Re_c)  

def lnprob_moffat0_s(theta, spec, specE, x_t, y_t, valsI, keysI, Infvalues, Supvalues, Namevalues):#db_m, At, e_t, tht_t, beta, ellip):
    lp = lnprior_mofft0_s(theta, At1=At, beta=beta, ellip=ellip)
    if not np.isfinite(lp):
        return -np.inf
    else:
        return lp + lnlike_moffat0_s(theta, spec, specE, x_t, y_t, db_m, e_t, tht_t, beta, ellip)

def lnprob_moffat0(theta, spec, specE, x_t, y_t, valsI, keysI, Infvalues, Supvalues, Namevalues):#db_m, At, e_t, tht_t, ellip):
    lp = lnprior_mofft0(theta, Infvalues, Supvalues, valsI, keysI)#At1=At, ellip=ellip)
    if not np.isfinite(lp):
        return -np.inf
    else:
        return lp + lnlike_moffat0(theta, spec, specE, x_t, y_t, valsI, keysI, Namevalues)#db_m, e_t, tht_t, ellip) 
    
def lnprob_moffat_s(theta, spec, specE, x_t, y_t, ds_m, db_m, At, e_t, tht_t, ellip):
    lp = lnprior_mofft_s(theta, At1=At)
    if not np.isfinite(lp):
        return -np.inf
    else:
        return lp + lnlike_moffat_s(theta, spec, specE, x_t, y_t, ds_m, db_m, e_t, tht_t, ellip)     
    
def lnprob_moffat(theta, spec, specE, x_t, y_t, ds_m, db_m, At):
    lp = lnprior_mofft(theta, At1=At)
    if not np.isfinite(lp):
        return -np.inf
    else:
        return lp + lnlike_moffat(theta, spec, specE, x_t, y_t, ds_m, db_m) 

def lnprob_gaussian(theta, spec, specE, x_t, y_t):
    lp = lnprior_gaussian(theta)
    if not np.isfinite(lp):
        return -np.inf
    else:
        return lp + lnlike_gaussian(theta, spec, specE, x_t, y_t)  