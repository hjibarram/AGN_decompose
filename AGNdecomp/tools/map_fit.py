#!/usr/bin/env python
import numpy as np
from astropy.wcs import WCS
from astropy.wcs.utils import pixel_to_skycoord
import AGNdecomp.tools.tools as tol
from AGNdecomp.tools.mcmc import evaluate_2dPSF
import warnings
warnings.filterwarnings("ignore")

def map_ana(map,mapE,hdr,sig=2,prior_config='priors_prop.yaml',prior_pathconf='',Usermods=['extern','pathext','extern_function.py'],mod_ind=0,mod_ind0=0,verbose=False,dir_o='',name='spectra',ncpu=10,logP=True,stl=False,smoth=True,sigm=1.8,ofsval=-1):
    Inpvalues, Infvalues, Supvalues, Namevalues, Labelvalues, Model_name=tol.get_priorsvalues(prior_pathconf+prior_config,verbose=verbose,mod_ind=mod_ind)
    if dir_o != '':
        tol.sycall('mkdir -p '+dir_o)
    nx,ny=map.shape
    p_vals=[]
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
    wcs = WCS(hdr)
    wcs=wcs.celestial
    valsI,Inpvalues=tol.define_initvals(p_vals,Namevalues,Namevalues0,Inpvalues,0,str_p=False)
    pars_max,psf1,Ft,FtF=evaluate_2dPSF(map,mapE,name=name,Usermods=Usermods,Model_name=Model_name,Labelvalues=Labelvalues,Namevalues=Namevalues,Inpvalues=Inpvalues,Infvalues=Infvalues,Supvalues=Supvalues,sig=sig,plot_f=True,ncpu=ncpu,valsI=valsI,path_out=dir_o,logP=logP,stl=stl,smoth=smoth,sigm=sigm,ofsval=ofsval)
    sky1=pixel_to_skycoord(pars_max['xo'],pars_max['yo'],wcs)
    val1=sky1.to_string('hmsdms')
    linet='FLUX='+str(FtF)+' FLUXN='+str(Ft)+' RADEC='+str(val1)+' PSF='+str(psf1*dpix)
    linev=''
    for val in Namevalues0:
        linev=linev+' '+val+'='+str(pars_max[val])
    print(linet+linev)