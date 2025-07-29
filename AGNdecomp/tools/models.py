#!/usr/bin/env python
import sys
import importlib.util
import numpy as np
import AGNdecomp.tools.tools as tol
from astropy.io import fits
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

def Dmoffat_model(theta, x_t=0, y_t=0,be_t=2.064,ds_t=3.47):
    At,dx,dy,Io,bn,Re,ns,Lt=theta
    r2=(x_t-dx)**2.0+(y_t-dy)**2.0
    r2_A=(x_t-(dx-Lt/(1.*np.sqrt(3.0))))**2.0+(y_t-(dy+0.0/2.0))**2.0
    r2_B=(x_t-(dx+Lt/(2.*np.sqrt(3.0))))**2.0+(y_t-(dy+Lt/2.0))**2.0
    r2_C=(x_t-(dx+Lt/(2.*np.sqrt(3.0))))**2.0+(y_t-(dy-Lt/2.0))**2.0
    spec_agn_A=At*(1.0 + (r2_A/ds_t**2.0))**(-be_t)
    spec_agn_B=At*(1.0 + (r2_B/ds_t**2.0))**(-be_t)
    spec_agn_C=At*(1.0 + (r2_C/ds_t**2.0))**(-be_t)
    spec_agn=spec_agn_A+spec_agn_B+spec_agn_C   
    r1=np.sqrt(r2)
    bt=r1
    spec_hst=Io*np.exp(-bn*((bt/Re)**(1./ns)-1))
    spec_t=spec_agn+spec_hst
    return spec_t  

def get_extern_function(Usermods=['moffat','path','extern_function.py'],verbose=False):
    # This function returns the external function for the costum user model
    # The name of the model must be defined in the file extern_function.py
    # The path is the path to the AGNdecomp package
    name=Usermods[0]
    path=Usermods[1]
    namef=Usermods[2]
    try:
        spec = importlib.util.spec_from_file_location(name, path + namef)
        extmod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(extmod)
        if verbose:
            print('Loading external function for',name)
        return getattr(extmod, name )
    except Exception as e:
        if verbose:
            print('Error loading external function:', e)
        sys.exit()

def multi_model(theta, valsI, Namevalues, Namemodel, Usermods, x_t=0, y_t=0, host=True):
    pars={}
    keys=list(valsI.keys())
    for key in keys:
        pars[key]=valsI[key]
    for i in range(0, len(Namevalues)):
        pars[Namevalues[i]]=theta[i]
    if Namemodel == 'moffat':
        spec_t=moffat_modelF(pars, x_t=x_t, y_t=y_t, host=host)
    elif Namemodel == 'gaussian':
        spec_t=gaussian_modelF(pars, x_t=x_t, y_t=y_t)
    elif Namemodel == Usermods[0]:
        extern_func=get_extern_function(Usermods=Usermods,verbose=False) 
        spec_t=extern_func(pars, x_t=x_t, y_t=y_t)
    else:
        print('Error, the model '+Namemodel+' is not implemented, available models are moffat gaussian or '+Usermods[0])
        sys.exit()
    return spec_t

def moffat_modelF(pars, x_t=0, y_t=0, host=True, agn=True):
    # This is the function for the Full Moffat model
    At=pars['At']
    ds=pars['alpha']
    be=pars['beta']
    dx=pars['xo']
    dy=pars['yo']
    Io=pars['Io']
    bn=pars['bn']
    Re=pars['Re']
    ns=pars['ns']
    es=pars['ellip']
    th=pars['theta']
    r1=tol.radi_ellip(x_t-dx,y_t-dy,es,th)
    if agn:
        spec_agn=At*(1.0 + (r1**2.0/ds**2.0))**(-be)
    else:
        spec_agn=r1*0.0
    if host:   
        spec_hst=Io*np.exp(-bn*((r1/Re)**(1./ns)-1))
    else:
        spec_hst=r1*0.0
    spec_t=spec_agn+spec_hst
    return spec_t

def gaussian_modelF(pars, x_t=0, y_t=0,):
    # This is the function for the basic gaussian model
    At=pars['At']
    dx=pars['xo']
    dy=pars['yo']
    ds=pars['sigma']
    spec_t=np.exp(-0.5*((((x_t-dx)/ds)**2.0)+((y_t-dy)/ds)**2.0))*At
    return spec_t   

def moffat_flux_psf_modelF(pars):
    psf=pars['alpha']*2.0*np.sqrt(2.0**(1./pars['beta'])-1)
    ft_fit=np.pi*pars['alpha']**2.0*pars['At']/(pars['beta']-1.0)
    return psf, ft_fit

def gaussian_flux_psf_modelF(pars):
    ft_fit=2*np.pi*pars['sigma']**2.0*pars['At']
    psf=pars['sigma']*2.0*np.sqrt(2.0*np.log10(2.0))
    return psf, ft_fit

def get_model(dir_o='./',dir_cube='./',vt='',hdri0=0,hdri1=1,hdri2=2,prior_config='priors_prop.yml',mod_ind0=0,prior_pathconf='',verbose=False,dir_cube_m='./',Usermods=['','',''],name='Name',sig=10,moffat=True,basename='NAME.cube.fits.gz'):
    modelname=tol.get_priorsvalues(prior_pathconf+prior_config,verbose=verbose,mod_ind=mod_ind0,onlymodel=True)
    psf_file='NAME_MODEL'.replace('NAME',name).replace('MODEL',modelname)
    valsT=tol.read_cvsfile(dir_o+psf_file+vt+'.csv',hid='wave')    
    keys=list(valsT.keys())
    cube_file=basename.replace('NAME',name)
    outf1='Model_'+basename.replace('.fits','').replace('.gz','').replace('NAME',name+vt)
    outf3='Residual_'+basename.replace('.fits','').replace('.gz','').replace('NAME',name+vt)
    [cube0, hdr0]=fits.getdata(dir_cube+cube_file, hdri0, header=True)
    try:
        [cube1, hdr1]=fits.getdata(dir_cube+cube_file, hdri1, header=True)
    except:
        [cube1, hdr1]=[cube0, hdr0]
    try:
        [cube2, hdr2]=fits.getdata(dir_cube+cube_file, hdri2, header=True)
    except:
        [cube2, hdr2]=[cube1, hdr1]
    nz,nx,ny=cube0.shape
    cube_mod=np.copy(cube0)
    cube_mod[:,:,:]=0.0
    try:
        model=get_extern_function(Usermods=Usermods,verbose=verbose)
        extern=True
    except:
        extern=False
        #model=getattr(mod, Model_name + '_modelF')
    if verbose:
        pbar=tqdm(total=nz)
    x_t=np.arange(ny)
    y_t=np.arange(nx)
    x_t=np.array([x_t]*nx)
    y_t=np.array([y_t]*ny).T    
    for k in range(0, nz):
        pars={}
        for key in keys:
            pars[key]=valsT[key][k]
        if extern == False:
            valt1=moffat_modelF(pars, x_t=x_t, y_t=y_t, host=False)
        else:
            try:
                valt1=model(pars, x_t=x_t, y_t=y_t, host=False)
            except:
                valt1=model(pars, x_t=x_t, y_t=y_t)
        for i in range(0, nx):
            for j in range(0, ny):
                if cube0[k,i,j] != 0:    
                    cube_mod[k,i,j]=valt1[i,j]
        if verbose:
            pbar.update(1)  
    h1=fits.PrimaryHDU(cube_mod)
    h=h1.header
    keys=list(hdr0.keys())
    for i in range(0, len(keys)):
        try:
            if not "COMMENT" in  keys[i] and not 'HISTORY' in keys[i]:
                h[keys[i]]=hdr0[keys[i]]
                h.comments[keys[i]]=hdr0.comments[keys[i]]
        except:
            continue
    hlist=fits.HDUList([h1])
    hlist.update_extend()
    out_fit=dir_cube_m+outf1+'.fits'
    hlist.writeto(out_fit, overwrite=True)
    tol.sycall('gzip -f '+out_fit)
    
    res=cube0-cube_mod
    h1=fits.PrimaryHDU(res)
    h2=fits.ImageHDU(cube1)
    h3=fits.ImageHDU(cube2)
    h_k=h1.header
    keys=list(hdr0.keys())
    for i in range(0, len(keys)):
        try:
            if not "COMMENT" in  keys[i] and not 'HISTORY' in keys[i]:
                h_k[keys[i]]=hdr0[keys[i]]
                h_k.comments[keys[i]]=hdr0.comments[keys[i]]
        except:
            continue
    h_k.update()
    h_t=h2.header
    keys=list(hdr1.keys())
    for i in range(0, len(keys)):
        try:
            if not "COMMENT" in  keys[i] and not 'HISTORY' in keys[i]:
                h_t[keys[i]]=hdr1[keys[i]]
                h_t.comments[keys[i]]=hdr1.comments[keys[i]]
        except:
            continue
    h_t['EXTNAME'] ='Error_cube'
    h_t.update()
    h_r=h3.header
    keys=list(hdr2.keys())
    for i in range(0, len(keys)):
        try:
            h_r[keys[i]]=hdr2[keys[i]]
            h_r.comments[keys[i]]=hdr2.comments[keys[i]]
        except:
            continue
    h_r['EXTNAME'] ='BADPIXELMASK'
    h_r.update()    
    hlist=fits.HDUList([h1,h2,h3])
    hlist.update_extend()
    out_fit=dir_cube_m+outf3+'.fits'
    hlist.writeto(out_fit, overwrite=True)
    tol.sycall('gzip -f '+out_fit)         