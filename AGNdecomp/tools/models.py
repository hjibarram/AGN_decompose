#!/usr/bin/env python
import glob, os,sys,timeit
import numpy as np
import AGNdecomp.tools.tools as tol
from astropy.io import fits

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

def moffat_model3(theta, valsI, keysI, Namevalues, x_t=0, y_t=0):
    ellip=keysI['ellip']
    re_int=keysI['re_int']
    fcenter=keysI['fcenter']
    be_t=valsI['db_m']
    bn=valsI['bn']
    ns=valsI['ns']
    e_t=valsI['e_m']
    tht_t=valsI['tht_m']
    Re=valsI['Re_c']
    dx=valsI['dx']
    dy=valsI['dy']
    if ellip:
        if fcenter:
            if re_int:
                At,Io,ds_t,e_t,tht_t=theta
            else:
                At,Io,Re,ds_t,e_t,tht_t=theta
        else:
            if re_int:
                At,dx,dy,Io,ds_t,e_t,tht_t=theta
            else:
                At,dx,dy,Io,Re,ds_t,e_t,tht_t=theta
    else:
        if fcenter:
            if re_int:
                At,Io,ds_t=theta
            else:
                At,Io,Re,ds_t=theta
        else:
            if re_int:
                At,dx,dy,Io,ds_t=theta
            else:
                At,dx,dy,Io,Re,ds_t=theta
    r1=tol.radi_ellip(x_t-dx,y_t-dy,e_t,tht_t) 
    spec_agn=At*(1.0 + (r1**2.0/ds_t**2.0))**(-be_t)
    spec_hst=Io*np.exp(-bn*((r1/Re)**(1./ns)-1))
    spec_t=spec_agn+spec_hst
    return spec_t             

def moffat_model0(theta, valsI, Namevalues, x_t=0, y_t=0, host=True):
    pars={}
    keys=list(valsI.keys())
    for key in keys:
        pars[key]=valsI[key]
    for i in range(0, len(Namevalues)):
        pars[Namevalues[i]]=theta[i]
    spec_t=moffat_modelF(pars, x_t=x_t, y_t=y_t, host=host)    
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


def gaussian_model(theta, valsI, Namevalues, x_t=0, y_t=0):
    pars={}
    keys=list(valsI.keys())
    for key in keys:
        pars[key]=valsI[key]
    for i in range(0, len(Namevalues)):
        pars[Namevalues[i]]=theta[i]
    spec_t=gaussian_modelF(pars, x_t=x_t, y_t=y_t)
    return spec_t  

def gaussian_modelF(pars, x_t=0, y_t=0,):
    # This is the function for the basic gaussian model
    At=pars['At']
    dx=pars['xo']
    dy=pars['yo']
    ds=pars['sigma']
    spec_t=np.exp(-0.5*((((x_t-dx)/ds)**2.0)+((y_t-dy)/ds)**2.0))*At
    return spec_t   


def get_model(dir_o='./',dir_cube='./',vt='',dir_cube_m='./',corr=False,name='Name',sig=10,cosmetic=False,moffat=True):
    if moffat:
        psf_file='NAME_moffat'.replace('NAME',name)
    else:
        psf_file='NAME'.replace('NAME',name)
    wave=[]
    FtF=[]
    Ft=[]
    psf=[]
    dx=[]
    dy=[]
    alpha=[]
    beta=[]
    At=[]
    et=[]
    th0=[]
    ft=open(dir_o+psf_file+vt+'.csv','r')
    for line in ft:
        if not 'WAVE' in line:
            data=line.replace('\n','')
            data=data.split(',')
            data=list(filter(None,data))
            wave.extend([float(data[0])])
            FtF.extend([float(data[1])])
            Ft.extend([float(data[2])])
            dx.extend([float(data[5])])
            dy.extend([float(data[6])])
            alpha.extend([float(data[7])])
            beta.extend([float(data[8])])
            psf.extend([float(data[9])])
            At.extend([float(data[14])])
            et.extend([float(data[15])])
            th0.extend([float(data[16])])
    ft.close()
    wave=np.array(wave)
    FtF=np.array(FtF)
    psf=np.array(psf)
    dx=np.array(dx)
    dy=np.array(dy)
    alpha=np.array(alpha)
    beta=np.array(beta)
    At=np.array(At)
    et=np.array(et)
    th0=np.array(th0)
    
 
    if corr:
        dxt=np.nanmean(dx[4450:4470])
        dyt=np.nanmean(dy[4450:4470])  
        dx[4479:len(dx)]=dxt
        dy[4479:len(dy)]=dyt
    if cosmetic:
        At=tol.conv(At,ke=sig)
        dx=tol.conv(dx,ke=sig)
        dy=tol.conv(dy,ke=sig)   
    
    cube_file='NAME.cube.fits.gz'.replace('NAME',name)
    outf1='Model_NAME.cube'.replace('NAME',name+vt)
    outf3='Residual_NAME.cube'.replace('NAME',name+vt)
    [cube, hdr0]=fits.getdata(dir_cube+cube_file, 0, header=True)
    [cube1, hdr1]=fits.getdata(dir_cube+cube_file, 1, header=True)
    [cube2, hdr2]=fits.getdata(dir_cube+cube_file, 2, header=True)
    nz,nx,ny=cube.shape
    cube_mod=np.copy(cube)
    cube_mod[:,:,:]=0.0
    for k in range(0, nz):
        if k < len(At):
            theta=At[k],dx[k],dy[k],alpha[k],beta[k],et[k],th0[k]
            for i in range(0, nx):
                for j in range(0, ny):
                    valt1=moffat_model_s(theta, x_t=j, y_t=i, ellip=True)       
                    if cube[k,i,j] != 0:    
                        cube_mod[k,i,j]=valt1
                
    h1=fits.PrimaryHDU(cube_mod)
    h=h1.header
    keys=list(hdr0.keys())
    for i in range(0, len(keys)):
        if not "COMMENT" in  keys[i] and not 'HISTORY' in keys[i]:
            h[keys[i]]=hdr0[keys[i]]
            h.comments[keys[i]]=hdr0.comments[keys[i]]
    hlist=fits.HDUList([h1])
    hlist.update_extend()
    out_fit=dir_cube_m+outf1+'.fits'
    hlist.writeto(out_fit, overwrite=True)
    tol.sycall('gzip  '+out_fit)
    
    res=cube-cube_mod
    h1=fits.PrimaryHDU(res)
    h2=fits.ImageHDU(cube1)
    h3=fits.ImageHDU(cube2)
    h_k=h1.header
    keys=list(hdr0.keys())
    for i in range(0, len(keys)):
        if not "COMMENT" in  keys[i] and not 'HISTORY' in keys[i]:
            h_k[keys[i]]=hdr0[keys[i]]
            h_k.comments[keys[i]]=hdr0.comments[keys[i]]
    h_k.update()
    h_t=h2.header
    keys=list(hdr1.keys())
    for i in range(0, len(keys)):
        if not "COMMENT" in  keys[i] and not 'HISTORY' in keys[i]:
            h_t[keys[i]]=hdr1[keys[i]]
            h_t.comments[keys[i]]=hdr1.comments[keys[i]]
    h_t['EXTNAME'] ='Error_cube'
    h_t.update()
    h_r=h3.header
    keys=list(hdr2.keys())
    for i in range(0, len(keys)):
        h_r[keys[i]]=hdr2[keys[i]]
        h_r.comments[keys[i]]=hdr2.comments[keys[i]]
    h_r['EXTNAME'] ='BADPIXELMASK'
    h_r.update()    
    hlist=fits.HDUList([h1,h2,h3])
    hlist.update_extend()
    out_fit=dir_cube_m+outf3+'.fits'
    hlist.writeto(out_fit, overwrite=True)
    tol.sycall('gzip -f '+out_fit)         