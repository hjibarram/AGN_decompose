#!/usr/bin/env python
import glob, os,sys,timeit
import numpy as np
import AGNdecomp.tools.tools as tol
from astropy.io import fits

def moffat_model_s_residual(theta, x_t=0, y_t=0):
    At,dx,dy,ds_t,be_t,be_t1,psf=theta
    alpha=psf/2.0/np.sqrt(2.0**(1/be_t1)-1)
    r2=(x_t-dx)**2.0+(y_t-dy)**2.0
    spec_agn_c=At*(1.0 + (r2/ds_t**2.0))**(-be_t)
    spec_agn_t=At*(1.0 + (r2/alpha**2.0))**(-be_t1)    
    spec_agn=spec_agn_t-spec_agn_c
    spec_t=spec_agn
    return spec_t

def Dmoffat_model_s(theta, x_t=0, y_t=0):
    At,dx,dy,ds_t,be_t,Lt=theta
    r2_A=(x_t-(dx-Lt/(1.*np.sqrt(3.0))))**2.0+(y_t-(dy+0.0/2.0))**2.0
    r2_B=(x_t-(dx+Lt/(2.*np.sqrt(3.0))))**2.0+(y_t-(dy+Lt/2.0))**2.0
    r2_C=(x_t-(dx+Lt/(2.*np.sqrt(3.0))))**2.0+(y_t-(dy-Lt/2.0))**2.0
    spec_agn_A=At*(1.0 + (r2_A/ds_t**2.0))**(-be_t)
    spec_agn_B=At*(1.0 + (r2_B/ds_t**2.0))**(-be_t)
    spec_agn_C=At*(1.0 + (r2_C/ds_t**2.0))**(-be_t)
    spec_t=spec_agn_A+spec_agn_B+spec_agn_C
    return spec_t

def moffat_model_s(theta, x_t=0, y_t=0, e_m=0.0, tht_m=0.0, ellip=False):
    if ellip:
        At,dx,dy,ds_t,be_t,e_t,tht_t=theta
        r2=tol.radi_ellip(x_t-dx,y_t-dy,e_t,tht_t)  
        r2=r2**2.0
    else:
        At,dx,dy,ds_t,be_t=theta
        r2=(x_t-dx)**2.0+(y_t-dy)**2.0
    spec_t=At*(1.0 + (r2/ds_t**2.0))**(-be_t)
    return spec_t  

def ring_model_s(theta, x_t=0, y_t=0):
    At,dx,dy,ds_t,r0=theta
    r=np.sqrt((x_t-dx)**2.0+(y_t-dy)**2.0)
    spec_t=np.exp(-0.5*(((r-r0)/ds_t)**2.0))*At           
    return spec_t   

def gaussian_model_s(theta, x_t=0, y_t=0):
    At,dx,dy,ds_t=theta
    spec_t=np.exp(-0.5*((((x_t-dx)/ds_t)**2.0)+((y_t-dy)/ds_t)**2.0))*At           
    return spec_t   


def Dmoffat_model3(theta, x_t=0, y_t=0, be_t=2.064, bn=1.0, ns=1.0, Lt=4.4):
    At,dx,dy,Io,Re,ds_t=theta
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

def Dmoffat_model2(theta, x_t=0, y_t=0, be_t=2.064, ds_t=3.47, bn=1.0, ns=1.0,Lt=4.4):
    At,dx,dy,Io,Re=theta
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

def Dmoffat_model0(theta, x_t=0, y_t=0, be_t=2.064):
    At,dx,dy,Io,bn,Re,ns,ds_t,Lt=theta
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


def ring_model_residual3(theta, x_t=0, y_t=0, bn=1.0, ns=1.0, dx=0.0, dy=0.0):
    At,Io,Re,ds_t,r0=theta
    r2=(x_t-dx)**2.0+(y_t-dy)**2.0
    r1=np.sqrt(r2)
    spec_agn=np.exp(-0.5*(((r1-r0)/ds_t)**2.0))*At
    bt=r1
    spec_hst=Io*np.exp(-bn*((bt/Re)**(1./ns)-1))
    spec_t=spec_agn+spec_hst
    return spec_t

def ring_model_residual2(theta, x_t=0, y_t=0, ds_t=3.47, r0=1.0, bn=1.0, ns=1.0, dx=0.0, dy=0.0):
    At,Io,Re=theta
    r2=(x_t-dx)**2.0+(y_t-dy)**2.0
    r1=np.sqrt(r2)
    spec_agn=np.exp(-0.5*(((r1-r0)/ds_t)**2.0))*At
    bt=r1
    spec_hst=Io*np.exp(-bn*((bt/Re)**(1./ns)-1))
    spec_t=spec_agn+spec_hst
    return spec_t

def ring_model_residual0(theta, x_t=0, y_t=0, dx=0.0, dy=0.0):
    At,Io,bn,Re,ns,ds_t,r0=theta
    r2=(x_t-dx)**2.0+(y_t-dy)**2.0
    r1=np.sqrt(r2)
    spec_agn=np.exp(-0.5*(((r1-r0)/ds_t)**2.0))*At
    bt=r1
    spec_hst=Io*np.exp(-bn*((bt/Re)**(1./ns)-1))
    spec_t=spec_agn+spec_hst
    return spec_t   

def ring_model_residual(theta, x_t=0, y_t=0, ds_t=3.47, r0=1.0, dx=0.0, dy=0.0):
    At,Io,bn,Re,ns=theta
    r2=(x_t-dx)**2.0+(y_t-dy)**2.0
    r1=np.sqrt(r2)
    spec_agn=np.exp(-0.5*(((r1-r0)/ds_t)**2.0))*At
    bt=r1
    spec_hst=Io*np.exp(-bn*((bt/Re)**(1./ns)-1))
    spec_t=spec_agn+spec_hst
    return spec_t   

def moffat_model_residual(theta, x_t=0, y_t=0, be_t=2.064, ds_t=3.47, At=1.0, psf=2.4):
    dx,dy,Io,bn,Re,ns=theta
    alpha=psf/2.0/np.sqrt(2.0**(1/be_t1)-1)
    r2=(x_t-dx)**2.0+(y_t-dy)**2.0
    spec_agn_c=At*(1.0 + (r2/ds_t**2.0))**(-be_t)
    spec_agn_t=At*(1.0 + (r2/alpha**2.0))**(-be_t1)    
    spec_agn=spec_agn_t-spec_agn_c
    r1=np.sqrt(r2)
    bt=r1
    spec_hst=Io*np.exp(-bn*((bt/Re)**(1./ns)-1))
    spec_t=spec_agn+spec_hst
    return spec_t   

def moffat_model3_s(theta, x_t=0, y_t=0, db_m=2.06, dx=0.0, dy=0.0, e_m=0.0, tht_m=0.0, al_m=5.0, beta=True, ellip=False, alpha=True):
    if ellip:
        if beta:
            At,ds_t,be_t,e_t,tht_t=theta
        else:
            At,ds_t,e_t,tht_t=theta
            be_t=db_m
        r2=tol.radi_ellip(x_t-dx,y_t-dy,e_t,tht_t) 
        r2=r2**2.0  
    else:
        if beta:
            At,ds_t,be_t=theta
        else:
            if alpha:
                At,ds_t=theta
                be_t=db_m
            else:
                At=theta
                be_t=db_m
                ds_t=al_m
        r2=tol.radi_ellip(x_t-dx,y_t-dy,e_m,tht_m) 
        r2=r2**2.0      
        #r2=(x_t-dx)**2.0+(y_t-dy)**2.0
    spec_agn=At*(1.0 + (r2/ds_t**2.0))**(-be_t)    
    spec_t=spec_agn
    return spec_t    

def moffat_model3(theta, x_t=0, y_t=0, be_t=2.064, bn=1.0, ns=1.0, e_m=0.0, tht_m=0.0, ellip=False, dxt=0.0, dyt=0.0, fcenter=False):
    if ellip:
        if fcenter:
            At,Io,Re,ds_t,e_t,tht_t=theta
            r2=tol.radi_ellip(x_t-dxt,y_t-dyt,e_t,tht_t) 
        else:
            At,dx,dy,Io,Re,ds_t,e_t,tht_t=theta
            r2=tol.radi_ellip(x_t-dx,y_t-dy,e_t,tht_t) 
    else:
        if fcenter:
            At,Io,Re,ds_t=theta
            r2=tol.radi_ellip(x_t-dxt,y_t-dyt,e_m,tht_m)
        else:
            At,dx,dy,Io,Re,ds_t=theta
            r2=tol.radi_ellip(x_t-dx,y_t-dy,e_m,tht_m) 
    r2=r2**2.0  
    spec_agn=At*(1.0 + (r2/ds_t**2.0))**(-be_t)    
    r1=np.sqrt(r2)
    bt=r1
    spec_hst=Io*np.exp(-bn*((bt/Re)**(1./ns)-1))
    spec_t=spec_agn+spec_hst
    return spec_t       

def moffat_model2_s(theta, x_t=0, y_t=0, be_t=2.064, ds_t=3.47, dx=0.0, dy=0.0, e_m=0.0, tht_m=0.0, ellip=False):
    At=theta
    if ellip:
        e_t=e_m
        tht_t=tht_m
        r2=tol.radi_ellip(x_t-dx,y_t-dy,e_t,tht_t) 
        r2=r2**2.0
    else:
        r2=(x_t-dx)**2.0+(y_t-dy)**2.0
    spec_agn=At*(1.0 + (r2/ds_t**2.0))**(-be_t)    
    spec_t=spec_agn
    return spec_t    

def moffat_model2(theta, x_t=0, y_t=0, be_t=2.064, ds_t=3.47, bn=1.0, ns=1.0, e_m=0.0, tht_m=0.0, ellip=False, dxt=0.0, dyt=0.0, fcenter=False, re_int=False, Re_c=1.0):
    if ellip:
        if fcenter:
            if re_int:
                At,Io,e_t,tht_t=theta 
                Re=Re_c
            else:
                At,Io,Re,e_t,tht_t=theta
            r2=tol.radi_ellip(x_t-dxt,y_t-dyt,e_t,tht_t)
        else:
            if re_int:
                At,dx,dy,Io,e_t,tht_t=theta
                Re=Re_c
            else:
                At,dx,dy,Io,Re,e_t,tht_t=theta
            r2=tol.radi_ellip(x_t-dx,y_t-dy,e_t,tht_t)
    else:
        if fcenter:
            if re_int:
                At,Io=theta
                Re=Re_c
            else:
                At,Io,Re=theta
            r2=tol.radi_ellip(x_t-dxt,y_t-dyt,e_m,tht_m)
        else:
            if re_int:
                At,dx,dy,Io=theta
                Re=Re_c
            else:
                At,dx,dy,Io,Re=theta
            r2=tol.radi_ellip(x_t-dx,y_t-dy,e_m,tht_m)
    r2=r2**2.0
    spec_agn=At*(1.0 + (r2/ds_t**2.0))**(-be_t)    
    r1=np.sqrt(r2)
    bt=r1
    spec_hst=Io*np.exp(-bn*((bt/Re)**(1./ns)-1))
    spec_t=spec_agn+spec_hst
    return spec_t        

def moffat_model0_s(theta, x_t=0, y_t=0, db_m=2.06, e_m=0.0, tht_m=0.0, beta=True, ellip=False):
    if ellip:
        if beta:
            At,dx,dy,ds_t,be_t,e_t,tht_t=theta
        else:
            At,dx,dy,ds_t,e_t,tht_t=theta
            be_t=db_m
        r2=tol.radi_ellip(x_t-dx,y_t-dy,e_t,tht_t)
        r2=r2**2.0 
    else:
        if beta:
            At,dx,dy,ds_t,be_t=theta
        else:
            At,dx,dy,ds_t=theta
            be_t=db_m
        r2=tol.radi_ellip(x_t-dx,y_t-dy,e_m,tht_m)
        r2=r2**2.0     
        #r2=(x_t-dx)**2.0+(y_t-dy)**2.0
    spec_agn=At*(1.0 + (r2/ds_t**2.0))**(-be_t)    
    spec_t=spec_agn
    return spec_t    

def moffat_model0(theta, x_t=0, y_t=0, be_t=2.064, e_m=0.0, tht_m=0.0, ellip=False):
    if ellip:
        At,dx,dy,Io,bn,Re,ns,ds_t,e_t,tht_t=theta
        r2=tol.radi_ellip(x_t-dx,y_t-dy,e_t,tht_t)
        r2=r2**2.0 
    else:
        At,dx,dy,Io,bn,Re,ns,ds_t=theta
        #e,th0=0,0
        #r2=tol.radi_ellip(x_t-dx,y_t-dy,e_m,tht_m)
        r2=(x_t-dx)**2.0+(y_t-dy)**2.0
    spec_agn=At*(1.0 + (r2/ds_t**2.0))**(-be_t)    
    r1=np.sqrt(r2)
    #tht=np.arctan2(y_t-dy,x_t-dx)
    bt=r1#ellipse2(tht,r1,e,th0)
    spec_hst=Io*np.exp(-bn*((bt/Re)**(1./ns)-1))
    spec_t=spec_agn+spec_hst
    return spec_t     

def moffat_model_st(theta, x_t=0, y_t=0, be_t=2.064, ds_t=3.47, e_m=0.0, tht_m=0.0, ellip=False):
    At,dx,dy=theta
    if ellip:
        e_t=e_m
        tht_t=tht_m
        r2=tol.radi_ellip(x_t-dx,y_t-dy,e_t,tht_t) 
        r2=r2**2.0
    else:
        r2=(x_t-dx)**2.0+(y_t-dy)**2.0
    spec_agn=At*(1.0 + (r2/ds_t**2.0))**(-be_t)    
    spec_t=spec_agn
    return spec_t      

def moffat_model(theta, x_t=0, y_t=0,be_t=2.064,ds_t=3.47):
    At,dx,dy,Io,bn,Re,ns=theta
    #e,th0=0,0
    r2=(x_t-dx)**2.0+(y_t-dy)**2.0
    spec_agn=At*(1.0 + (r2/ds_t**2.0))**(-be_t)    
    r1=np.sqrt(r2)
    #tht=np.arctan2(y_t-dy,x_t-dx)
    bt=r1#ellipse2(tht,r1,e,th0)
    spec_hst=Io*np.exp(-bn*((bt/Re)**(1./ns)-1))
    spec_t=spec_agn+spec_hst
    return spec_t          

def gaussian_model(theta, x_t=0, y_t=0):
    At,dx,dy,ds_t=theta
    spec_t=np.exp(-0.5*((((x_t-dx)/ds_t)**2.0)+((y_t-dy)/ds_t)**2.0))*At           
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
    tol.wfits_ext(out_fit,hlist)
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
    tol.wfits_ext(out_fit,hlist)
    tol.sycall('gzip -f '+out_fit)         