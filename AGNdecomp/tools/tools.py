#!/usr/bin/env python
import glob, os,sys,timeit
import numpy as np
from scipy.special import gamma, gammaincinv, gammainc
from scipy.ndimage.filters import gaussian_filter1d as filt1d
import os.path as ptt
import yaml
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.coordinates import ICRS, Galactic, FK4, FK5
from astropy import units as u
from astropy.wcs.utils import skycoord_to_pixel
from astropy.wcs import WCS

def get_priorsvalues(filename):
    """
    Reads the priors values from a YAML file.
    """
    data_lines=read_config_file(filename)
    if data_lines:
        n_types=len(data_lines['types'])
        #pac=['AoN','dvoN','fwhmoN']
        #pacL=[r'$A_{N}$',r'$\Delta v_{N}$',r'$FWHM_{N}$']
        #pacH=['N_Amplitude','N_Velocity','N_FWHM']
        #waves0=[]
        model_name=[]
        model_pars=[]
        #vals0=[]
        #vals=[]
        #valsL=[]
        #valsH=[]
        #fac0=[]
        #facN0=[]
        #velfac0=[]
        #velfacN0=[]
        #fwhfac0=[]
        #fwhfacN0=[]
        for i in range(0, n_types):
            parameters=data_lines['types'][i]
            npar=len(parameters)
            kesys=parameters.keys()
            mpars={}
            for j in range(0, npar):
                if keys[j] == 'name':
                    model_name.extend([parameters['name']])
                mpars[keys[j]]=parameters[keys[j]]
            print('Model parameters:',mpars)
            sys.exit()
            try:
                facN0.extend([parameters['fac_Name']])
                fac0.extend([parameters['fac']])
                facp=True
            except:
                facN0.extend(['NoNe'])
                fac0.extend([None])
                facp=False
            try:
                velfacN0.extend([parameters['vel_Name']])
                velfac0.extend([parameters['velF']])
                velfacp=True
            except:
                velfacN0.extend(['NoNe'])
                velfac0.extend([None])
                velfacp=False    
            try:
                fwhfacN0.extend([parameters['fwh_Name']])
                fwhfac0.extend([parameters['fwhF']])
                fwhfacp=True
            except:
                fwhfacN0.extend(['NoNe'])
                fwhfac0.extend([None])
                fwhfacp=False    
            inr=0    
            for a in pac:
                val_t=a.replace('N',str(i))
                val_tL=pacL[inr].replace('N',names0[i])
                val_tH=pacH[inr].replace('N',names0[i])
                if 'AoN' in a:
                    if facp == False:
                        vals.extend([val_t])
                        valsL.extend([val_tL])
                elif 'dvoN' in a:
                    if velfacp == False:
                        vals.extend([val_t])
                        valsL.extend([val_tL])        
                elif 'fwhmoN' in a:
                    if fwhfacp == False:
                        vals.extend([val_t])
                        valsL.extend([val_tL])
                else:    
                    vals.extend([val_t])
                    valsL.extend([val_tL])
                valsH.extend([val_tH])
                vals0.extend([val_t])    
                inr=inr+1
        valsp=data_lines['priors']
        Inpvalues=[]
        Infvalues=[]
        Supvalues=[]
        for valt in vals:
            try:
                Inpvalues.extend([valsp[valt]])
            except:
                print('The keyword '+valt+' is missing in the line config file')
                return
            try:
                Infvalues.extend([valsp[valt.replace('o','i')]])
            except:
                print('The keyword '+valt.replace('o','i')+' is missing in the line config file')
                return
            try:
                Supvalues.extend([valsp[valt.replace('o','s')]])
            except:
                print('The keyword '+valt.replace('o','s')+' is missing in the line config file')
                return
    else:
        print('No configuration line model file')
        return



    data_lines = read_config_file(filename)
    if data is None:
        return None
    if 'priors' in data:
        priors = data['priors']
        if 'values' in priors:
            return priors['values']
        else:
            print('No values found in priors')
            return None
    else:
        print('No priors found in the configuration file')
        return None

def read_config_file(file):
    try:
        with open(file, 'r') as stream:
            try:
                data = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        return data
    except:
        print('Config File not found')
        return None

def get_spectra(vect_pyqsfit,wave):
    ct=299792.458
    Fe_flux,PL_norm,PL_slop=vect_pyqsfit[0:3]
    lines=vect_pyqsfit[3:len(vect_pyqsfit)-1]
    #fwhm_hb,   sigma_hb,   ew_hb,   peak_hb,   area_hb,   fwhm_ha,   sigma_ha,   ew_ha,   peak_ha,   area_ha,\
    #fwhm_hbn,  sigma_hbn,  ew_hbn,  peak_hbn,  area_hbn,  fwhm_oiii1,sigma_oiii1,ew_oiii1,peak_oiii1,area_oiii1,\
    #fwhm_oiii2,sigma_oiii2,ew_oiii2,peak_oiii2,area_oiii2,fwhm_han,  sigma_han,  ew_han,  peak_han,  area_han,\
    #fwhm_nii1, sigma_nii1, ew_nii1, peak_nii1, area_nii1, fwhm_nii2, sigma_nii2, ew_nii2, peak_nii2, area_nii2,\
    #fwhm_sii1, sigma_sii1, ew_sii1, peak_sii1, area_sii1, fwhm_sii2, sigma_sii2, ew_sii2, peak_sii2, area_sii2,\
    #fwhm_oi,   sigma_oi,   ew_oi,   peak_oi,   area_oi,   fwhm_mg,   sigma_mg,   ew_mg,   peak_mg,   area_mg,\
    #fwhm_hr,   sigma_hr,   ew_hr,   peak_hr,   area_hr=theta
    nl=np.int(len(lines)/5)
    spec=0.0
    for i in range(0, nl):
        #sigma=lines[1+i*5]
        At=lines[4+i*5]
        dx=lines[3+i*5]
        fwhm=lines[0+i*5]
        if At > 0:
            sigma=fwhm/ct*dx/(2.0*np.sqrt(2.0*np.log(2.0)))
            Amp=At/(np.sqrt(2.0*np.pi)*sigma)
            theta=[Amp,dx,sigma]
            print(fwhm,dx,sigma,Amp)
            spec=gaussian_single(theta,x_t=wave)+spec
    return spec

def gaussian_single(theta, x_t=0):
    At,dx,ds_t=theta
    spec_t=np.exp(-0.5*((x_t-dx)/ds_t)**2.0)*At    
    return spec_t   

def b_s(n):
    # Normalisation constant
    return gammaincinv(2*n, 0.5)

def create_sersic_function(Ie, re, n):
    # Not required for integrals - provided for reference
    # This returns a "closure" function, which is fast to call repeatedly with different radii
    neg_bn = -b_s(n)
    reciprocal_n = 1.0/n
    f = neg_bn/re**reciprocal_n
    def sersic_wrapper(r):
        return Ie * exp(f * r ** reciprocal_n - neg_bn)
    return sersic_wrapper

def sersic_lum(Ie, re, n):
    # total luminosity (integrated to infinity)
    bn = b_s(n)
    g2n = gamma(2*n)
    return Ie * re**2 * 2*pi*n * exp(bn)/(bn**(2*n)) * g2n

def sersic_enc_lum(r, Ie, re, n):
    # luminosity enclosed within a radius r
    x = b_s(n) * (r/re)**(1.0/n)
    return sersic_lum(Ie, re, n) * gammainc(2*n, x)

def sycall(comand):
    import os
    linp=comand
    os.system(comand)

def sycallo(comand):
    import os
    out=os.popen(comand, 'r')
    line=out.readline()
    return line    

def conv(xt,ke=2.5):
    nsf=len(xt)
    krn=ke
    xf=filt1d(xt,ke)
    return xf

def ellipse2(th,r,e,th0):
    b=r/(np.sqrt(1-e**2.0)/np.sqrt(1-e**2.0*np.sin(th-th0)**2.0))
    return b

def radi_ellip(dx,dy,e,tht):
    b=((1-(e*np.sin(np.arctan2(dy,dx)-tht*np.pi/180.))**2.0)*(dx**2.0+dy**2.0)/(1-e**2.0))**(1/2.0)
    return b


def get_somoth_val(name,dir='./',sigma=20,sp=10,val=10,out_p=False,deg=5,tp='',convt=False,mask_val=[]):
    if sp > 0:
        spt='_sp'+str(int(sp))
    else:
        spt=''
    file=dir+name+'_moffat'+spt+tp+'.csv'
    f=open(file,'r')
    #flux=[]
    wave=[]
    val_v=[]
    for line in f:
        if not 'WAVE' in line:
            data=line.replace('\n','').split(',')
            data=list(filter(None,data))
            wave.extend([float(data[0])])
            #flux.extend([np.float(data[1])])
            val_v.extend([float(data[val])])
    f.close()
    #flux=np.array(flux)
    wave=np.array(wave)
    val_v=np.array(val_v)
    #flux=conv(flux,ke=sigma)
    val_vt=step_vect_Mean(val_v,sp=20,pst=True,mask_val=mask_val)
    if sigma==0:
        val_s=val_vt
    else:
        val_s=conv(val_vt,ke=sigma)
    if convt:
        val_c,p=continum_fit(wave,val_s,deg=deg)
    else:    
        val_c,p=continum_fit(wave,val_vt,deg=deg)
    if out_p:
        file2=dir+name+'_moffat'+spt+tp+'_smoth_'+str(int(np.round(sigma)))+'_val'+str(val)+'.csv'
        f=open(file2,'w')
        f.write('WAVE , Val, Val_s, Val_c \n')
        for i in range(0, len(wave)):
            f.write(str(wave[i])+' , '+str(val_v[i])+' , '+str(val_s[i])+' , '+str(val_c[i])+' \n')
        f.close()
    if sigma == 0:
        return val_v
    else:    
        return p
    
def continum_fit(wave,flux,deg=5):
    nt=np.where(flux > 0)[0]
    x=wave[nt]
    y=flux[nt]
    coef=np.polyfit(x, y, deg)
    p = np.poly1d(coef)
    cont=p(wave)
    #print(coef)
    return cont,p

def step_vect_Mean(flux,sp=20,pst=True,mask_val=[]):
    nz=len(flux)
    flux_t=np.copy(flux)
    flux_0=np.copy(flux)
    if len(mask_val) > 0:
        nt=np.where((flux_0 >= (mask_val[0]-mask_val[1])) & (flux_0 <= (mask_val[0]+mask_val[1])))[0]
        if len(nt) > 0:
           flux_0[nt]=np.nan
    for i in range(0, nz):
        i0=int(i-sp/2.0)
        i1=int(i+sp/2.0)
        if i1 > nz:
            i1=nz
        if i0 > nz:
            i0=int(nz-sp)
        if i0 < 0:
            i0=0
        if i1 < 0:
            i1=sp   
        if pst:
            lt0=np.nanpercentile(flux_0[i0:i1],50)
            flux_t[i]=lt0
        else:
            flux_t[i]=np.nanmean(flux_0[i0:i1])
    return flux_t    

def step_vect(flux,sp=20,pst=True):
    nz=len(flux)
    flux_t=np.copy(flux)
    for i in range(0, nz):
        i0=int(i-sp/2.0)
        i1=int(i+sp/2.0)
        if i1 > nz:
            i1=nz
        if i0 > nz:
            i0=int(nz-sp)
        if i0 < 0:
            i0=0
        if i1 < 0:
            i1=sp   
        #print(i0,i1) 
        if pst:
            lts=np.nanpercentile(flux[i0:i1],78)
            lt0=np.nanpercentile(flux[i0:i1],50)
            lti=np.nanpercentile(flux[i0:i1],22)
            val=(np.abs(lts-lt0)+np.abs(lti-lt0))/2.0
            flux_t[i]=val#mean
        else:
            flux_t[i]=np.nanstd(flux[i0:i1])#mean
    return flux_t

def get_error_vec(name,dir='./',sigma=20,sp=1,out_p=True,deg=5,zt=0,sptt=50,vt=''):
    if sp > 0:
        spt='_sp'+str(int(sp))
    else:
        spt=''
    file=dir+name+'_moffat'+spt+vt+'.csv'
    f=open(file,'r')
    flux=[]
    wave=[]
    flux_t=[]
    for line in f:
        if not 'WAVE' in line:
            data=line.replace('\n','').split(',')
            data=list(filter(None,data))
            wave.extend([float(data[0])])
            flux.extend([float(data[1])])
            flux_t.extend([float(data[2])])
    f.close()
    flux=np.array(flux)
    wave=np.array(wave)
    flux_t=np.array(flux_t)
    wave_z=np.copy(wave)/(1+zt)
    flux_sm=conv(flux,ke=sigma)
    flux_r=flux_sm-flux
    #flux_r[np.where(np.abs(flux_r) > 1)]=0
    flux_e=step_vect(flux_r,sp=sptt,pst=True)
    #flux_r[np.where(flux_r < -1)]=0
    #flux_e=np.abs(flux_r)
    #flux_e=conv(flux_e,ke=sigma*2)
    #val_c,p=continum_fit(wave,val_v,deg=deg)
    if out_p:
        file2=dir+name+'_moffat'+spt+'_error_sig'+str(int(np.round(sigma)))+'.csv'
        f=open(file2,'w')
        f.write('WAVE , Flux, Flux_s, Flux_r, Flux_e, WAVE_z, Flux_T \n')
        for i in range(0, len(wave)):
            f.write(str(wave[i])+' , '+str(flux[i])+' , '+str(flux_sm[i])+' , '+str(flux_r[i])+' , '+str(flux_e[i])+' , '+str(wave_z[i])+' , '+str(flux_t[i])+' \n')
        f.close()
    #return p

def get_error_vec_simple(name,dir='./',sigma=20,sp=1,out_p=True,deg=5,zt=0,sptt=50,vt=''):
    if sp > 0:
        spt='_sp'+str(int(sp))
    else:
        spt=''
    file=dir+name+'.csv'
    f=open(file,'r')
    flux=[]
    wave=[]
    for line in f:
        if not 'Wave' in line:
            data=line.replace('\n','').split(',')
            data=list(filter(None,data))
            wave.extend([float(data[0])])
            flux.extend([float(data[1])])
    f.close()
    flux=np.array(flux)
    wave=np.array(wave)
    wave_z=np.copy(wave)/(1+zt)
    flux_sm=conv(flux,ke=sigma)
    flux_r=flux_sm-flux
    flux_e=step_vect(flux_r,sp=sptt,pst=True)*5
    if out_p:
        file2=dir+name+'_moffat'+spt+'_error_sig'+str(int(np.round(sigma)))+'.csv'
        f=open(file2,'w')
        f.write('WAVE , Flux, Flux_s, Flux_r, Flux_e, WAVE_z, Flux_T \n')
        for i in range(0, len(wave)):
            f.write(str(wave[i])+' , '+str(flux[i])+' , '+str(flux_sm[i])+' , '+str(flux_r[i])+' , '+str(flux_e[i])+' , '+str(wave_z[i])+' , '+str(0)+' \n')
        f.close()    

def get_beta(val):
    bet_t=np.exp(np.arange(1000)/(999.)*(np.log(1e5)-np.log(1.1))+np.log(1.1))
    ft=(2**(1/bet_t)-1)*(bet_t-1)
    bet_f=interp1d(ft,bet_t,bounds_error=False,fill_value=0)(val)
    return bet_f

def extract_spec(filename,dir_cube_m='',ra='',dec='',rad=1.5,sig=10,smoth=False,avgra=False,dered=False,fErrr=False):
    file=dir_cube_m+filename

    [cube0, hdr0]=fits.getdata(file, 0, header=True)
    nz,nx,ny=cube0.shape
    try:
        dx=np.sqrt((hdr0['CD1_1'])**2.0+(hdr0['CD1_2'])**2.0)*3600.0
        dy=np.sqrt((hdr0['CD2_1'])**2.0+(hdr0['CD2_2'])**2.0)*3600.0
    except:
        try:
            dx=hdr0['CD1_1']*3600.0
            dy=hdr0['CD2_2']*3600.0
        except:
            dx=hdr0['CDELT1']*3600.
            dy=hdr0['CDELT2']*3600.
    pix=(np.abs(dx)+np.abs(dy))/2.0    
    try:
        [cube1, hdr1]=fits.getdata(file, 1, header=True)
        Error=True
        #print("Using error cube")
    except:
        Error=False
        #print("Calculationg errors")
    if fErrr:
        Error=False

    if ra != '':
        sky1=SkyCoord(ra+' '+dec,frame=FK5, unit=(u.hourangle,u.deg))
        val1=sky1.ra.deg
        val2=sky1.dec.deg
        wcs = WCS(hdr0)
        wcs=wcs.celestial
        ypos,xpos=skycoord_to_pixel(sky1,wcs)
    else:
        xpos=ny/2.0
        ypos=nx/2.0
        
    radis=np.zeros([nx,ny])
    for i in range(0, nx):
        for j in range(0, ny):
            x_n=i-xpos
            y_n=j-ypos
            r_n=np.sqrt((y_n)**2.0+(x_n)**2.0)*pix
            radis[i,j]=r_n
    single_T=np.zeros(nz)
    single_ET=np.zeros(nz)
    nt=np.where(radis <= rad)
    if avgra:
        ernt=len(nt[0])
    else:
        ernt=1.0
    for i in range(0, nz):
        tmp=cube0[i,:,:]
        tmp[np.where(tmp <= 0)]=np.nan
        if avgra:
            single_T[i]=np.nanmean(tmp[nt])
        else:
            single_T[i]=np.nansum(tmp[nt])
        if Error:
            tmp=cube1[i,:,:]
            print(np.nansum(tmp[nt])/ernt)
            single_ET[i]=np.nansum(tmp[nt])/ernt
    
    if Error == False:
        single_ET=step_vect(single_T,sp=60)*2.0*np.sqrt(2.0*np.log10(2.0))
    
        
    crpix=hdr0["CRPIX3"]
    try:
        cdelt=hdr0["CD3_3"]
    except:
        cdelt=hdr0["CDELT3"]
    crval=hdr0["CRVAL3"]
    wave_f=(crval+cdelt*(np.arange(nz)+1-crpix))
    
    
    #if dered:
    #    ra_t=sky1.ra.deg
    #    dec_t=sky1.dec.deg
    #    single_T,single_ET=DeRedden(wave_f, single_T, single_ET, ra_t, dec_t)
    
    if smoth:
        single_T=conv(single_T,ke=sig)
    
    return wave_f,single_T,single_ET

def plot_outputs(vt='',dir_cube_m='',name='Name',rad=1.5,smoth=False,ra='',dec=''):
    outf1='Model_NAME.cube.fits.gz'.replace('NAME',name+vt)
    outf2='Residual_NAME.cube.fits.gz'.replace('NAME',name+vt)
    wave1,spec_mod,spec_modE=extract_spec(outf1,dir_cube_m=dir_cube_m,rad=rad,sig=10,smoth=smoth,fErrr=True,ra=ra,dec=dec)
    wave2,spec_res,spec_resE=extract_spec(outf2,dir_cube_m=dir_cube_m,rad=rad,sig=10,smoth=smoth,fErrr=True,ra=ra,dec=dec)
    spec0=spec_res+spec_mod

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    fig = make_subplots(rows=1, cols=1, row_heights=(3,))
    
    fig.add_trace(go.Scatter( x = wave1, y = spec0 ,    mode="lines", line=go.scatter.Line(color="white", width=1), name='Input Spectra', legendrank=1, showlegend=True), row=1, col=1)
    fig.add_trace(go.Scatter( x = wave1, y = spec_resE, mode="lines", line=go.scatter.Line(color="#FE00CE"  , width=1), name='Noise Spectra',     legendrank=2, showlegend=True), row=1, col=1)
    fig.add_trace(go.Scatter( x = wave1, y = spec_res,  mode="lines", line=go.scatter.Line(color="lime", width=1), name='Host Galaxy Spectra',    legendrank=3, showlegend=True), row=1, col=1)
    fig.add_trace(go.Scatter( x = wave2, y = spec_mod,  mode="lines", line=go.scatter.Line(color="red", width=1), name='AGN Spectra',legendrank=4, showlegend=True), row=1, col=1)
        
        
    #fig.add_hline(y=0.0, line=dict(color="gray", width=2), row=1, col=1)  
    #fig.add_trace(go.Scatter( x = wave1, y = spec_res, mode="lines", line=go.scatter.Line(color="white"  , width=1), name="Residuals", showlegend=False), row=2, col=1)
    if ra != '':
        post=' at '+ra+' '+dec
    else:
        post=''
    fig.update_layout(
        autosize=False,
        width=1500,
        height=600,
        margin=dict(
            l=100,
            r=100,
            b=100,
            t=100,
            pad=1
        ),
        title= name+' central '+str(rad)+' arcsec aperture'+post,
        font_family="Times New Roman",
        font_size=16,
        font_color="white",
        legend_title_text="Components",
        legend_bgcolor="black",
        paper_bgcolor="black",
        plot_bgcolor="black",
    )
    fig.update_xaxes(title=r"$\Large\rm{Wavelength}\;\left[Å\right]$", linewidth=0.5, linecolor="gray", mirror=True, 
                     gridwidth=1, gridcolor="#222A2A", zerolinewidth=2, zerolinecolor="#222A2A",
                     row=1, col=1)
    fig.update_yaxes(title=r"$\Large\rm{Density}\;{Flux}\;\left[10^{16}\rm{erg}\;\rm{cm}^{-2}\;\rm{s}^{-1}\;Å^{-1}\right]$", linewidth=0.5, linecolor="gray",  mirror=True,
                     gridwidth=1, gridcolor="#222A2A", zerolinewidth=2, zerolinecolor="#222A2A",
                     row=1, col=1)
        
    fig.update_xaxes(matches='x')
    # fig.update_yaxes(matches='y')
    # fig.show()
    
    if ra != '':
        file_f=dir_cube_m+'NAME_R_ra_dec_bestfit.html'.replace('NAME',name).replace('R',str(rad)).replace('ra',ra).replace('dec',dec)
    else:
        file_f=dir_cube_m+'NAME_R_bestfit.html'.replace('NAME',name).replace('R',str(rad))
    fig.write_html(file_f,include_mathjax="cdn")
    fig.write_image(file_f.replace('.html','.pdf'))

    return
