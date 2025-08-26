#!/usr/bin/env python
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np
import corner 
from astropy.io import fits
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import AGNdecomp.tools.tools as tol

def plot_models_maps(inMap,modelAGN,modelHST,samples,name='Name',path_out='',savefig=False,Labelvalues=[],logP=True,stl=False,smoth=True,sig=1.8,ofsval=-1):
    if stl:
        try:
            import MapLines.tools.tools as mptol
        except:
            print('No module MapLine installed. Please install it to use this function with pip install mapline')
            stl=False
    # Plot the original map, model AGN, model HST, residuals and corner plot
    nameO='Original_NAME'.replace('NAME',name)
    nameM='Model_NAME'.replace('NAME',name)
    nameR1='Residual1_NAME'.replace('NAME',name)
    nameR2='Residual2_NAME'.replace('NAME',name)
    cm=plt.cm.get_cmap('jet')
    lev=np.sqrt(np.arange(0.0,10.0,1.5)+0.008)/np.sqrt(10.008)*np.amax(inMap)
    fig, ax = plt.subplots(figsize=(6.8*1.1,5.5*1.2))
    if logP:
        ict=plt.imshow(np.log10(inMap),cmap=cm) 
    else:
        ict=plt.imshow(inMap,cmap=cm) 
    cbar=plt.colorbar(ict)
    ics=plt.contour(inMap,lev,colors='k',linewidths=1)            
    cbar.set_label(r"Relative Density")
    fig.tight_layout()
    if savefig:
        fig.savefig(path_out+nameO+'.pdf')
    else:
        plt.show()
    if stl:
        if logP:
            maxval=np.nanmax(np.log10(inMap)) 
        else:
            maxval=np.nanmax(inMap)
        minval=-0.1#1.7
        mptol.get_map_to_stl(inMap, nameid=nameO, path_out=path_out,sig=sig,smoth=smoth, pval=27, mval=0, border=True,logP=logP,ofsval=ofsval,maxval=maxval,minval=minval)    

    fig, ax = plt.subplots(figsize=(6.8*1.1,5.5*1.2))
    if logP:
        ict=plt.imshow(np.log10(modelAGN),cmap=cm,alpha=0.6) 
    else:
        ict=plt.imshow(modelAGN,cmap=cm,alpha=0.6) 
    cbar=plt.colorbar(ict)
    ics=plt.contour(modelAGN,lev,colors='k',linewidths=1)
    ics=plt.contour(inMap,lev,colors='red',linewidths=1)            
    cbar.set_label(r"Relative Density")
    fig.tight_layout()
    if savefig:
        fig.savefig(path_out+nameM+'.pdf')
    else:
        plt.show()
    if stl:
        mptol.get_map_to_stl(modelAGN, nameid=nameM, path_out=path_out,sig=sig,smoth=smoth, pval=27, mval=0, border=True,logP=logP,ofsval=ofsval,maxval=maxval,minval=minval)    
            
    fig, ax = plt.subplots(figsize=(6.8*1.1,5.5*1.2))
    if logP:
        ict=plt.imshow(np.log10(inMap-modelAGN),cmap=cm)
    else:
        ict=plt.imshow((inMap-modelAGN),cmap=cm)
    cbar=plt.colorbar(ict)
    ics=plt.contour((inMap-modelAGN),lev,colors='k',linewidths=1)
    cbar.set_label(r"Relative Density")
    fig.tight_layout()
    if savefig:
        fig.savefig(path_out+nameR1+'.pdf')
    else:
        plt.show()
    if stl:
        mptol.get_map_to_stl(inMap-modelAGN, nameid=nameR1, path_out=path_out,sig=sig,smoth=smoth, pval=27, mval=0, border=True,logP=logP,ofsval=ofsval,maxval=maxval,minval=minval)    
            
    fig, ax = plt.subplots(figsize=(6.8*1.1,5.5*1.2))
    if logP:
        ict=plt.imshow(np.log10(inMap-modelAGN-modelHST),cmap=cm) 
    else:
        ict=plt.imshow((inMap-modelAGN-modelHST),cmap=cm) 
    cbar=plt.colorbar(ict)
    ics=plt.contour((inMap-modelAGN-modelHST),lev,colors='k',linewidths=1)
    cbar.set_label(r"Relative Density")
    fig.tight_layout()
    if savefig:
        fig.savefig(path_out+nameR2+'.pdf')
    else:
        plt.show()
    if stl:
        mptol.get_map_to_stl(inMap-modelAGN-modelHST, nameid=nameR2, path_out=path_out,sig=sig,smoth=smoth, pval=27, mval=0, border=True,logP=logP,ofsval=ofsval,maxval=maxval,minval=minval) 
            
    labels = [*Labelvalues]
    fig = corner.corner(samples,show_titles=True,labels=labels,plot_datapoints=True,quantiles=[0.16, 0.5, 0.84],title_kwargs={"fontsize": 16},label_kwargs={"fontsize": 16})
    fig.set_size_inches(15.8*len(labels)/8.0, 15.8*len(labels)/8.0)
    fig.savefig(path_out+'corners_NAME.pdf'.replace('NAME',name))    


def plot_outputs(vt='',dir_cube_m='',name='Name',rad=1.5,smoth=False,ra='',dec='',basename='NAME.cube.fits.gz'):
    #PLOT CODE BASED ON the interactive ploting tool from the Bayesian AGN Decomposition Analysis for SDSS Spectra (BADASS3D) package
    #see https://github.com/remingtonsexton/BADASS3 and Sexton et al.2021
    outf1='Model_'+basename.replace('NAME',name+vt)
    outf2='Residual_'+basename.replace('NAME',name+vt)
    wave1,spec_mod,spec_modE=tol.extract_spec(outf1,dir_cube_m=dir_cube_m,rad=rad,sig=10,smoth=smoth,fErrr=True,ra=ra,dec=dec)
    wave2,spec_res,spec_resE=tol.extract_spec(outf2,dir_cube_m=dir_cube_m,rad=rad,sig=10,smoth=smoth,fErrr=True,ra=ra,dec=dec)
    spec0=spec_res+spec_mod

    fig = make_subplots(rows=1, cols=1, row_heights=(3,))
    
    fig.add_trace(go.Scatter( x = wave1, y = spec0 ,    mode="lines", line=go.scatter.Line(color="white", width=1), name='Input Spectra', legendrank=1, showlegend=True), row=1, col=1)
    fig.add_trace(go.Scatter( x = wave1, y = spec_resE, mode="lines", line=go.scatter.Line(color="#FE00CE"  , width=1), name='Noise Spectra',     legendrank=2, showlegend=True), row=1, col=1)
    fig.add_trace(go.Scatter( x = wave1, y = spec_res,  mode="lines", line=go.scatter.Line(color="lime", width=1), name='Host Galaxy Spectra',    legendrank=3, showlegend=True), row=1, col=1)
    fig.add_trace(go.Scatter( x = wave2, y = spec_mod,  mode="lines", line=go.scatter.Line(color="red", width=1), name='AGN Spectra',legendrank=4, showlegend=True), row=1, col=1)
        
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
    
    if ra != '':
        file_f=dir_cube_m+'NAME_R_ra_dec_bestfit.html'.replace('NAME',name).replace('R',str(rad)).replace('ra',ra).replace('dec',dec)
    else:
        file_f=dir_cube_m+'NAME_R_bestfit.html'.replace('NAME',name).replace('R',str(rad))
    fig.write_html(file_f,include_mathjax="cdn")
    fig.write_image(file_f.replace('.html','.pdf'))

    return    

def get_plotmap(plt,flux,vmax,vmin,pix=0.5,tit='flux',lab='[10^{-16}erg/s/cm^2/arcsec^2]',clb=False,logt=True):
    nx,ny=flux.shape
    max_f=vmax-(vmax-vmin)*0.05
    min_f=vmin+(vmax-vmin)*0.05
    cm=plt.cm.get_cmap('jet')
    plt.xlabel(r'$\Delta \alpha\ [arcsec]$',fontsize=22)
    plt.ylabel(r'$\Delta \delta\ [arcsec]$',fontsize=22)
    if logt:#,vmax=vmax,vmin=vmin
        ict=plt.imshow(flux,cmap=cm,origin='lower',extent=[-ny*pix/2.,ny*pix/2.,-nx*pix/2.,nx*pix/2.],alpha=0.6,aspect='auto',norm=colors.SymLogNorm(vmax=vmax,vmin=vmin,linthresh=10**-2))#,norm=colors.SymLogNorm(10**-2))#norm=LogNorm(0.2,7.0))#colors.SymLogNorm(10**-1))#50  norm=colors.SymLogNorm(10**-0.1)
    else:
        ict=plt.imshow(flux,cmap=cm,origin='lower',extent=[-ny*pix/2.,ny*pix/2.,-nx*pix/2.,nx*pix/2.],vmax=vmax,vmin=vmin,alpha=0.6,aspect='auto')#norm=LogNorm(0.2,7.0))#colors.SymLogNorm(10**-1))#50  norm=colors.SymLogNorm(10**-0.1)
    plt.xlim(-ny*pix/2,ny*pix/2)
    plt.ylim(-nx*pix/2,nx*pix/2)
    if clb:
        return ict


def plot_mapmodelress(fig_path='',lab='[10^{-16}erg/s/cm^2/arcsec^2]',basefigname='maps_NAME',sumc=False,scale=0,sb=False,fwcs=False,logs=False,zerofil=False,valz=None,maxmin=[],vt='',name='Name',basename='NAME.cube.fits.gz',path='',hd=0,indx=0,indx2=None,scalef=1.0,facs=1,av=[0.15,0.2,0.12,0.03]):
    outf1='Model_'+basename.replace('NAME',name+vt)
    outf2='Residual_'+basename.replace('NAME',name+vt)
    

    [data1,hdr1]=fits.getdata(path+outf1, hd, header=True)
    [data2,hdr2]=fits.getdata(path+outf1, hd, header=True)
    data0=data1+data2
    try:
        dx=np.sqrt((hdr1['CD1_1'])**2.0+(hdr1['CD1_2'])**2.0)*3600.0
        dy=np.sqrt((hdr1['CD2_1'])**2.0+(hdr1['CD2_2'])**2.0)*3600.0
    except:
        try:
            dx=hdr1['CD1_1']*3600.0
            dy=hdr1['CD2_2']*3600.0
        except:
            try:
                dx=hdr1['CDELT1']*3600.
                dy=hdr1['CDELT2']*3600.
            except:
                dx=hdr1['PC1_1']*3600.
                dy=hdr1['PC2_2']*3600.
    pix=(np.abs(dx)+np.abs(dy))/2.0
    if sumc:
        try:
            map_val0=np.nansum(data0[indx,:,:],axis=0)*scalef
            map_val1=np.nansum(data1[indx,:,:],axis=0)*scalef
            map_val2=np.nansum(data2[indx,:,:],axis=0)*scalef
        except:
            print('It is not possible to integrate the data cube within the indexes provided, we will integrate all the cube')
            map_val0=np.nansum(data0,axis=0)*scalef
            map_val1=np.nansum(data1,axis=0)*scalef
            map_val2=np.nansum(data2,axis=0)*scalef
    else:
        map_val0=data0[indx,:,:]*scalef
        map_val1=data1[indx,:,:]*scalef
        map_val2=data2[indx,:,:]*scalef
    if indx2 != None:
        val20=data0[indx2,:,:]*scalef
        val21=data1[indx2,:,:]*scalef
        val22=data2[indx2,:,:]*scalef
        map_val0=map_val0/val20
        map_val1=map_val0/val21
        map_val2=map_val0/val22
    if zerofil:
        if valz == None:
            map_val0[np.where(map_val0 == 0)]=np.nan
            map_val1[np.where(map_val1 == 0)]=np.nan
            map_val2[np.where(map_val2 == 0)]=np.nan
        else:
            map_val0[np.where(map_val0 <= valz)]=np.nan
            map_val1[np.where(map_val1 <= valz)]=np.nan
            map_val2[np.where(map_val2 <= valz)]=np.nan
    if sb:
        map_val0=map_val0/pix**2
        map_val1=map_val1/pix**2
        map_val2=map_val2/pix**2
    if logs:
        map_val0=np.log10(map_val0)
        map_val1=np.log10(map_val1)
        map_val2=np.log10(map_val2)

    if len(maxmin) > 0:
        vmax=maxmin[1]
        vmin=maxmin[0]
    else:
        vmax=np.nammax(map_val0)*1.1
        vmin=0.001

    
    facx=0.99
    facy=0.99
    nx=3
    ny=1
    dx1=av[0]/facx
    dx2=av[1]/facx
    dy1=av[2]/facy
    dy2=av[3]/facy
    dx=(1.0-(dx1+dx2))/float(1.0)
    dy=(1.0-(dy1+dy2))/float(1.0)
    dx1=dx1/(1.0+(nx-1)*dx)
    dx2=dx2/(1.0+(nx-1)*dx)
    dy1=dy1/(1.0+(ny-1)*dy)
    dy2=dy2/(1.0+(ny-1)*dy)
    dx=(1.0-(dx1+dx2))/float(nx)
    dy=(1.0-(dy1+dy2))/float(ny)
    xfi=6*nx*facx*facs#6
    yfi=6*ny*facy#5.5
    fig = plt.figure(figsize=(xfi,yfi))
    pro1=[0,1,2]
    pro2=[0,0,0]
    ax = fig.add_axes([dx1+pro1[0]*dx, dy1+pro2[0]*dy, dx, dy])
    flux=map_val0
    get_plotmap(plt,flux,vmax,vmin,pix=0.499,tit='flux',lab=lab,logt=logs)
    plt.text(0.05, 0.96, r'Input', fontsize=20, va='center',transform=ax.transAxes)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))

    ax = fig.add_axes([dx1+pro1[1]*dx, dy1+pro2[1]*dy, dx, dy])
    flux=map_val1
    get_plotmap(plt,flux,vmax,vmin,pix=0.499,tit='flux',lab=lab,logt=logs)
    plt.text(0.05, 0.96, r'AGN Model', fontsize=20, va='center',transform=ax.transAxes)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_ylabel('').set_visible(False)
    plt.setp( ax.get_yticklabels(), visible=False)
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))

    ax = fig.add_axes([dx1+pro1[2]*dx, dy1+pro2[2]*dy, dx, dy])
    flux=map_val2
    sc=get_plotmap(plt,flux,vmax,vmin,pix=0.499,tit='flux',lab=lab,clb=True,logt=logs)
    plt.text(0.05, 0.96, r'Residual', fontsize=20, va='center',transform=ax.transAxes)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.set_ylabel('').set_visible(False)
    plt.setp( ax.get_yticklabels(), visible=False)
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))
    ax2 = fig.add_axes([dx1+pro1[2]*dx+dx, dy1+pro2[2]*dy, dx*0.05, dy]) 
    ax2.tick_params(axis='both', which='major', labelsize=18)
    cbar=plt.colorbar(sc, cax=ax2, orientation="vertical")#,ticks=[0.02,0.06,0.10,0.20])
    cbar.set_label(r'$'+lab+'$',fontsize=20)

    plt.savefig(fig_path+basefigname.replace('NAME',name)+'.pdf')
    plt.show()
    plt.close()