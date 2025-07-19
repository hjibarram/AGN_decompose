import numpy as np
import AGNdecomp.tools.tools as tol
from astropy.io import fits
#This is an example of how define the user costum external function to model the PSF and Host Galaxies
#Usefull predefined function:
#radi_ellip(x,y,e,th) --> this function returns the radial distance from the center of an ellipse with eccentricity e and PA angle th, values of e=0, th=0 returns a circle

def ExternFunctionName(pars, x_t=0, y_t=0):
    # This is an example of a single Moffat model
    At=pars['At']
    ds=pars['alpha']
    be=pars['beta']
    dx=pars['xo']
    dy=pars['yo']
    es=pars['ellip']
    th=pars['theta']
    r1=tol.radi_ellip(x_t-dx,y_t-dy,es,th)
    spec_t=At*(1.0 + (r1**2.0/ds**2.0))**(-be)
    return spec_t

def ExternFunctionName_flux_psf(pars, x_t=0, y_t=0,):
    psf=pars['alpha']*2.0*np.sqrt(2.0**(1./pars['beta'])-1)
    ft_fit=np.pi*pars['alpha']**2.0*pars['At']/(pars['beta']-1.0)
    return psf, ft_fit