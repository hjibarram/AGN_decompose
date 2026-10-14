#!/usr/bin/env python
import numpy as np
import AGNdecomp.tools.models as mod

'''                   
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
'''


def lnlike_multmodel(theta, spec, specE, x_t, y_t,
                     valsI, Namevalues, Namemodel, Usermods,
                     datapsf, max_oversub):
    """
    Compute the Gaussian log-likelihood of the complete model.

    An optional upper constraint prevents the total model
    from exceeding the observed data by more than a
    specified number of observational uncertainties.

    Parameters
    ----------
    theta : array_like
        Model parameters.
    spec : ndarray
        Observed data.
    specE : ndarray
        Observational uncertainties.
    x_t, y_t : ndarray
        Spatial coordinate grids.
    valsI : dict
        Fixed model parameters.
    Namevalues : list
        Names of fitted parameters.
    Namemodel : str
        Model identifier.
    Usermods : list
        External model configuration.
    datapsf : object
        Arbitrary user-defined data passed unchanged
        to the model.
    max_oversub : float or None
        Maximum model overestimation in sigma units.
        None disables the constraint.
        0.0 forbids any overestimation.

    Returns
    -------
    float
        Gaussian log-likelihood or -np.inf when the
        constraint is violated.
    """
    model = mod.multi_model(theta, valsI, Namevalues, Namemodel, 
        Usermods, x_t=x_t, y_t=y_t, datapsf=datapsf)

    valid = (np.isfinite(spec) & np.isfinite(specE) &
        (specE > 0))

    if not np.any(valid):
        return -np.inf
    # Reject models with invalid predictions.
    if not np.all(np.isfinite(model[valid])):
        return -np.inf
    # Optional total-model overestimation constraint.
    if max_oversub is not None:
        if max_oversub < 0:
            raise ValueError("max_oversub must be >= 0")
        if np.any(
            model[valid] >
            spec[valid] + max_oversub * specE[valid]):
            return -np.inf
    # Original Gaussian likelihood.
    chi2 = np.sum(
        ((spec[valid] - model[valid]) / specE[valid])**2)

    return -0.5 * chi2


def lnprior_multmodel(theta, Infvalues, Supvalues):
    """
    Evaluate uniform priors for the model parameters.

    Parameters
    ----------
    theta : array_like
        Model parameters.
    Infvalues : array_like
        Lower bounds for each parameter.
    Supvalues : array_like
        Upper bounds for each parameter.

    Returns
    -------
    float
        0.0 if all parameters are within their bounds,
        otherwise -np.inf.
    """
    boolf = True
    for i in range(len(theta)):
        bool1 = theta[i] <= Supvalues[i]
        bool2 = theta[i] >= Infvalues[i]
        boolf = (bool1 & bool2) & boolf
    if boolf:
        return 0.0
    else:
        return -np.inf


def lnprob_multmodel(theta, spec, specE, x_t, y_t,
                     valsI, Infvalues, Supvalues,
                     Namevalues, Namemodel, Usermods,
                     datapsf, max_oversub):
    """
    Compute the log-posterior probability with an optional
    constraint on total model overestimation.

    Parameters
    ----------
    theta : array_like
        Model parameters.
    spec : ndarray
        Observed data.
    specE : ndarray
        Observational uncertainties.
    x_t, y_t : ndarray
        Spatial coordinate grids.
    valsI : dict
        Fixed model parameters.
    Infvalues, Supvalues : array_like
        Lower and upper bounds for the model parameters.
    Namevalues : list
        Names of fitted parameters.
    Namemodel : str
        Model identifier.
    Usermods : list
        External model configuration.
    datapsf : object
        Arbitrary user-defined data passed to the model.
        Can be None.
    max_oversub : float or None
        Maximum allowed model overestimation in sigma units.
        None disables the constraint.

    Returns
    -------
    float
        Log-posterior probability, or -np.inf if the
        parameter bounds or model constraints are violated.
    """
    lp = lnprior_multmodel(theta, Infvalues, Supvalues)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike_multmodel(
        theta, spec, specE, x_t, y_t,
        valsI, Namevalues, Namemodel, Usermods,
        datapsf, max_oversub)