# Copyright (c) 2016-2018, The University of Texas at Austin 
# & University of California, Merced.
#
# All Rights reserved.
# See file COPYRIGHT for details.
#
# This file is part of the hIPPYlib library. For more information and source code
# availability see https://hippylib.github.io.
#
# hIPPYlib is free software; you can redistribute it and/or modify it under the
# terms of the GNU General Public License (as published by the Free
# Software Foundation) version 2.0 dated June 1991.

from __future__ import absolute_import, division, print_function

import numpy as np

def _acorr(mean_free_samples, lag, norm = 1):
    #http://stackoverflow.com/questions/14297012/estimate-autocorrelation-using-python    
    return (mean_free_samples[:mean_free_samples.size-lag]*mean_free_samples[lag:]).ravel().mean() / norm

def _acorr_vs_lag(samples, max_lag):
    mean = samples.mean()
    mean_free_samples = samples - mean
    
    norm = _acorr(mean_free_samples, 0)
    
    lags = np.arange(0,max_lag+1)
    acorrs = np.ones(max_lag+1)
    
    for lag in lags[1:]:
        acorrs[lag] = _acorr(mean_free_samples, lag, norm)
        
    return lags, acorrs

def integratedAutocorrelationTime(samples, max_lag = None):
    assert len(samples.shape) == 1
    
    if max_lag is None:
        max_lag = samples.shape[0] // 10
        
    lags, acorrs = _acorr_vs_lag(samples, max_lag)
    
    iact = 1. + 2.* np.max( acorrs.cumsum() )
    
    return iact, lags, acorrs