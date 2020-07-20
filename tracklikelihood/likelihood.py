#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  likelihood.py
#  
#  Copyright 2020 Florian Thomas <flthomas@students.uni-mainz.de>
#  
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#  
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#  
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#  
#  

import numpy as np
import matplotlib.pyplot as plt
from . import track
from . import spectrogram as sp
from scipy.stats import expon
import time

def get_likelihood_vals(spec_in, t, f, track):
    intensity_0 = track.get_signal(t, f)
    probabilities = expon.pdf(spec_in.spec, loc=intensity_0)
    
    return np.log(np.maximum(probabilities, 1e-30))#1e-100))

def get_likelihood(spec_in, t, f, track):
    
    return - np.sum(get_likelihood_vals(spec_in, t, f, track))
    
def plot_hypothesis(spec_in, t, f, track):
	
	spectrogram_hypothesis = sp.Spectrogram.from_tracks(t, f, [track], add_noise=False)
	print('Track hypothesis')
	spectrogram_hypothesis.plot()
	
	spec_out = get_likelihood_vals(spec_in, t, f, track)
	
	print('Likelihood values')
	sp.plot_spectrogram(spec_out, t, f)
    
def scan_likelihood(scan_vals, likelihood_function):
    
    lh = np.empty(shape=scan_vals.shape)
    for i, val in enumerate(scan_vals):
        lh[i] = likelihood_function(val)
    
    return lh
    
def scan_likelihood_2d(scan_vals_x, scan_vals_y, likelihood_function):
    #lh = []
    lh = np.empty(shape=[scan_vals_x.shape[0], scan_vals_y.shape[0]])
    for i, x in enumerate(scan_vals_x):
     #   lhx = []
        for j, y in enumerate(scan_vals_y):
            #lhx.append(likelihood_function(x,y))
            lh[i,j] = likelihood_function(x,y)
      #  lh.append(lhx)
    
    return lh 
    
def plot_likelihood(scan_vals, lh_res, x_real, label, name):
    
    fig, ax = plt.subplots()
    ax.plot(scan_vals, lh_res)
    ax.vlines([x_real], min(lh_res), max(lh_res), color='r', ls='--', label='real '+ label)
    ax.set_xlabel(label)
    ax.set_ylabel('-loglikelihood')
    ax.legend(loc='best')
    plt.savefig(name + '_' + label + '_scan.png', dpi=600)
    plt.show()  
    
def plot_likelihood_2d(scan_vals_x, scan_vals_y, lh_np, x_real, y_real, 
                                x_label, y_label, min_ind, name):
	
    fig, ax = plt.subplots()
    extent = [scan_vals_x[0],scan_vals_x[-1],scan_vals_y[0],scan_vals_y[-1]]
    aspect = sp.get_aspect(extent, lh_np.shape)
    im=ax.imshow(lh_np, origin='lower', extent= extent, aspect=aspect)
    ax.plot(x_real, y_real, marker='x', c='r')
    ax.plot(scan_vals_x[min_ind[0]], scan_vals_y[min_ind[1]], marker='v', c='g')
    cbar = fig.colorbar(im)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    cbar.set_label('log-likelihood')
    plt.savefig(name + '_' + x_label + '_' + y_label +'_scan.png',dpi=600)
    plt.show()
    
def scan_and_plot(lh_func, x_min, x_max, dx, x_real, label, name):
    scan_vals = np.arange(x_min, x_max, dx)
    start = time.time()
    lh_res = scan_likelihood(scan_vals, lh_func)
    end = time.time()
    
    print("Scanning time: ", end-start)
    
    min_ind = lh_res.argmin()
    
    print("----------- " + label + " -----------")
    print("True val: ", x_real, "Found minimum: ", scan_vals[min_ind])

    plot_likelihood(scan_vals, lh_res, x_real, label, name)
    
def scan_and_plot_2d(lh_func, x_min, x_max, dx, x_real, y_min, y_max, dy, y_real, x_label, y_label, name):
    scan_vals_x = np.arange(x_min, x_max, dx)
    scan_vals_y = np.arange(y_min, y_max, dy)
    
    start = time.time()
    lh_np = scan_likelihood_2d(scan_vals_x, scan_vals_y, lh_func)
    end = time.time()
    
    print("Scanning time: ", end-start)
    
    min_ind = np.unravel_index(lh_np.argmin(), lh_np.shape)
    
    print("True vals: ", x_real, y_real, "Found minimum: ", scan_vals_x[min_ind[0]], scan_vals_y[min_ind[1]])
    
    plot_likelihood_2d(scan_vals_x, scan_vals_y, lh_np, x_real, y_real, 
                                            x_label, y_label, min_ind, name)

def main(args):
    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
