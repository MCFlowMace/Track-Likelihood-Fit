#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  spectrogram.py
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
from scipy.stats import expon

def get_aspect(extent, shape):
    
    x_res = (extent[0] - extent[1])/shape[0]
    y_res = (extent[2] - extent[3])/shape[1]
    
    return x_res/y_res
    
def plot_spectrogram(spec, t, f, tracks=[], vscale='default', log_scale=False,
                                name='spectrogram', save=False):

    fig, ax = plt.subplots()  
    
    dt = t[1]-t[0]
    df = f[1]-f[0]  

    extent = [t[0], t[-1]+dt, f[0]-df/2, f[-1]+df/2]
    aspect = get_aspect(extent, spec.shape)
    
    data = np.transpose(spec)
    
    if log_scale:
        data = np.log(data)

    if vscale=='default':
        vmin = np.nanmin(data[data != -np.inf])
        vmax = np.max(data)
    else:
        print(vscale)
        vmin = vscale[0]
        vmax = vscale[1]
    
    cmap = ax.imshow(data, extent=extent, origin='lower', vmin=vmin, vmax=vmax, 
                                                    aspect=aspect)
                                            
    for tr in tracks:
        t_start = tr[0]
        f_start = tr[1]
        t_end = tr[2]
        f_end = tr[3]
        
        f_track, _ = track.line_segment(t_start, t_end, f_start, f_end)
    
        t_probe = np.linspace(t_start, t_end, t.shape[0]*100)
        f_probe = f_track(t_probe)
        ax.plot(t_start, f_start, c='r', marker='x', markersize=2)
        ax.plot(t_end, f_end, c='r', marker='x', markersize=2)
        ax.plot(t_probe, f_probe, c='g', linewidth=1)
    
    cbar = fig.colorbar(cmap)
    cbar.set_label('Power')
    
    ax.set_xlabel('t[s]')
    ax.set_ylabel('f[Hz]')

    if save:
        plt.savefig(name+'.png', dpi=600)

    plt.show()
    
def plot_power_distribution(spec, name='power_dist', bins=50, save=False):
	
    fig, ax = plt.subplots()    
    
    flat_spec = spec.flatten()
    maximum = np.max(flat_spec)
    
    ax.hist(flat_spec, bins=bins, density=True)
    #ax.plot(expon.pdf(np.linspace(0,maximum, 1000)))
    
    ax.set_xlabel('power')
    ax.set_ylabel('count')

    if save:
        plt.savefig(name+'.png', dpi=1200)

    plt.show()

class Spectrogram:
	
		
    def __init__(self, spec, t, f, name='spectrogram'):
        self.spec = spec
        self.t = t
        self.f = f
        self.name = name

    @property
    def name(self):
        return self._name
    
    @name.setter
    def name(self, name):
        if isinstance(name, str):
            self._name = name
        else: 
            raise TypeError('Expected a string')
    
    @property
    def spec(self):
        return self._spec
    
    @spec.setter
    def spec(self, spec):
        if isinstance(spec, np.ndarray) and spec.ndim==2:
            self._spec = spec
        else: 
            raise TypeError('Expected numpy array with ndim==2')
            
    @property
    def t(self):
        return self._t
    
    @t.setter
    def t(self, t):
        if isinstance(t, np.ndarray) and t.ndim==1:
            self._t = t
        else: 
            raise TypeError('Expected numpy array with ndim==1')
            
    @property
    def f(self):
        return self._f
    
    @f.setter
    def f(self, f):
        if isinstance(f, np.ndarray) and f.ndim==1:
            self._f = f
        else: 
            raise TypeError('Expected numpy array with ndim==1')
            
    @classmethod
    def from_tracks(cls, t, f, tracks, name='spectrogram', add_noise=True):
		
        signal = np.zeros((t.shape[0], f.shape[0]))

        for track in tracks:
            signal += track.get_signal(t,f)
            
        if add_noise:
            signal += np.random.exponential(1, (t.shape[0], f.shape[0]))

        return cls(signal, t, f, name)
		
    def plot(self, tracks=[], vscale='default', log_scale=False, save=False):
        
        plot_spectrogram(self.spec, self.t, self.f, 
                                tracks, vscale, log_scale, self.name, save)
        
    def normalize(self):
		
        mean = np.mean(self.spec)
        self.spec /= mean
		
    def power_dist(self, save=False):
        
        plot_power_distribution(self.spec, name=self.name+"_power_dist", save=save)

def main(args):
	
    t = np.linspace(0, 50, 100)
    f = np.linspace(10, 200, 100)
    sp = Spectrogram.from_tracks(t, f, [])
    sp.plot()
	
    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
