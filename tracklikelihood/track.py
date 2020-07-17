#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  track.py
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

def line(m, b, x):
    return m*x + b

def line_segment(t_start, t_end, f_start, f_end):
    
    dt = t_end - t_start
    df = f_end - f_start
    
    m = df/dt
    b = f_end - m*t_end
    
    return lambda x: line(m, b, x)

def get_b_m(t_start, t_end, f_start, f_end):
    
    dt = t_end - t_start
    df = f_end - f_start
    
    m = df/dt
    b = f_end - m*t_end
    
    return b, m

def gauss(y, mu, sigma):
    
    return 1/(sigma*np.sqrt(2*np.pi))*np.exp(-(y-mu)**2/sigma)

class Track:
    
    def __init__(self, t_start, t_end, f_start, f_end, sigma, snr):
        self.t_start = t_start
        self.t_end = t_end
        self.f_start = f_start
        self.f_end = f_end
        self.sigma = sigma
        self.snr = snr
    
    def get_delta(self):
        
        dt = self.t_end - self.t_start
        df = self.f_end - self.f_start
        
        return dt, df
    
    def get_slope(self):
        
        dt, df = self.get_delta()

        return df/dt
    
    def get_length(self):

        dt, df = self.get_delta()
        
        return np.sqrt(df**2 + dt**2)
        
    def get_signal(self, t, f):

        track_signal = np.zeros(shape=[t.shape[0], f.shape[0]])

        track_ind = (t>(self.t_start))&(t<(self.t_end))

        t_track = t[track_ind]

        ff, tt = np.meshgrid(f, t_track)

        mu = line_segment(self.t_start, self.t_end, self.f_start, self.f_end)

        track_signal[track_ind] = gauss(ff, mu(tt), self.sigma)

        return self.snr*track_signal
    
    @classmethod
    def from_slope_and_length(cls, t_start, f_start, slope, length, sigma, snr):
        
        dt = length
        df = dt*slope
        return cls(t_start, t_start+dt, f_start, f_start+df, sigma, snr)
        

def main(args):
    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
