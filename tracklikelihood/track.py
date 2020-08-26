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
from scipy import special
from math import sqrt

def line(m, b, x):
    return m*x + b

def line_segment(t_start, t_end, f_start, f_end):

    dt = t_end - t_start
    df = f_end - f_start

    m = df/dt
    b = f_end - m*t_end

    return lambda x: line(m, b, x), m

def get_b_m(t_start, t_end, f_start, f_end):

    dt = t_end - t_start
    df = f_end - f_start

    m = df/dt
    b = f_end - m*t_end

    return b, m

def gauss(y, mu, sigma):

    return 1/(sigma*np.sqrt(2*np.pi))*np.exp(-1/2*((y-mu)/sigma)**2)

def erf_cont(diff, sigma):
	#diff = mu - y
	return diff*special.erf(diff/(sqrt(2)*sigma))

def exp_cont(diff, sigma):
	return np.exp(-1/2*(diff/sigma)**2)

def exp_erf_cont(y, mu, sigma):

	diff = mu - y
	return sqrt(2/np.pi)*sigma*exp_cont(diff, sigma) + erf_cont(diff, sigma)

def integrate_bin(mumin, mumax, ymin, ymax, sigma, slope):

    return 1/(2*slope) * (exp_erf_cont(ymax, mumin, sigma) \
                    + exp_erf_cont(ymin, mumax, sigma) \
                    - exp_erf_cont(ymax, mumax, sigma) \
                    - exp_erf_cont(ymin, mumin, sigma))

def rect_spectrum(f, M):

    res = np.zeros(f.shape)

    res[f==0] = M
    res[f!=0] = np.sin(np.pi*f[f!=0]*M/2)/np.sin(np.pi*f[f!=0])

    return res**2

def f_frac(f, SR):
    return f/SR

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

    def get_integration_region(self, x, y, dx, dy):

        ymin = y-dy#-dy/2
        ymax = y+dy#/2
        xmin = x#-dx/2
        xmax = x+dx#/2

        xmax = np.minimum(xmax, self.t_end)
        xmin = np.maximum(xmin, self.t_start)

        #print(xmin[:,0], xmax[:,0])

        mu, _ = line_segment(self.t_start, self.t_end, self.f_start, self.f_end)

        mumin = mu(xmin)
        mumax = mu(xmax)

        return ymin, ymax, mumin, mumax

    def get_signal(self, t, f, N, BW):

        dt = (t[1]-t[0])
        df = (f[1]-f[0])/2

        track_signal = np.zeros(shape=[t.shape[0], f.shape[0]])
      #  track_signal = -100*track_signal

        track_ind = (t>(self.t_start-dt))&(t<self.t_end)

      #  first = np.argmax(t>self.t_start)
      #  print(first, t[first])

        t_track = t[track_ind]

        ff, tt = np.meshgrid(f, t_track)

        fmin, fmax, mumin, mumax = self.get_integration_region(tt, ff, dt, df)

        #mu, _ = line_segment(self.t_start, self.t_end, self.f_start, self.f_end)

       # track_signal[track_ind] = integrate_bin(mumin, mumax, fmin, fmax,
       #                                                     self.sigma,
       #                                                     self.get_slope())  #gauss(ff, mu(tt), self.sigma)

        #track_signal[track_ind]/=dt*df

        mean_f = (mumax+mumin)/2

        track_signal[track_ind] = rect_spectrum((ff-mean_f)/BW, N)

        #t_in0 = t_track[1]-self.t_start
        #t_in1 = self.t_end-t_track[-1]

       # print(t_in0, t_in1, dt, t_in0/dt, t_in1/dt)

        #track_signal[track_ind[0]] *= t_in0/dt
        #track_signal[track_ind[-1]] *= t_in1/dt

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
