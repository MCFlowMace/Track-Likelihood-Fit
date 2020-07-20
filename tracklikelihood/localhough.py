#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  localhough.py
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

def get_voting(img, xx, yy, window, angles):
    
    #plt.imshow(window)
    #plt.show()
    
    #m_x=ma.masked_array(xx, mask=np.invert(window))
    #m_y=ma.masked_array(yy, mask=np.invert(window))
    
    #result=m_x*np.cos(angles[:,None,None])+m_y*np.sin(angles[:,None,None])
    result=xx*np.cos(angles[:,None,None])+yy*np.sin(angles[:,None,None])

    voting_matrix = np.zeros(shape=result.shape, dtype=bool)
    
    #bool_mask = (~result.mask)&(np.abs(result)<0.5)
    bool_mask = window[None,:,:]&(np.abs(result)<0.5)
    #bool_mask = np.abs(result)<0.5

    voting_matrix[bool_mask]=1

    return voting_matrix
    
def get_local_result(binary_result, pixel_threshold):
    
    hist = np.sum(binary_result,axis=(1,2))
    #plt.plot(hist)
    #plt.show()
    result_angles = hist>pixel_threshold

    return np.any(binary_result[result_angles],axis=0)
    
def get_window_centers(binary_img, width):
    ind = np.argwhere(binary_img)
    ind_c=(ind[:,0]>=width)&(ind[:,0]<binary_img.shape[0]-width)&\
            (ind[:,1]>=width)&(ind[:,1]<binary_img.shape[1]-width)
    ind = ind[ind_c]
    return ind

def get_binary_img(img,n_std):
    threshold = np.mean(img)+n_std*np.std(img)
    binary_img = img>threshold
    return binary_img
    
def local_hough_track_reconstruction(img, window_size, angle_dim, pixel_threshold, n_std):

    width = int(window_size/2)

    binary_img = get_binary_img(img,n_std)
    binary_final = np.zeros(shape=binary_img.shape, dtype=bool)
    
    angles = np.linspace(-np.pi/2,np.pi/2,angle_dim)
    
    print(angles)

    ind = get_window_centers(binary_img, width)
    
    x = np.arange(-width, width+1, 1)
    y = np.arange(-width, width+1, 1)
    xx, yy = np.meshgrid(x,y)

    for k in range(ind.shape[0]):

        i=ind[k,0]
        j=ind[k,1]

        x_min = i-width
        x_max = i+width+1
        y_min = j-width
        y_max = j+width+1
    
        window = binary_img[x_min:x_max,y_min:y_max]

       # start=time.time()
        binary_result = get_voting(binary_img, xx, yy, window, angles)
       # end=time.time()
       # print(end-start)
        #print(binary_result[0])
        #plt.imshow(binary_result[10].transpose())
        #plt.show()
        hist = np.sum(binary_result,axis=(1,2))
        #plt.plot(hist)
        #plt.show()
        result_angles = hist>pixel_threshold

        local_result = get_local_result(binary_result,pixel_threshold)

        binary_final[x_min:x_max,y_min:y_max] = binary_final[x_min:x_max,y_min:y_max]|local_result


    return binary_img, binary_final

def main(args):
    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main(sys.argv))
