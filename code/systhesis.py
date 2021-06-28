#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 21:54:36 2021

@author: weiliao
"""

# matplotlib inline
import numpy as np
from scipy import ndimage
import math
import cv2
import random
import sys
import math
import requests
import matplotlib.pyplot as plt
np.set_printoptions(threshold=sys.maxsize)
import scipy.stats as st
#%%

def im2double(im):
    info = np.iinfo(im.dtype) # Get the data type of the input image
    return im.astype(np.float) / info.max # Divide all values by the largest possible value in the datatype

'''
Functions that might be helpful
'''
def im2col_sliding(A, BSZ, stepsize=1):
    '''
    :param A: image
    :param BSZ: kernel size
    :param stepsize: the step size of moving the kernel
    :return: All blocks that have the same kernel size, which are selected from the image by the sliding kernel.
    All blocks have been reshaped to a column vector.
    '''
    # This function is similar to the `im2col` function from MATLAB. It rearrange image blocks into columns.
    # 
    # Paste the following command to a coding block and check out the results to get more intuitions!
    # r = np.arange(25).reshape(5, 5); s = (3, 3); print(im2col_sliding(r, s).shape); print(im2col_sliding(r, s))
    # Parameters
    m,n = A.shape[:2]
    s0, s1 = A.strides[:2]
    nrows = m-BSZ[0]+1
    ncols = n-BSZ[1]+1
    shp = BSZ[0],BSZ[1],nrows,ncols
    strd = s0,s1,s0,s1
    out_view = np.lib.stride_tricks.as_strided(A, shape=shp, strides=strd)
    return out_view.reshape(BSZ[0]*BSZ[1],-1)[:,::stepsize]

def sub2ind(array_shape, rows, cols):
    return rows*array_shape[1] + cols    



def Find_matches(template, sample, G):
    ### Note: below is just a provided sketch, you can have your own code flow 
    ### parameters, as used by Efros and Leung
    epsilon = 0.1
    delta = 0.3

    #### TODO:
    # validMask is a square mask of width w that is 1 where template is filled
    # the unfilled pixel is -1
    validMask = np.logical_not(np.isnan(template))
    template[np.isnan(template)] = 0

    #### TODO:
    # Play with im2col_sliding()! 
    # partition sample to blocks (represented by column vectors). 
    # We can actually do this only once, and pass this representation to this 
    # function, but we leave it as is in order not to change function signature 
    # that was instructed.
    blocks = im2col_sliding(sample, template.shape).T # size: n * k^2
    template1D = template.reshape((1, -1)) # size: 1 * k^2
    validMask1D = validMask.reshape((1, -1))# size: 1 * k^2
    G1D = G.reshape((1, -1))
    dist_mat= validMask1D * (blocks - template1D)**2 #size: n * k^2
    error_vec = np.sum(G1D * dist_mat, axis=-1)
    valid_indices = np.logical_or(error_vec < (1 + epsilon) * np.min(error_vec), error_vec == 0)
    valid_indices = np.logical_and(valid_indices, error_vec < delta)
    # valid_indices = error_vec < (1 + epsilon) * np.min(error_vec+1e-3)
    return blocks[valid_indices], error_vec[valid_indices]


def dnorm(x, mu, sd):
    return 1 / (np.sqrt(2 * np.pi) * sd) * np.e ** (-np.power((x - mu) / sd, 2) / 2)


def gaussian_kernel(w, sigma=1, verbose=False):
    kernel_1D = np.linspace(-(w // 2), w // 2, w)
    for i in range(w):
        kernel_1D[i] = dnorm(kernel_1D[i], 0, sigma)
    kernel_2D = np.outer(kernel_1D.T, kernel_1D.T)
    kernel_2D *= 1.0 / np.sum(kernel_2D)

    if verbose:
        plt.imshow(kernel_2D, interpolation='none', cmap='gray')
        plt.title("Image")
        plt.show()
    return kernel_2D


def Synth_texture(sample, w, s):
    ###Texture Synthesis by Non-parameteric Sampling / Efros and Leung
    ###Note: below is just a provided sketch, you can have your own code flow
    
    ## Normalizing pixel intensity
    sample = im2double(sample)
    seed_size = 3
    [sheight, swidth, nChannels] = sample.shape
    theight = s[0]
    twidth = s[1]
    synthIm = np.full((theight, twidth, nChannels),np.nan)

    ### TODO: Fill in mu, sigma, G
    ### G is a 2D zero-mean Gaussian with standard deviation w/6.4 sampled on a w x w grid centered about its mean
    sigma = w/6.4
    G = gaussian_kernel(w, sigma, verbose=False)

    ### Initialization: pick a random 3x3 patch from sample and place in the middle of the synthesized image.
    ### Just for convenience, keep some space (SEED_SIZE) from the boundary
    i0 = 31
    j0 = 3
    c = [round(.5 * x) for x in s]
    synthIm[c[0]: c[0] + seed_size , c[1]: c[1] + seed_size ,:] = sample[i0: i0 + seed_size , j0: j0 + seed_size,:]
     
    ### bitmap indicating filled pixels
    filled = np.zeros(s)
    filled[c[0]: c[0] + seed_size , c[1]: c[1] + seed_size ] = 1
    n_filled = int(np.sum(filled))
    n_pixels = s[0]*s[1]

    ### Main Loop
    next_p = n_pixels / 10
    while(n_filled < n_pixels):
        #report progress
        if(n_filled > next_p):
            print( round(100 * n_filled / n_pixels), '% complete', )
            next_p += n_pixels / 10
            
        ### dilate current boundary, find the next round of un-filled pixels
        ### (ii, jj) represents the locations
        border = ndimage.binary_dilation(filled).astype(filled.dtype) - filled
        ii = []
        jj = []
        for i in range(s[0]):
            for j in range(s[1]):
                if(int(border[i,j]) == 1):
                    ii.append(i)
                    jj.append(j)
        ii = np.asarray(ii)
        jj = np.asarray(jj)        
       
        
        ### Permute (just to insert some random noise, not a must, but recommended. play with it!)
        #perm = np.random.permutation(len(ii))
        #ii = ii[perm]
        #jj = jj[perm]        

        for i in range(len(ii)):
            ### Place window at the center. Use find_matches to get the best matches from the src image.
            ic = [x for x in range(math.ceil(ii[i] - w/2), math.floor(ii[i] + w / 2)+1)]
            ic = np.asarray(ic)
            jc = [x for x in range(math.ceil(jj[i] - w/2), math.floor(jj[i] + w / 2)+1)]
            jc = np.asarray(jc)
            inbounds_ic = (ic >= 0) & (ic< theight)
            inbounds_jc = (jc >=0) & (jc < twidth)
            template = np.full((w, w, nChannels), np.nan)

            nix_1 = np.ix_(np.nonzero(inbounds_ic)[0],np.nonzero(inbounds_jc)[0])
            nix_2 = np.ix_(ic[inbounds_ic],jc[inbounds_jc])
            template[nix_1] = synthIm[nix_2]
            #
            # cv2.imshow('template', template)
            # cv2.imshow('sample', sample)
            # cv2.waitKey(10)

            ### TODO: 
            ### Implement find_matches().
            [best_matches, errors] = Find_matches(template, sample, G)
            # if len(best_matches) > 1:
            ### TODO:
            idx = np.random.randint(0, len(best_matches))
            best_match = best_matches[idx].reshape((w, w, -1))
            ### Sample from best matches and update synthIm
            idx = math.floor(w / 2)
            synthIm[ii[i], jj[i],:] = best_match[idx, idx]
            ### update bitmap indicating the corresponding pixel is filled
            filled[ii[i], jj[i]] = 1
            n_filled = n_filled + 1

    print(synthIm.shape)
    return synthIm

def main():
    source = cv2.imread('data/rings.jpg')
    source = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY).reshape((source.shape[0],source.shape[1],-1))
    w = 13
    target = Synth_texture(source, w, [100, 100])
    target = (target * 255).astype(np.uint8)
    cv2.imwrite('data/rings_out.jpg', target)
    cv2.imshow('target',target)
    cv2.waitKey(30)

    plt.imshow(target)
    plt.title('w =' + str(w))
    plt.show()

if __name__ == '__main__':
    main()
