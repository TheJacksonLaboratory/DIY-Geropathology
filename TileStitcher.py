# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 09:52:23 2022

@author: mawes
"""

import os
import cv2
import re
import numpy as np

def KidneyRGBSticher(target_dir,out_dir,down_sample,interval=512,tile_size=512):
    slide_dirs=os.listdir(target_dir)
    slide_dirs =[sdir for sdir in slide_dirs if os.path.isdir(target_dir+sdir)]
    for slide_dir in slide_dirs:
        tile_files=np.array(os.listdir(target_dir+slide_dir))
        xs=[]
        ys=[]
        sub_tile_files=[]
        for tile_file in tile_files:
            if ".tif" in tile_file:
                continue
            x=re.search(r"x=[0-9]+", tile_file)
            y=re.search(r"y=[0-9]+", tile_file)
            xs.append(np.uint64(x[0][2:]))
            ys.append(np.uint64(y[0][2:]))
            sub_tile_files.append(tile_file)
        xs=np.uint64(np.array(xs)/(interval*down_sample))
        ys=np.uint64(np.array(ys)/(interval*down_sample))
        tile_files=sub_tile_files
        if not sub_tile_files:
            continue
        for ii in range(np.uint64(np.max(ys)+1)):
            for jj in range(np.uint64(np.max(xs)+1)):
                file_idx=np.logical_and(np.equal(ys,ii),np.equal(xs,jj))
                if np.any(file_idx):
                    tile_file=np.array(tile_files)[file_idx][0]
                    tile=cv2.imread(target_dir+slide_dir+"/"+tile_file)
                    if tile.shape[0]<tile_size:
                        tile_pad=np.zeros((tile_size-tile.shape[0],tile.shape[1],tile.shape[2]),np.uint8)
                        tile_pad.fill(255)
                        tile=np.vstack((tile,tile_pad))
                    if tile.shape[1]<tile_size:
                        tile_pad=np.zeros((tile.shape[0],tile_size-tile.shape[1],tile.shape[2]),np.uint8)
                        tile_pad.fill(255)
                        tile=np.hstack((tile,tile_pad))
                else:
                    tile=np.zeros((tile_size,tile_size,3),np.uint8)
                if jj==0:
                    row=np.copy(tile)
                else:
                    row=np.hstack((row,np.copy(tile)))
            if ii==0:
                img=np.copy(row)
            else:
                img=np.vstack((img,np.copy(row)))
        keep_rows=np.full(img.shape[0], True)
        for ii in range(img.shape[0]):
            if np.equal(img[ii,:,:],0).all():
                keep_rows[ii]=False
        img=img[keep_rows,:,:]
        keep_cols=np.full(img.shape[1], True)
        for jj in range(img.shape[1]):
            if np.equal(img[:,jj,:],0).all():
                keep_cols[jj]=False
        img=img[:,keep_cols,:]
        zero_idx=np.logical_and(np.logical_and(np.equal(img[:,:,0],0),np.equal(img[:,:,1],0)),np.equal(img[:,:,2],0))
        zero_idx=np.reshape(zero_idx, (zero_idx.shape[0],zero_idx.shape[1],1))
        zero_idx=np.concatenate((zero_idx,zero_idx,zero_idx),axis=2)
        img[zero_idx]=255
        png_name=out_dir+slide_dir+"_stiched.png"
        cv2.imwrite(png_name, img)
    
        
            
