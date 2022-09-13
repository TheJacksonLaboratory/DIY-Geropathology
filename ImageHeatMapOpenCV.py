# -*- coding: utf-8 -*-
"""
Created on Thu Jan 20 10:42:23 2022

@author: Seamus Mawe
"""
from PIL import Image
import numpy as np
import zarr
import cv2
import time


def paint_image(tissue_image,mask,mask_bounds=None,colorspace=cv2.COLORMAP_JET):
    """Creates a heatmap from a mask and overlays grayscale intensity of assosiated image. 

    Parameters
    ----------
    tissue_image : Pillow Image
        Image to be painted.
    mask : 2d array
        Mask of values for painting.
    mask_bounds : Tuple or None, optional
        Theoretical upper and lower bounds of values in mask if None uses min and max of mask array. ex: (0,1), The default is None.
    colorspace : Matplotlib colormap object, optional
        The matplotlib colormap to be used for painting. The default is cm.plasma.

    Returns
    -------
    psudocolor_mask : Pillow Image
        heatmap of mask with tissue_image overlayed as a transparency layer.

    """
    if mask_bounds is None:
        mask_bounds=(np.min(mask),np.max(mask))
        
    mask[mask<mask_bounds[0]]=mask_bounds[0]
    mask[mask>mask_bounds[1]]=mask_bounds[1]
    mask=(mask-mask_bounds[0])/(mask_bounds[1]-mask_bounds[0])
    mask=np.round(mask*255).astype(np.uint8)
    mask=cv2.applyColorMap(mask,colorspace)
    mask=mask[:,:,(2,1,0)]
    
    gray = cv2.cvtColor(tissue_image, cv2.COLOR_RGB2GRAY)
    gray=np.reshape(gray, (gray.shape[0],gray.shape[1],1))
    gray=np.concatenate((gray,gray,gray),axis=2)
    print(mask.shape)
    print(gray.shape)
    heatmap_img=cv2.addWeighted(mask,0.5, gray, 0.5, 0)
    return heatmap_img




def load_image_and_mask(tissue_image_path,mask_path):
    """Loads image and mask from files

    Parameters
    ----------
    tissue_image_path : string
        path to image to be painted.
    mask_path : None or String
        path for the mask file if None creates a randomised mask, 
        otherwise can be a one channel image file or a 2d zarr file.

    Returns
    -------
    tissue_image : Pillow Image
        Image to be painted.
    mask : 2d array
        Mask of values for painting.

    """
    tissue_images=[]
    if ".zarr" in tissue_image_path:
        z_tissue=zarr.open(tissue_image_path)
        tissue_image_matrix=z_tissue[:,:,:,:]
        
        for tt in range(tissue_image_matrix.shape[3]):
            tissue_image=tissue_image_matrix[:,:,:,tt]
            tissue_image=tissue_image.astype(np.uint8)
            tissue_images.append(tissue_image)
    else:
        tissue_image=cv2.imread(tissue_image_path)
        tissue_image=tissue_image[:,:,(2,1,0)]
        tissue_images.append(tissue_image)
    
    z_mask=zarr.open(mask_path)
    mask_matrix=z_mask[:,:,:]
    masks=[]
    for mm in range(mask_matrix.shape[2]):
        mask=mask_matrix[:,:,mm]
        mask=cv2.GaussianBlur(mask,(9,9),0)
        masks.append(mask)
    for ii in range(len(masks)):
        if tissue_images[ii].shape[0]<masks[ii].shape[0]:
            tissue_images[ii]=np.vstack((tissue_images[ii],np.ones((masks[ii].shape[0]-tissue_images[ii].shape[0],tissue_images[ii].shape[1],3),np.uint8)*255))
        if tissue_images[ii].shape[1]<masks[ii].shape[1]:
            tissue_images[ii]=np.hstack((tissue_images[ii],np.ones((tissue_images[ii].shape[0],masks[ii].shape[1]-tissue_images[ii].shape[1],3),np.uint8)*255))
    return tissue_images,masks
                
            
def load_image_and_mask2(tissue_image_path,mask_path,mask_thresholds):
    """Loads image and mask from files

    Parameters
    ----------
    tissue_image_path : string
        path to image to be painted.
    mask_path : None or String
        path for the mask file if None creates a randomised mask, 
        otherwise can be a one channel image file or a 2d zarr file.

    Returns
    -------
    tissue_image : Pillow Image
        Image to be painted.
    mask : 2d array
        Mask of values for painting.

    """
    z_tissue=zarr.open(tissue_image_path)
    tissue_image_matrix=z_tissue[:,:,:,:]
    tissue_images=[]
    for tt in range(tissue_image_matrix.shape[3]):
        tissue_image=tissue_image_matrix[:,:,:,tt]
        tissue_image=tissue_image.astype(np.uint8)
        tissue_images.append(tissue_image)
    
    z_mask=zarr.open(mask_path)
    mask_matrix=z_mask[:,:,:,:]
    for ii in range(mask_matrix.shape[2]):
        for mm in range(mask_matrix.shape[3]):
            mask_matrix[:,:,ii,mm]=cv2.GaussianBlur(mask_matrix[:,:,ii,mm],(9,9),0)
    m_thresh=np.zeros(len(mask_thresholds))
    for ii in range(len(mask_thresholds)):
        m_thresh[ii]=np.percentile(mask_matrix[:,:,ii,:], mask_thresholds[ii])
    masks=[]
    for mm in range(mask_matrix.shape[3]):
        mask=np.zeros([mask_matrix.shape[0],mask_matrix.shape[1]])
        for ii in range(mask_matrix.shape[0]):
            for jj in range(mask_matrix.shape[1]):
                for cc in range(len(m_thresh)):
                    if mask_matrix[ii,jj,cc,mm]<m_thresh[cc]:
                        mask[ii,jj]=cc
                        break
                    else:
                        mask[ii,jj]=len(m_thresh)
        masks.append(mask)
    return tissue_images,masks

def image_heatmap_test(test_image_path="13236M-320-0001.tif",colorspace=cv2.COLORMAP_JET,image_heatmap_path="image_heatmap_test.png"):
    """ Quick test of image painting with random mask.
    

    Parameters
    ----------
    test_image_path : string, optional
        test image to be painted, will try and include the default image file when sharing. The default is "13236M-320-0001.tif".
    colorspace : TYPE, optional
        The matplotlib colormap to be used for painting. The default is cm.plasma.
    image_heatmap_path : TYPE, optional
        path for saving painted image (must be a png file). The default is "image_heatmap_test.png".

    Output
    -------
    painted image PNG file

    """
    tissue_image,mask=load_image_and_mask(test_image_path, None)
    tic=time.perf_counter()
    painted_image=paint_image(tissue_image, mask,None,colorspace)
    toc=time.perf_counter()
    painted_image.save(image_heatmap_path,"PNG")
    print(f"painted {tissue_image.size[1]:0.0f}x{tissue_image.size[0]:0.0f} image in {toc - tic:0.4f} seconds")
    
            
def image_heatmap(tissue_image_path,mask_path,image_heatmap_path,mask_bounds,colorspace=cv2.COLORMAP_JET):
    """Paints a target image with a heatmap based on mask
    

    Parameters
    ----------
    tissue_image_path : string
        path to image to be painted.
    mask_path : None or String
        path for the mask file if None creates a randomised mask, 
        otherwise can be a one channel image file or a 2d zarr file.
    image_heatmap_path : string
        path for saving painted image (must be a png file).
    mask_bounds : Tuple or None, optional
        Theoretical upper and lower bounds of values in mask if None uses min and max of mask array. ex: (0,1), The default is None.
    colorspace : Matplotlib colormap object, optional
        The matplotlib colormap to be used for painting. The default is cm.plasma.

    Output
    -------
    painted image PNG file

    """
    tissue_images,masks=load_image_and_mask(tissue_image_path, mask_path)
    
    for mm in range(len(tissue_images)):
        painted_image=paint_image(tissue_images[mm], masks[mm], mask_bounds, colorspace)
        painted_image=Image.fromarray(painted_image)
        painted_image.save(image_heatmap_path,"PNG")#+"{run:02d}".format(run=mm+1)+".png","PNG")

