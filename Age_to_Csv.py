# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 12:31:50 2022
@author: mawes
"""

import os
import zarr
import numpy as np 
import pandas as pd
from skimage import measure

def age_scores2csv(age_score_dir_path,label_dir_path,pxl_level_out,slide_level_out,section_level_out):
    age_files=os.listdir(age_score_dir_path)
    age_files=[age_file for age_file in age_files if '.zarr' in age_file]
    slide_id_short=[]
    slide_id=[]
    i=[]
    j=[]
    section_slide_id=[]
    section_i=[]
    section_j=[]
    section_label=[]
    section_size=[]
    score=[]
    section_score=[]
    mean_score=[]
    for ff in range(len(age_files)):
        file =age_files[ff]
        slide_id_short.append(file[:-16])
        print(age_score_dir_path+file)
        z1=zarr.open(age_score_dir_path+file,'r')
        image=z1[:,:,0]
        bwmask=np.not_equal(image,0)
        comp_labels=measure.label(bwmask)
        np.save(label_dir_path+file[:-16]+"_labels.npy",comp_labels)
        for cc in range(1,np.max(comp_labels)+1):
            if np.sum(np.equal(comp_labels,cc))>15000:
                section_score.append(np.mean(image[np.equal(comp_labels,cc)]))
                section_slide_id.append(file[:-16])
                for props in measure.regionprops(np.equal(comp_labels,cc).astype(np.uint8)):
                    si, sj=props.centroid
                    section_i.append(si)
                    section_j.append(sj)
                    section_label.append(cc)
                    section_size.append(np.sum(np.equal(comp_labels,cc)))
        mean_score.append(np.mean(image[np.not_equal(image,0)]))
        if pxl_level_out:
           for ii in range(image.shape[0]):
               for jj in range(image.shape[1]):
                   if (ii%100==0) and (jj==0):
                       print([[ff+1,ii+1],[len(age_files),image.shape[0]]])
                   if image[ii,jj]==0:
                       continue
                   else:
                       slide_id.append(file[:-16])
                       i.append(ii+1)
                       j.append(jj+1)
                       score.append(image[ii,jj])
    section_data={"slide":section_slide_id, "label":section_label, "size":section_size,"i":section_i,"j":section_j,"score":section_score}
    mean_data={"slide":slide_id_short,"mean_score":mean_score}
    section_age_df=pd.DataFrame.from_dict(section_data)
    mean_age_df=pd.DataFrame.from_dict(mean_data)
    section_age_df.to_csv(section_level_out,index=False)
    mean_age_df.to_csv(slide_level_out,index=False)
    if pxl_level_out:
       data={"slide":slide_id,"i":i,"j":j,"score":score}
       age_df=pd.DataFrame.from_dict(data)
       age_df.to_csv(pxl_level_out,index=False)
