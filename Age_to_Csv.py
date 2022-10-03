# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 12:31:50 2022

@author: mawes
"""

import os
import zarr
import numpy as np 
import pandas as pd
def age_scores2csv(age_score_dir_path,pxl_level_out,slide_level_out):
    age_files=os.listdir(age_score_dir_path)
    age_files=[age_file for age_file in age_files if '.zarr' in age_file]
    slide_id_short=[]
    slide_id=[]
    i=[]
    j=[]
    score=[]
    mean_score=[]
    for ff in range(len(age_files)):
        file =age_files[ff]
        slide_id_short.append(file[:-16])
        z1=zarr.open(age_score_dir_path+file,'r')
        image=z1[:,:,0]
        mean_score.append(np.mean(image[np.not_equal(image,0)]))
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
    data={"slide":slide_id,"i":i,"j":j,"score":score}
    mean_data={"slide":slide_id_short,"mean_score":mean_score}
    age_df=pd.DataFrame.from_dict(data)
    mean_age_df=pd.DataFrame.from_dict(mean_data)
    age_df.to_csv(pxl_level_out)
    mean_age_df.to_csv(slide_level_out)
