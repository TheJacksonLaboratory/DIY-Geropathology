# DIY-Geropathology
This Repository will provide code to help you apply our geropathology tools to your datasets

Overall this repository is being built to be used in conjunction with https://www.geropathology-imaging.org/

More detailed directions can be found there, for those with more advanced computational skills directly working with codes via github is privided as an option.  

If using Qupath to download tiles for the aging classifers use the file KidneyTileDownload.groovy   This file needs to be adjusted in lines ~ 20 to 24 for the labels relevant to your project and slides.   If you are using a different classifer than the one provided for the kidney, then you need to pay attention to the .tileSize() and .overlap() options as these numbers might need to be adjusted.  

The TileSticker.py code is used to stick the tiles together to make one Image to feed into the classifier

After the Images are stiched together they are run through the aging classifer we use two code 
JuliaSetup.jl and LinkNet.jl to prepare Julia to handle the classifiers. The classifiers themselves can be found XXX becuse they are too large for Github.  Use the runningClassifier.jl script  changing 3 lines: 

  StitchedFile Path = Path to where the sitched Images are 
  Path_TO_SCORE= path where to put raw Age Data 
  JLD2.@Load = Path where the classifier is 

After the classifier is run, we need to combine the age predictions into an age Score this is done in Jupyter notebook First run the CombinedAgeSCore.py code 
Then Use Providing 3 things 
 CombinedAgeScore("AgeScoresRaw/", "AgeScoresCombined/", image_dir="Path To StichedImages", thresh_meathod="GaussianBlur", thresh_param=200, 
                     dist=scipy.stats.norm, loc=0, scale=1)
  
 
First= Directory to raw age scores from runningclassifier.jl
Second, Directoty to put combined age score in 
Third= Path the stitched Images can be found in 

To go from combined age scores to a csv file for analysis use the Age_toCsv.py function called by 
age_scores2csv("Path to_AgeScoresCombined/","pixel.csv","slide.csv")
Where you change 3 things 
First = Path to combinedAge scores as the result of CombinedAgeScore 
Pixel.Csv= path to pixel level data .csv file for output 
Slide.csv= path of sldie level data in .csv file for output 

WE can also use the ImageHeatmapOpenCV.py to initialize the paintitngs which need to be adjusted 

tissue_image_path= "Path to stitched Images"
mask_path="Path to AgeScoresCombined/"
image_heatmap_path="Path for Output paintings/"
mask_bounds=[-4,4]




