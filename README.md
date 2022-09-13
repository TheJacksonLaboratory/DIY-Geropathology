# DIY-Geropathology
This Repository will provide code to help you apply our geropathology tools to your datasets

Overall this repository is being built to be used in conjunction with https://www.geropathology-imaging.org/

More detailed directions can be found there, for those with more advanced computational skills directly working with codes via github is privided as an option.  

If using Qupath to download tiles for the aging classifers use the file KidneyTileDownload.groovy   This file needs to be adjusted in lines ~ 20 to 24 for the labels relevant to your project and slides.   If you are using a different classifer than the one provided for the kidney, then you need to pay attention to the .tileSize() and .overlap() options as these numbers might need to be adjusted.  

The TileSticker.py code is used to stick the tiles together to make one Image to feed into the classifier

After the Images are stiched together they are run through the aging classifer we use two code 
JuliaSetup.jl and LinkNet.jl to prepare Julia to handle the classifiers
