StitchedFilePath =("/Users/sms/Library/CloudStorage/Box-Box/GeropathologyResearchNetwork/QupathAnnotations/JAX_NSC_Kidney/StitchedAllSet/")
ListofImages=readdir(StitchedFilePath,join=false)
#FileNames=readdir(StitchedFilePath,join=false)
PATH_TO_SCORE=("/Users/sms/Library/CloudStorage/Box-Box/GeropathologyResearchNetwork/QupathAnnotations/JAX_NSC_Kidney/AgeScoresRaw/")

JLD2.@load "/Users/sms/LinkNet_age__tile_22_test_3_cutoff_0_ADADelta_checkpoint_30.jld2" l

print("
Directory to Process",StitchedFilePath)

print("
Directory for scores",PATH_TO_SCORE)

print("
List of Images to Process full path",ListofImages)

#print("
#List of file names to Procees names only",FileNames)




for image in ListofImages
        print("
        Name of image",image)
        
        ImageFilePath=(StitchedFilePath*image)

        print("
        Name of Path to load", ImageFilePath)

        img=FileIO.load(ImageFilePath)

        img = channelview(img)
        img = permutedims(img,[2,3,1])
        img = reshape(img,(size(img,1),size(img,2),size(img,3),1))
        print("
        size of image",size(img))

        img = Float32.(img[:,:,:,:].*255)
   
        patch=Float32.(img)
        score=Flux.σ.(l(patch))
        image=replace(image,".png"=>".zarr")
#=
        for name in FileNames
            print("
            name of original file",name) 
            name=replace(name,".png"=>".zarr")
            print("
            Name modifided",name)
            scorepath =(PATH_TO_SCORE*name)
            print("
            Final Directory",scorepath)
=#        

        py"""
        import zarr
        import numpy as np
        zarr.save($(PATH_TO_SCORE)+$(image),$score)
        """
    
end

