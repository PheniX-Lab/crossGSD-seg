# crossGSD-seg
We developed a cross-spatial resolution semantic segmentation model for rice crops by integrating real and sim2real RGB images, which was based on the SegFormer and FADE module.  
This is a well-trained model that can be directly used, just configure the required virtual environment.  
The link for the trained model:  
https://drive.google.com/file/d/15VBJi9whdML-iOX0dafTnMR5FhSlLS_y/view?usp=drive_link  

# Virtual environment configuration
To address the dependency on Segformer and FADE when running the model, please refer to the link below for environment configuration.  
SegFormer:  
https://github.com/NVlabs/SegFormer  
FADE:  
http://lnkiy.in/fade_in  

# Hardware platform
CPU: Intel(R) Xeon(R) Platinum 8358P CPU @ 2.0 GHz  
GPU: NVIDIA GTX 4090 Ti  
CUDA version: 12.0  

# Test
We provide two sets of images, a dataset with 4 spatial resolution levels and a challenging dataset with complex field conditions.  
To test the images in the folder, run the pre_mask.py  

