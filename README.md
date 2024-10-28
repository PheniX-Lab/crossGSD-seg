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

# Citation
If our code or models help your work, please cite our paper:  
@online{https://doi.org/10.1016/j.isprsjprs.2024.10.007,  
  title={Bridging real and simulated data for cross-spatial-resolution vegetation segmentation with application to rice crops},  
  author={Yangmingrui Gao, Linyuan Li, Marie Weiss, Wei Guo, Ming Shi, Hao Lu, Ruibo Jiang, Yanfeng Ding, Tejasri Nampally, P. Rajalakshmi, Frédéric Baret, Shouyang Liu},  
  Journal={ISPRS Journal of Photogrammetry and Remote Sensing},  
  pages={133-150},  
  year={2024}  
}
