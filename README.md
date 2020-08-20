# Deep Neuron Computerized Tomography Facial Reconstruction
CT (Computerized Tomography) Facial Reconstruction is a project that aims to utilize neural networks in the field of forensic facial reconstruction. The project aims to re-construct 3D accurate and representative facial images from .dicom formatted CT scans. In particular the team is implementing a generative adversarial network (GAN) and aims to compare this models performance against other methods of forensic facial reconstruction.  

## Dependencies  
* fastai  
* mayavi  
* matplotlib  
* nibabel  
* numpy  
* pydicom  
* pytorch  
* pytorch-lightning 
* scipy  
* torchvision  
* wandb  

## Approach
The project aims to recreate the faces using generative image translation techniques in both 3D and 2D. 
The expected data is in the form of 3D Dicom files of faces, which are preprocessed, cleaned and thresholded into 3D numpy arrays for the bone and flesh of the face. These serve as the input and label respectively, and are utilized in a 3D convolutional generative neural network modelled off of the V-net (https://github.com/mattmacy/vnet.pytorch).
Different ResNet 3D configurations are utilized as discriminator and down-sample path of the generator (https://github.com/kenshohara/3D-ResNets-PyTorch). Additionally, Fast Ai's dynamic U-Net has been utilized and adapted for a 2D generative model and a dynamic configuration for the V-net (https://docs.fast.ai/vision.models.unet.html).


