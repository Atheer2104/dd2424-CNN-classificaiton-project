# Building and Optimizing a CNN for CIFAR-10 Classification from Scratch

This project is our submission for the KTH course DD2424 Deep Learning for Data Science. The project is about exploring and building various Convolutional Neural Networks 
for image classification using PyTorch on the Cifar 10 & Cifar 100 dataset. The models that we have built are VGG style networks which are E-level grade (passing grade) but also 
we have worked on upgrades for A-level grade (the highest grade) and this includes experimenting with ResNet style networks and experimenting with noisy labels. 
See the assignment description for more detail [here](https://github.com/Atheer2104/dd2424-project/blob/main/Assignment_Description.pdf)

Our Best network achieved an accuracy of **95.59 %** on Cifar 10 and **75.45 %** on Cifar 100 using ResNet110 with sophisticated data augmentations such as CutMix and MixUp. 
For more details read our [report](https://github.com/Atheer2104/dd2424-project/blob/main/dd2424_project_report.pdf). Finally, you can see our video presentation of the project 
[here](https://github.com/Atheer2104/dd2424-project/blob/main/Group_4_video.mp4.zip)

##### Note: These models were all trained using Google Cloud platform VMs using Nvidia Tesla T4 GPU.

## File structure 

- **Baseline (E-level grade)**
this folder contains VGG networks with increasing VGG blocks added to the network, VGG1 contains only 1 VGG block whilst VGG3 contains 3 VGG blocks

- **baseline_dropout (E-level grade)**
This folder contains dropout implemented for the VGG network

- **baseline_weightdecay (E-level grade)**
This folder contains the VGG network trained with weight decay

- **baseline_data_augmentation (E-level grade)**
This folder contains the VGG network trained with data augmentations

- **baseline_combined_reguralizations (E-level grade)**
  This folder contains the combined regularization VGG model this model is trained with data augmentations, batch normalization and dropout

- **part_E_further_exploration (E-level grade)**
This folder contains 4 subfolders where each folder is a single experiment which was done on top of the VGG network, The experiments are

  -  Normalizing training data to 0 mean and 1 std
  -  Replacing SGD with momentum to Adam and AdamW optimizer
  - Experimenting with different learning rate schedulers such as Step Decay, Warm-up + cosine annealing, cosine annealing with restarts
  - Experimenting with different order of batch normalization and dropout and experimenting with VGG with only dropout and batch normalization separately
  
- **Architectural upgrades (A-Level grade)**
This folder contains the architectural upgrades which were implemented in ResNet with various depths namely ResNet20, ResNet56, and ResNet110. Furthermore experimented on top of ResNet
by adding Squeeze and Excitation ResNet blocks, adding Patchify and Embed (the first operation in Visual Transformer) to the original ResNet architecture and finally experimenting with
more sophisticated data augmentations such as random erasing, CutMix and MixUp

- **noisy labels (A-level grade)** 
This folder contains the experiments on noisy labels using the Symmetric Cross Entropy Loss on the VGG network



