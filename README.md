# Brain MRI-based 3D Convolutional Neural Networks for Classification of Schizophrenia and Controls

## Background 

The ability of automatic feature learning makes Convolutional Neural Network (CNN) potentially suitable to uncover the complex and widespread brain changes in schizophrenia. Despite that, limited studies have been done on schizophrenia identification using interpretable deep learning approaches on multimodal neuroimaging data. Here, we developed a deep feature approach based on pre-trained 2D CNN and naive 3D CNN models trained from scratch for schizophrenia classification by integrating 3D structural and diffusion magnetic resonance imaging (MRI) data. We found that the naive 3D CNN models outperformed the pretrained 2D CNN models and the handcrafted feature-based machine learning approach using support vector machine during both cross-validation and testing on an independent dataset. Multimodal neuroimaging-based models accomplished performance superior to models based on a single modality. Furthermore, we identified brain grey matter and white matter regions critical for illness classification at the individual- and group-level which supported the salience network and striatal dysfunction hypotheses in schizophrenia. Our findings underscore the potential of CNN not only to automatically uncover and integrate multimodal 3D brain imaging features for schizophrenia identification, but also to provide relevant neurobiological interpretations which are crucial for developing objective and interpretable imaging-based probes for prognosis and diagnosis in psychiatric disorders.  

## Model Architecture

Sequential models followed the typical CNN sequential architecture with convolutional layers, pooling layers and fully connected layers. The convolutional kernel and pooling kernel were set with 3x3x3 dimensions using grid search. Independent input maps were connected to different network branches. The resulting feature maps from each branch were flattened and connected to a fully connected layer with 128 neurons. Output was obtained by sigmoid function.

![Figure 2](https://user-images.githubusercontent.com/44959050/120256340-7a855580-c2c0-11eb-8df4-da1e6345d81a.png)


Inspired by the GoogLeNet, a 3D inception module was utilized in inception models. The inception module divides the network into multiple branches with different convolutional kernels thus allowing operating convolutions with different kernels on the same level. The inception module not only improves the performance of the network but also controls overfitting and reduces computational expenses. 

![Figure 3](https://user-images.githubusercontent.com/44959050/120256514-eb2c7200-c2c0-11eb-957d-5c198e7b80a7.png)


Inspired by the residual module, inception_resnet models combined inception architecture and residual module to utilize information from previous layers. Inception_resnet_1 model has the same arterial structure as Inception_1 model with an extra connection that adds up the output from the previous layer and output from the inception module. Similarly, Inception_resnet_2 model has two extra connections adding outputs from different layers together. 

![Figure 4](https://user-images.githubusercontent.com/44959050/120256595-0e572180-c2c1-11eb-8aa5-ff9c9fc917dd.png)

## Setup models

Python version: Python3.6

Tensorflow version: Tensorflow 14.0

## Citation 

Please cite our work "Hu M, Sim K, Zhou JH, Jiang X, Guan C. Brain MRI-based 3D Convolutional Neural Networks for Classification of Schizophrenia and Controls. Annu Int Conf IEEE Eng Med Biol Soc. 2020 Jul;2020:1742-1745. doi: 10.1109/EMBC44109.2020.9176610. PMID: 33018334.". 

## Contact

Pre-trained weights are avaialble upon request. Please contact hu.mengjiao01@gmail.com if you need any clarification or would like to contribute. 


