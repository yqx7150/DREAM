# Diffusion Transformer Meets Random Masks: An Advanced PET Reconstruction Framework

Author: Bin Huang, Binzhong He, Yanhan Chen, Zhili Liu, Xinyue Wang, Binxuan Li, Qiegen Liu

Deep learning has significantly advanced PET image reconstruction, achieving remarkable improvements in image quality through direct training on sinogram or image data. Traditional methods often utilize masks for inpainting tasks, but their incorporation into PET reconstruction frameworks introduces transformative potential. In this study, we propose an advanced PET reconstruction framework called Diffusion tRansformer mEets rAndom Masks (DREAM). To the best of our knowledge, this is the first work to integrate mask mechanisms into both the sinogram domain and the latent space, pioneering their role in PET reconstruction and demonstrating their ability to enhance reconstruction fidelity and efficiency. The framework employs a high-dimensional stacking approach, transforming masked data from two to three dimensions to expand the solution space and enable the model to capture richer spatial relationships. Additionally, a mask-driven latent space is designed to accelerate the diffusion process by leveraging sinogram-driven and mask-driven compact priors, which reduce computational complexity while preserving essential data characteristics. A hierarchical masking strategy is also introduced, guiding the model from focusing on fine-grained local details in the early stages to capturing broader global patterns over time. This progressive approach ensures a balance between detailed feature preservation and comprehensive context understanding. Experimental results demonstrate that DREAM not only improves the overall quality of reconstructed PET images but also preserves critical clinical details, highlighting its potential to advance PET imaging technology. By integrating compact priors and hierarchical masking, DREAM offers a promising and efficient avenue for future research and application in PET imaging. The open-source code is available at: https://github.com/yqx7150/DREAM.


## Training

To pretrain DiffIR_S1, run
```
sh trainS1.sh
```

To train DiffIR_S2, run
```
#set the 'pretrain_network_g' and 'pretrain_network_S1' in ./options/train_DiffIRS2_sino_nomin.yml to be the path of DiffIR_S1's pre-trained model

sh trainS2.sh
```


## Evaluation



- Testing
```
# modify the dataset path in ./options/test_DiffIRS2.yml

sh test.sh 
```



