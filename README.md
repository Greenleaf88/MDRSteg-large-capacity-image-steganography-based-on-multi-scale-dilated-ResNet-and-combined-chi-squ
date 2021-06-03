Please cite below paper if the codes are helpful to you
[1] Mo, L. ,  Zhu, L. ,  Ma, J. ,  Wang, D. , &  Wang, H. . (2021). Mdrsteg: large-capacity image steganography based on multi-scale dilated resnet and combined chi-square distance loss. Journal of Electronic Imaging, 30(1).

# Experiment summary

- dataset: VOC2012

  train: 11540

  test: 5585 



**note**: test set uses half size images to cover other half, with the same crop and resize method



# 1. Gray Image hiding

#### 1.1 first experiment

- network: vanilla convolution net
- training details: other setting is similar to the original paper, hiding secret image in Y (YUV) space, loss use mean square error add ssim loss. **but without mssim loss, and not using any hyper parameters(alpha, beta, gamma) in the mix loss**
- test performance:

|secret mse|cover mse  |secret ssim |cover ssim |
| ------- | ----------- | ------ | -------- |
|0.0021 | 0.0426 |  0.0509 | 0.0536|





#### 1.2 second experiment

- network: vanilla convolution net with dilation convolution  to expand the receptive field 
- training details: same as above

| Layer # | Kernel Size | Stride | Dilation | Padding | Input Size | Output Size | Receptive Field |
| ------- | ----------- | ------ | -------- | ------- | ---------- | ----------- | --------------- |
| 1       | 3           | 1      | 1        | 2       | 256        | 256         | 3               |
| 2       | 3           | 1      | 1        | 2       | 256        | 256         | 5               |
| 3       | 3           | 1      | 2        | 4       | 256        | 256         | 9               |
| 4       | 3           | 1      | 2        | 4       | 256        | 256         | 13              |
| 5       | 3           | 1      | 4        | 8       | 256        | 256         | 21              |
| 6       | 3           | 1      | 4        | 8       | 256        | 256         | 29              |
| 7       | 3           | 1      | 8        | 16      | 256        | 256         | 45              |
| 8       | 3           | 1      | 8        | 16      | 256        | 256         | 61              |
| 9       | 3           | 1      | 16       | 32      | 256        | 256         | 93              |
| 10      | 3           | 1      | 16       | 32      | 256        | 256         | 125             |
| 11      | 3           | 1      | 32       | 64      | 256        | 256         | 189             |
| 12      | 3           | 1      | 32       | 64      | 256        | 256         | 253             |



- test performance:

| secret mse | cover mse | secret ssim | cover ssim |
| ---------- | --------- | ----------- | ---------- |
| 0.0056     | 0.0010    | 0.0864      | 0.0633     |



#### 1.3 third experiment

- network: restore first vanilla convolution and use smaller learning rate to finetune its weights
- training details: same as above
- test performance:

| secret mse | cover mse | secret ssim | cover ssim |
| ---------- | --------- | ----------- | ---------- |
| **1253.8170**  | 0.0106    | 0.0665      | 0.0408     |





#### 1.4 fourth experiment

- network: use first vanilla net add GAN training
- training details: same as above. As for the GAN part, the discriminator is also bit different with the original repository (XuNet is used in the original repo), **I use relu to replaced the tanh, max pooling to replace the average pooling.**
- test performance:

![](img/0417-1820/loss_result.jpg)

![](img/0417-1820/ssim_result.jpg)

| secret mse | cover mse | secret ssim | cover ssim |
| ---------- | --------- | ----------- | ---------- |
| 0.0018     | 0.0012    | 0.0630      | 0.0542     |



#### 1.5 fifth experiment

- network: use resnet50 in both hiding and reveal network to replace vanilla convolution net, without GAN

- training details: same as above

- Architecture (Drawio )

  ![](img/resnet.jpg)

- resnet receptive field calculation

  | Layer # | Kernel Size | Stride | Dilation | Padding | Input Size | Output Size | Receptive Field |
  | ------- | ----------- | ------ | -------- | ------- | ---------- | ----------- | --------------- |
  | 1       | 3           | 1      | 1        | 2       | 256        | 256         | 3               |
  | 2       | 3           | 1      | 1        | 2       | 256        | 256         | 5               |
  | 3       | 3           | 1      | 1        | 2       | 256        | 256         | 7               |
  | 4       | 3           | 1      | 1        | 2       | 256        | 256         | 9               |
  | 5       | 3           | 1      | 1        | 2       | 256        | 256         | 11              |
  | 6       | 3           | 1      | 1        | 2       | 256        | 256         | 13              |
  | 7       | 3           | 1      | 1        | 2       | 256        | 256         | 15              |
  | 8       | 3           | 1      | 1        | 2       | 256        | 256         | 17              |
  | 9       | 3           | 1      | 1        | 2       | 256        | 256         | 19              |
  | 10      | 3           | 1      | 1        | 2       | 256        | 256         | 21              |
  | 11      | 3           | 1      | 1        | 2       | 256        | 256         | 23              |
  | 12      | 3           | 1      | 1        | 2       | 256        | 256         | 25              |
  | 13      | 3           | 1      | 1        | 2       | 256        | 256         | 27              |
  | 14      | 3           | 1      | 1        | 2       | 256        | 256         | 29              |
  | 15      | 3           | 1      | 1        | 2       | 256        | 256         | 31              |
  | 16      | 3           | 1      | 1        | 2       | 256        | 256         | 33              |
  | 17      | 3           | 1      | 1        | 2       | 256        | 256         | 35              |
  | 18      | 3           | 1      | 1        | 2       | 256        | 256         | 37              |
  | 19      | 3           | 1      | 1        | 2       | 256        | 256         | 39              |
  | 20      | 3           | 1      | 1        | 2       | 256        | 256         | 41              |
  | 21      | 3           | 1      | 1        | 2       | 256        | 256         | 43              |

- test performance:



![](img/0419-1121/ssim_result.jpg)



| secret mse | cover mse | secret ssim | cover ssim |
| ---------- | --------- | ----------- | ---------- |
| 0.0017     | 0.0009    | 0.0697      | 0.0495     |



#### 1.6 sixth experiment

- network: restore fifth network, changed data supply, use random pair(buffer size=100) cover and secret to finetune the weights
- training details: same as above
- test performance:



![](img/0421-0857/ssim_result.jpg)

**note**:  figure use different test data(different pair of cover and secret like above experiments) during the validation, it also shows that different pair of cover and secret image can vary in the final ssim score.



| secret mse | cover mse | secret ssim | cover ssim |
| ---------- | --------- | ----------- | ---------- |
| 0.0012     | 0.0011    | 0.0463      | 0.0559     |
| 0.0014     | 0.0013    | 0.0575      | 0.0476     |





#### All together test result comparison


||secret mse|cover mse  |secret ssim |cover ssim |
|-| ------- | ----------- | ------ | -------- |
|1|0.0021 | 0.0426 |  0.0509 | 0.0536|
|2| 0.0056     | 0.0010    | 0.0864      | 0.0633     |
|3| **1253.8170**  | 0.0106    | 0.0665      | 0.0408     |
|4| 0.0018     | 0.0012    | 0.0630      | 0.0542     |
|5| 0.0017     | 0.0009    | 0.0697      | 0.0495     |
|6| 0.0012     | 0.0011    | 0.0463      | 0.0559     |





# 2. RGB Image Hiding

#### 2.1 

- network: use resnet50 in hiding and reveal network, without GAN

- training details: transform RGB cover and secret image both to YUV space, use MAE loss and SSIM loss, both calculated in YUV space.

- test performance:

  | secret mae | cover mae | secret ssim | cover ssim |
  | ---------- | --------- | ----------- | ---------- |
  | 0.0155     | 0.0138    | 0.0676      | 0.1269     |



#### 2.2
- network: use resnet50 but add dilation convolution in some of its residual blocks , without GAN
- training details: same as above
- Architecture

#### dilation resnet

| Layer # | Kernel Size | Stride | Dilation | Padding | Input Size | Output Size | Receptive Field |
| ------- | ----------- | ------ | -------- | ------- | ---------- | ----------- | --------------- |
| 1       | 3           | 1      | 1        | 2       | 256        | 256         | 3               |
| 2       | 3           | 1      | 2        | 4       | 256        | 256         | 7               |
| 3       | 3           | 1      | 2        | 4       | 256        | 256         | 11              |
| 4       | 3           | 1      | 2        | 4       | 256        | 256         | 15              |
| 5       | 3           | 1      | 4        | 8       | 256        | 256         | 23              |
| 6       | 3           | 1      | 4        | 8       | 256        | 256         | 31              |
| 7       | 3           | 1      | 4        | 8       | 256        | 256         | 39              |
| 8       | 3           | 1      | 4        | 8       | 256        | 256         | 47              |
| 9       | 3           | 1      | 4        | 8       | 256        | 256         | 55              |
| 10      | 3           | 1      | 8        | 16      | 256        | 256         | 71              |
| 11      | 3           | 1      | 8        | 16      | 256        | 256         | 87              |
| 12      | 3           | 1      | 8        | 16      | 256        | 256         | 103             |
| 13      | 3           | 1      | 8        | 16      | 256        | 256         | 119             |
| 14      | 3           | 1      | 8        | 16      | 256        | 256         | 135             |
| 15      | 3           | 1      | 8        | 16      | 256        | 256         | 151             |
| 16      | 3           | 1      | 8        | 16      | 256        | 256         | 167             |
| 17      | 3           | 1      | 16       | 32      | 256        | 256         | 199             |
| 18      | 3           | 1      | 16       | 32      | 256        | 256         | 231             |
| 19      | 3           | 1      | 16       | 32      | 256        | 256         | 263             |
| 20      | 3           | 1      | 16       | 32      | 256        | 256         | 295             |
| 21      | 3           | 1      | 1        | 2       | 256        | 256         | 297             |

- test performance:

  | secret mae | cover mae | secret ssim | cover ssim |
  | ---------- | --------- | ----------- | ---------- |
  | 0.0200     | 0.0138    | 0.1840      | 0.1445     |
  
