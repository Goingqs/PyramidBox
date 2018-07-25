# PyramidBox
This repo implements PyramidBox with pytorch
The paper [PyramidBox](https://arxiv.org/abs/1803.07737) is based on VGG16. But this repo is based on resnet50.

Here, I implements data-anchor-sampling, LFPN, max-in-out layer.

The result on WIDER FACE Val set:

AP Easy | AP Medium | AP Hard
--------|-----------|---------
  95.3  |    94.3   |  89.0   

The trained models will be released later.


## Usage
### Prerequisites
*Python3
*Pytorch3.1
*OpenCV3

### annoPath
annoPath is path to your label file.
The label file should in the following format:
path_to_img1 num_face1 X1 Y1 W1 H1 X2 Y2 ... Wn1 Hn1
path_to_img2 num_face2 X1 Y1 W1 H1 X2 Y2 ... Wn2 Hn2
