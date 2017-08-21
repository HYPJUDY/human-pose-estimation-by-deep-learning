# Human Pose Estimation by Deep Learning
A simple regression based implementation/VGG16 of pose estimation with tensorflow (python) based on [JakeRenn's repository](https://github.com/JakeRenn/pose_estimation).
Please read my [post](https://hypjudy.github.io/2017/05/04/pose-estimation/) for more details about approaches, results, analyses and comprehension of papers: S.-E. Wei, V. Ramakrishna, T. Kanade, and Y. Sheikh. Convolutional pose machines. In CVPR, 2016.

## How to Run
The images in `data/input/` is not complete. If you just want to run the code with demo images, the codes can run without modification. If you want to train and test with complete images:

* `human-pose-estimation-by-deep-learning/data/input/test_imgs`: ~4000 test images
* `human-pose-estimation-by-deep-learning/data/input/train_imgs`: ~60000 train images, use half of them. Each of them are labeled with 15 articulate points.

Please contact with [JakeRenn](https://github.com/JakeRenn/pose_estimation) for data (maybe not available currently). And modify the parametaer `TAG = "_demo"` to `TAG = ""` in corresponding files like `train.py`, `test.py` and `draw_point.py`.

### Train
``` shell
cd human-pose-estimation-by-deep-learning
python train.py # or: nohup python train.py &
```

A directory `log` will be generated to log every training. A directory `params_demo` will also be generated to store your models. With the log and models, you can use tensorboard for visualisation.

``` shell
tensorboard --logdir ./log/train_log/ --port=8008
# then visit: http://hostname:8008/
```

Also, you can modify parameters like batch size, number of max training iteration, number of checkpoint iteration in `train.py`.

### Test
``` shell
python test.py
```

Then an annos file with predicted position of demo test images should be generated in `human-pose-estimation-by-deep-learning/labels/txt/output/test_annos_demo.txt"`.

### Utils
Files in `human-pose-estimation-by-deep-learning\labels\python\` are some utils (in python). For example, you can use `draw_point.py` to draw the points indicated with annos file to images.

``` shell
cd human-pose-estimation-by-deep-learning\labels\python\
python draw_point.py
```

Then the images with 15 points should be generated in `human-pose-estimation-by-deep-learning\data\output\train_imgs_demo\` (I have generated them for your reference). Modify the parameter `PHASE` to `test` will draw test images' points.

For example, the annos file of image `01060538.png` is
```
01060538.png 137,68,136,80,138,105,121,90,107,102,87,114,152,90,168,105,189,118,133,131,130,163,128,201,146,130,149,161,153,199,1
```
This line means image `01060538.png` is labeled with 15 points and their coordinates are (137,68), (136,80), ..., (153,199) (please see the following example output to understand their order). And this line is ended with `'1'`.
By running `draw_point.py`, the output is
```
drawing image 01060538.png
coordinate of point # 1 : ( 137 , 68 )
coordinate of point # 2 : ( 136 , 80 )
coordinate of point # 3 : ( 138 , 105 )
coordinate of point # 4 : ( 121 , 90 )
coordinate of point # 5 : ( 107 , 102 )
coordinate of point # 6 : ( 87 , 114 )
coordinate of point # 7 : ( 152 , 90 )
coordinate of point # 8 : ( 168 , 105 )
coordinate of point # 9 : ( 189 , 118 )
coordinate of point # 10 : ( 133 , 131 )
coordinate of point # 11 : ( 130 , 163 )
coordinate of point # 12 : ( 128 , 201 )
coordinate of point # 13 : ( 146 , 130 )
coordinate of point # 14 : ( 149 , 161 )
coordinate of point # 15 : ( 153 , 199 )
```
![](https://github.com/HYPJUDY/human-pose-estimation-by-deep-learning/blob/master/data/output/train_imgs_demo/01060538.png)
