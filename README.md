# Human Pose Estimation by Deep Learning
A simple regression based implementation/VGG16 of pose estimation with tensorflow (python) based on [JakeRenn's repository](https://github.com/JakeRenn/pose_estimation).
Please read my [post](https://hypjudy.github.io/2017/05/04/pose-estimation/) for more details about approaches, results, analyses and comprehension of papers: S.-E. Wei, V. Ramakrishna, T. Kanade, and Y. Sheikh. Convolutional pose machines. In CVPR, 2016.

## How to Run
The images in `data/input/` is not complete. If you just want to run the code with demo images, the codes can run without modification. If you want to train and test with complete images:

pose\_estimation/data/input/test_imgs: ~4000 test images
pose\_estimation/data/input/train_imgs: ~60000 train images, use half of them
Each of them are labeled with 15 articulate points.

Please contact with me for data. And modify the parametaer `TAG = "_demo"` to `TAG = ""` in corresponding files like `train.py`, `test.py` and `draw_point.py`.

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

Then the images with 15 points should be generated in `human-pose-estimation-by-deep-learning\data\output\train_imgs_demo\`. Modify the parameter `PHASE` to `test` will draw test images' points.