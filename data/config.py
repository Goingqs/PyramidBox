# config.py
import os.path

face = {
    'feature_maps' : [160, 80, 40, 20, 10, 5],

    'min_dim' : 640,

    'steps' : [4, 8, 16, 32, 64, 128],

    'min_sizes' : [16, 32, 64, 128, 256, 512],

    'max_sizes' : [1,1,1,1,1,1],

    'aspect_ratios' : [[],[],[],[],[],[]],

    'variance' : [0.1, 0.2],

    'clip' : False,

    'name' : 'v2',
}
