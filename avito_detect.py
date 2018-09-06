
import argparse
import keras

# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

# import miscellaneous modules
import cv2
import os
import numpy as np
import time

# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

# use this environment flag to change which GPU to use
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# set the modified tf session as backend in keras
keras.backend.tensorflow_backend.set_session(get_session())

# adjust this to point to your downloaded/trained model
# models can be downloaded here: https://github.com/fizyr/keras-retinanet/releases
model_path = os.path.join('keras-retinanet', 'snapshots', 'resnet50_coco_best_v2.1.0.h5')

# load retinanet model
model = models.load_model(model_path, backbone_name='resnet50')


def arg_parse():
    """
    Parse arguements to the detect module
    """
    parser = argparse.ArgumentParser(description='Detection Module')
    parser.add_argument("--image", dest = 'image', help = 
                        "Image path to perform detection upon",
                        default = "img", type = str)
    
    return parser.parse_args()


def area(coords):
    return (coords[2]-coords[0])*(coords[3]-coords[1])



def set_normalize_function(height, width):
    
    def normalize_boxes(box_coords):
        x1 = box_coords[0] / width
        y1 = box_coords[1] / height
        x2 = box_coords[2] / width
        y2 = box_coords[3] / height
        return np.array([x1, y1, x2, y2])
    
    return normalize_boxes



def get_localization(image):
    # copy to draw on
    draw = image.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

    # preprocess image for network
    #print(image.shape)
    image = preprocess_image(image)
    image, scale = resize_image(image)
    #print(image.shape)

    # process image
    #start = time.time()
    boxes, probas, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
    
    inds = np.where(~(probas[0] == -1))[0]
    probas = probas[0][inds]
    boxes = boxes[0][inds]

    new_h, new_w, _ = image.shape
    
    normalize_boxes = set_normalize_function(new_h, new_w)
    boxes = np.array(list(map(normalize_boxes, boxes)))
    
    
    # hyperparameters
    ALPHA = np.linspace(0,1,20)[13] # 0.6842105263157894
    TH = 0.25
    k = 3
    CONST_BOX = (0.04, 0.04, 0.96, 0.96)
    
    if len(boxes) == 0:
    	x1, y1, x2, y2 = CONST_BOX
        return x1, y1, x2, y2
    
    biggest_area_inds = np.argsort(list(map(area, boxes)))[::-1]
    biggest_area = np.max(list(map(area, boxes)))
    biggest_proba = probas[biggest_area_inds[0]]
    proba_area_inds = (np.argsort((np.array(list(map(area, boxes))) / biggest_area)\
                                   * (probas / biggest_proba))[::-1])

    boxes = boxes[proba_area_inds[:k]]
    boxes = np.hstack([boxes[:,:2].min(axis=0), boxes[:,2:].max(axis=0)])
    answer = ALPHA * boxes + (1 - ALPHA) * np.array(list(CONST_BOX))
    if area(answer) <= TH:
        answer = CONST_BOX
    
    x1, y1, x2, y2 = answer
    return x1, y1, x2, y2


if __name__ ==  '__main__':
    args = arg_parse()
    image_path = args.image
    image = read_image_bgr(image_path)
    localication_coordinates = get_localization(image)
    print(localication_coordinates)
