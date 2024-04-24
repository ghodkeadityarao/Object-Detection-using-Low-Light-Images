import os

import cv2
import keras
import numpy as np
from huggingface_hub import from_pretrained_keras
from PIL import Image


def image_enhancement(path, filename):

    model = from_pretrained_keras("keras-io/lowlight-enhance-mirnet")

    low_light_img = Image.open(path).convert('RGB')
    low_light_img = low_light_img.resize((640,640))

    image = keras.utils.img_to_array(low_light_img)
    image = image / 255.0
    image = np.expand_dims(image, axis = 0)

    output = model.predict(image)
    output_image = output[0] * 255.0
    output_image = output_image.clip(0, 255)

    output_image = output_image.reshape(output_image.shape[0], output_image.shape[1], 3)
    output_image = np.uint32(output_image)
    arr = Image.fromarray(output_image.astype('uint8'),'RGB')
    arr.save('static/enhanced_image/'+filename)

    