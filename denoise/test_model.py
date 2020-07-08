import numpy as np
from pathlib import Path
import cv2
from model import get_model
from noise_model import get_noise_model



def get_image(image):
    image = np.clip(image, 0, 255)
    return image.astype(dtype=np.uint8)


def start():
    image_dir = './after.png'
    weight_file = './denoise/weights.gauss_clean.hdf5'
    val_noise_model = get_noise_model("clean")
    model = 'srresnet'
    output_dir = './denoise/results'
    model = get_model(model)
    model.load_weights(weight_file)

    image = cv2.imread(image_dir)
    h, w, _ = image.shape
    image = image[:(h // 16) * 16, :(w // 16) * 16]  # for stride (maximum 16)
    h, w, _ = image.shape

    out_image = np.zeros((h, w * 3, 3), dtype=np.uint8)
    noise_image = val_noise_model(image)
    pred = model.predict(np.expand_dims(noise_image, 0))
    denoised_image = get_image(pred[0])
    out_image[:, :w] = image
    out_image[:, w:w * 2] = noise_image
    out_image[:, w * 2:] = denoised_image
    cv2.imwrite('./denoise/results/denoise.png', out_image[:, w * 2:])
