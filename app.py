import gradio as gr
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import os

CLASS_NAMES = ['Mild Dementia', 'Moderate Dementia', 'Very mild Dementia']
IMG_SIZE = (224, 224)
IMG_SIZE_HYBRID = (380, 380)

try:
    resnet = keras.models.load_model('models/resnet_final.keras')
    eff = keras.models.load_model('models/efficientnetb4_final.keras')
    hyb = keras.models.load_model('models/hybrid_final_fixed.keras')
except:
    print('Loading models...')

def preprocess_image(img_array, size):
    img = Image.fromarray(img_array).convert('RGB')
    img = img.resize(size)
    return np.expand_dims(np.array(img, dtype=np.float32) / 255.0, axis=0)

def predict_alzheimer(img):
    if img is None:
        return 'Please upload a brain MRI image'
    try:
        img_res = preprocess_image(img, IMG_SIZE)
        img_eff = preprocess_image(img, IMG_SIZE_HYBRID)
        pred_r = resnet.predict(img_res, verbose=0)[0]
        pred_e = eff.predict(img_eff, verbose=0)[0]
        pred_h = hyb.predict([img_res, img_eff], verbose=0)[0]
        avg = (pred_r + pred_e + pred_h) / 3
        idx = np.argmax(avg)
        conf = avg[idx] * 100
        result = f'**{CLASS_NAMES[idx]}** ({conf:.1f}%)\n'
        for i, c in enumerate(CLASS_NAMES):
            result += f'  {c}: {avg[i]*100:.1f}%\n'
        return result
    except Exception as e:
        return f'Error: {e}'

demo = gr.Interface(predict_alzheimer,
                   gr.Image(type='numpy', label='Upload Brain MRI'),
                   gr.Textbox(label='Prediction'),
                   title='🧠 Alzheimer Detection',
                   description='Detect Alzheimer disease stages from brain MRI scans')

if __name__ == '__main__':
    demo.launch()
