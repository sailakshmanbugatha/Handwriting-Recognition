from flask import Flask,escape,request,render_template
import tensorflow as tf
import io
import os
import cv2
import keras
import easyocr
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from spellchecker import SpellChecker
from gtts import gTTS


reader = easyocr.Reader(['en'],gpu=False)
app = Flask(__name__)
mymodel = tf.keras.models.load_model('mymodel_new.h5',compile=False)

from tensorflow.keras.layers.experimental.preprocessing import StringLookup
characters = sorted(['!', '"', '#', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '?', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'])
AUTOTUNE = tf.data.AUTOTUNE
char_to_num = StringLookup(vocabulary=list(characters), mask_token=None)
num_to_char = StringLookup(vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True)


image_width = 128
image_height = 32
batch_size = 64
padding_token = 99
max_len = 21

def distortion_free_resize(image, img_size):
    w, h = img_size
    image = tf.image.resize(image, size=(h, w), preserve_aspect_ratio=True)
    pad_height = h - tf.shape(image)[0]
    pad_width = w - tf.shape(image)[1]
    
    if pad_height % 2 != 0:
        height = pad_height // 2
        pad_height_top = height + 1
        pad_height_bottom = height
    else:
        pad_height_top = pad_height_bottom = pad_height // 2

    if pad_width % 2 != 0:
        width = pad_width // 2
        pad_width_left = width + 1
        pad_width_right = width
    else:
        pad_width_left = pad_width_right = pad_width // 2

    image = tf.pad(image, paddings=[ [pad_height_top, pad_height_bottom], [pad_width_left, pad_width_right], [0, 0],],)
    image = tf.transpose(image, perm=[1, 0, 2])
    image = tf.image.flip_left_right(image)
    return image

def segmentation(image):
    img = cv2.imdecode(np.frombuffer(image.read(), np.uint8), 1)
    result = reader.readtext(img)
    
    for count,detection in enumerate(result): 
        top_left = tuple(detection[0][0])
        bottom_right = tuple(detection[0][2])
        try:
            temp = img[top_left[1]:bottom_right[1],top_left[0]:bottom_right[0]]
            cv2.imwrite("Segmented/{}{}.jpg".format("0",str(count).zfill(3)), temp)
            count += 1
        except TypeError:
            pass

def preprocess_image(image_path, img_size=(image_width, image_height)):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, 1)
    image = distortion_free_resize(image, img_size)
    image = tf.cast(image, tf.float32) / 255.0
    return image


def vectorize_label(label):
    label = char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
    length = tf.shape(label)[0]
    pad_amount = max_len - length
    label = tf.pad(label, paddings=[[0, pad_amount]], constant_values=padding_token)
    return label

def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][:,:max_len]
    output_text = []
    for res in results:
        res = tf.gather(res, tf.where(tf.math.not_equal(res, -1)))
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8")
        output_text.append(res)
    return output_text

def process_images_labels(image_path, label):
    image = preprocess_image(image_path)
    label = vectorize_label(label)
    return {"image": image, "label": label}


def prepare_dataset(image_paths, labels):
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels)).map(process_images_labels, num_parallel_calls=AUTOTUNE)
    return dataset.batch(batch_size).cache().prefetch(AUTOTUNE)

@app.route('/')
def hello():
    return render_template("index.html")

@app.route('/index',methods=['GET','POST'])
def index():
    if request.method == 'POST':
        image = request.files['image']
        # Performing the segmentation
        segmentation(image)

        # Setting path for Segmentation
        seg_path = 'Segmented'

        # Getting the path of Segmented Images
        images_paths = os.listdir(seg_path)
        images_paths = [os.path.join(seg_path,i) for i in images_paths]

        # Making a temporary labels
        labels = [' '] * len(images_paths)

        # Prepraing the Input Pipeline
        test=prepare_dataset(images_paths,labels)

        # Extracting only the Image
        temp = [x['image'] for x in test]

        # Predicing the results
        preds = mymodel.predict(temp)

        # Getting the text output from the numerical ones
        pred_texts = decode_batch_predictions(preds)

        # Creating the Spell Checker Object
        spell = SpellChecker()
        output = []

        # Iterating over the predicted words and modifying the using spell checker
        for word in pred_texts:
            Corr_Word = spell.correction(word)
            if type(Corr_Word) != type(None):
                output.append(Corr_Word)
            else:
                output.append(word)

        # Coverting the corpus of words to a String
        output = " ".join(map(str,output)).lower().title()

        # Preparing the Audio file gTTs
        language = 'en'
        myobj = gTTS(text=output, lang=language, slow=False)

        # Saving the Audio file in Static Folder
        myobj.save("static/hwt.mp3")
        os.system("mpg321 hwt.mp3")

        # Deleting all the Segemented Images in Segmented folder for Multiple Submissions in front end.
        for i in images_paths:
            if os.path.isfile(i):
                os.remove(i)
                
        return render_template("index.html",prediction_text="{}".format(output))
    else:
        return render_template("index.html")


if __name__ == '__main__':
    app.run(debug=True)