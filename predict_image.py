import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

model = tf.keras.models.load_model("model/model.h5")

img_path = "sample.jpg"  

img = image.load_img(img_path, target_size=(128, 128))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = img_array / 255.0

prediction = model.predict(img_array)

if prediction[0][0] > 0.5:
    print("REAL IMAGE")
    print("Confidence:", prediction[0][0] * 100)
else:
    print("AI GENERATED IMAGE")
    print("Confidence:", (1 - prediction[0][0]) * 100)