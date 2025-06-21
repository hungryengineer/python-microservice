import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

#load the model
model = tf.keras.models.load_model('pneumonia_densenet_model.h5')

img_path = 'pneumonic.jpg'

img = image.load_img(img_path, target_size=(224, 224))  # default is RGB
img_array = image.img_to_array(img)  # shape (224, 224, 3)
img_array = img_array / 255.0  # normalize to [0,1]
img_array = np.expand_dims(img_array, axis=0)  # shape (1, 224, 224, 3)

#run inference
prediction = model.predict(img_array)

# Assuming sigmoid activation, output will be between 0 and 1
if prediction[0][0] > 0.5:
    print(f"Pneumonia detected with confidence {prediction[0][0]:.2f}")
else:
    print(f"Normal chest X-ray with confidence {1 - prediction[0][0]:.2f}")

#visualization
plt.imshow(img, cmap='gray')
plt.title("Pneumonia" if prediction[0][0] > 0.5 else "Normal")
plt.axis('off')
plt.show()
