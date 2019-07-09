from keras.preprocessing import image
import numpy as np
from keras.models import Model, load_model 
from mobilenet_v2 import MobileNetv2

model = MobileNetv2((224,224, 3), 34)
model.load_weights("./model/weights.h5")

img_path = '/home/eric/data/flower_photos/roses/353897245_5453f35a8e.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
# x = preprocess_input(x)
x=x/255.0
print(x[0])

preds = model.predict(x)
print(preds)
# preds=np.argmax(preds)
# # print('Predicted:', decode_predictions(preds))
# print(preds)
# print(label_dict[str(preds)])