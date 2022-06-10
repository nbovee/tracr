selected = "MobileNet"

image_size = (224, 224)
if selected == "VGG":
    from tensorflow.keras.applications.vgg16 import VGG16 as model
    from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
    image_size = (224, 224)
if selected == "EfficientNet":
    from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2S as model
    from tensorflow.keras.applications.efficientnet_v2 import preprocess_input, decode_predictions
    image_size = (384, 384)
if selected == "MobileNet":
    from tensorflow.keras.applications.mobilenet import MobileNet as model
    from tensorflow.keras.applications.mobilenet import preprocess_input, decode_predictions
    image_size = (224, 224)
else:
    from tensorflow.keras.applications.resnet50 import ResNet50 as model
    from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
    image_size = (224, 224)

from tensorflow.keras.preprocessing import image
import numpy as np

import colab_vision_client as cv_client
import colab_vision_server as cv_server


model = model(weights='imagenet')
timer = cv_client.cv.timer
print('model is loaded.')
start = timer()
target_dict = cv_client.demo_funct()
target_file = './tmp/morbius_out.jpg'

img_path = 'cause problems'

img_path = target_file if target_file is not None else img_path

start_proc = timer()
print('Begin preprocessing')
img = image.load_img(img_path, target_size=image_size)
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)
start_inf = timer()
# check if tf is lazily loading by running second image
preds = model.predict(x)
end = timer()
target_dict['pre_processing'] = start_inf - start_proc
target_dict['inference'] = end - start_inf
target_dict['overall'] = end - start
# decode the results into a list of tuples (class, description, probability)
# (one such list for each sample in the batch)
print('Predicted:', decode_predictions(preds, top=1)[0])
# print(target_dict)
sum = 0
for e in target_dict:
    val = target_dict[e]/1e9
    val = 0 if val < 0 else val
    print(f"{e} : {val:.4f}s")
    if e is not 'overall':
        sum += val
print(f'measured overall time: {target_dict["overall"]/(1e9)}')
# print((sum(target_dict.values()) - target_dict['overall'])/(1e9))
# print(f'sum of measure times: {sum}')
