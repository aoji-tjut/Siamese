import os
import numpy as np
from PIL import Image
from siamese import Siamese

model = Siamese()
target = Image.open("./datasets/images_background/10/10_1.png")

res = []
imgs = os.listdir("./all/")
for img in imgs:
    temp = Image.open("./all/" + img)
    probability = model.detect_image(target, temp)
    res.append(probability)
    if probability > 0.95:
        print(img, probability)

# res = np.array(res)
# res_sort = np.sort(res)
# print(res_sort[:4])
# print(res.argsort())
# for i in range(4):
#     print(imgs[res.argsort()[i]])
#     print(imgs[res.argsort()[i]])
