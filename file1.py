from pycocotools.coco import COCO
import os
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
%matplotlib inline

from google.colab import drive
drive.mount('/content/drive')
coco = '/content/drive/My Drive/COCO_Data/result.json'

img_dir = '/content/drive/My Drive/COCO_Data/img.png'

image = np.array(Image.open(img_dir))
plt.imshow(image, interpolation='nearest')
plt.show()

coco=COCO(coco)

cat_ids = coco.getCatIds()
anns_ids = coco.getAnnIds(imgIds=img['id'], catIds=cat_ids, iscrowd=None)
anns = coco.loadAnns(anns_ids)
anns_img = np.zeros((img['height'],img['width']))
for ann in anns:
    anns_img = np.maximum(anns_img,coco.annToMask(ann)*ann['category_id'])

plt.imshow(anns_img, interpolation='nearest')
plt.show()
