from pycocotools.coco import COCO
import os
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
%matplotlib inline

from google.colab import drive
drive.mount('/content/drive')
coco = '/content/drive/My Drive/COCO_Data/grid/result.json'
coco=COCO(coco)
cat_ids = coco.getCatIds()
for i in range(50):
  image_id = i
  img = coco.imgs[image_id]

  print(img['file_name'][81:-4])

  anns_ids = coco.getAnnIds(imgIds=img['id'], catIds=cat_ids, iscrowd=None)
  anns = coco.loadAnns(anns_ids)
  anns_img = np.zeros((img['height'],img['width']))
  for ann in anns:
      anns_img = np.maximum(anns_img,coco.annToMask(ann)*ann['category_id'])


  save_path = '/content/drive/My Drive/COCO_Data/masks'
  plt.imsave(os.path.join(save_path,str(img['file_name'][81:-4])+".png"),anns_img)
