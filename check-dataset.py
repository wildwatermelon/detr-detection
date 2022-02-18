import pycocotools.coco as coco
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab

pylab.rcParams['figure.figsize'] = (10.0, 8.0)
dataDir='D:/ITSS/balloon_dataset/balloon/'
dataType='train2017'
annFile='{}annotations/custom_train.json'.format(dataDir)

# initialize COCO api for instance annotations
coco=COCO(annFile)

# display COCO categories and supercategories
cats = coco.loadCats(coco.getCatIds())

nms=[cat['name'] for cat in cats]
print('Categories: {}'.format(nms))

nms = set([cat['supercategory'] for cat in cats])
print('Super-categories: {}'.format(nms))

# load and display image
catIds = coco.getCatIds(catNms=['balloon'])
imgIds = coco.getImgIds(catIds=catIds )

#img_id = imgIds[np.random.randint(0,len(imgIds))]
img_id = imgIds[0]
print('Image n°{}'.format(img_id))

img = coco.loadImgs(img_id)[0]

print(img)
img_name = '%s/%s/%s'%(dataDir, dataType, img['file_name'])
print('Image name: {}'.format(img_name))

I = io.imread(img_name)
plt.figure()
plt.imshow(I)
plt.show()

annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds)
anns = coco.loadAnns(annIds)

# load and display instance annotations
plt.imshow(I)
coco.showAnns(anns, draw_bbox=False)
plt.show()

plt.imshow(I)
coco.showAnns(anns, draw_bbox=True)
plt.show()