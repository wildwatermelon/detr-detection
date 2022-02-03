import torch, torchvision
import torchvision.transforms as T
import helper as helper
from PIL import Image
import requests

model = torch.hub.load('facebookresearch/detr', 'detr_resnet50', pretrained=True)
model.eval()

url = 'http://images.cocodataset.org/train2017/000000310645.jpg'

# standard PyTorch mean-std input image normalization
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

im = Image.open(requests.get(url, stream=True).raw)

# mean-std normalize the input image (batch-size: 1)
img = transform(im).unsqueeze(0)

# propagate through the model
outputs = model(img)

for threshold in [0.9, 0.7, 0.0]:
    probas_to_keep, bboxes_scaled = helper.filter_bboxes_from_outputs(im, outputs, threshold=threshold)
    helper.plot_results(im, probas_to_keep, bboxes_scaled)