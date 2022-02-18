import torch
import matplotlib.pyplot as plt
import torchvision.transforms as T
import helper as helper
from PIL import Image

num_classes = 1
finetuned_classes = ['balloon',]

model = torch.hub.load('facebookresearch/detr',
                       'detr_resnet50',
                       pretrained=False,
                       num_classes=num_classes)

checkpoint = torch.load('outputs100/checkpoint.pth',
                        map_location='cpu')

model.load_state_dict(checkpoint['model'],
                      strict=False)

model.eval()

# colors for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

# standard PyTorch mean-std input image normalization
transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def plot_finetuned_results(pil_img, prob=None, boxes=None):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    if prob is not None and boxes is not None:
      for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
          ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                    fill=False, color=c, linewidth=3))
          cl = p.argmax()
          text = f'{finetuned_classes[cl]}: {p[cl]:0.2f}'
          ax.text(xmin, ymin, text, fontsize=15,
                  bbox=dict(facecolor='yellow', alpha=0.5))

          print('grids:' + str(xmin) + '-' + str(ymin) + '-' + text)
    # plt.axis('off')
    plt.show()


def run_worflow(my_image, my_model):
    # mean-std normalize the input image (batch-size: 1)
    img = transform(my_image).unsqueeze(0)

    # propagate through the model
    outputs = my_model(img)

    for threshold in [0.9, 0.7, 0.5]:
    # for threshold in [.995]:
        probas_to_keep, bboxes_scaled = helper.filter_bboxes_from_outputs(my_image, outputs,
                                                                   threshold=threshold)

        plot_finetuned_results(my_image,
                               probas_to_keep,
                               bboxes_scaled)


#img_name = 'D:/ITSS/balloon_dataset/balloon/train2017/145053828_e0e748717c_b.jpg'
img_name = './final_test/Screenshot_2.jpg'
im = Image.open(img_name)

run_worflow(im,model)
