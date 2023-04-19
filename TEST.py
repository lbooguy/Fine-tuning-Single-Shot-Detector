import torch
from PIL import Image
from torchvision.io.image import read_image
from torchvision.models.detection import SSD300_VGG16_Weights
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import draw_bounding_boxes

image_path = "./1.jpg"
img = Image.open(image_path)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model = torch.load(
    "./SSD.pt")
model.score_thresh = 0.5
model = model.to('cpu')
model.eval()

w = SSD300_VGG16_Weights.DEFAULT
preprocess = w.transforms()
batch = [preprocess(img)]
prediction = model(batch)[0]
print(prediction)

image = read_image(image_path)
labels = ['robot arm' + ' ' + str(100 * round(prediction['scores'].item(), 3)) + '%' for i in prediction["labels"]]
box = draw_bounding_boxes(image, boxes=prediction["boxes"],
                          labels=labels,
                          colors="white",
                          width=4,
                          font='C:/Windows/Fonts/Arial.ttf',
                          font_size=100)
im = to_pil_image(box.detach())
im.show()
