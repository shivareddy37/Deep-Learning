import cv2
import torch
import urllib.request
import matplotlib.pyplot as plt
import numpy as np

url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
urllib.request.urlretrieve(url, filename)

model_type = "DPT_Hybrid"
midas = torch.hub.load("intel-isl/MiDaS", model_type)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
midas.to(device)
midas.eval()
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform

im = cv2.imread(filename)
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

input_batch = transform(im).to(device)

with torch.no_grad():
    prediction = midas(input_batch)

    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=im.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

output = prediction.cpu().numpy()
plt.figure(figsize=(5, 3))
plt.imshow(im)
plt.show()
plt.figure(figsize=(5, 3))
plt.imshow(output)
plt.show()
