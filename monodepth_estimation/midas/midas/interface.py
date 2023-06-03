import os
import glob
import torch
import cv2
import numpy as np
from enum import Enum
from midas.dpt_depth import DPTDepthModel
from midas.transforms import Resize, NormalizeImage, PrepareForNet
from torchvision.transforms import Compose


def load_model(model_path, model_type="dpt_large_384", optimize=True, height=None, square=False):
    keep_aspect_ratio = not square
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_type == "dpt_large_384":
        model = DPTDepthModel(
            path=model_path,
            backbone="vitl16_384",
            non_negative=True,
        )
        net_w, net_h = 384, 384
        resize_mode = "minimal"
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    elif model_type == "dpt_swin2_large_384":
        model = DPTDepthModel(
            path=model_path,
            backbone="swin2l24_384",
            non_negative=True,
        )
        net_w, net_h = 384, 384
        keep_aspect_ratio = False
        resize_mode = "minimal"
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    else:
        print(f"model_type '{model_type}' not implemented, use: --model_type large")
        assert False
    print("Model loaded, number of parameters = {:.0f}M".format(sum(p.numel() for p in model.parameters()) / 1e6))
    if height is not None:
        net_w, net_h = height, height
    transform = Compose(
        [
            Resize(
                net_w,
                net_h,
                resize_target=None,
                keep_aspect_ratio=keep_aspect_ratio,
                ensure_multiple_of=32,
                resize_method=resize_mode,
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            normalization,
            PrepareForNet(),
        ]
    )
    if optimize and (device == torch.device("cuda")):
        model = model.to(memory_format=torch.channels_last)
        model = model.half()
    model.to(device)
    return model, transform, net_w, net_h


avalibale_model_types = ["dpt_swin2_large_384", "dpt_large_384"]


class DepthEstimator:
    def __init__(self, path: str, model_type: str = "dpt_large_384"):
        assert model_type in avalibale_model_types, f"Model {model_type} is not in {avalibale_model_types}"
        self.model, self.transform, self.net_w, self.net_h = load_model(path, model_type)
        # set torch options
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        self.model.eval()

    def predict(self, image: np.ndarray):  # assumes BGR
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0

        image_input = self.transform({"image": image})["image"]

        with torch.no_grad():
            sample = torch.from_numpy(image_input).to(device).unsqueeze(0)
            prediction = self.model.forward(sample)
            prediction = (
                torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=image.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                )
                .squeeze()
                .cpu()
                .numpy()
            )

        return prediction
