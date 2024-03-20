import argparse
import os
import json
from typing import Tuple

import cv2
import torch
import numpy as np
import pandas as pd
import yaml

import cvrt
from models.common import DetectMultiBackend
from utils.torch_utils import select_device, smart_inference_mode
from utils.general import check_img_size, non_max_suppression, scale_boxes
from utils.augmentations import letterbox


def pre_process(
    image: np.array,
    img_size: Tuple[int, int],
    stride: int,
    pt,
    device: str
) -> torch.tensor:
    """
    Preprocessing to be performed on the image before inference
    :param image: Image read using cv2.imread()
    :param img_size: e.g. (640, 640) size expected by the model
    :param stride: Model stride.
    :param pt: Whether or not the model is a PyTorch model
    :param device: PyTorch device string
    :returns: Tensor which can be fed directly into the model for inference.
    """
    im = letterbox(image, img_size, stride=stride, auto=pt)[0]  # padded resize
    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)  # contiguous
    im = torch.from_numpy(im).to(device)
    im = im.float()
    im /= 255  # 0 - 255 to 0.0 - 1.0
    im = im.unsqueeze(0)
    return im


def post_process(post_nms_pred, src_shape, in_tensor_shape, class_labels):
    """
    :param post_nms_pred: Preds from the model after passing through non max suppression
    :param src_shape: Original image shape
    :param in_tensor_shape: Shape of the tensor which you fed into the model
    """
    # Its actually a list, with the length = number of images for that particular batch.
    # However, in this demo we doing 1 by 1 so we expect a length of 1
    assert len(post_nms_pred) == 1, "Demo should be single img inference, not batched"
    det = post_nms_pred[0]

    # Rescale back to original image
    det[:, :4] = scale_boxes(in_tensor_shape[2:], det[:, :4], src_shape).round()
    det = det.cpu().numpy()

    # Manipulation to make it in the format I want
    results = []
    for detection in det:
        x1, y1, x2, y2, conf, class_idx = detection
        results.append({
            "xyxy": list(map(int, [x1, y1, x2, y2])),
            "conf": float(conf),
            "class": class_labels[class_idx]
        })

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_dir", type=str, required=True, help="Path to input folder containing images")
    parser.add_argument("-o", "--output_csv", type=str, required=True, help="Path to output csv")
    parser.add_argument("-d", "--device", type=str, default="cpu", help="Device to use e.g. 'cuda:0', 'cuda:1', 'cpu'")
    parser.add_argument("-w", "--weights", type=str, required=True, help="Path to weights")
    parser.add_argument("--infer_size", type=tuple, default=(640, 640), help="Inference image size")
    parser.add_argument("--conf_thres", type=float, default=0.25, help="Pred minimum confidence")
    parser.add_argument("--iou_thres", type=float, default=0.45, help="Min IOU threshold for NMS")
    parser.add_argument("--max_det", type=int, default=1000, help="Max number of detections")
    parser.add_argument("--config", type=str, default="data/coco.yaml", help="Config which contains label info")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    det_classes = config["names"]

    # Load model
    device = select_device(args.device)
    model = DetectMultiBackend(args.weights, device=device, dnn=False, data=args.config, fp16=False)
    imgsz = check_img_size(args.infer_size, s=model.stride)  # check image size
    model.warmup(imgsz=(1, 3, *args.infer_size))  # warmup

    image_paths = cvrt.path.walk(args.input_dir, {".png"})

    data = {"image_path": [], "detections": []}

    for image_path in image_paths:
        in_image = cv2.imread(image_path)
        in_tensor = pre_process(in_image, args.infer_size, model.stride, model.pt, args.device)
        # Augment -> Augmented inference, Visualize -> Whether to visualize results
        pred = model(in_tensor, augment=False, visualize=False)
        # classes => Filter out classes, Agnostic nms => Class agnostic NMS, Max det = 1000 detections
        pred = non_max_suppression(pred, args.conf_thres, args.iou_thres, classes=None, agnostic=False, max_det=args.max_det)

        obj_labels = post_process(pred, in_image.shape, in_tensor.shape, det_classes)

        data["image_path"].append(image_path)
        data["detections"].append(json.dumps(obj_labels))

        # Viz related
        for obj_label in obj_labels:
            xyxy = obj_label["xyxy"]
            cv2.rectangle(in_image, xyxy[:2], xyxy[2:], (0, 255, 255), 1)
        cv2.imshow("in", in_image)
        key = cv2.waitKey(0)
        if key & 255 == 27:
            break

    cv2.destroyAllWindows()

    df = pd.DataFrame(data)
    df.to_csv(args.output_csv, index=False)
