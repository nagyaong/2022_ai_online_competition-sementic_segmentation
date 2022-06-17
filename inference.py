import argparse
import os
from collections import defaultdict

import torch
from torch.nn import functional

from transformers import AutoFeatureExtractor

from PIL import Image

import pandas as pd

from tqdm import tqdm

from config import load_config

import numpy as np
from PIL import Image
import scipy.misc
import matplotlib.pyplot as plt 
import albumentations as A
parser = argparse.ArgumentParser(description="Image segmentation")
parser.add_argument("--model_path", type=str, required=True)

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = load_config(os.path.join(os.path.dirname(__file__), "config.yaml"))

model_name = config["pretrained_model_name"]

label2id = {k: v + 1 for k, v in config["label2id"].items()}
label2id["background"] = 0
label2id["container"]=1
label2id["reach"]=2
print(label2id)
id2label = {v: k for k, v in label2id.items()}

feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
feature_extractor.reduce_labels = False

df = pd.read_csv(config["submission_frame_path"])

model = torch.load(args.model_path, map_location=device)
model.eval()

pred_classes = []
pred_segments = []
transform = A.Compose([
    A.CLAHE(clip_limit=(1,4), p=1)
])
for file_name in tqdm(df["file_name"]):
    num=label2id[file_name.split("_")[0]]
    img = Image.open(os.path.join(config["test_image_directory"], file_name))
    image = np.array(img)
    transformed = transform(image = image)
    image = transformed['image']
    encoded_inputs = feature_extractor(image, return_tensors="pt")
    pixel_values = encoded_inputs["pixel_values"].to(device)

    with torch.no_grad():
        outputs = model(pixel_values=pixel_values)
        logits = outputs.logits

        upsampled_logits = functional.interpolate(
            logits,
            size=img.size[::-1],
            mode="bilinear",
            align_corners=False
        )
        predicted = upsampled_logits.argmax(dim=1)[0]
        p=np.array(predicted.cpu().detach().numpy())
        p=np.where(p>0,num,p)
        # print(p.max())
        # im=Image.fromarray(p)
        # im.save(file_name+".png")
        # scipy.misc.toimage(p).save(file_name+".png")
        
        plt.imsave("../data/final_data/train/unlabels/"+file_name[:-4]+".png", p)
    # predicted = predicted.view(-1)

#     labels = defaultdict(list)
#     cnts = defaultdict(int)

#     id = 0
#     start = -1
#     cnt = 0
#     for idx, _id in enumerate(predicted.tolist()):
#         if _id != id:
#             if start == -1:
#                 start = idx
#             else:
#                 labels[id].extend([start, cnt])
#                 cnts[id] += cnt

#                 if _id == 0:
#                     start = -1
#                 else:
#                     start = idx
#                 cnt = 0

#             id = _id

#         if _id == 0:
#             continue

#         cnt += 1

#     max_cnt_id = sorted(cnts.keys(), key=lambda k: cnts[k])[-1]
#     pred_classes.append(id2label[max_cnt_id])
#     pred_segments.append(" ".join(map(str, labels[max_cnt_id])))

# df["class"] = pred_classes
# df["prediction"] = pred_segments

# df.to_csv(f"{args.model_path.split('_')[0]}_submission.csv", index=False)
