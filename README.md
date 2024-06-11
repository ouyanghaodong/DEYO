<h2 align="center">DEYO: DETR with YOLO for End-to-End Object Detection</h2>

![deyo_overview](https://github.com/ouyanghaodong/DEYO/blob/main/figs/1.png)

This is the official implementation of the paper "[DEYO: DETR with YOLO for End-to-End Object Detection](https://arxiv.org/abs/2402.16370)".

| Model | Epoch | Shape | $AP^{val}$ | $AP^{val}_{50}$ | Params(M) | FLOPs(G) | T4 TensorRT FP16(FPS) |
|:------|:-----:|:-----------:|:----------:|:---------------:|:---------:|:--------:|:---------------------:|
| [DEYO-tiny](https://github.com/ouyanghaodong/DEYO/releases/download/v0.1/deyo-tiny.pt) | 96 | 640 | 37.6 | 52.8 | 4 | 8 | 497 |
| [DEYO-N](https://github.com/ouyanghaodong/DEYO/releases/download/v0.1/deyo-n.pt) | 96 | 640 | 39.7 | 55.6 | 6 | 10 | 396 |
| [DEYO-S](https://github.com/ouyanghaodong/DEYO/releases/download/v0.1/deyo-s.pt) | 96 | 640 | 45.8 | 62.9 | 14 | 26 | 299 |
| [DEYO-M](https://github.com/ouyanghaodong/DEYO/releases/download/v0.1/deyo-m.pt) | 96 | 640 | 50.7 | 68.4 | 33 | 78 | 140 |
| [DEYO-L](https://github.com/ouyanghaodong/DEYO/releases/download/v0.1/deyo-l.pt) | 96 | 640 | 52.7 | 70.2 | 51 | 155 | 100 |
| [DEYO-X](https://github.com/ouyanghaodong/DEYO/releases/download/v0.1/deyo-x.pt) | 96 | 640 | 53.7 | 71.3 | 78 | 242 | 65 |

The upgraded version of DEYO, [DEYOv1.5](https://github.com/ouyanghaodong/DEYOv1.5) has been released🔥🔥🔥

The end-to-end speed results following the method proposed in RT-DETR. We believe this does not fully reflect the speed relationship between DEYO and YOLO, and should only be considered as a reference. In fact, the latency of NMS is related to the edge device and the number of objects in the processed image. We suggest using DEYO when NMS becomes a bottleneck in detection speed or in detecting dense scenes (DEYO eliminates the reliance on NMS, ensuring that even if two objects overlap significantly, they will not be mistakenly filtered out by NMS).

| Model        | Shape | score_threshold | iou | $AP^{val}$ | $AP^{val}_{50}$ | T4 TensorRT FP16 (FPS) |
|:-------------|:-----:|:---------------:|:---:|:------:|:---------:|:----------------------:|
| YOLOv8-N     | 640   | 0.001           | 0.7 | 37.3   | 52.5      | 163                    |
| YOLOv8-N     | 640   | 0.005           | 0.7 | --     | --        | 640                    |
| YOLOv8-N     | 640   | 0.250           | 0.7 | --     | --        | 643                    |

Based on the findings, when NMS becomes a speed bottleneck (score_threshold=0.001), DEYO-tiny's FPS is three times that of YOLOv8-N. However, when the NMS post-processing time is shorter than the computation time for DEYO's one-to-one branch (score_threshold=0.005), DEYO-tiny does not maintain a speed advantage. It's important to note that in deployment, we typically would not use such a low threshold as score_threshold=0.001, but on edge devices, the execution time for NMS could become even slower. We recommend testing the speed of YOLOv8 and DEYO separately according to your actual use case scenarios.

| Model | Epoch | End-to-End | $AP^{val}$ | $AP^{val}_{50}$ | Params(M) | FLOPs(G) | T4 TRT FP16(FPS) |
|:------|:-----:|:-----------:|:----------:|:---------------:|:---------:|:--------:|:---------------:|
| YOLOv8-N | --  | ✔ | 37.3 | 52.5 | 3    | 9     | 565 | 
| DEYO-tiny | 96 | ✔ | 37.6 | 52.8 | 4    | 8     | 497 |

It is noteworthy that during the integration of EfficientNMS, optimizations were made to the ONNX model, and DEYO-tiny can, at a minimum, achieve 88% of the speed of YOLOv8 in the worst-case scenario.

## Introduction
We propose a brand-new end-to-end real-time object detector called DEYO. DEYO surpasses all existing real-time object detectors in terms of speed and accuracy. It is worth noting that the comprehensive DEYO series can complete its second phase training on the COCO dataset using a single 8GB RTX 4060 GPU.
<div align="center">
  <img src="https://github.com/ouyanghaodong/DEYO/blob/main/figs/2.png" width=500 >
</div>

## Install
```bash
pip install ultralytics
```

## Config
| Model              | Queries           | Neck                | Hidden Dimension | GPU Memory(Training) |
|:-------------------|:-----------------:|:--------------------|:----------------:|-----------:|
| YOLOv8-N           |         N/A       | (64, 128, 256)      |       N/A        |  3247 MiB  |
| YOLOv8-S           |         N/A       | (64, 128, 512)      |       N/A        |  4857 MiB  |
| YOLOv8-M           |         N/A       | (192, 384, 576)     |       N/A        |  7081 MiB  |
| YOLOv8-L           |         N/A       | (256, 512, 512)     |       N/A        | 10503 MiB  |
| YOLOv8-X           |         N/A       | (320, 640, 640)     |       N/A        | 13069 MiB  |
| DEYO-tiny          |        100        | (64, 128, 256)      |        64        |  2238 MiB  |
| DEYO-N             |        300        | (64, 128, 256)      |       128        |  4746 MiB  |
| DEYO-S             |        300        | (64, 128, 256)      |       128        |  5062 MiB  |
| DEYO-M             |        300        | (192, 384, 576)     |       256        |  6444 MiB  |
| DEYO-L             |        300        | (256, 512, 512)     |       256        |  6476 MiB  |
| DEYO-X             |        300        | (320, 640, 640)     |       320        |  6888 MiB  |
| DEYO-tiny (No CDN) |        100        | (64, 128, 256)      |        64        |  1514 MiB  |
| DEYO-N (No CDN)    |        300        | (64, 128, 256)      |       128        |  2700 MiB  |
| DEYO-S (No CDN)    |        300        | (64, 128, 512)      |       128        |  3108 MiB  |
| DEYO-M (No CDN)    |        300        | (192, 384, 576)     |       256        |  3948 MiB  |
| DEYO-L (No CDN)    |        300        | (256, 512, 512)     |       256        |  4216 MiB  |
| DEYO-X (No CDN)    |        300        | (320, 640, 640)     |       320        |  5194 MiB  |

### You can configure DEYO according to the table above, controlling the scale of DEYO through yolov8-rtdetr.yaml. You can simply select the scale by using "#". Below is an example of selecting a scale of X.
```python
# Parameters
nc: 80  # number of classes
scales: # model compound scaling constants, i.e. 'model=yolov8n.yaml' will call yolov8.yaml with scale 'n'
  # [depth, width, max_channels]
  # n: [0.33, 0.25, 1024]  # YOLOv8n summary: 225 layers,  3157200 parameters,  3157184 gradients,   8.9 GFLOPs
  # s: [0.33, 0.50, 1024]  # YOLOv8s summary: 225 layers, 11166560 parameters, 11166544 gradients,  28.8 GFLOPs
  # m: [0.67, 0.75, 768]   # YOLOv8m summary: 295 layers, 25902640 parameters, 25902624 gradients,  79.3 GFLOPs
  # l: [1.00, 1.00, 512]   # YOLOv8l summary: 365 layers, 43691520 parameters, 43691504 gradients, 165.7 GFLOPs
  x: [1.00, 1.25, 512]   # YOLOv8x summary: 365 layers, 68229648 parameters, 68229632 gradients, 258.5 GFLOPs
```

### While modifying the scale, you need to configure the RTDETRDecoder.
```python
# Open ultralytics/nn/modules/head.py 
# Find RTDETRDecoder
# DEYO-tiny nq=100 hd=64  o2m: yolov8n
# DEYO-N    nq=300 hd=128 o2m: yolov8n
# DEYO-S    nq=300 hd=128 o2m: yolov8s
# DEYO-M    nq=300 hd=256 o2m: yolov8m
# DEYO-L    nq=300 hd=256 o2m: yolov8l
# DEYO-X    nq=300 hd=320 o2m: yolov8x
def __init__(
   self,
   nc=80,
   ch=(512, 1024, 2048),
   hd=64,  # hidden dim
   nq=100,  # num queries
   ndp=4,  # num decoder points
   nh=8,  # num head
   ndl=6,  # num decoder layers
   d_ffn=1024,  # dim of feedforward
   dropout=0.0,
   act=nn.ReLU(),
   eval_idx=-1,
   # Training args
   nd=100,  # num denoising
   label_noise_ratio=0.5,
   box_noise_scale=1.0,
   learnt_init_query=False,
):

```
## Step-by-step Training

```python
from ultralytics import RTDETR

# Load a model
model = RTDETR("yolov8-rtdetr.yaml")
model.load("yolov8n.pt")

# Use the model
model.train(data = "coco.yaml", epochs = 96, lr0 = 0.0001, lrf = 0.0001, weight_decay = 0.0001, optimizer = 'AdamW', warmup_epochs = 0, mosaic = 1.0, close_mosaic = 24)

# Eval the model
model = RTDETR("DEYO-tiny.pt")
model.val(data = "coco.yaml")  # for DEYO-tiny: 37.6 AP
```

### When encountering out-of-memory (OOM) issues, you can opt to turn off the CDN.
```python
# Open ultralytics/nn/modules/head.py 
# Find RTDETRDecoder
# Find "Prepare denoising training"
dn_embed, dn_bbox, attn_mask, dn_meta = get_cdn_group(
						batch,
						self.nc,
						self.num_queries,
						self.denoising_class_embed.weight,
						self.num_denoising,
						self.label_noise_ratio,
						self.box_noise_scale,
						False,
				)
```



## Multi GPUs

Replace `ultralytics/engine/trainer.py` with our modified `ddp/trainer.py`
```bash
rm -rf Path/ultralytics
cp -r ultralytics Path/  # Path：The location of the ultralytics package
```

```python
import torch
from ultralytics import RTDETR

# Load a model
model = RTDETR("yolov8-rtdetr.yaml")
model.load("yolov8l.pt")
torch.save({"epoch":-1, "model": model.model.half(), "optimizer":None}, "init.pt")

# Load a model
model = RTDETR("init.pt")

# Use the model
model.train(data = "coco.yaml", epochs = 96, lr0 = 0.0001, lrf = 0.0001, weight_decay = 0.0001, optimizer = 'AdamW', warmup_epochs = 0, mosaic = 1.0, close_mosaic = 24, device = '0, 1, 2, 3, 4, 5, 6, 7')
```

## Benchmark
```bash
python3 deyo_onnx.py
trtexec --onnx=./deyo.onnx --saveEngine=deyo.trt --buildOnly --fp16
python3 speed.py
```

## License
This project builds heavily off of [ultralytics](https://github.com/ultralytics/ultralytics). Please refer to their original licenses for more details.

## Citation
If you use `DEYO` in your work, please use the following BibTeX entries:
```
@article{Ouyang2024DEYO,
         title={DEYO: DETR with YOLO for End-to-End Object Detection},
         author={Haodong Ouyang},
         journal={ArXiv},
         year={2024},
         volume={abs/2402.16370},
}
```

