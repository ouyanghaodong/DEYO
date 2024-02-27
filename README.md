<h2 align="center">DEYO: DETR with YOLO for End-to-End Object Detection</h2>

![deyov3_overview](https://github.com/ouyanghaodong/DEYO/blob/main/figs/1.png)

This is the official implementation of the paper "[DEYO: DETR with YOLO for End-to-End Object Detection](https://arxiv.org/abs/2402.16370)".

| Model | Epoch | Input Shape | $AP^{val}$ | $AP^{val}_{50}$ | Params(M) | FLOPs(G) | T4 TensorRT FP16(FPS) |
|:------|:-----:|:-----------:|:----------:|:---------------:|:---------:|:--------:|:---------------------:|
| [DEYO-tiny](https://raw.githubusercontent.com/ouyanghaodong/assets/main/deyo-tiny.pt) | 96 | 640 | 37.6 | 52.8 | 4 | 8 | 497 |
| [DEYO-N](https://raw.githubusercontent.com/ouyanghaodong/assets/main/deyo-n.pt) | 96 | 640 | 39.7 | 55.6 | 6 | 10 | 396 |
| [DEYO-S](https://raw.githubusercontent.com/ouyanghaodong/assets/main/deyo-s.pt) | 96 | 640 | 45.8 | 62.9 | 14 | 26 | 299 |
| [DEYO-M](https://raw.githubusercontent.com/ouyanghaodong/assets/main/deyo-m.pt) | 96 | 640 | 50.7 | 68.4 | 33 | 78 | 140 |
| [DEYO-L](https://raw.githubusercontent.com/ouyanghaodong/assets/main/deyo-l.pt) | 96 | 640 | 52.7 | 70.2 | 51 | 155 | 100 |
| [DEYO-X](https://raw.githubusercontent.com/ouyanghaodong/assets/main/deyo-x.pt) | 96 | 640 | 53.7 | 71.3 | 78 | 242 | 65 |




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
```python
# Open ultralytics/nn/modules/head.py 
# Find RTDETRDecoder
# You can configure DEYO by referring to Table 9 in our paper.
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
### When encountering out-of-memory (OOM) issues, you can opt to turn off the CDN.

```python
from ultralytics import RTDETR

# Load a model
model = RTDETR("yolov8-rtdetr.yaml")
model.load("yolov8n.pt")

# Use the model
model.train(data = "coco.yaml", epochs = 96, lr0 = 0.0001, lrf = 0.0001, weight_decay = 0.0001, optimizer = 'AdamW', warmup_epochs = 0, mosaic = 1.0, close_mosaic = 24)

# Eval the model
model = RTDETR("DEYO-tiny.pt")
model.val(data = "coco.yaml")  # for DEYO-tiny: 37.3 AP
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

