# The following code is based on the RT-DETR project by lyuwenyu
# Original source: https://github.com/lyuwenyu/RT-DETR/tree/main/benchmark
# modifyed by ouyanghaodong


import torch 
import torchvision

import numpy as np 
import onnxruntime as ort 

class DEYO(torch.nn.Module):
    def __init__(self, name) -> None:
        super().__init__()
        from ultralytics import RTDETR
        model = RTDETR('deyo-tiny.pt')  
        self.model = model.model

    def forward(self, x):
        '''https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/tasks.py#L216
        '''
        pred: torch.Tensor = self.model(x)[0] # n 84 8400,
        boxes, scores = pred.split([4, 80], dim=-1)
        boxes = torchvision.ops.box_convert(boxes, in_fmt='cxcywh', out_fmt='xyxy')

        return boxes, scores

def export_onnx(name='yolov8n'):
    '''export onnx
    '''
    m = DEYO(name)

    x = torch.rand(1, 3, 640, 640)
    dynamic_axes = {
        'image': {0: '-1'}
    }
    torch.onnx.export(m, x, f'{name}.onnx', 
                      input_names=['image'], 
                      output_names=['boxes', 'scores'], 
                      opset_version=16, 
                      dynamic_axes=dynamic_axes)

    data = np.random.rand(1, 3, 640, 640).astype(np.float32)
    sess = ort.InferenceSession(f'{name}.onnx')
    _ = sess.run(output_names=None, input_feed={'image': data})

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='deyo')
    args = parser.parse_args()

    export_onnx(name=args.name)

  