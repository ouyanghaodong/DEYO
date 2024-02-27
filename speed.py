from trtinfer import TRTInference
from dataset import Dataset
from tqdm import tqdm

trt_dataset = Dataset('val2017')

trt = TRTInference(engine_path = 'deyo-tiny.trt')

for blob in trt_dataset:
    trt.warmup(blob, 500)
    break
    
total_time = 0

for blob in tqdm(trt_dataset):
    total_time += trt.speed(blob, 1)

latency = total_time / 5000 * 1000
fps = 1000 / latency

print("latency: %.2fms" % latency, end = ' ')
print("fps: %.2f" % fps)
    
