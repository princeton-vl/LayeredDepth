import os
import jsonlines
import numpy as np
from PIL import Image
import json
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datasets import load_dataset
import cv2

class MetricTracker:
    def __init__(self):
        self.metrics = {}
        self.reset()
    
    def reset(self):
        self.metrics = {}
    
    def update(self, metrics):
        for key, value in metrics.items():
            if key not in self.metrics:
                self.metrics[key] = {
                    'sum': 0,
                    'count': 0
                }
            self.metrics[key]['sum'] += value
            self.metrics[key]['count'] += 1

    def merge(self, other):
        for key, value in other.metrics.items():
            if key not in self.metrics:
                self.metrics[key] = {
                    'sum': 0,
                    'count': 0
                }
            self.metrics[key]['sum'] += value['sum']
            self.metrics[key]['count'] += value['count']

    def get_sum(self):
        return {key: self.metrics[key]['sum'] for key in self.metrics}

    def get_count(self):
        return {key: self.metrics[key]['count'] for key in self.metrics}
    
    def get_average(self):
        return {key: self.metrics[key]['sum'] / self.metrics[key]['count'] for key in self.metrics}
    
    def get_raw_metrics(self):
        return {key: {'sum': self.metrics[key]['sum'], 'count': self.metrics[key]['count']} for key in self.metrics}

def get_depth(pd, x, y, l, mask=None):
    pred_depths = pd[:, y, x]
    if mask is not None:
        pred_mask = mask[:, y, x]
        pred_depths = pred_depths[pred_mask]

    if len(pred_depths) <= l:
        return None
    return pred_depths[l]

def layereddepth_tuple_correct(single_tuple, pd, mask=None):
    last_depth = None
    points = [single_tuple[key] for key in single_tuple.keys() if key.startswith('p')]
    is_fake = not single_tuple['is_real']
    for point in points:
        x, y, l = point
        d = get_depth(pd, x, y, l, mask)
        if not is_fake:
            if d is None: 
                return False
            elif last_depth is not None and last_depth >= d:
                return False
            last_depth = d
        else:
            if d is not None:
                return False
    return True

def get_layer_name(single_tuple):
    current_layer = None
    points = [single_tuple[key] for key in single_tuple.keys() if key.startswith('p')]
    for point in points:
        x, y, l = point
        if current_layer is None:
            current_layer = l
        elif current_layer != l:
            return 'mixed'
    assert current_layer is not None
    return str(int(current_layer))

def check_tuple_valid(depth, single_tuple):
    h, w = depth.shape[1:]
    points = [single_tuple[key] for key in single_tuple.keys() if key.startswith('p')]
    for point in points:
        x, y, l = point
        if x < 0 or x >= w or y < 0 or y >= h or l % 2 == 0 or l > 7 or l < 0:
            return False
    return True

def read_depth(index, prediction_path='/n/fs/pvl-transobj/layereddepth/depth_methods/prediction/multiout'):
    depths = []
    for i in [1, 3, 5, 7]:
        depth_file = os.path.join(prediction_path, f'{index}_{(i - 1) // 2}.npy')
        if os.path.exists(depth_file):
            depth = 1.0 / np.load(depth_file)
        else:
            depth_file = os.path.join(prediction_path, f'{index}_{(i - 1) // 2}.png')
            depth = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000
        depths.append(depth)

    depth = np.stack(depths, axis=0)
    return depth

def evaluate(args):
    dataset = load_dataset("princeton-vl/LayeredDepth")
    metric_tracker = MetricTracker()
    for item in dataset['validation']:
        index = int(item['__key__'])
        relative_depth_tuples = item['tuples.json'][args.subset]
        image = item['image.png'] # PIL Image
        height, width = image.size
        # depth = ... make a depth prediction for this image of shape (L, H, W), where L is the number of layers
        depth = read_depth(index)
        # depth = np.zeros((4, height, width))
        print(index, depth.shape)
        for tuple_type in ['pairs', 'trips', 'quads']:
            for single_tuple in relative_depth_tuples[tuple_type]:
                valid = check_tuple_valid(depth, single_tuple)
                if not valid:
                    continue

                layer = get_layer_name(single_tuple)
                correctness = int(layereddepth_tuple_correct(single_tuple, depth))
                metric_tracker.update({f'{args.subset}/{tuple_type}/{layer}': correctness})
                metric_tracker.update({f'{args.subset}/{tuple_type}/all': correctness})

    return metric_tracker.get_average()
                    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--subset', type=str, required=True, choices=['layer_all', 'layer_first'])
    args = parser.parse_args()
    results = evaluate(args)
    print(json.dumps(results, indent=4))
