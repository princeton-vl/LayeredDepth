import os
import jsonlines
import numpy as np
import cv2
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

def verify_depth(depth, single_tuple):
    last_depth = None
    for point in single_tuple:
        x, y, l = point
        x, y = int(x), int(y)
        if x < 0 or x >= depth.shape[2] or y < 0 or y >= depth.shape[1]:
            return None
        if l > 7:
            return None
        d = depth[l - 1, y, x]
        if last_depth is not None and last_depth >= d:
            return False
        last_depth = d
    return True

def get_layer(single_tuple):
    layer = None
    for point in single_tuple:
        x, y, l = point
        if layer is None:
            layer = l
        elif layer != l:
            return 'mixed'
    return str(layer)

def process_image(index, method, prediction_path, data_path, test_subset):
    # Load the depth image
    depths = []
    for i in range(8):
        depth_file = os.path.join(prediction_path, index + f'_{i+1}.png')
        if not os.path.exists(depth_file):
            return index, None
        depth = cv2.imread(depth_file, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000
        depths.append(depth)
    depth = np.stack(depths, axis=0)

    # Prepare result dictionary for all subsets
    results = {
        layer_subsection: {
            subset: {"correct_point": 0, "total_point": 0} 
                for subset in test_subset
        }
        for layer_subsection in ['all', '1', '3', '5', '7', 'mixed']
    }
    
    # Process each test subset
    for subset in test_subset:
        tuples_file = os.path.join(data_path, subset, index + '.jsonl')
        if not os.path.exists(tuples_file):
            continue

        tuples = list(jsonlines.open(tuples_file))

        for single_tuple in tuples:
            tuple_layer = get_layer(single_tuple)
            if tuple_layer not in ['1', '3', '5', '7', 'mixed']:
                continue
            
            valid = verify_depth(depth, single_tuple)
            if valid is None:
                continue
            elif valid:
                results[tuple_layer][subset]["correct_point"] += 1
                results['all'][subset]["correct_point"] += 1
            results[tuple_layer][subset]["total_point"] += 1
            results['all'][subset]["total_point"] += 1

    return index, results

ESTIMATIONS_PATH = './estimations/'
DATA_PATH = './data/'

def main():
    methods = [
        'your_method', # replace with your method name
    ]

    test_subset = [
        'pairs_layerall_test',
        'triplets_layerall_test',
        'quadruplets_layerall_test',
    ]
    data_path = DATA_PATH

    for method in methods:
        start_time = time.time()
        print(f"Processing {method}...")
        prediction_path = f'{ESTIMATIONS_PATH}/{method}/'

        # Initialize aggregate counters for each subset
        aggregate_results = {
            layer_subsection: {
                subset: {"correct_point": 0, "total_point": 0} 
                    for subset in test_subset
            }
            for layer_subsection in ['all', '1', '3', '5', '7', 'mixed']
        }

        indices = [str(i) for i in range(1500)]
        # Process images in parallel using 16 processes (true parallelism across CPU cores)
        with ProcessPoolExecutor(max_workers=40) as executor:
            futures = [executor.submit(process_image, idx, method, prediction_path, data_path, test_subset)
                       for idx in indices]
            for future in as_completed(futures):
                index, results = future.result()
                if results is None:
                    continue
                for layer_subsection in results:
                    for subset in results[layer_subsection]:
                        aggregate_results[layer_subsection][subset]["correct_point"] += results[layer_subsection][subset]["correct_point"]
                        aggregate_results[layer_subsection][subset]["total_point"] += results[layer_subsection][subset]["total_point"]
                

        # Print the aggregated results for each subset
        print(f"Method: {method}")
        for layer_subsection in aggregate_results:
            for subset in test_subset:
                correct_point = aggregate_results[layer_subsection][subset]["correct_point"]
                total_point = aggregate_results[layer_subsection][subset]["total_point"]
                accuracy = correct_point / total_point if total_point > 0 else 0
                print(f"Layer: {layer_subsection}, Subset: {subset}, Accuracy: {accuracy:.4f} ({correct_point}/{total_point})")

        elapsed = time.time() - start_time
        print(f"Elapsed time: {elapsed:.2f} seconds\n", flush=True)

if __name__ == "__main__":
    main()
