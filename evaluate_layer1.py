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
        if x < 0 or x >= depth.shape[1] or y < 0 or y >= depth.shape[0]:
            return None
        d = depth[y, x]
        if last_depth is not None and last_depth >= d:
            return False
        last_depth = d
    return True

def process_image(index, method, prediction_path, data_path, test_subset):
    # Load the depth image
    depth_file = os.path.join(prediction_path, index + '.npy')
    depth_png = os.path.join(prediction_path, index + '_0.png')
    
    if not os.path.exists(depth_png):
        depth = np.load(depth_file)
    else:
        depth = cv2.imread(depth_png, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000

    # Prepare result dictionary for all subsets
    results = {subset: {"correct_point": 0, "total_point": 0, "correct_image": 0, "total_image": 0} 
               for subset in test_subset}

    # Process each test subset
    for subset in test_subset:
        tuples_file = os.path.join(data_path, subset, index + '.jsonl')
        if not os.path.exists(tuples_file):
            continue

        tuples = list(jsonlines.open(tuples_file))
        image_correct = True

        for single_tuple in tuples:
            valid = verify_depth(depth, single_tuple)
            if valid is None:
                continue
            elif valid:
                results[subset]["correct_point"] += 1
            else:
                image_correct = False
            results[subset]["total_point"] += 1

        if image_correct:
            results[subset]["correct_image"] += 1
        results[subset]["total_image"] += 1

    return index, results

ESTIMATIONS_PATH = './estimations/'
DATA_PATH = './data/'

def main():
    methods = [
        'your_method', # replace with your method name
    ]

    test_subset = [
        'pairs_layer1_val',
        'triplets_layer1_val',
        'quadruplets_layer1_val',
    ]
    data_path = DATA_PATH

    for method in methods:
        start_time = time.time()
        print(f"Processing {method}...")
        prediction_path = f'{ESTIMATIONS_PATH}/{method}/'

        # Initialize aggregate counters for each subset
        aggregate_results = {subset: {"correct_point": 0, "total_point": 0, 
                                      "correct_image": 0, "total_image": 0} 
                             for subset in test_subset}

        indices = [str(i) for i in range(1500)]

        # Process images in parallel using 16 processes
        with ProcessPoolExecutor(max_workers=16) as executor:
            futures = [executor.submit(process_image, idx, method, prediction_path, data_path, test_subset)
                       for idx in indices]
            for future in as_completed(futures):
                index, results = future.result()
                for subset in test_subset:
                    aggregate_results[subset]["correct_point"] += results[subset]["correct_point"]
                    aggregate_results[subset]["total_point"] += results[subset]["total_point"]

        # Print the aggregated results for each subset
        print(f"Method: {method}")
        for subset in test_subset:
            print(f"Subset: {subset}")
            correct_point = aggregate_results[layer_subsection][subset]["correct_point"]
            total_point = aggregate_results[layer_subsection][subset]["total_point"]
            accuracy = correct_point / total_point if total_point > 0 else 0
            print(f"Layer 1, Subset: {subset}, Accuracy: {accuracy:.4f} ({correct_point}/{total_point})")

        elapsed = time.time() - start_time
        print(f"Elapsed time: {elapsed:.2f} seconds\n", flush=True)

if __name__ == "__main__":
    main()
