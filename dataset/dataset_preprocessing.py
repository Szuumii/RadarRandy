import os
import numpy as np
import math
import pandas as pd
import pickle

def find_clostest_match(searched_timestamp, images_timestamps):
    value = np.int(searched_timestamp) / 1e9
    images_numbers = np.copy(images_timestamps) / 1e9
    retv = [abs(image_timestamp - value) for image_timestamp in images_numbers ]
    min_idx = 0
    for idx, element in enumerate(retv):
        if element <= 1 and element < retv[min_idx]:
            min_idx = idx

    return images_timestamps[min_idx]


def get_images_timestamps(images_path):
    images = os.listdir(images_path)
    for idx, img in enumerate(images):
        images[idx] = np.int(img.split(".")[0])
    return np.array(images)

def preprocess_positions(dataframes):
    print("Processing positions")

    retv = []
    images_path = "/content/datasets/ParkingLot/polar"
    images = get_images_timestamps(images_path)

    for _, row in dataframes.iterrows():
        matching_image = find_clostest_match(row["timestamp"], images)
        position = np.zeros((3,1), dtype = np.float64)
        position[0] = np.array([row["pos_x"]])
        position[1] = np.array([row["pos_y"]])
        position[2] = np.array([row["pos_z"]])
        retv.append({
            "timestamp": matching_image,
            "position": position
        })

    return retv

def is_negative(x1, x2, y1, y2, z1, z2):
    return math.sqrt(math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2) + math.pow(z1 - z2, 2)) >= 50

def is_positive(x1, x2, y1, y2, z1, z2):
    return math.sqrt(math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2) + math.pow(z1 - z2, 2)) <= 10

def preprocess_queries(dataset_path, position_filepath):

    assert os.path.exists(position_filepath), f"Cannot access position filepath: {position_filepath}"
    assert os.path.exists(dataset_path), f"Cannot access dataset path: {dataset_path}"

    queries_filepath = os.path.join(dataset_path, "processed_queries.pickle")

    my_dataset = {}

    if os.path.exists(queries_filepath):
        print(f"Loading file {queries_filepath}")
        with open(queries_filepath, "rb") as handle:
            my_dataset = pickle.load(handle)
        return my_dataset       

    df = pd.read_csv(position_filepath, 
                 names = ["timestamp", "rot_x1", "rot_x2", "rot_x3", "pos_x", "rot_y1", "rot_y2", "rot_y3", "pos_y", "rot_z1", "rot_z2", "rot_z3", "pos_z", ], 
                 header = None,
                 dtype = {"timestamp" : np.str_})
    
    dataframes = preprocess_positions(df)
    
    for idx, elem in enumerate(dataframes):
        positives = [0] * len(dataframes)
        negatives = [0] * len(dataframes)
        x1 = elem["position"][0]
        y1 = elem["position"][1]
        z1 = elem["position"][2]
        for index, example in enumerate(dataframes):
            if index == idx:
                continue
            x2 = example["position"][0]
            y2 = example["position"][1]
            z2 = example["position"][2]
            if is_positive(x1, x2, y1, y2, z1, z2):
                positives[index] = 1
            elif is_negative(x1, x2, y1, y2, z1, z2):
                negatives[index] = 1

        single_frame = {
            "query": elem["timestamp"],
            "positives" : positives,
            "negatives" : negatives
        }
        my_dataset[idx] = single_frame

    
    with open(queries_filepath, "wb") as handle:
        pickle.dump(my_dataset, handle)

    return my_dataset
