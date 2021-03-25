import numpy as np
import math
import pandas as pd
import pickle
import os

def find_position(image, dataframes: np.array) -> np.array:
    value = np.int(image) / 1e9
    images_timestamps = np.copy(dataframes[:, 0])
    images_timestamps = images_timestamps / 1e9
    time_diff = np.abs(images_timestamps - value)
    if np.amin(time_diff) > 1:
        # print(f"Cannot find matching position for file: {image}")
        return None
    result = np.where(time_diff == np.amin(time_diff))
    result_idx = result[0][0]
    # match = str((dataframes[result_idx, 0] / 1e9)).replace(".","")
    # print(f"Value: {image} is matched with {match} with timediff {np.amin(time_diff)}")
    return np.array([dataframes[result_idx, 1], dataframes[result_idx, 2], dataframes[result_idx, 3]])

def get_images_timestamps(images_path):
    images = os.listdir(images_path)
    for idx, img in enumerate(images):
        images[idx] = np.int(img.split(".")[0])
    return np.array(images)

def dataframes_to_numpy(dataframes) -> np.array:
    retv = []
    for _, row in dataframes.iterrows():
        frame = [row["timestamp"], row["pos_x"], row["pos_y"], row["pos_z"]]
        retv.append(np.array(frame))

    return np.array(retv)

def preprocess_positions(dataframes : np.array, images_path):
    retv = []
    images = get_images_timestamps(images_path)

    for image in images:
        matching_position = find_position(image, dataframes)
        if matching_position is not None:
            retv.append({
                "timestamp": image,
                "position" : matching_position
            })

    return retv

def is_negative(position_1, position_2):
    temp = np.power(position_1 - position_2, 2)
    return math.sqrt(temp[0] + temp[1] + temp[2]) >= 30

def is_positive(position_1, position_2):
    temp = np.power(position_1 - position_2, 2)
    return math.sqrt(temp[0] + temp[1] + temp[2]) <= 10


def create_dataset_from_frames(dataframes):
    dataset = {}

    for idx, elem in enumerate(dataframes):
        positives = [0] * len(dataframes)
        negatives = [0] * len(dataframes)
        position = elem["position"]

        for index, example in enumerate(dataframes):
            if index == idx:
                continue
            exaple_position = example["position"]
            if is_positive(position, exaple_position):
                positives[index] = 1
            elif is_negative(position, exaple_position):
                negatives[index] = 1

        single_frame = {
            "query": elem["timestamp"],
            "positives" : positives,
            "negatives" : negatives
        }

        dataset[idx] = single_frame

    return dataset

def devide_frames(dataframes):
    train_frames = []
    test_frames = []

    for frame in dataframes:
        if frame["position"][0] < 3.47e5 and frame["position"][1] < 4.04e6:
            test_frames.append(frame)
        else:
            train_frames.append(frame)

    return { "train": train_frames, "test": test_frames}

def preprocess_queries(dataset_path, position_filepath):

    assert os.path.exists(position_filepath), f"Cannot access position filepath: {position_filepath}"
    assert os.path.exists(dataset_path), f"Cannot access dataset path: {dataset_path}"

    train_queries_filepath = os.path.join(dataset_path, "train_processed_queries.pickle")
    test_queries_filepath = os.path.join(dataset_path, "test_processed_queries.pickle")
    
    images_path = os.path.join(dataset_path, "polar")

    dataset = {}

    if os.path.exists(train_queries_filepath):
        print(f"Loading file {train_queries_filepath}")
        with open(train_queries_filepath, "rb") as handle:
            dataset["train"] = pickle.load(handle)

    if os.path.exists(test_queries_filepath):
        print(f"Loading file {test_queries_filepath}")
        with open(test_queries_filepath, "rb") as handle:
            dataset["test"] = pickle.load(handle)

    if "train" in dataset and "test" in dataset:
        return dataset

    df = pd.read_csv(position_filepath, 
                 names = ["timestamp", "rot_x1", "rot_x2", "rot_x3", "pos_x", "rot_y1", "rot_y2", "rot_y3", "pos_y", "rot_z1", "rot_z2", "rot_z3", "pos_z", ], 
                 header = None,
                 dtype = {"timestamp" : np.float64})
    
    df_array = dataframes_to_numpy(df)
    
    postion_frames = preprocess_positions(df_array, images_path)

    dataframes = devide_frames(postion_frames)

    if "train" not in dataset:
        dataset["train"] = create_dataset_from_frames(dataframes["train"])

        with open(train_queries_filepath, "wb") as handle:
            pickle.dump(dataset["train"], handle)

    
    if "test" not in dataset:
        dataset["test"] = create_dataset_from_frames(dataframes["test"])

        with open(test_queries_filepath, "wb") as handle:
            pickle.dump(dataset["test"], handle)

    return dataset


if __name__ == "__main__":
    dataset_path = ''
    position_filepath = ''

    print("Starting querries preprocessing")
    preprocess_queries(dataset_path, position_filepath)
    print("Finished querries preprocessing")