import numpy as np
import math
import pandas as pd
import pickle
import os
import glob
import random


def merge_position_files(dataset_path):
    all_files = glob.glob(os.path.join(dataset_path, "*_pose.csv"))
    [print(f"Merging file {f}") for f in all_files]
    df_merged = pd.concat([pd.read_csv(file, names=["timestamp", "rot_x1", "rot_x2", "rot_x3", "pos_x", "rot_y1",
                                                    "rot_y2", "rot_y3", "pos_y", "rot_z1", "rot_z2", "rot_z3", "pos_z", ],
                                       header=None,
                                       dtype={"timestamp": np.float64}) for file in all_files])
    out_file = os.path.join(dataset_path, "merged_positions.csv")
    df_merged.to_csv(out_file, index=False)


def show_csv(csv_file):
    df = pd.read_csv(csv_file,
                     names=["timestamp", "rot_x1", "rot_x2", "rot_x3", "pos_x", "rot_y1",
                            "rot_y2", "rot_y3", "pos_y", "rot_z1", "rot_z2", "rot_z3", "pos_z", ],
                     header=None,
                     dtype={"timestamp": np.float64})

    print(df)


def find_matching_position(image, dataframes: np.array) -> np.array:
    value = int(image) / 1e9
    images_timestamps = np.copy(dataframes[:, 0])
    images_timestamps = images_timestamps / 1e9
    time_diff = np.abs(images_timestamps - value)
    smallest_time_diff = np.amin(time_diff)
    if smallest_time_diff > 1:
        # print(
        # f"Cannot find matching position for file: {image}. Smallest difference is {smallest_time_diff}")
        return None
    result = np.where(time_diff == smallest_time_diff)
    result_idx = result[0][0]
    # match = str((dataframes[result_idx, 0] / 1e9)).replace(".","")
    # print(f"Value: {image} is matched with {match} with timediff {np.amin(time_diff)}")
    return np.array([dataframes[result_idx, 1], dataframes[result_idx, 2], dataframes[result_idx, 3]])


def get_images_timestamps(images_path):
    images = os.listdir(images_path)
    for idx, img in enumerate(images):
        images[idx] = int(img.split(".")[0])
    return np.array(images)


def dataframes_to_numpy(dataframes) -> np.array:
    retv = []
    for _, row in dataframes.iterrows():
        frame = [row["timestamp"], row["pos_x"], row["pos_y"], row["pos_z"]]
        retv.append(np.array(frame))

    return np.array(retv)


def preprocess_positions(dataframes: np.array, images_path: str):
    retv = []
    images = get_images_timestamps(images_path)

    print(f"Looking for positions for: {len(images)}")

    for image in images:
        matching_position = find_matching_position(image, dataframes)
        if matching_position is not None:
            retv.append({
                "query": image,
                "position": matching_position
            })

    print(f"Found positions for: {len(retv)} images")

    return retv


def is_negative(position_1, position_2):
    temp = np.power(position_1 - position_2, 2)
    return math.sqrt(temp[0] + temp[1] + temp[2]) >= 50


def is_positive(position_1, position_2):
    temp = np.power(position_1 - position_2, 2)
    return math.sqrt(temp[0] + temp[1] + temp[2]) <= 10


def create_train_dataframes(dataframes):
    retv = {}

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
            "query": elem["query"],
            "positives": positives,
            "negatives": negatives
        }

        retv[idx] = single_frame

    return retv


def get_frames(dataframes, test):
    if test is True:
        frames = [frame for frame in dataframes if frame["position"]
                  [0] < 3.47e5 and frame["position"][1] < 4.04e6]
    else:
        frames = [frame for frame in dataframes if frame["position"]
                  [0] > 3.47e5 or frame["position"][1] > 4.04e6]

    return frames


def get_validation_frames(dataframes):
    k = round(0.1 * len(dataframes))
    random.shuffle(dataframes)
    return dataframes[:k], dataframes[k:]


def get_small_set_of_frames(dataframes):
    sample_ratio = round(0.2 * len(dataframes))
    small_set = random.sample(dataframes, sample_ratio)
    return small_set


def preprocess_queries(dataset_path, position_filepath, images_path, test=False, small_set=False):

    assert os.path.exists(
        dataset_path), f"Cannot access position filepath: {dataset_path}"

    assert os.path.exists(
        position_filepath), f"Cannot access position filepath: {position_filepath}"

    querries_filename = "test_processed_queries.pickle" if test is True else "train_processed_queries.pickle"

    if small_set is True:
        querries_filename = "small_" + querries_filename

    queries_filepath = os.path.join(
        dataset_path, querries_filename)

    dataset = {}

    if os.path.exists(queries_filepath):
        print(f"Loading file {queries_filepath}")
        with open(queries_filepath, "rb") as handle:
            dataset = pickle.load(handle)

        return dataset

    assert os.path.exists(
        images_path), f"Cannot access images path: {images_path}"

    df = pd.read_csv(position_filepath,
                     names=["timestamp", "rot_x1", "rot_x2", "rot_x3", "pos_x", "rot_y1",
                            "rot_y2", "rot_y3", "pos_y", "rot_z1", "rot_z2", "rot_z3", "pos_z", ],
                     header=None,
                     dtype={"timestamp": np.float64})

    df_array = dataframes_to_numpy(df)

    frames_with_positions = preprocess_positions(
        df_array, images_path)

    relevant_frames = get_frames(frames_with_positions, test)

    if small_set is True:
        relevant_frames = get_small_set_of_frames(relevant_frames)

    print(f"Length of querries dataset: {len(relevant_frames)}")

    print("Creating dataset")
    if test is True:
        dataset = relevant_frames        
    else:
        validation_set, train_set = get_validation_frames(relevant_frames)

        print(f"Lenght of validation set: {len(validation_set)}")
        print(f"Lenght of train set: {len(train_set)}")

        validation_filename = "val_processed_queries.pickle"

        if small_set is True:
            validation_filename = "small_" + validation_filename

        validation_filepath = os.path.join(
            dataset_path, validation_filename)

        dataset = create_train_dataframes(validation_set)

        with open(validation_filepath, "wb") as handle:
            pickle.dump(dataset, handle)

        dataset = create_train_dataframes(train_set)

    with open(queries_filepath, "wb") as handle:
        pickle.dump(dataset, handle)

    return dataset


if __name__ == "__main__":
    dataset_path = '/home/jszumski/mulran_dataset'
    # test1_position_filepath = '/data3/mulran/Sejong01/global_pose.csv'
    # test2_position_filepath = '/data3/mulran/Sejong02/global_pose.csv'
    # images_path = '/data3/mulran/Sejong02/polar/'

    train_position_filepath = '/home/jszumski/mulran_dataset/merged_positions.csv'
    train_images_path = '/home/jszumski/mulran_dataset/images'

    print("Starting querries preprocessing")
    preprocess_queries(dataset_path, train_position_filepath,
                       train_images_path)
    # preprocess_queries(dataset_path, test2_position_filepath,
    #    images_path, True)
    print("Finished querries preprocessing")

    # show_csv(position_filepath)
