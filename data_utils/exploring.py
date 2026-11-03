from pathlib import Path

# [patient id] [filename] [class] [data source] 


DATASET_PATH = Path("./data")

TRAIN_DATASET_PATH = DATASET_PATH / "train"
TEST_DATASET_PATH = DATASET_PATH / "test"
VAL_DATASET_PATH = DATASET_PATH / "val"


def explore_dataset(dataset_path: Path):
    """
    Explore the dataset and print the number of files in each class.
    """
    if not dataset_path.exists():
        print(f"Dataset path {dataset_path} does not exist.")
        return

    _type = "train" if "train" in str(dataset_path) else "test" if "test" in str(dataset_path) else "val"
    data_file = DATASET_PATH / f"{_type}.txt"

    stats = {}

    with open(data_file, "r") as f:
        lines = f.readlines()

        for line in lines:
            patient_id, filename, class_name, data_source = line.strip().split()
            file_path = dataset_path  / filename

            if not file_path.exists():
                print(f"File {file_path} does not exist.")
                continue

            if class_name not in stats:
                stats[class_name] = {}

            if data_source not in stats[class_name]:
                stats[class_name][data_source] = 0

            stats[class_name][data_source] += 1

    print(stats)
            

if __name__ == "__main__":
    print(f"Train Dataset Path: {TRAIN_DATASET_PATH}")
    print(f"Test Dataset Path: {TEST_DATASET_PATH}")
    print(f"Validation Dataset Path: {VAL_DATASET_PATH}")

    explore_dataset(TEST_DATASET_PATH)
    