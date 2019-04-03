import os

PATH = os.path.abspath(os.path.dirname(__file__))
# print(sorted(os.listdir(PATH)))
# sequence_path = os.path.join(PATH, "training_logs", "sequence_labels")
sequence_path = "/home/stoplime/workspace/python/vid2vid/datasets/Cityscapes/sequence_original"

def sort_files_to_dirs():
    for root, dirs, files in os.walk(sequence_path):
        for name in files:
            # print("name", name)
            file_path = os.path.join(root, name)
            data_location = str(name).split("_")[0]
            sequence_id = str(name).split("_")[1]
            data_id = data_location + "_" + sequence_id
            # print("data_location", data_location)
            location_file = os.path.join(root, data_id)
            # print("location_file", location_file)
            # print(os.path.join(location_file, name))
            os.makedirs(location_file, exist_ok=True)

            # move file
            os.rename(file_path, os.path.join(location_file, name))
        print(len(files))

def remove_dirs_from_files():
    for root, dirs, files in os.walk(sequence_path):
        for name in files:
            root_prev = os.path.abspath(os.path.join(root, ".."))
            if sequence_path == root:
                continue
            # print(root, root_prev)
            current_file = os.path.join(root, name)
            move_file = os.path.join(root_prev, name)

            # move file
            os.rename(current_file, move_file)
        print(len(files))

def remove_empty_dirs():
    for root, dirs, files in os.walk(sequence_path):
        if not os.listdir(root):
            print(root)
            # exit()
            os.rmdir(root)


if __name__ == "__main__":
    sort_files_to_dirs()
    # remove_dirs_from_files()
    # remove_empty_dirs()