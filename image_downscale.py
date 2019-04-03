import os
import cv2
import tqdm

PATH = os.path.abspath(os.path.dirname(__file__))

downscale_path = os.path.abspath(os.path.join(PATH, "..", "vid2vid", "datasets", "Cityscapes", "sequence_original_test"))

raw_image_path = "/media/stoplime/My Passport/SteffenResearchDatasets/Cityscapes/sequence/leftImg8bit_sequence"

def downscale(img_path, w=512, h=256):
    img = cv2.imread(img_path, -1) # (shape: (1024, 2048, 3))
    # resize img without interpolation:
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST) # (shape: (256, 512, 3))
    return img

def walk_images(mode="train"):
    root_folder = os.path.join(raw_image_path, mode)

    dirs = os.listdir(root_folder)
    for location_data_path in dirs:
        print(location_data_path)
        downscale_location_path = os.path.join(downscale_path, location_data_path)
        os.makedirs(downscale_location_path, exist_ok=True)
        img_dir_path = os.path.join(root_folder, location_data_path)
        if not os.path.isdir(img_dir_path):
            print("Not a Directory", img_dir_path)
            continue

        file_names = os.listdir(img_dir_path)
        for file_name in tqdm.tqdm(file_names):
            img_path = os.path.join(img_dir_path, file_name)
            down_img_path = os.path.join(downscale_location_path, file_name)
            down_img = downscale(img_path)

            cv2.imwrite(down_img_path, down_img)

if __name__ == "__main__":
    walk_images(mode="test")