import os
from random import shuffle
from subprocess import call
import cv2
import os.path
import urllib
import tarfile
import caffe


def prepare_data():
    data_file = os.curdir + "/lfw.tgz"
    data_folder = os.curdir + "/lfw"
    if not os.path.isfile(data_file):
        print("File 'lfw.tgz' doesn't existed in current directory!")
        download_lfw()
    if not os.path.isdir(data_folder):
        print("Images folder './lfw' cannot be find!")
        extract_lfw(data_file)
    print("Images folder is ready!")


def download_lfw():
    lfw_data_url = 'http://vis-www.cs.umass.edu/lfw/lfw.tgz'
    lfw_data_file = urllib.URLopener()
    print("Downloading 'lfw.tgz' from 'http://vis-www.cs.umass.edu/lfw/lfw.tgz' ")
    lfw_data_file.retrieve(lfw_data_url, './lfw.tgz')
    print("File 'lfw.tg' downloaded!")


def extract_lfw(data_file_path):
    tgz_file = tarfile.open(data_file_path)
    print("Extracting images from 'lfw.tgz' to './lfw'")
    tgz_file.extractall()
    tgz_file.close()
    print("Images have been extracted and stores under './lfw'")


def create_image_list():
    base_dir = "/home/amax/mlp_practice/face_recognize/lfw/"
    file_list = []
    name_list = dict()
    name_id = 0
    for i in os.listdir(base_dir):
        if i not in name_list:
            name_list[i] = name_id
            name_id += 1
        for j in os.listdir(os.path.join(base_dir, i)):
            file_list.append(base_dir + str(i) + "/" + str(j) + " " + str(name_list[i]))
    shuffle(file_list)
    x_train, x_val, x_test = split_data(file_list)
    save_list_to_file("./train.list", x_train)
    print("Training Data: " + str(len(x_train)) + " has been saved in to ./train.list")
    save_list_to_file("./val.list", x_val)
    print("Validation Data: " + str(len(x_val)) + " has been saved in to ./val.list")
    save_list_to_file("./test.list", x_test)
    print("Testing Data: " + str(len(x_test)) + " has been saved in to ./test.list")

    name_mapping = open("./name_map.list", 'w')
    for name_map_id in name_list:
        print>> name_mapping, name_map_id + " " + str(name_list[name_map_id])
    name_mapping.close()


def save_list_to_file(file_name, list_name):
    saved_file_name = open(file_name, 'w')
    for line in list_name:
        print>> saved_file_name, line
    saved_file_name.close()


def split_data(data_list):
    # x_train_list, x_val_list, x_test_list (70%, 20%, 10%)
    data_amount = len(data_list)
    print("Total Data amount: " + str(data_amount))
    print("Data will be split into 3 parts, 70% for training, 20% for validation and 10% for testing!")
    train_data_amount = data_amount * 7 / 10
    val_data_amount = data_amount * 2 / 10
    test_data_amount = data_amount - train_data_amount - val_data_amount

    x_train_list = data_list[:train_data_amount]
    x_val_list = data_list[train_data_amount: train_data_amount + val_data_amount]
    x_test_list = data_list[-test_data_amount:]

    return x_train_list, x_val_list, x_test_list


def create_image_lmdb_mean(file_name):
    ROOT_FOLDER = "/home/amax/mlp_practice/face_recognize/"
    CAFFE_CMD_PATH = "/home/amax/caffe/build/tools/"
    CMD_CONV_IMG = "convert_imageset.bin"
    CMD_COMP_MEAN = "compute_image_mean.bin"
    conv_cmd_array = [os.path.join(CAFFE_CMD_PATH, CMD_CONV_IMG),
                      '-resize_height=256',
                      '-resize_width=256',
                      '-shuffle',
                      '/',
                      os.path.join(ROOT_FOLDER, file_name),
                      os.path.join(ROOT_FOLDER, file_name + ".lmdb")]

    mean_cmd_array = [os.path.join(CAFFE_CMD_PATH, CMD_COMP_MEAN),
                      os.path.join(ROOT_FOLDER, file_name + ".lmdb"),
                      os.path.join(ROOT_FOLDER, file_name + '.mean')]

    if os.path.isdir(os.path.join(ROOT_FOLDER, file_name + ".lmdb")) \
            or os.path.isfile(os.path.join(ROOT_FOLDER, file_name + ".mean")):
        print("LMDB or Mean for " + file_name + " exists. If you want to re-generate, please remove it "
                                                "manually then run the script again!")
        return
    else:
        print("Generate Image LMDB. Please Wait.......")
        if call(conv_cmd_array):
            print("Convert Image Set failed!")
            exit(1)
        print("Image Set LMDB has been generated. Located at " + os.path.join(ROOT_FOLDER, file_name) + ".lmdb")
        print("Computing Image Mean. Please Wait.......")
        if call(mean_cmd_array):
            print("Computing Image mean failed!")
            exit(1)
        print("Compute Image Mean has been done. Located at " + os.path.join(ROOT_FOLDER, file_name) + ".mean")


def main():
    prepare_data()
    create_image_list()
    create_image_lmdb_mean("./train.list")
    create_image_lmdb_mean("./val.list")
    create_image_lmdb_mean("./test.list")


if __name__ == '__main__':
    main()
