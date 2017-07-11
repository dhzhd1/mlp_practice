import pandas as pd
import os
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
    pass


def main():
    prepare_data()


if __name__ == '__main__':
    main()
