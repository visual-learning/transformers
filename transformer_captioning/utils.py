import os, json
import random
import torch
import numpy as np
import h5py
import urllib.request, urllib.error, urllib.parse, os, tempfile
from torch.utils.data import Dataset
from imageio import imread
from PIL import Image


dir_path = os.path.dirname(os.path.realpath(__file__))
BASE_DIR = os.path.join(dir_path, "datasets/coco_captioning")

def set_all_seeds(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.deterministic = True

def load_coco_data(base_dir=BASE_DIR, max_train=None, max_val = None, pca_features=True):
    print('base dir ', base_dir)
    data = {}
    caption_file = os.path.join(base_dir, "coco2014_captions.h5")
    with h5py.File(caption_file, "r") as f:
        for k, v in f.items():
            data[k] = np.asarray(v)

    if pca_features:
        train_feat_file = os.path.join(base_dir, "train2014_vgg16_fc7_pca.h5")
    else:
        train_feat_file = os.path.join(base_dir, "train2014_vgg16_fc7.h5")
    with h5py.File(train_feat_file, "r") as f:
        data["train_features"] = np.asarray(f["features"])

    if pca_features:
        val_feat_file = os.path.join(base_dir, "val2014_vgg16_fc7_pca.h5")
    else:
        val_feat_file = os.path.join(base_dir, "val2014_vgg16_fc7.h5")
    with h5py.File(val_feat_file, "r") as f:
        data["val_features"] = np.asarray(f["features"])

    dict_file = os.path.join(base_dir, "coco2014_vocab.json")
    with open(dict_file, "r") as f:
        dict_data = json.load(f)
        for k, v in dict_data.items():
            data[k] = v

    train_url_file = os.path.join(base_dir, "train2014_urls.txt")
    with open(train_url_file, "r") as f:
        train_urls = np.asarray([line.strip() for line in f])
    data["train_urls"] = train_urls

    val_url_file = os.path.join(base_dir, "val2014_urls.txt")
    with open(val_url_file, "r") as f:
        val_urls = np.asarray([line.strip() for line in f])
    data["val_urls"] = val_urls

    # Maybe subsample the training data
    if max_train is not None:
        num_train = data["train_captions"].shape[0]
        mask = np.random.randint(num_train, size=max_train)
        data["train_captions"] = data["train_captions"][mask]
        data["train_image_idxs"] = data["train_image_idxs"][mask]

    if max_val is not None:
        num_val = data["val_captions"].shape[0]
        mask = np.random.randint(num_val, size=max_val)
        data["val_captions"] = data["val_captions"][mask]
        data["val_image_idxs"] = data["val_image_idxs"][mask]
    return data


def decode_captions(captions, idx_to_word):
    singleton = False
    if captions.ndim == 1:
        singleton = True
        captions = captions[None]
    decoded = []
    N, T = captions.shape
    for i in range(N):
        words = []
        for t in range(T):
            word = idx_to_word[captions[i, t]]
            if word != "<NULL>":
                words.append(word)
            if word == "<END>":
                break
        decoded.append(" ".join(words))
    if singleton:
        decoded = decoded[0]
    return decoded

class CocoDataset(Dataset):
    def __init__(self, data, mode='train'):
        self.data = data
        self.mode = mode

    def __len__(self):
        return self.data["%s_captions" % self.mode].shape[0]
        
    def __getitem__(self, idx):
        split = self.mode
        split_size = self.data["%s_captions" % split].shape[0]
        captions = self.data["%s_captions" % split][idx]
        image_idxs = self.data["%s_image_idxs" % split][idx]
        image_features = self.data["%s_features" % split][idx]
        return image_features, captions.astype(np.long), image_idxs

def image_from_url(url):
    """
    Read an image from a URL. Returns a numpy array with the pixel data.
    We write the image to a temporary file then read it back. Kinda gross.
    """
    try:
        f = urllib.request.urlopen(url)
        _, fname = tempfile.mkstemp()
        with open(fname, "wb") as ff:
            ff.write(f.read())
        img = imread(fname)
        os.remove(fname)
        return img
    except urllib.error.URLError as e:
        print("URL Error: ", e.reason, url)
    except urllib.error.HTTPError as e:
        print("HTTP Error: ", e.code, url)
