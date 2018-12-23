# -*- coding: utf-8 -*-

# Some image captioning preprocessing functions #

import os
import queue
import threading
import zipfile
import json
from collections import defaultdict, Counter
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pickle
import keras.backend as K
from keras.applications import InceptionV3
from keras.applications import inception_v3
from keras.models import Model
from keras.layers import GlobalAveragePooling2D
import tqdm
import itertools
import re


def get_cnn_encoder():
    # Transfer Learning : we take the last hidden layer of IncetionV3 as an image embedding
    K.set_learning_phase(False)
    model = InceptionV3(include_top=False)
    preprocess_for_model = inception_v3.preprocess_input
    model = Model(model.inputs, GlobalAveragePooling2D()(model.output))
    return model, preprocess_for_model


# special tokens
PAD = "#PAD#"
UNK = "#UNK#"
START = "#START#"
END = "#END#"


# split sentence into tokens (split into lowercased words)
def split_sentence(sentence):
    return list(filter(lambda x: len(x) > 0, re.split('\W+', sentence.lower())))


def generate_vocabulary(train_captions):
    """
    Return {token: index} for all train tokens (words) that occur 5 times or more, 
        `index` should be from 0 to N, where N is a number of unique tokens in the resulting dictionary.
    Also, PAD (for batch padding), UNK (unknown, out of vocabulary), 
        START (start of sentence) and END (end of sentence) tokens into the vocabulary.
    """
    concat_sentenc = itertools.chain.from_iterable(train_captions)
    l = itertools.chain.from_iterable(map(lambda x: split_sentence(x), concat_sentenc))
    counter = Counter(l)
    concat_sentenc = itertools.chain.from_iterable(train_captions)
    l = itertools.chain.from_iterable(map(lambda x: split_sentence(x), concat_sentenc))
    vocab = [w for w in set(l) if counter[w] >= 5] + [PAD, UNK, START, END]
    return {token: index for index, token in enumerate(sorted(vocab))}


def caption_tokens_to_indices(captions, vocab):
    """
    `captions` argument is an array of arrays:
    [
        [
            "image1 caption1",
            "image1 caption2",
            ...
        ],
        [
            "image2 caption1",
            "image2 caption2",
            ...
        ],
        ...
    ]
    Replace all tokens with vocabulary indices, use UNK for unknown words (out of vocabulary).
    Add START and END tokens to start and end of each sentence respectively.
    One example would be the following:
    [
        [
            [vocab[START], vocab["image1"], vocab["caption1"], vocab[END]],
            [vocab[START], vocab["image1"], vocab["caption2"], vocab[END]],
            ...
        ],
        ...
    ]
    """

    f = lambda s: [vocab['#START#']] + [vocab[w] if w in vocab else vocab['#UNK#']
                                        for w in split_sentence(s)] + [vocab['#END#']]
    res = [list(map(f, captions[:][i])) for i in range(0, len(captions))]
    return res


# we will use this during training
def batch_captions_to_matrix(batch_captions, pad_idx, max_len=None):
    """
    `batch_captions` is an array of arrays:
    [
        [vocab[START], ..., vocab[END]],
        [vocab[START], ..., vocab[END]],
        ...
    ]
    Put vocabulary indexed captions into np.array of shape (len(batch_captions), columns),
        where "columns" is max(map(len, batch_captions)) when max_len is None
        and "columns" = min(max_len, max(map(len, batch_captions))) otherwise.
    Add padding with pad_idx where necessary.
    Input example: [[1, 2, 3], [4, 5]]
    Output example: np.array([[1, 2, 3], [4, 5, pad_idx]]) if max_len=None
    Output example: np.array([[1, 2], [4, 5]]) if max_len=2
    Output example: np.array([[1, 2, 3], [4, 5, pad_idx]]) if max_len=100
    """

    if max_len:
        batch_captions = [x[:max_len] for x in batch_captions]
        n_pad = min(max_len, max(map(len, batch_captions))) - np.array([len(s) for s in batch_captions])
    else:
        n_pad = max(map(len, batch_captions)) - np.array([len(s) for s in batch_captions])

    padding = [[pad_idx] * n for n in n_pad]

    matrix = np.array([batch_captions[i] + padding[i] for i in range(0, len(padding))])
    return matrix


# generate batch via random sampling of images and captions for them,
# we use `max_len` parameter to control the length of the captions (truncating long captions)
def generate_batch(images_embeddings, indexed_captions, batch_size, decoder, max_len=None):
    """
    `images_embeddings` is a np.array of shape [number of images, IMG_EMBED_SIZE].
    `indexed_captions` holds 5 vocabulary indexed captions for each image:
    [
        [
            [vocab[START], vocab["image1"], vocab["caption1"], vocab[END]],
            [vocab[START], vocab["image1"], vocab["caption2"], vocab[END]],
            ...
        ],
        ...
    ]
    Generate a random batch of size `batch_size`.
    Take random images and choose one random caption for each image.
    Return feed dict {decoder.img_embeds: ..., decoder.sentences: ...}.
    """

    batch = np.random.choice(images_embeddings.shape[0], batch_size, replace=False)
    batch_image_embeddings = images_embeddings[batch, :]
    batch_captions = indexed_captions[batch]
    batch_captions_sample = [batch_captions[i][int(np.random.choice(5, 1))]
                             for i in range(0, len(batch_captions))]
    batch_captions_matrix = batch_captions_to_matrix(batch_captions_sample, pad_idx=1,
                                                     max_len=max_len)
    return {decoder.img_embeds: batch_image_embeddings,
            decoder.sentences: batch_captions_matrix}


# Image Preprocessing Functions

def get_captions_for_fns(fns, zip_fn, zip_json_path):
    # extract captions from zip
    zf = zipfile.ZipFile(zip_fn)
    j = json.loads(zf.read(zip_json_path).decode("utf8"))
    id_to_fn = {img["id"]: img["file_name"] for img in j["images"]}
    fn_to_caps = defaultdict(list)
    for cap in j['annotations']:
        fn_to_caps[id_to_fn[cap['image_id']]].append(cap['caption'])
    fn_to_caps = dict(fn_to_caps)
    return list(map(lambda x: fn_to_caps[x], fns))


def image_center_crop(img):
    h, w = img.shape[0], img.shape[1]
    pad_left = 0
    pad_right = 0
    pad_top = 0
    pad_bottom = 0
    if h > w:
        diff = h - w
        pad_top = diff - diff // 2
        pad_bottom = diff // 2
    else:
        diff = w - h
        pad_left = diff - diff // 2
        pad_right = diff // 2
    return img[pad_top:h - pad_bottom, pad_left:w - pad_right, :]


def decode_image_from_buf(buf):
    img = cv2.imdecode(np.asarray(bytearray(buf), dtype=np.uint8), 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def crop_and_preprocess(img, input_shape, preprocess_for_model):
    img = image_center_crop(img)  # take center crop
    img = cv2.resize(img, input_shape)  # resize for our model
    img = img.astype("float32")  # prepare for normalization
    img = preprocess_for_model(img)  # preprocess for model
    return img


def apply_model(zip_fn, model, preprocess_for_model, extensions=(".jpg",), input_shape=(224, 224), batch_size=32):
    # queue for cropped images
    q = queue.Queue(maxsize=batch_size * 10)

    # when read thread put all images in queue
    read_thread_completed = threading.Event()

    # time for read thread to die
    kill_read_thread = threading.Event()

    def reading_thread(zip_fn):
        zf = zipfile.ZipFile(zip_fn)
        for fn in tqdm.tqdm_notebook(zf.namelist()):
            if kill_read_thread.is_set():
                break
            if os.path.splitext(fn)[-1] in extensions:
                buf = zf.read(fn)  # read raw bytes from zip for fn
                img = decode_image_from_buf(buf)  # decode raw bytes
                img = crop_and_preprocess(img, input_shape, preprocess_for_model)
                while True:
                    try:
                        q.put((os.path.split(fn)[-1], img), timeout=1)  # put in queue
                    except queue.Full:
                        if kill_read_thread.is_set():
                            break
                        continue
                    break

        read_thread_completed.set()  # read all images

    # start reading thread
    t = threading.Thread(target=reading_thread, args=(zip_fn,))
    t.daemon = True
    t.start()

    img_fns = []
    img_embeddings = []

    batch_imgs = []

    def process_batch(batch_imgs):
        batch_imgs = np.stack(batch_imgs, axis=0)
        batch_embeddings = model.predict(batch_imgs)
        img_embeddings.append(batch_embeddings)

    try:
        while True:
            try:
                fn, img = q.get(timeout=1)
            except queue.Empty:
                if read_thread_completed.is_set():
                    break
                continue
            img_fns.append(fn)
            batch_imgs.append(img)
            if len(batch_imgs) == batch_size:
                process_batch(batch_imgs)
                batch_imgs = []
            q.task_done()
        # process last batch
        if len(batch_imgs):
            process_batch(batch_imgs)
    finally:
        kill_read_thread.set()
        t.join()

    q.join()

    img_embeddings = np.vstack(img_embeddings)
    return img_embeddings, img_fns


def save_pickle(obj, fn):
    with open(fn, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def read_pickle(fn):
    with open(fn, "rb") as f:
        return pickle.load(f)


# look at training example (each has 5 captions)
def show_trainig_example(train_img_fns, train_captions, example_idx=0):
    """
    You can change example_idx and see different images
    """
    zf = zipfile.ZipFile("data/train2014_sample.zip")
    captions_by_file = dict(zip(train_img_fns, train_captions))
    all_files = set(train_img_fns)
    found_files = list(filter(lambda x: x.filename.rsplit("/")[-1] in all_files, zf.filelist))
    example = found_files[example_idx]
    img = decode_image_from_buf(zf.read(example))
    plt.imshow(image_center_crop(img))
    plt.title("\n".join(captions_by_file[example.filename.rsplit("/")[-1]]))
    plt.show()
