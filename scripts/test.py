import copy
import os
import sys
import time
import torch
import tensorflow as tf
import chainer
import cv2
import numpy as np
from chainer import serializers
from tqdm import tqdm

import State_color as State_color
import State_crop as State_crop
from MyFCN_color import MyFcn_color
from MyFCN_crop import MyFcn_crop
from a3c import A3C_InnerState
from dataloader_test import MiniBatchLoader
from pixelwise_a3c import PixelWiseA3C_InnerState


# _/_/_/ paths _/_/_/
IMAGE_DIR_PATH = ""
TESTING_DATA_PATH = ""
TEST_MODEL_PATH_COLOR = ""  # model path
TEST_MODEL_PATH_CROP = ""  # model path
SAVE_PATH = ""
LOG_PATH = ""

# _/_/_/ training parameters _/_/_/
LEARNING_RATE = 0.0001
TEST_BATCH_SIZE = 1  # must be 1
GAMMA_COLOR = 0.95  # discount factor
GAMMA_CROP = 0.99  # discount factor
BETA_COLOR = 0.01
BETA_CROP = 0.05

# _/_/_/ other parameters _/_/_/
EPISODE_LEN_COLOR = 6
EPISODE_LEN_CROP = 10
N_ACTIONS_COLOR = 13
N_ACTIONS_CROP = 16
CROP_SIZE = 224

# _/_/_/ device parameters _/_/_/
GPU_ID = 0


gpus = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(gpus[GPU_ID], 'GPU')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
chainer.cuda.get_device_from_id(GPU_ID).use()


def draw_dashed_rectangle(img, box, color, thickness=1, dash_length=10, type="draw"):
    x1, x2, y1, y2 = box
    if type == "crop":
        #执行裁剪操作
        img = img[:, y1:y2, x1:x2]
    if type == "mask":
        mask = np.zeros_like(img)
        mask[:, y1:y2, x1:x2] = 1
        img = img * mask
    img = np.transpose(img, (1, 2, 0))
    img = np.maximum(img, 0)
    img = np.minimum(img, 1)

    if img is None:
        raise ValueError("Image is not loaded correctly")

    if img.dtype == np.uint16:
        img = (img / 256).astype(np.uint8)
    elif img.dtype == np.float32 or img.dtype == np.float64:
        img = (img * 255).astype(np.uint8)
    img = np.ascontiguousarray(img)

    if type == "draw":
        # 执行绘制框图操作
        for i in range(x1, x2, dash_length * 2):
            cv2.line(img, (i, y1), (min(i + dash_length, x2), y1), color, thickness)
            cv2.line(img, (i, y2), (min(i + dash_length, x2), y2), color, thickness)
        for i in range(y1, y2, dash_length * 2):
            cv2.line(img, (x1, i), (x1, min(i + dash_length, y2)), color, thickness)
            cv2.line(img, (x2, i), (x2, min(i + dash_length, y2)), color, thickness)

    return img


def bgr2lab_tensor_converter(src):
    b, c, h, w = src.shape
    src_t = np.transpose(src, (0, 2, 3, 1))
    dst = np.zeros(src_t.shape, src_t.dtype)
    for i in range(0, b):
        dst[i] = cv2.cvtColor(src_t[i], cv2.COLOR_BGR2Lab)
    return np.transpose(dst, (0, 3, 1, 2))


def model_test(loader, agent_color, agent_crop, fout):
    test_data_size = MiniBatchLoader.count_paths(TESTING_DATA_PATH)

    state_color = State_color.State((TEST_BATCH_SIZE, 3, CROP_SIZE, CROP_SIZE))
    state_crop = State_crop.State((TEST_BATCH_SIZE, 3, CROP_SIZE, CROP_SIZE))

    for i in tqdm(range(0, test_data_size, TEST_BATCH_SIZE), desc="Testing Progress", unit="ep"):
        raw_x, filename = loader.load_testing_data(np.array(range(i, i + TEST_BATCH_SIZE)))

        state_color.reset(raw_x)
        state_crop.reset(raw_x)

        current_image_lab = bgr2lab_tensor_converter(state_color.image)

        for t in range(0, EPISODE_LEN_COLOR):
            state_color.set(current_image_lab)
            action_color, inner_state_color = agent_color.act(state_color.tensor)
            state_color.step(action_color, inner_state_color)
            current_image_lab = bgr2lab_tensor_converter(state_color.image)

        for t in range(0, EPISODE_LEN_CROP):
            state_crop.set()
            action_crop, inner_state_crop = agent_crop.act(state_crop.tensor)
            state_crop.step(action_crop, inner_state_crop)

        agent_color.stop_episode()
        agent_crop.stop_episode()

        # 保存test结果
        currentImage = copy.copy(state_color.image[0])
        # 处理图像, 裁剪or绘制裁剪框or掩码裁剪
        currentImage = draw_dashed_rectangle(currentImage, state_crop.c_box[0], (0, 255, 0), thickness=2, type="mask")

        cv2.imwrite(SAVE_PATH + filename + '.png', currentImage)


    sys.stdout.flush()


def main(fout):
    # _/_/_/ load dataset _/_/_/
    mini_batch_loader = MiniBatchLoader(
        TESTING_DATA_PATH,
        IMAGE_DIR_PATH,
        CROP_SIZE)

    chainer.cuda.get_device_from_id(GPU_ID).use()

    # load myfcn model
    model_color = MyFcn_color(N_ACTIONS_COLOR)
    model_crop = MyFcn_crop(N_ACTIONS_CROP)

    # _/_/_/ setup _/_/_/
    optimizer = chainer.optimizers.Adam(alpha=LEARNING_RATE)
    optimizer.setup(model_color)
    optimizer.setup(model_crop)

    agent_color = PixelWiseA3C_InnerState(model_color, optimizer, int(EPISODE_LEN_COLOR / 2), GAMMA_COLOR, BETA_COLOR, act_deterministically = True)
    serializers.load_npz(TEST_MODEL_PATH_COLOR, agent_color.model)
    agent_color.model.train = False
    agent_color.model.to_gpu()

    agent_crop = A3C_InnerState(model_crop, optimizer, int(EPISODE_LEN_CROP / 2), GAMMA_CROP, BETA_CROP, act_deterministically = True)
    serializers.load_npz(TEST_MODEL_PATH_CROP, agent_crop.model)
    agent_crop.model.train = False
    agent_crop.model.to_gpu()

    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    # _/_/_/ testing _/_/_/
    model_test(mini_batch_loader, agent_color, agent_crop, fout)


if __name__ == '__main__':
    fout = open(LOG_PATH, "w")
    start = time.time()
    main(fout)
    end = time.time()
    print("{s}[s]".format(s=end - start))
    print("{s}[m]".format(s=(end - start) / 60))
    print("{s}[h]".format(s=(end - start) / 60 / 60))
    fout.write("{s}[s]\n".format(s=end - start))
    fout.write("{s}[m]\n".format(s=(end - start) / 60))
    fout.write("{s}[h]\n".format(s=(end - start) / 60 / 60))
    fout.close()

