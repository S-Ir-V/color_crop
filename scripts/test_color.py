import chainer
import cv2
import numpy as np
import torch
import tensorflow as tf
import tensorflow_hub as hub
from chainer import serializers
from tqdm import tqdm
import os
import sys
import time
from dataloader_test import MiniBatchLoader
from MyFCN_color import MyFcn_color
import State_color as State
from pixelwise_a3c import PixelWiseA3C_InnerState

# _/_/_/ paths _/_/_/
IMAGE_DIR_PATH = ""  # data
TESTING_DATA_PATH = ""  # test_data_list
TEST_MODEL_PATH = ""  # model path
SAVE_PATH = ""  # images are saved here
LOG_PATH = ""  # test_log is saved here

# _/_/_/ training parameters _/_/_/
LEARNING_RATE = 0.0001
TEST_BATCH_SIZE = 1  # must be 1
EPISODE_LEN = 6
TEST_EPISODES = 300
GAMMA = 0.95  # discount factor
BETA = 0.01

N_ACTIONS = 13
CROP_SIZE = 224

# _/_/_/ device parameters _/_/_/
GPU_ID = 0


gpus = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(gpus[GPU_ID], 'GPU')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
chainer.cuda.get_device_from_id(GPU_ID).use()

def bgr2lab_tensor_converter(src):
    b, c, h, w = src.shape
    src_t = np.transpose(src, (0, 2, 3, 1))
    dst = np.zeros(src_t.shape, src_t.dtype)
    for i in range(0, b):
        dst[i] = cv2.cvtColor(src_t[i], cv2.COLOR_BGR2Lab)
    return np.transpose(dst, (0, 3, 1, 2))


def model_test(loader, agent, fout):
    sum_l2_error = 0
    sum_reward = 0
    n_pixels = 0
    test_data_size = MiniBatchLoader.count_paths(TESTING_DATA_PATH)
    current_state = State.State((TEST_BATCH_SIZE, 3, CROP_SIZE, CROP_SIZE))
    for i in tqdm(range(0, test_data_size, TEST_BATCH_SIZE), desc="Testing Progress", unit="ep"):
        raw_x, filename = loader.load_testing_data(np.array(range(i, i + TEST_BATCH_SIZE)))
        current_state.reset(raw_x)
        current_image_lab = bgr2lab_tensor_converter(current_state.image)

        for t in range(0, EPISODE_LEN):
            current_state.set(current_image_lab)
            action, inner_state = agent.act(current_state.tensor)
            current_state.step(action, inner_state)
            current_image_lab = bgr2lab_tensor_converter(current_state.image)

        agent.stop_episode()

        current_state.image = np.transpose(current_state.image[0], (1, 2, 0))
        current_state.image = np.maximum(current_state.image, 0)
        current_state.image = np.minimum(current_state.image, 1)
        u16image = (current_state.image * (2 ** 16 - 1) + 0.5).astype(np.uint16)
        cv2.imwrite(SAVE_PATH + filename + '.png', u16image)

        current_state.image = cv2.cvtColor(current_state.image, cv2.COLOR_BGR2Lab)

    print("test total reward {a}, l2_error {b}".format(a=sum_reward / test_data_size, b=sum_l2_error / test_data_size))
    fout.write(
        "test total reward {a}, l2_error {b}\n".format(a=sum_reward / test_data_size, b=sum_l2_error / test_data_size))
    sys.stdout.flush()


def main(fout):
    # _/_/_/ load dataset _/_/_/
    mini_batch_loader = MiniBatchLoader(
        TESTING_DATA_PATH,
        IMAGE_DIR_PATH,
        CROP_SIZE)

    chainer.cuda.get_device_from_id(GPU_ID).use()

    # load myfcn model
    model = MyFcn_color(N_ACTIONS)

    # _/_/_/ setup _/_/_/

    optimizer = chainer.optimizers.Adam(alpha=LEARNING_RATE)
    optimizer.setup(model)

    agent = PixelWiseA3C_InnerState(model, optimizer, int(EPISODE_LEN / 2), GAMMA, BETA, act_deterministically = True)
    serializers.load_npz(TEST_MODEL_PATH, agent.model)
    agent.model.train = False
    agent.model.to_gpu()

    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)

    # _/_/_/ testing _/_/_/
    model_test(mini_batch_loader, agent, fout)


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

