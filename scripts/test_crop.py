import torch
import chainer
import tensorflow as tf
import tensorflow_hub as hub
from chainer import serializers
import os
import sys
import cv2
import time
import numpy as np
from tqdm import tqdm
import State_crop as State
from a3c import A3C_InnerState
from dataloader_test import MiniBatchLoader
from MyFCN_crop import MyFcn_crop
from models.model import *



# _/_/_/ paths _/_/_/
IMAGE_DIR_PATH = "/"
TESTING_DATA_PATH = ""
PREMODEL_PATH = ""
MODEL_PATH = ""
SAVE_PATH = ""
SAVE_PATH_TXT = ""
SAVE_PATH_ROOT = ""


# _/_/_/ training parameters _/_/_/
LEARNING_RATE = 0.0001
TEST_BATCH_SIZE  = 1 #must be 1
EPISODE_LEN = 10
GAMMA = 0.99  # discount factor
BETA = 0.05

# _/_/_/ other parameters _/_/_/
N_ACTIONS = 16
CROP_SIZE = 224

# _/_/_/ device parameters _/_/_/
GPU_ID = 0


gpus = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(gpus[GPU_ID], 'GPU')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
chainer.cuda.get_device_from_id(GPU_ID).use()


def get_AesScore(current_state, aes_model):
    predict_fn = aes_model.signatures['serving_default']
    b, c, h, w = current_state.image.shape
    aesthetic_scores = []

    for i in range(b):
        currentImage = copy.copy(current_state.image[i, ::])
        x1, x2, y1, y2 = current_state.c_box[i]
        currentImage = currentImage[:, y1:y2, x1:x2]
        currentImage = np.transpose(currentImage, (1, 2, 0))
        currentImage = cv2.resize(currentImage, (224, 224))
        currentImage = np.maximum(currentImage, 0)
        currentImage = np.minimum(currentImage, 1)
        currentImage = (currentImage * 255).astype(np.uint8)
        _, image_bytes = cv2.imencode('.jpg', currentImage)
        image_bytes = image_bytes.tobytes()
        image_bytes = tf.convert_to_tensor(image_bytes, dtype=tf.string)
        predictions = predict_fn(image_bytes)
        aesthetic_score = predictions['predictions'].numpy()
        aesthetic_scores.append(aesthetic_score)
    aesthetic_scores = np.array(aesthetic_scores)
    return aesthetic_scores


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


def model_test(loader, agent, model, fout):
    test_data_size = MiniBatchLoader.count_paths(TESTING_DATA_PATH)
    current_state = State.State((TEST_BATCH_SIZE, 3, CROP_SIZE, CROP_SIZE))
    for i in tqdm(range(0, test_data_size, TEST_BATCH_SIZE), desc="Testing Progress", unit="ep"):
        agent.model.reset_state()
        raw_x, filename = loader.load_testing_data(np.array(range(i, i + TEST_BATCH_SIZE)))
        current_state.reset(raw_x)

        for t in range(0, EPISODE_LEN):
            current_state.set()
            action, inner_state = agent.act(current_state.tensor)
            current_state.step(action, inner_state)

        agent.stop_episode()

        # 保存test结果
        # 保存裁剪框
        x1, x2, y1, y2 = current_state.c_box[0]
        img_name = filename[0]
        fout.write(f"{img_name}.jpg {int(y1)} {int(y2)} {int(x1)} {int(x2)}\n")
        fout.flush()

        # 保存图像
        currentImage0 = copy.copy(current_state.image[0])
        # 处理图像, 裁剪or绘制裁剪框
        currentImage = draw_dashed_rectangle(currentImage0, current_state.c_box[0], (0, 255, 0), thickness=2, type="draw")
        cv2.imwrite(SAVE_PATH + "_draw/" + filename[0] + '.jpg', currentImage)
        currentImage = draw_dashed_rectangle(currentImage0, current_state.c_box[0], (0, 255, 0), thickness=2, type="crop")
        cv2.imwrite(SAVE_PATH + "_crop/" + filename[0] + '.jpg', currentImage)
        currentImage = draw_dashed_rectangle(currentImage0, current_state.c_box[0], (0, 255, 0), thickness=2, type="mask")
        cv2.imwrite(SAVE_PATH + "_mask/" + filename[0] + '.jpg', currentImage)

        # cv2.imwrite(SAVE_PATH + filename[0] + '.jpg', currentImage)


def main(fout):
    # _/_/_/ load dataset _/_/_/
    mini_batch_loader = MiniBatchLoader(
        TESTING_DATA_PATH,
        IMAGE_DIR_PATH,
        CROP_SIZE)

    chainer.cuda.get_device_from_id(GPU_ID).use()

    # load myfcn model
    model = MyFcn_crop(N_ACTIONS)

    # _/_/_/ setup _/_/_/
    optimizer = chainer.optimizers.Adam(alpha=LEARNING_RATE)
    optimizer.setup(model)

    agent = A3C_InnerState(model, optimizer, int(EPISODE_LEN / 2), GAMMA, BETA, act_deterministically = True)
    serializers.load_npz(MODEL_PATH, agent.model)
    # agent.model.train = False
    agent.model.reset_state()
    agent.model.to_gpu()

    # # Vila
    # aes_model = hub.load("../premodel/vila")
    aes_model = ''

    # if not os.path.exists(SAVE_PATH):
    #     os.makedirs(SAVE_PATH)
    if not os.path.exists(SAVE_PATH + "_draw/"):
        os.makedirs(SAVE_PATH + "_draw/")
    if not os.path.exists(SAVE_PATH + "_crop/"):
        os.makedirs(SAVE_PATH + "_crop/")
    if not os.path.exists(SAVE_PATH + "_mask/"):
        os.makedirs(SAVE_PATH + "_mask/")

    # _/_/_/ testing _/_/_/
    model_test(mini_batch_loader, agent, aes_model, fout)


if __name__ == "__main__":
    if not os.path.exists(SAVE_PATH_ROOT):
        os.makedirs(SAVE_PATH_ROOT)
    if not os.path.exists(SAVE_PATH_TXT):
        with open(SAVE_PATH_TXT, 'w', encoding='utf-8') as f:
            pass
    fout = open(SAVE_PATH_TXT, "w")
    # start = time.time()
    main(fout)
    # end = time.time()
    # print("{s}[s]".format(s=end - start))
    # print("{s}[m]".format(s=(end - start) / 60))
    # print("{s}[h]".format(s=(end - start) / 60 / 60))
    # fout.write("{s}[s]\n".format(s=end - start))
    # fout.write("{s}[m]\n".format(s=(end - start) / 60))
    # fout.write("{s}[h]\n".format(s=(end - start) / 60 / 60))
    fout.close()
