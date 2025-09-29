import cv2
import torch
import chainer
import tensorflow as tf
import tensorflow_hub as hub
import sys
import time
import numpy as np
from tqdm import tqdm

from mini_batch_loader_crop import MiniBatchLoader
from MyFCN_crop import MyFcn_crop
import State_crop as State
from a3c import A3C_InnerState
from models.model import *


# _/_/_/ paths _/_/_/
IMAGE_DIR_PATH = "/"
TRAINING_DATA_PATH = "train.txt"
TESTING_DATA_PATH = "test.txt"
PREMODEL_PATH = "premodel/"
SAVE_PATH = "model/fpop_myfcn_"
LOG_PATH = "log.txt"
MODEL_NAME = ""


# _/_/_/ training parameters _/_/_/
LEARNING_RATE = 0.0001
TRAIN_BATCH_SIZE = 2
TEST_BATCH_SIZE  = 1 #must be 1
EPISODE_LEN = 10
N_EPISODES = 30000
SNAPSHOT_EPISODES = 300
TEST_EPISODES = 300
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
        predictions = predict_fn(image_bytes=image_bytes)
        aesthetic_score = predictions['predictions'].numpy()
        aesthetic_scores.append(aesthetic_score)
    aesthetic_scores = np.array(aesthetic_scores)
    return aesthetic_scores


def model_test(loader, agent, aes_model, fout):
    sum_AesScore = 0.0
    test_data_size = MiniBatchLoader.count_paths(TESTING_DATA_PATH)
    current_state = State.State((TEST_BATCH_SIZE, 3, CROP_SIZE, CROP_SIZE))
    for i in range(0, test_data_size, TEST_BATCH_SIZE):
        agent.model.reset_state()
        raw_x, filename = loader.load_testing_data(np.array(range(i, i + TEST_BATCH_SIZE)))
        current_state.reset(raw_x)

        for t in range(0, EPISODE_LEN):
            current_state.set()
            action, inner_state = agent.act(current_state.tensor)
            current_state.step(action, inner_state)
        agent.stop_episode()

        sum_AesScore += get_AesScore(current_state, aes_model).mean()
    # print("test total AesScore {a}".format(a=sum_AesScore / test_data_size))
    fout.write(
        "test total AesScore {a}".format(a=sum_AesScore / test_data_size))
    sys.stdout.flush()
    return sum_AesScore / test_data_size


def main(fout):
    # _/_/_/ load dataset _/_/_/
    mini_batch_loader = MiniBatchLoader(
        TRAINING_DATA_PATH,
        TESTING_DATA_PATH,
        IMAGE_DIR_PATH,
        CROP_SIZE)

    current_state = State.State((TRAIN_BATCH_SIZE, 3, CROP_SIZE, CROP_SIZE))

    # load myfcn model
    model = MyFcn_crop(N_ACTIONS)

    # _/_/_/ setup _/_/_/
    optimizer = chainer.optimizers.Adam(alpha=LEARNING_RATE)
    optimizer.setup(model)

    agent = A3C_InnerState(model, optimizer, EPISODE_LEN, GAMMA, BETA)
    agent.model.to_gpu(GPU_ID)

    # Vila
    aes_model = hub.load("../premodel/vila")

    # _/_/_/ training _/_/_/
    train_data_size = MiniBatchLoader.count_paths(TRAINING_DATA_PATH)
    indices = np.random.permutation(train_data_size)
    i = 0
    for episode in tqdm(range(1, N_EPISODES + 1), desc="Training Progress", unit="ep"):
        # print("episode %d" % episode)
        fout.write("episode %d\n" % episode)
        sys.stdout.flush()
        agent.model.reset_state()

        # load data
        r = indices[i:i + TRAIN_BATCH_SIZE]
        raw_x, filename = mini_batch_loader.load_training_data(r)

        # 初始化状态
        current_state.reset(raw_x)

        # 初始化奖励参数
        reward = np.zeros(raw_x.shape, raw_x.dtype)
        sum_reward = 0.0

        # 计算原始图像美学得分
        premean = get_AesScore(current_state, aes_model)
        oral = premean

        # 执行动作
        for t in range(0, EPISODE_LEN):
            current_state.set()
            action, inner_state = agent.act_and_train(current_state.tensor, reward)
            rewardAct = current_state.step(action, inner_state)

            # 计算一批次图片的平均美学得分
            mean = get_AesScore(current_state, aes_model)

            # _/_/_/ reward function _/_/_/
            rewardAes = np.subtract(mean, premean)
            # rewardAes2 = np.subtract(mean, oral)
            # rewardAes2 = np.sign(rewardAes2)
            # rewardAes2 = np.maximum(rewardAes2, 0)

            reward = 0.1*rewardAct.mean() + rewardAes.mean()

            sum_reward += np.mean(reward) * np.power(GAMMA, t)
            premean = mean


        with writer.as_default():
            tf.summary.scalar('Reward', sum_reward, step=episode)
        agent.stop_episode_and_train(current_state.tensor, reward, True)
        # print("train total reward {a}".format(a=sum_reward))
        fout.write("train total reward {a}\n".format(a=sum_reward))
        sys.stdout.flush()

        # if episode % 300 == 0:
        #     # _/_/_/ testing _/_/_/
        #     mean_AesScore = model_test(mini_batch_loader, agent, aes_model, fout)
        #     with writer.as_default():
        #         tf.summary.scalar('Mean_AesScore', mean_AesScore, step=episode)

        if episode % SNAPSHOT_EPISODES == 0:
            agent.save(SAVE_PATH + str(episode))

        if i + TRAIN_BATCH_SIZE >= train_data_size:
            i = 0
            indices = np.random.permutation(train_data_size)
        else:
            i += TRAIN_BATCH_SIZE

        if i + 2 * TRAIN_BATCH_SIZE >= train_data_size:
            i = train_data_size - TRAIN_BATCH_SIZE

        # if episode % EPISODE_BORDER == 0:
        #    optimizer.alpha *= 0.1
        optimizer.alpha = LEARNING_RATE * ((1 - episode / N_EPISODES) ** 0.9)


if __name__ == "__main__":
    log_dir = f'../logs/crop_{MODEL_NAME}_{time.strftime("%Y-%m-%d_%H-%M-%S")}'
    writer = tf.summary.create_file_writer(log_dir)

    fout = open(LOG_PATH, "w")
    print("LEARNING_RATE is {a}, GAMMA is {b}, BETA is {c}".format(a=LEARNING_RATE, b=GAMMA, c=BETA))
    fout.write("LEARNING_RATE is {a}, GAMMA is {b}, BETA is {c}\n".format(a=LEARNING_RATE, b=GAMMA, c=BETA))

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