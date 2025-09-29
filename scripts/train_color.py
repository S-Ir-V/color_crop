import torch
import chainer
import tensorflow as tf
import tensorflow_hub as hub
import sys
import time
import numpy as np
from tqdm import tqdm
from mini_batch_loader_color import *
from MyFCN_color import MyFcn_color
import State_color as State
from pixelwise_a3c import PixelWiseA3C_InnerState
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
TEST_BATCH_SIZE = 1  # must be 1
EPISODE_LEN = 6
N_EPISODES = 30000
SNAPSHOT_EPISODES = 300
TEST_EPISODES = 300
GAMMA = 0.95  # discount factor
BETA = 0.01
EPISODE_BORDER = 15000  # decreas the learning rate at this epoch

N_ACTIONS = 13
CROP_SIZE = 224

# _/_/_/ device parameters _/_/_/
GPU_ID = 0


gpus = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(gpus[GPU_ID], 'GPU')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
chainer.cuda.get_device_from_id(GPU_ID).use()

def get_score(y_pred):
    w = torch.from_numpy(np.linspace(1, 10, 10))
    w = w.type(torch.FloatTensor)
    w = w.to(device)

    w_batch = w.repeat(y_pred.size(0), 1)

    score = (y_pred * w_batch).sum(dim=1)
    score_np = score.data.cpu().numpy()
    return score, score_np

def get_AesScore(image, model):
    b, c, h, w = image.shape
    imt = torch.from_numpy(np.zeros((b, c, 224, 224), dtype=np.float32))
    imt = imt.to(device)
    for i in range(b):
        currentImage = copy.copy(image[i, ::])
        currentImage = np.transpose(currentImage, (1, 2, 0))
        currentImage -= [0.485, 0.456, 0.406]
        currentImage /= [0.229, 0.224, 0.225]
        currentImage = np.transpose(currentImage, (2, 0, 1))
        currentImage = torch.tensor(currentImage)
        currentImage = currentImage.unsqueeze(dim=0)
        imt[i] = copy.copy(currentImage)
    with torch.no_grad():
        out = model(imt)
    pscore, pscore_np = get_score(out)

    return pscore.sum().cpu().numpy()

# def get_AesScore(image, aes_model):
#     predict_fn = aes_model.signatures['serving_default']
#     b, c, h, w = image.shape
#     aesthetic_scores = []
#
#     for i in range(b):
#         currentImage = copy.copy(image[i, ::])
#         currentImage = np.transpose(currentImage, (1, 2, 0))
#         currentImage = np.maximum(currentImage, 0)
#         currentImage = np.minimum(currentImage, 1)
#         currentImage = (currentImage * 255).astype(np.uint8)
#         _, image_bytes = cv2.imencode('.jpg', currentImage)
#         image_bytes = image_bytes.tobytes()
#         image_bytes = tf.convert_to_tensor(image_bytes, dtype=tf.string)
#         predictions = predict_fn(image_bytes)
#         aesthetic_score = predictions['predictions'].numpy()
#         aesthetic_scores.append(aesthetic_score)
#     aesthetic_scores = np.array(aesthetic_scores)
#     return aesthetic_scores


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
    for i in range(0, test_data_size, TEST_BATCH_SIZE):
        raw_x, raw_y, filename = loader.load_testing_data(np.array(range(i, i + TEST_BATCH_SIZE)))
        current_state.reset(raw_x)
        current_image_lab = bgr2lab_tensor_converter(current_state.image)

        for t in range(0, EPISODE_LEN):
            current_state.set(current_image_lab)
            action, inner_state = agent.act(current_state.tensor)
            current_state.step(action, inner_state)
            current_image_lab = bgr2lab_tensor_converter(current_state.image)

        agent.stop_episode()

        raw_y = np.transpose(raw_y[0], (1, 2, 0))
        raw_y = np.round(raw_y * 255) / 255
        raw_y = cv2.cvtColor(raw_y, cv2.COLOR_BGR2Lab)
        h, w, c = raw_y.shape
        n_pixels += h * w
        current_state.image = np.transpose(current_state.image[0], (1, 2, 0))
        current_state.image = np.maximum(current_state.image, 0)
        current_state.image = np.minimum(current_state.image, 1)
        current_state.image = np.round(current_state.image * 255) / 255
        current_state.image = cv2.cvtColor(current_state.image, cv2.COLOR_BGR2Lab)
        sum_l2_error += np.sum(np.sqrt(np.sum(np.square(current_state.image - raw_y), axis=2))) / (h * w)

    return sum_reward / test_data_size, sum_l2_error / test_data_size


def main(fout):
    # _/_/_/ load dataset _/_/_/
    mini_batch_loader = MiniBatchLoader(
        TRAINING_DATA_PATH,
        TESTING_DATA_PATH,
        IMAGE_DIR_PATH,
        CROP_SIZE)

    current_state = State.State((TRAIN_BATCH_SIZE, 3, CROP_SIZE, CROP_SIZE))

    # load myfcn model
    model = MyFcn_color(N_ACTIONS)

    # _/_/_/ setup _/_/_/
    optimizer = chainer.optimizers.Adam(alpha=LEARNING_RATE)
    optimizer.setup(model)

    agent = PixelWiseA3C_InnerState(model, optimizer, EPISODE_LEN, GAMMA, BETA)
    agent.model.to_gpu(GPU_ID)

    # TANet
    aes_model = TANet()
    aes_model.load_state_dict(torch.load(PREMODEL_PATH + "tanet.pth", map_location='cuda:0'))
    aes_model = aes_model.to(device)
    aes_model.eval()

    # # Vila
    # aes_model = hub.load(PREMODEL_PATH +"vila")

    # _/_/_/ training _/_/_/
    train_data_size = MiniBatchLoader.count_paths(TRAINING_DATA_PATH)
    indices = np.random.permutation(train_data_size)
    i = 0
    for episode in tqdm(range(1, N_EPISODES + 1), desc="Training Progress", unit="ep"):
        # print("episode %d" % episode)
        fout.write("episode %d\n" % episode)
        sys.stdout.flush()

        # load data
        r = indices[i:i + TRAIN_BATCH_SIZE]
        raw_x, raw_y, filename = mini_batch_loader.load_training_data(r)

        current_state.reset(raw_x)
        current_image_lab = bgr2lab_tensor_converter(current_state.image)
        raw_y_lab = bgr2lab_tensor_converter(raw_y)

        reward = np.zeros(raw_x.shape, raw_x.dtype)
        sum_reward = 0
        premean = get_AesScore(current_state.image, aes_model)
        # premean = 0.0

        for t in range(0, EPISODE_LEN):
            previous_image_lab = current_image_lab.copy()
            current_state.set(current_image_lab)
            # current_state.set()
            action, inner_state = agent.act_and_train(current_state.tensor, reward)
            current_state.step(action, inner_state)
            current_image_lab = bgr2lab_tensor_converter(current_state.image)

            # reward function
            mean = get_AesScore(current_state.image, aes_model)
            rewardAes = np.subtract(mean, premean)

            rewardFea = np.sqrt(np.sum(np.square(raw_y_lab - previous_image_lab), axis=1)[:, np.newaxis, :, :]) - np.sqrt(np.sum(np.square(raw_y_lab - current_image_lab), axis=1)[:, np.newaxis, :, :])

            reward = rewardFea + 0.1 * rewardAes

            sum_reward += np.mean(reward) * np.power(GAMMA, t)
            premean = mean

        with writer.as_default():
            tf.summary.scalar('Reward', sum_reward, step=episode)
        agent.stop_episode_and_train(current_state.tensor, reward, True)
        fout.write("train total reward {a}\n".format(a=sum_reward))
        sys.stdout.flush()

        # if episode % 300 == 0:
        #     # _/_/_/ testing _/_/_/
        #     reward, l2_error = model_test(mini_batch_loader, agent, fout)
        #     with writer.as_default():
        #         tf.summary.scalar('Mean_Reward', reward, step=episode)
        #         tf.summary.scalar('Mean_L2_Error', l2_error, step=episode)

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


if __name__ == '__main__':
    log_dir = f'../logs/color_{MODEL_NAME}_{time.strftime("%Y-%m-%d_%H-%M-%S")}'
    writer = tf.summary.create_file_writer(log_dir)
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
