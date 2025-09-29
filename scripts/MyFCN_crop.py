import chainer
import chainer.links as L
import chainer.functions as F
import chainerrl
import numpy as np
from chainerrl.agents import a3c
import cupy as cp

def set_random_seed(seed):
    np.random.seed(seed)
    cp.random.seed(seed)
    chainer.config.cudnn_deterministic = True
    chainer.config.use_cudnn = 'never'

set_random_seed(42)

def bbox_from_mask(mask):
    mask_np = chainer.cuda.to_cpu(mask.array)
    bboxes = []
    for m in mask_np[:, 0]:
        ys, xs = np.where(m > 0)
        if len(xs) == 0:
            x1, x2, y1, y2 = 0, m.shape[1], 0, m.shape[0]
        else:
            x1, x2 = xs.min(), xs.max() + 1
            y1, y2 = ys.min(), ys.max() + 1
        bboxes.append((y1, y2, x1, x2))
    return bboxes

class MyFcn_crop(chainer.Chain, a3c.A3CModel):

    def __init__(self, n_actions):
        super(MyFcn_crop, self).__init__(
            conv1=L.Convolution2D(3, 96, ksize=11, stride=4, pad=0),
            conv2 = L.Convolution2D(96, 256, ksize=5, stride=1, pad=2),
            conv3 = L.Convolution2D(256, 384, ksize=3, stride=1, pad=1),
            conv4 = L.Convolution2D(384, 384, ksize=3, stride=1, pad=1),
            conv5 = L.Convolution2D(384, 256, ksize=3, stride=1, pad=1),
            fc1=L.Linear(256, 256),
            fc2=L.Linear(256, 128),
            fc3=L.Linear(128, 128),
            lstm_pi=chainerrl.policies.SoftmaxPolicy(L.LSTM(128, n_actions)),
            lstm_v=L.LSTM(128, 1),
        )
        self.train = True

    def reset_state(self):
        self.lstm_pi.model.reset_state()
        self.lstm_v.reset_state()

    def pi_and_v(self, x):
        h = x[:, 0:3, :, :]
        h = F.relu(self.conv1(h))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = F.relu(self.conv4(h))
        h = F.relu(self.conv5(h))

        mask = x[:, -3:, :, :]
        mask = F.max_pooling_2d(mask, ksize=11, stride=4, pad=0)
        mask = F.max_pooling_2d(mask, ksize=5, stride=1, pad=2)
        mask = F.max_pooling_2d(mask, ksize=3, stride=1, pad=1)
        target_height, target_width = h.shape[2], h.shape[3]
        current_height, current_width = mask.shape[2], mask.shape[3]
        if current_height > target_height or current_width > target_width:
            mask = mask[:, :, :target_height, :target_width]
        elif current_height < target_height or current_width < target_width:
            pad_height = target_height - current_height
            pad_width = target_width - current_width
            pad_top = pad_height // 2
            pad_bottom = pad_height - pad_top
            pad_left = pad_width // 2
            pad_right = pad_width - pad_left
            mask = F.pad(mask, ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)), mode='constant',
                         constant_values=0)

        mask = F.expand_dims(mask[:, 0, :, :], 1)
        mask = F.broadcast_to(mask, h.shape)
        h1 = h * mask
        h1 = F.average_pooling_2d(h1, ksize=h1.shape[2:], stride=h1.shape[2:])

        h = F.dropout(F.relu(self.fc1(h1)))
        h = F.dropout(F.relu(self.fc2(h)))
        h = F.dropout(F.relu(self.fc3(h)))

        pout = self.lstm_pi(h)
        vout = self.lstm_v(h)

        return pout, vout, h