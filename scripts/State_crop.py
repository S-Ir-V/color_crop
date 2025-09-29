import copy

import cv2
import numpy as np


class State():
    def __init__(self, size):
        self.tensor = None
        self.image = np.zeros(size, dtype=np.float32)  # 输入的原始图像
        self.c_box = np.zeros((size[0], 4), dtype=np.int_)  # x1, x2, y1, y2

    def reset(self, x):
        self.image = x
        b, c, h, w = self.image.shape
        self.c_box[:, 0] = 0
        self.c_box[:, 1] = w
        self.c_box[:, 2] = 0
        self.c_box[:, 3] = h

        self.tensor = np.zeros((b, 6, h, w), dtype=np.float32)

    def set(self):
        temp = copy.copy(self.image)
        mask = np.ones_like(temp)
        self.tensor[:, 0:3, :, :] = temp
        self.tensor[:, 3:6, :, :] = mask

    def step(self, act, inner_state):
        b, c, h, w = self.image.shape
        rewardAct = np.zeros((b, 1), dtype=np.int_)
        ratio_x = w * 0.05
        ratio_y = h * 0.05

        for i in range(b):
            x1, x2, y1, y2 = self.c_box[i]

            # action space
            # 单边
            if act[i] == 0:
                x1 += ratio_x
            elif act[i] == 1:
                x2 -= ratio_x
            elif act[i] == 2:
                y1 += ratio_y
            elif act[i] == 3:
                y2 -= ratio_y
            # 双边
            elif act[i] == 4:
                x1 += ratio_x
                x2 -= ratio_x
            elif act[i] == 5:
                x1 += ratio_x
                y1 += ratio_y
            elif act[i] == 6:
                x1 += ratio_x
                y2 -= ratio_y
            elif act[i] == 7:
                x2 -= ratio_x
                y1 += ratio_y
            elif act[i] == 8:
                x2 -= ratio_x
                y2 -= ratio_y
            elif act[i] == 9:
                y1 += ratio_y
                y2 -= ratio_y
            # 三边
            elif act[i] == 10:
                x1 += ratio_x
                x2 -= ratio_x
                y1 += ratio_y
            elif act[i] == 11:
                x1 += ratio_x
                x2 -= ratio_x
                y2 -= ratio_y
            elif act[i] == 12:
                x1 += ratio_x
                y1 += ratio_y
                y2 -= ratio_y
            elif act[i] == 13:
                x2 -= ratio_x
                y1 += ratio_y
                y2 -= ratio_y
            # 四边
            elif act[i] == 14:
                x1 += ratio_x
                x2 -= ratio_x
                y1 += ratio_y
                y2 -= ratio_y


            # 裁剪框合法处理
            curnt_w = x2 - x1
            curnt_h = y2 - y1
            if curnt_w < w/5 or curnt_h < h/5:
                [x1, x2, y1, y2] = self.c_box[i]
                rewardAct[i] += -1
            if 2*curnt_w < curnt_h or 2*curnt_h < curnt_w:
                [x1, x2, y1, y2] = self.c_box[i]
                rewardAct[i] += -1

            self.c_box[i] = [x1, x2, y1, y2]

        # 执行裁剪操作（掩码版）
        mask = np.zeros_like(self.image)
        for i in range(b):
            x1, x2, y1, y2 = self.c_box[i]
            mask[i, :, y1:y2, x1:x2] = 1

        self.tensor[:, 3:6, :, :] = mask

        return rewardAct
