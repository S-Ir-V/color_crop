import numpy as np
import sys
import cv2


class State():
    def __init__(self, size):
        self.image = np.zeros(size, dtype=np.float32)

    def reset(self, x):
        self.image = x
        size = self.image.shape
        prev_state = np.zeros((size[0], 64, size[2], size[3]), dtype=np.float32)
        self.tensor = np.concatenate((self.image, prev_state), axis=1)

    def set(self, x):
        temp = np.copy(x)
        temp[:, 0, :, :] /= 100
        temp[:, 1, :, :] /= 127
        temp[:, 2, :, :] /= 127
        self.tensor[:, :self.image.shape[1], :, :] = temp

    def step(self, act, inner_state):
        b, c, h, w = self.image.shape
        image = self.image

        bgr_t = np.transpose(image, (0, 2, 3, 1))  # (B, H, W, C)
        temp3 = np.zeros_like(bgr_t)
        temp4 = np.zeros_like(bgr_t)

        mask3 = (act == 3).any(axis=(1, 2))  # (B,)
        mask4 = (act == 4).any(axis=(1, 2))

        if mask3.any():
            for i in np.where(mask3)[0]:
                temp = cv2.cvtColor(bgr_t[i], cv2.COLOR_BGR2HSV)
                temp[..., 1] *= 0.95
                temp3[i] = cv2.cvtColor(temp, cv2.COLOR_HSV2BGR)
        if mask4.any():
            for i in np.where(mask4)[0]:
                temp = cv2.cvtColor(bgr_t[i], cv2.COLOR_BGR2HSV)
                temp[..., 1] *= 1.05
                temp4[i] = cv2.cvtColor(temp, cv2.COLOR_HSV2BGR)

        bgr3 = np.transpose(temp3, (0, 3, 1, 2))
        bgr4 = np.transpose(temp4, (0, 3, 1, 2))

        # 预计算所有分支图像（避免多次拷贝）
        bgr1 = image * 0.98 + 0.025
        bgr2 = image * 1.02 - 0.025
        bgr5 = image - 0.025
        bgr6 = image + 0.025
        bgr7 = image.copy()
        bgr7[:, 0] *= 0.95
        bgr8 = image.copy()
        bgr8[:, 0] *= 1.05
        bgr9 = image.copy()
        bgr9[:, 1] *= 0.95
        bgr10 = image.copy()
        bgr10[:, 1] *= 1.05
        bgr11 = image.copy()
        bgr11[:, 2] *= 0.95
        bgr12 = image.copy()
        bgr12[:, 2] *= 1.05

        act_3channel = np.broadcast_to(act[:, None, :, :], (b, c, h, w))

        result = np.empty_like(image)
        masks = {
            1: bgr1,
            2: bgr2,
            3: bgr3,
            4: bgr4,
            5: bgr5,
            6: bgr6,
            7: bgr7,
            8: bgr8,
            9: bgr9,
            10: bgr10,
            11: bgr11,
            12: bgr12,
        }

        result[:] = image
        for act_val, img_val in masks.items():
            mask = (act_3channel == act_val)
            np.copyto(result, img_val, where=mask)

        self.image = result
        self.tensor[:, :c, :, :] = self.image
        self.tensor[:, -64:, :, :] = inner_state