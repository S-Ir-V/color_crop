import collections
import os
import numpy as np
import cv2
import _pickle


# PathInfo = collections.namedtuple('PathInfo', ['image_path'])


class MiniBatchLoader(object):

    def __init__(self, test_path, image_dir_path, crop_size):

        # load data paths
        self.testing_path_infos = self.read_test_paths(test_path, image_dir_path)
        self.crop_size = crop_size

    @staticmethod
    def path_label_generator_test(txt_path, src_path):
        for line in open(txt_path):
            line = line.strip()
            # line = line.split()[0].rsplit('.', 1)[0]
            # image_name = 'input/' + line + '.jpg'
            # image_name = line + '.png'
            image_name = line + '.jpg'
            # dst_image_name = 'target/' + line + '.jpg'
            dst_image_name = line + '.jpg'
            dst_full_path = os.path.join(src_path, dst_image_name)
            full_path = os.path.join(src_path, image_name)
            if os.path.isfile(full_path) and os.path.isfile(dst_full_path):
                yield full_path, dst_full_path
            else:
                print(dst_full_path)
                print(full_path)

    # test ok
    @staticmethod
    def count_paths(path):
        c = 0
        for _ in open(path):
            c += 1
        return c

    @staticmethod
    def read_test_paths(txt_path, src_path):
        cs = []
        for pair in MiniBatchLoader.path_label_generator_test(txt_path, src_path):
            cs.append(pair)
        return cs

    def load_testing_data(self, indices):
        return self.load_data(self.testing_path_infos, indices)

    # test ok
    def load_data(self, path_infos, indices):
        mini_batch_size = len(indices)
        in_channels = 3

        for i, index in enumerate(indices):
            img_path, s_img_path = path_infos[index]
            file_name = os.path.splitext(os.path.basename(img_path))[0]

            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            s_img = cv2.imread(s_img_path, cv2.IMREAD_COLOR)
            if img is None or s_img is None:
                raise RuntimeError("invalid image: {i}".format(i=img_path))

        h, w, c = img.shape
        xt = np.zeros((mini_batch_size, in_channels, h, w)).astype(np.float32)
        # xs = np.zeros((mini_batch_size, in_channels, h, w)).astype(np.float32)
        img = (img / 255).astype(np.float32)
        # s_img = (s_img / 255).astype(np.float32)
        xt[0, :, :, :] = np.transpose(img, (2, 0, 1))
        # xs[0, :, :, :] = np.transpose(s_img, (2, 0, 1))

        # return xt, xs, file_name
        return xt, file_name
