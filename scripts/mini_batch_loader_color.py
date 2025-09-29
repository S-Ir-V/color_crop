import collections
import os
import numpy as np
import cv2
import _pickle


# PathInfo = collections.namedtuple('PathInfo', ['image_path'])


class MiniBatchLoader(object):

    def __init__(self, train_path, test_path, image_dir_path, crop_size):

        # load data paths
        self.training_path_infos = self.read_paths(train_path, image_dir_path)
        self.testing_path_infos = self.read_test_paths(test_path, image_dir_path)

        self.crop_size = crop_size

    # test ok
    @staticmethod
    def path_label_generator(txt_path, src_path):
        for line in open(txt_path):
            line = line.strip()
            image_name = 'train/input/' + line + '.jpg'
            dst_image_name = 'train/target/' + line + '.jpg'
            dst_full_path = os.path.join(src_path, dst_image_name)
            full_path = os.path.join(src_path, image_name)
            if os.path.isfile(full_path) and os.path.isfile(dst_full_path):
                yield full_path, dst_full_path
            else:
                print(dst_full_path)
                print(full_path)

    @staticmethod
    def path_label_generator_test(txt_path, src_path):
        for line in open(txt_path):
            line = line.strip()
            image_name = 'test/input/' + line + '.jpg'
            dst_image_name = 'test/target/' + line + '.jpg'
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

    # test ok
    @staticmethod
    def read_paths(txt_path, src_path):
        cs = []
        for pair in MiniBatchLoader.path_label_generator(txt_path, src_path):
            cs.append(pair)
        return cs

    @staticmethod
    def read_test_paths(txt_path, src_path):
        cs = []
        for pair in MiniBatchLoader.path_label_generator_test(txt_path, src_path):
            cs.append(pair)
        return cs

    def load_training_data(self, indices):
        return self.load_data(self.training_path_infos, indices, augment=True)

    def load_testing_data(self, indices):
        return self.load_data(self.testing_path_infos, indices)

    # test ok
    def load_data(self, path_infos, indices, augment=False):
        mini_batch_size = len(indices)
        in_channels = 3

        if augment:
            xt = np.zeros((mini_batch_size, in_channels, self.crop_size, self.crop_size)).astype(np.float32)
            xs = np.zeros((mini_batch_size, in_channels, self.crop_size, self.crop_size)).astype(np.float32)

            for i, index in enumerate(indices):
                img_path, s_img_path = path_infos[index]

                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                s_img = cv2.imread(s_img_path, cv2.IMREAD_COLOR)
                # print(s_img_path)
                file_name = os.path.splitext(os.path.basename(img_path))[0]
                if img is None or s_img is None:
                    raise RuntimeError("invalid image: {i}, {j}".format(i=img_path, j=s_img_path))
                # if img.shape != s_img.shape:
                #    raise RuntimeError("invalid image: {i}:{n}, {j}:{m}".format(i=img_path,n=img.shape,j=s_img_path,m=s_img.shape))
                h, w, c = img.shape

                if np.random.rand() > 0.5:
                    img = np.fliplr(img)
                    s_img = np.fliplr(s_img)

                if np.random.rand() > 0.5:
                    angle = 45 * np.random.rand()
                    if np.random.rand() > 0.5:
                        angle *= -1
                    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
                    img = cv2.warpAffine(img, M, (w, h))
                    s_img = cv2.warpAffine(s_img, M, (w, h))

                rand_range_h = h - self.crop_size
                rand_range_w = w - self.crop_size
                x_offset = np.random.randint(rand_range_w)
                y_offset = np.random.randint(rand_range_h)
                img = img[y_offset:y_offset + self.crop_size, x_offset:x_offset + self.crop_size, :]
                s_img = s_img[y_offset:y_offset + self.crop_size, x_offset:x_offset + self.crop_size, :]
                img = (img / 255).astype(np.float32)
                s_img = (s_img / 255).astype(np.float32)
                # img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
                # s_img = cv2.cvtColor(s_img, cv2.COLOR_BGR2Lab)
                xt[i, :, :, :] = np.transpose(img, (2, 0, 1))
                xs[i, :, :, :] = np.transpose(s_img, (2, 0, 1))


        elif mini_batch_size == 1:
            for i, index in enumerate(indices):
                img_path, s_img_path = path_infos[index]
                file_name = os.path.splitext(os.path.basename(img_path))[0]

                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                s_img = cv2.imread(s_img_path, cv2.IMREAD_COLOR)
                if img is None or s_img is None:
                    raise RuntimeError("invalid image: {i}".format(i=img_path))

            h, w, c = img.shape
            xt = np.zeros((mini_batch_size, in_channels, h, w)).astype(np.float32)
            xs = np.zeros((mini_batch_size, in_channels, h, w)).astype(np.float32)
            img = (img / 255).astype(np.float32)
            s_img = (s_img / 255).astype(np.float32)
            xt[0, :, :, :] = np.transpose(img, (2, 0, 1))
            xs[0, :, :, :] = np.transpose(s_img, (2, 0, 1))


        else:
            raise RuntimeError("mini batch size must be 1 when testing")

        return xt, xs, file_name
