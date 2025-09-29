import os
import cv2
import numpy as np

class MiniBatchLoader(object):

    def __init__(self, train_path, test_path, image_dir_path, crop_size):

        # load data paths
        self.training_path_infos = self.read_paths(train_path, image_dir_path)
        self.testing_path_infos = self.read_test_paths(test_path, image_dir_path)
        # load defalt size
        self.crop_size = crop_size

    # test ok
    @staticmethod
    def path_label_generator(txt_path, src_path):
        for line in open(txt_path):
            name = line.strip()
            image_name = 'train/' + name + '.jpg'
            full_path = os.path.join(src_path, image_name)
            if os.path.isfile(full_path):
                yield full_path
            else:
                print(full_path)

    @staticmethod
    def path_label_generator_test(txt_path, src_path):
        for line in open(txt_path):
            name = line.strip()
            image_name = 'test/' + name + '.jpg'
            full_path = os.path.join(src_path, image_name)
            if os.path.isfile(full_path):
                yield full_path
            else:
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

    def load_data(self, path_infos, indices, augment=False):
        mini_batch_size = len(indices)
        in_channels = 3
        xs = np.zeros((mini_batch_size, in_channels, self.crop_size, self.crop_size)).astype(np.float32)
        if augment:
            file_names = np.empty((mini_batch_size), dtype=object)
            file_names.fill('')

            for i, index in enumerate(indices):
                img_path = path_infos[index]
                img = cv2.imread(img_path, cv2.IMREAD_COLOR)
                img = cv2.resize(img, (self.crop_size, self.crop_size))
                file_name = os.path.splitext(os.path.basename(img_path))[0]
                if img is None:
                    raise RuntimeError("invalid image: {i}".format(i=img_path))

                # 数据增强
                h, w, c = img.shape
                if np.random.rand() > 0.5:
                    img = np.fliplr(img)

                if np.random.rand() > 0.5:
                    angle = 45 * np.random.rand()
                    if np.random.rand() > 0.5:
                        angle *= -1
                    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
                    img = cv2.warpAffine(img, M, (w, h))

                img = (img / 255).astype(np.float32)

                xs[i, :, :, :] = np.transpose(img, (2, 0, 1))
                file_names[i] = file_name
        elif mini_batch_size == 1:
            file_names = np.empty((mini_batch_size), dtype=object)
            file_names.fill('')

            img_path = path_infos[indices[0]]
            file_name = os.path.splitext(os.path.basename(img_path))[0]
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError("invalid image: {i}".format(i=img_path))

            h, w, c = img.shape
            xs = np.zeros((mini_batch_size, in_channels, h, w)).astype(np.float32)
            img = (img / 255).astype(np.float32)

            xs[0, :, :, :] = np.transpose(img, (2, 0, 1))
            file_names[0] = file_name
        else:
            raise RuntimeError("mini batch size must be 1 when testing")

        return xs, file_names
