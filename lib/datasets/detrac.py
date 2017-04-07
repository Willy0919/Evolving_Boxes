import os
from datasets.imdb import imdb
import xml.etree.ElementTree as ET
import numpy as np
import scipy.sparse
import scipy.io as sio
import utils.cython_bbox
import cPickle
import subprocess
import uuid
from evb.config import cfg

class detrac(imdb):
    def __init__(self, image_set):
        imdb.__init__(self, 'detrac_' + image_set)
        self._image_set = image_set


        self._data_path = cfg.DATASET_DIR

        self._classes = ('__background__', # always index 0
                         'car')
        self._class_to_ind = dict(zip(self.classes, xrange(self.num_classes)))
        self._image_ext = '.jpg'
        self._image_index = self._load_image_set_index()
        # Default to roidb handler
        self._roidb_handler = None


        self.config = {'cleanup'     : True,
                       'min_size'    : 2}

        assert os.path.exists(self._data_path), \
                'Path does not exist: {}'.format(self._data_path)

    def image_path_at(self, i):
        """
        Return the absolute path to image i in the image sequence.
        """
        return self.image_path_from_index(self._image_index[i])

    def image_index_name(self, i):
        return self._image_index[i]

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """

        image_path = os.path.join(self._data_path, 'Insight-MVT_Annotation_Train', index + self._image_ext)
        if self._image_set == 'test':
            image_path = os.path.join(self._data_path, 'Insight-MVT_Annotation_Test', index + self._image_ext)
        assert os.path.exists(image_path), \
                'Path does not exist: {}'.format(image_path)
        return image_path

    def _load_image_set_index(self):
        """
        Load the indexes listed in this dataset's image set file.
        """

        image_set_file = os.path.join(self._data_path, self._image_set + '.txt')
        assert os.path.exists(image_set_file), \
                'Path does not exist: {}'.format(image_set_file)
        with open(image_set_file) as f:
            image_index = [x.strip() for x in f.readlines()]
        return image_index

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                roidb = cPickle.load(fid)
            print '{} gt roidb loaded from {}'.format(self.name, cache_file)
            return roidb

        gt_roidb = [self._load_detrac_annotation(index)
                    for index in self.image_index]
        with open(cache_file, 'wb') as fid:
            cPickle.dump(gt_roidb, fid, cPickle.HIGHEST_PROTOCOL)
        print 'wrote gt roidb to {}'.format(cache_file)

        return gt_roidb


    def _load_detrac_annotation(self, index):

        filename = os.path.join(self._data_path, "anno", index)

        f = open(filename,'r')
        img = f.readline().strip('\n')
        num_objs = int(f.readline().strip('\n'))

        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)

        for i in range(num_objs):
            line = f.readline().strip('\n')
            axis = line.split(' ')
            x1 = float(axis[0])
            y1 = float(axis[1])
            x2 = x1 + float(axis[2]) - 1
            y2 = y1 + float(axis[3]) - 1
            cls = self._class_to_ind['car']
            boxes[i, :] = [x1, y1, x2, y2]
            gt_classes[i] = cls
            overlaps[i, cls] = 1.0
        overlaps = scipy.sparse.csr_matrix(overlaps)
        if f:
            f.close()
        return {'boxes' : boxes,
                'gt_classes': gt_classes,
                'gt_overlaps' : overlaps,
                'flipped' : False}

