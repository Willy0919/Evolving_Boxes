import caffe
import numpy as np
from evb.config import cfg
from evb.bbox_transform import bbox_transform_inv, clip_boxes
from evb.nms_wrapper import nms
import time

DEBUG = False

class NMSLayer(caffe.Layer):
    """
	applying NMS on the rois from PN 
    """

    def setup(self, bottom, top):
        top[0].reshape(1, 5)

        top[1].reshape(1,1)

    def forward(self, bottom, top):

        cfg_key = str(self.phase) # either 'TRAIN' or 'TEST'
        pre_nms_topN  = cfg[cfg_key].PRE_NMS_TOP_N
        post_nms_topN = cfg[cfg_key].POST_NMS_TOP_N
        nms_thresh    = cfg[cfg_key].NMS_THRESH
        min_size      = cfg[cfg_key].MIN_SIZE

        scores = bottom[0].data[:, 1:]

        bbox_deltas = bottom[1].data
        im_info = bottom[2].data[0, :]
        all_rois = bottom[3].data

        proposals = bbox_transform_inv(all_rois[:,1:5], bbox_deltas)

        proposals = clip_boxes(proposals, im_info[:2])

        keep = _filter_boxes(proposals, min_size * im_info[2])
        proposals = proposals[keep, :]
        scores = scores[keep]

        order = scores.ravel().argsort()[::-1]
        if pre_nms_topN > 0:
            order = order[:pre_nms_topN]
        proposals = proposals[order, :]
        scores = scores[order]

        keep = nms(np.hstack((proposals, scores)), nms_thresh)

        if post_nms_topN > 0:
            keep = keep[:post_nms_topN]
        proposals = proposals[keep, :]
        batch_inds = np.zeros((proposals.shape[0], 1), dtype=np.float32)
        blob = np.hstack((batch_inds, proposals.astype(np.float32, copy=False)))
        top[0].reshape(*(blob.shape))
        top[0].data[...] = blob

        top[1].reshape(*(np.array(keep).shape))
        top[1].data[...] = np.array(keep)


    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass

def _filter_boxes(boxes, min_size):
    """Remove all boxes with any side smaller than min_size."""
    ws = boxes[:, 2] - boxes[:, 0] + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1
    keep = np.where((ws >= min_size) & (hs >= min_size))[0]
    return keep
