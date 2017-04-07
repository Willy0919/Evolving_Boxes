import caffe
import yaml
import numpy as np
import numpy.random as npr
from evb.config import cfg
from evb.bbox_transform import bbox_transform,bbox_transform_inv,clip_boxes
from utils.cython_bbox import bbox_overlaps


class ProposalNetLayer(caffe.Layer):
    '''
	propose candidates directly from proposal_layer where generates anchor boxes 
    '''

    def setup(self, bottom, top):
        layer_params = yaml.load(self.param_str_)
        self._num_classes = layer_params['num_classes']
        self._allowed_border = layer_params.get('allowed_border', 0)

        # labels
        top[0].reshape(1, 1)
        # bbox_targets
        top[1].reshape(1, 4)
        # bbox_inside_weights
        top[2].reshape(1, 4)
        # bbox_outside_weights
        top[3].reshape(1, 4)

    def forward(self, bottom, top):

        all_rois = bottom[0].data
        gt_boxes = bottom[1].data

        labels = np.empty((len(all_rois),), dtype=np.float32)
        labels.fill(-1)


        overlaps = bbox_overlaps(
            np.ascontiguousarray(all_rois[:,1:5], dtype=np.float),
            np.ascontiguousarray(gt_boxes, dtype=np.float))
        argmax_overlaps = overlaps.argmax(axis=1)
        max_overlaps = overlaps[np.arange(len(all_rois)), argmax_overlaps]
        gt_argmax_overlaps = overlaps.argmax(axis=0)
        gt_max_overlaps = overlaps[gt_argmax_overlaps,
                                   np.arange(overlaps.shape[1])]
        gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]
        if not cfg.TRAIN.PN_CLOBBER_POSITIVES:
            # assign bg labels first so that positive labels can clobber them
            labels[max_overlaps < cfg.TRAIN.PN_NEGATIVE_OVERLAP] = 0

        # fg label: for each gt, anchor with highest overlap
        labels[gt_argmax_overlaps] = 1

        # fg label: above threshold IOU
        labels[max_overlaps >= cfg.TRAIN.PN_POSITIVE_OVERLAP] = 1

        if cfg.TRAIN.PN_CLOBBER_POSITIVES:
            # assign bg labels last so that negative labels can clobber positives
            labels[max_overlaps < cfg.TRAIN.PN_NEGATIVE_OVERLAP] = 0

        # subsample positive labels if we have too many
        num_fg = int(cfg.TRAIN.PN_FG_FRACTION * cfg.TRAIN.PN_BATCHSIZE)
        fg_inds = np.where(labels == 1)[0]
        if len(fg_inds) > num_fg:
            disable_inds = npr.choice(
                fg_inds, size=(len(fg_inds) - num_fg), replace=False)
            labels[disable_inds] = -1

        num_bg = cfg.TRAIN.PN_BATCHSIZE - np.sum(labels == 1)
        bg_inds = np.where(labels == 0)[0]
        if len(bg_inds) > num_bg:
            disable_inds = npr.choice(
                bg_inds, size=(len(bg_inds) - num_bg), replace=False)
            labels[disable_inds] = -1

        bbox_targets = np.zeros((len(all_rois), 4), dtype=np.float32)
        bbox_targets[:,:] = _compute_targets(all_rois[:,1:5], gt_boxes[argmax_overlaps, :])

        bbox_inside_weights = np.zeros((len(all_rois), 4), dtype=np.float32)
        bbox_inside_weights[labels == 1, :] = np.array(cfg.TRAIN.PN_BBOX_INSIDE_WEIGHTS)

        bbox_outside_weights = np.zeros((len(all_rois), 4), dtype=np.float32)

        if cfg.TRAIN.PN_POSITIVE_WEIGHT < 0:
            # uniform weighting of examples (given non-uniform sampling)
            num_examples = np.sum(labels >= 0)
            positive_weights = np.ones((1, 4)) * 1.0 / num_examples
            negative_weights = np.ones((1, 4)) * 1.0 / num_examples
        else:
            assert ((cfg.TRAIN.PN_POSITIVE_WEIGHT > 0) &
                    (cfg.TRAIN.PN_POSITIVE_WEIGHT < 1))
            positive_weights = (cfg.TRAIN.PN_POSITIVE_WEIGHT /
                                np.sum(labels == 1))
            negative_weights = ((1.0 - cfg.TRAIN.PN_POSITIVE_WEIGHT) /
                                np.sum(labels == 0))
        bbox_outside_weights[labels == 1, :] = positive_weights
        bbox_outside_weights[labels == 0, :] = negative_weights
        # labels
        top[0].reshape(*labels.shape)

        top[0].data[...] = labels

        # bbox_targets
        top[1].reshape(*bbox_targets.shape)
        top[1].data[...] = bbox_targets
        # bbox_inside_weights
        top[2].reshape(*bbox_inside_weights.shape)
        top[2].data[...] = bbox_inside_weights

        # bbox_outside_weights
        top[3].reshape(*bbox_outside_weights.shape)
        top[3].data[...] = bbox_outside_weights


    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass
def _compute_targets(ex_rois, gt_rois):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    assert ex_rois.shape[1] == 4
    assert gt_rois.shape[1] == 5

    return bbox_transform(ex_rois, gt_rois[:, :4]).astype(np.float32, copy=False)

def _get_bbox_regression_labels(bbox_target_data, num_classes):

    clss = bbox_target_data[:, 0]
    bbox_targets = np.zeros((clss.size, 4 * num_classes), dtype=np.float32)
    bbox_inside_weights = np.zeros(bbox_targets.shape, dtype=np.float32)
    inds = np.where(clss > 0)[0]
    for ind in inds:
        cls = clss[ind]
        start = 4 * cls
        end = start + 4
        bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
        bbox_inside_weights[ind, start:end] = cfg.TRAIN.BBOX_INSIDE_WEIGHTS
    return bbox_targets, bbox_inside_weights
