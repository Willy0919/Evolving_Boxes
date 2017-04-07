__author__ = 'willy'
import caffe
import numpy as np


class ConcatLayer(caffe.Layer):
    def setup(self, bottom, top):
	pass

    def reshape(self, bottom, top):
	#bottom[0] fc1 N*256*1*1
	#bottom[1] fc2 K*4096*1*1
	#bottom[2] keep1 N N->M from NMS
	#bottom[3] keep2 M M->K from FTN
	#top[0] fc K*(512+256)*1*1

	K = bottom[1].shape[0]
	self.Dim1 = bottom[0].shape[1]#256
	self.Dim2 = bottom[1].shape[1]#512
	top[0].reshape(K,self.Dim1+self.Dim2)
	self.keep1 = bottom[2].data
	self.keep2 = bottom[3].data

    def forward(self, bottom, top):
	fc1 = bottom[0].data
	fc2 = bottom[1].data
        print 'keep1++++++++++',self.keep1
	print 'keep2++++++++++',self.keep2
        # initialize
	top[0].data[:,0:self.Dim1] = 0.0
        # to ensure the indices from FTN are contained in the indices from PN
	self.selectDim = np.where(self.keep2<self.keep1.shape[0])
        # the new features for rois are [fc1, fc2]
	top[0].data[self.selectDim,0:self.Dim1] = (fc1[self.keep1.tolist()])[self.keep2[self.selectDim].tolist()]
	top[0].data[:,self.Dim1:self.Dim1+self.Dim2] = fc2

    def backward(self, top, propagate_down, bottom):
	diff_fc1 = bottom[0].diff
	diff_fc2 = bottom[1].diff
	diff_fc = top[0].diff
	
	diff_fc1[...] = 0.0
	(diff_fc1[self.keep1.tolist()])[self.keep2[self.selectDim].tolist()] = diff_fc[self.selectDim,0:self.Dim1]
	diff_fc2[...] = diff_fc[:,self.Dim1:self.Dim1+self.Dim2]
