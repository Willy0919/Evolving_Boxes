import caffe
import numpy as np


class ConcatTestLayer(caffe.Layer):
    def setup(self, bottom, top):
	pass

    def reshape(self, bottom, top):
	#bottom[0] fc1 N*256*1*1
	#bottom[1] fc2 K*4096*1*1
	#bottom[2] keep1 N N->M
	#top[0] fc K*(512+256)*1*1

	K = bottom[1].shape[0]
	self.Dim1 = bottom[0].shape[1]#256
	self.Dim2 = bottom[1].shape[1]#512
	top[0].reshape(K,self.Dim1+self.Dim2)
	self.keep1 = bottom[2].data
        # only use the rois from NMS
	self.keep2 = np.array([])

    def forward(self, bottom, top):
	fc1 = bottom[0].data
	fc2 = bottom[1].data

	top[0].data[:,0:self.Dim1] = 0.0
	self.selectDim = np.where(self.keep2<self.keep1.shape[0])
	top[0].data[self.selectDim,0:self.Dim1] = (fc1[self.keep1.tolist()])[self.keep2[self.selectDim].tolist()]
	top[0].data[:,self.Dim1:self.Dim1+self.Dim2] = fc2

    def backward(self, top, propagate_down, bottom):
	pass
