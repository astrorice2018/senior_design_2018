#it will output the image every certain period of bb_id code to predict
#bb_id code will read the image and write the preidction to a npy file containing all the coordinates of targets
#it will draw bb according to the coords given in the file


import numpy as np
import cv2
import time 
import pickle as pkl
import random
inp_dim=160
target=['vest','no_vest','unknown']
def prep_image(img, inp_dim):
	"""
	Prepare image for inputting to the neural network. 
	
	Returns a Variable 
	"""

	orig_im = img
	dim = orig_im.shape[1], orig_im.shape[0]
	img = cv2.resize(orig_im, (inp_dim, inp_dim))
	#img_ = img[:,:,::-1].transpose((2,0,1)).copy()
	#img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
	return orig_im, dim

counter=0
cap=cv2.VideoCapture(0) 
assert cap.isOpened(), "cannot open capture source"
while cap.isOpened():
	
	ret, frame = cap.read()
	if ret:
		orig_im, dim = prep_image(frame, inp_dim)
		if counter==0:
			cv2.imwrite('/home/luke/test.png',orig_im)
		try:
			coord=np.load('coord0.npy')
		except Exception as e:
			print(e)
		for i in range(0,len(coord)):
			cv2.rectangle(orig_im, tuple(coord[i,0:2].astype(int)), tuple(coord[i,2:4].astype(int)),255, 1)
			t_size = cv2.getTextSize(target[int(coord[i,-1])], cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
			cv2.putText(orig_im, target[int(coord[i,-1])], (int(coord[i,0]), int(coord[i,1] + t_size[1] + 4)), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1);

		cv2.imshow('frame',orig_im)
		counter=(counter+1)%30
		key = cv2.waitKey(1)
		if key & 0xFF == ord('q'):
			break
		continue

