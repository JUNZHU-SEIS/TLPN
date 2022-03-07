# Purpose:	Generate dataset for PhaseNet 
# Date:		7 Mar 2021
# Author:	Jun ZHU
# Email:	Jun__Zhu@outlook.com


import os
import pandas as pd
import numpy as np
from scipy import signal
import torch


def routine(waveform):
	"""
		routine preprocessing for waveform data
	"""
	detrended = signal.detrend(waveform, axis=1)
	normalized = detrended / np.max(np.std(detrended, axis=1))
	return normalized

def gaus(width=30):
	label = np.exp(-((np.arange(-width//2, width//2+1))**2)/(2*(width/5)**2))
	return label

def cut(sample, p, s, mode='train', length=3001, halfwin=15, fname=''):
	maxn = sample['input'].shape[1]
	if mode=="train":
		select_range = [s-p-length+halfwin, -halfwin] # make sure both P&S picks involved
		shift = np.random.randint(*select_range)
	elif mode=="test":
		shift = int((p+s-length)/2) - p
	sample = {x:sample[x][:,shift+p:shift+p+length] for x in sample}
	sample['p'] = -shift
	sample['s'] = s-p-shift
	sample['fname'] = fname
	return sample

def mklabel(p, s, shape, halfwin=50):
	target = np.zeros(shape)
	maxn = shape[1]
	# the order of response is "P", "S", "N"
	target[0, max(0,p-halfwin):min(p+(halfwin+1),maxn)] = gaus(2*halfwin)[max(halfwin-p,0):2*halfwin+1-max(p+(halfwin+1)-maxn,0)]
	target[1, max(0,s-halfwin):min(s+(halfwin+1),maxn)] = gaus(2*halfwin)[max(halfwin-s,0):2*halfwin+1-max(s+(halfwin+1)-maxn,0)]
	target[2,...] = 1 - target[0,...] - target[1,...]
	return torch.tensor(np.array(target))

class Dataset(torch.utils.data.Dataset):
	"""dataset for TLPN-bench"""
	def __init__(self, dfile, data_folder, transform=None, mode='train'):
		self.dfile = pd.read_csv(dfile)
		self.data_folder = data_folder
		self.transform = transform
		self.mode = mode
	def __len__(self):
		return len(self.dfile)
	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()
		p, s = (self.dfile.iloc[idx]['itp'],
			self.dfile.iloc[idx]['its'])
		npz = np.load(os.path.join(self.data_folder,
					self.dfile.iloc[idx]['fname']))
		input = torch.tensor(npz['data'].transpose()) # [channel, feature]
		# order by ZNE
		input = torch.flip(input, [0])
		# detrend&normalization
		input = routine(input)
		target = mklabel(p, s, input.shape)
		sample = {'input': input, 'target': target}
		cutsample = cut(sample, p, s, fname=self.dfile.iloc[idx]['fname'],
				mode=self.mode)
		if self.transform:
			sample = self.transform(sample)
		return cutsample
