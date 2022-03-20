# Purpose:	Phase picking using the transfer-learned PhaseNet (tlstead) for continuous data
# Date:		8 Mar 2022
# Author:	Jun ZHU
# Email:	Jun__Zhu@outlook.com


import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from obspy import read
import seisbench.models as sbm
from config import *


def read_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--model",
						default="../model/tlstead.pt",
						type=str,
						help="../model/[retrain, scedc, tlscedc, stead, tlstead]")
	parser.add_argument("--fname",
						default="../dataset/sac/*",
						type=str,
						help="../dataset/sac/*.sac")
	parser.add_argument("--result_dir",
						default="../results",
						type=str,
						help="../results")
	args = parser.parse_args()
	return args

def routine(stream):
	st = stream.copy()
	st.detrend()
#	st.taper(max_percentage=.05)
#	st.filter('bandpass', freqmin=1, freqmax=15, zerophase='True')
	st.normalize()
	return st

def plot(stream, response, folder='./'):
	stream.normalize()
	maxstart, minend = max([x.stats.starttime for x in stream]), min([x.stats.endtime for x in stream])
	stream.trim(maxstart, minend)
	print(stream, response)
	st = np.vstack([tr.data for tr in stream]).transpose()
	re= np.vstack([tr.data for tr in response])[:-1].transpose()
	meta = stream[0].stats
	title = "%s_%s_%s"%(meta['network'], meta['station'], meta['starttime'])
	lw = .8
	t = np.arange(st.shape[0]) / 100
	fig = plt.figure()
	ax = fig.subplots(2, 1, sharex=True, gridspec_kw={'hspace':.15})
#	ax[0].plot(t, st[:,0], lw=lw, label=stream[0].stats.channel)
#	ax[0].plot(t, st[:,1]-2, lw=lw, label=stream[1].stats.channel)
#	ax[0].plot(t, st[:,2]-4, lw=lw, label=stream[2].stats.channel)
	ax[0].plot(t, st, lw=lw, label=[trace.stats.channel for trace in stream])
	ax[1].plot(t, re, lw=lw, label=['P', 'S'])
	ax[0].legend(loc='best')
	ax[1].legend(loc='best')
	ax[0].set_title(title)
	ax[1].set_xlabel('Time (s)')
	ax[0].set_ylabel('Waveform')
	ax[1].set_ylabel('Response')
#	ax[0].set_yticks([])
	plt.savefig(os.path.join(folder, title+".png"), dpi=600)
#	plt.show()
	return

if __name__ == "__main__":
	args = read_args()
	# test waveform in sac format
	sac = read(args.fname)
	sac = routine(sac)
	# instantiate a seisbench model
	model = sbm.PhaseNet(phases='PSN')
	# choose a model to load
	model.load_state_dict(torch.load(args.model).state_dict())
	if not os.path.exists(os.path.join(args.result_dir, 'figures')):
		os.makedirs(os.path.join(args.result_dir, 'figures'))
	# pick P & S
	picks = model.classify(sac, P_threshold=.41, S_threshold=.36)
	# log the pick
	with open(os.path.join(args.result_dir, 'picks.csv'), 'a') as f:
		f.write(os.path.join(os.path.abspath(os.path.join(args.fname,
			os.pardir)), os.path.basename(args.fname))+',')
		f.write(','.join([','.join([str(pick.peak_time), pick.phase]) for pick in picks])+'\n')
	# plot the waveform and response
	plot(sac, model.annotate(sac), folder=os.path.join(args.result_dir,
		'figures'))
