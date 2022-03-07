# Purpose:	Visualize the responses of 5 different models (retrain, scedc, tlscedc, stead, tlstead)
# Date:		7 Mar 2022
# Author:	Jun ZHU
# Email:	Jun__Zhu@outlook.com

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
# device
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# data type for torch
dtype = torch.FloatTensor
from torch.autograd import Variable
from torch.utils.data import DataLoader
from config import *
from dataset import Dataset


def read_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--model_dir",
						default="../model",
						type=str,
						help="../model")
	parser.add_argument("--data_list",
						default="../dataset/waveform.csv",
						type=str,
						help="../dataset/waveform.csv")
	parser.add_argument("--data_dir",
						default="../dataset/waveform_pred",
						type=str,
						help="../dataset/waveform_pred")
	args = parser.parse_args()
	return args

def plo(input, target, model_dict, p=-100, s=-100, error=10, title=""):
	fig = plt.figure(figsize=(15, 10))
	ax = fig.subplots(7, 1, sharex=True, gridspec_kw={'hspace':.15})
	npts = input.shape[1]
	t = np.arange(npts) / 100
	p, s = p/100, s/100
	error /= 100
	p_color, s_color = 'b', 'r'
	lw = .8
	legend_loc = 'upper left'
	ax[0].plot(t, input[0], label='Z', lw=lw)
	ax[0].plot(t, input[1], label='N', lw=lw)
	ax[0].plot(t, input[2], label='E', lw=lw)
	ax[0].legend(loc=legend_loc)
	ax[1].plot(t, target[0], label='Target P', c=p_color, lw=lw)
	ax[1].plot(t, target[1], label='Target S', c=s_color, lw=lw)
	ax[1].legend(loc=legend_loc)
	ax[2].plot(t, model_dict['RETRAIN'][0], label='Retrain P', c=p_color, lw=lw)
	ax[2].plot(t, model_dict['RETRAIN'][1], label='Retrain S', c=s_color, lw=lw)
	ax[2].legend(loc=legend_loc)
	ax[3].plot(t, model_dict['SCEDC'][0], label='SCEDC P', c=p_color, lw=lw)
	ax[3].plot(t, model_dict['SCEDC'][1], label='SCEDC S', c=s_color, lw=lw)
	ax[3].legend(loc=legend_loc)
	ax[4].plot(t, model_dict['TL SCEDC'][0], label='TL SCEDC P', c=p_color, lw=lw)
	ax[4].plot(t, model_dict['TL SCEDC'][1], label='TL SCEDC S', c=s_color, lw=lw)
	ax[4].legend(loc=legend_loc)
	ax[5].plot(t, model_dict['STEAD'][0], label='STEAD P', c=p_color, lw=lw)
	ax[5].plot(t, model_dict['STEAD'][1], label='STEAD S', c=s_color, lw=lw)
	ax[5].legend(loc=legend_loc)
	ax[6].plot(t, model_dict['TL STEAD'][0], label='TL STEAD P', c=p_color,	lw=lw)
	ax[6].plot(t, model_dict['TL STEAD'][1], label='TL STEAD S', c=s_color,	lw=lw)
	ax[6].legend(loc=legend_loc)
	ax[0].set_title(title, fontsize=15)
	ax[6].set_xlabel('Time (s)')
	for i in range(6):
		ax[i+1].set_ylim(0, 1)
		ax[i+1].grid(ls='--')
	ax[0].axvline(x=p, c='k')
	ax[0].axvline(x=s, c='k')
	for i in range(6):
		if p>0 and s>0:
			ax[i+1].axvspan(p-error, p+error, color='lavender')
			ax[i+1].axvspan(s-error, s+error, color='salmon', alpha=.3)
	plt.show()
	plt.close()
	return


if __name__ == "__main__":
	args = read_args()
	# instantiate 5 different models
	retrain_model = torch.load(os.path.join(args.model_dir, 'retrain.pt')).to(device)
	scedc_model = torch.load(os.path.join(args.model_dir, 'scedc.pt')).to(device)
	tl_scedc_model = torch.load(os.path.join(args.model_dir, 'tlscedc.pt')).to(device)
	stead_model = torch.load(os.path.join(args.model_dir, 'stead.pt')).to(device)
	tl_stead_model = torch.load(os.path.join(args.model_dir, 'tlstead.pt')).to(device)
	# create test dataloader
	test_loader = DataLoader(Dataset(args.data_list, args.data_dir, mode='test'),
			batch_size=batch_size, shuffle=False, num_workers=20)
	# show example one by one
	for i,sample in enumerate(test_loader):
		input = Variable(sample['input'].type(dtype)).to(device)
		target = sample['target']
		retrain = retrain_model(input)
		scedc = scedc_model(input)
		tl_scedc = tl_scedc_model(input)
		stead = stead_model(input)
		tl_stead = tl_stead_model(input)
		model_dict = {'RETRAIN':retrain, 'SCEDC':scedc, 'TL SCEDC':tl_scedc,
				'STEAD':stead, 'TL STEAD':tl_stead}
		for x in model_dict:
			model_dict[x] = model_dict[x].cpu().detach().numpy()
		response = {}
		for test in range(batch_size):
			for x in model_dict:
				response[x] = model_dict[x][test]
			plo(sample['input'][test], target[test], response,
				p=sample['p'][test], s=sample['s'][test], error=40,
				title=sample['fname'][test].split(".")[0])
