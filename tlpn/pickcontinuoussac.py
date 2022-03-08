# Purpose:	Phase picking using the transfer-learned PhaseNet (tlstead) for continuous data
# Date:		8 Mar 2022
# Author:	Jun ZHU
# Email:	Jun__Zhu@outlook.com


import os
import argparse
from detect_peaks import detect_peaks
import torch
# device
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# data type for torch
dtype = torch.FloatTensor
from torch.utils.data import DataLoader
from torch.autograd import Variable
from obspy import read
import seisbench.models as sbm
from config import *
from dataset import Dataset


def read_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--model",
						default="../model/retrain.pt",
						type=str,
						help="../model/[retrain, scedc, tlscedc, stead, tlstead]")
	parser.add_argument("--data_list",
						default="../dataset/waveform.csv",
						type=str,
						help="../dataset/waveform.csv")
	parser.add_argument("--data_dir",
						default="../dataset/waveform_pred",
						type=str,
						help="../dataset/waveform_pred")
	parser.add_argument("--result_dir",
						default="../results",
						type=str,
						help="../results")
	parser.add_argument("--num_workers",
						default=20,
						type=int,
						help=8)
	args = parser.parse_args()
	return args

if __name__ == "__main__":
	args = read_args()
#	# create the test dataloader
	test_loader = DataLoader(Dataset(args.data_list, args.data_dir, mode='test'),
			batch_size=batch_size, shuffle=False, num_workers=args.num_workers)
	# test waveform in sac format
	sac = read('/home/as/Data/tlPhasenet/xichang/20151001.274.174337.050/*')
	# instantiate a seisbench model
	model = sbm.PhaseNet(phases='PSN')
	# choose a model to load
	model.load_state_dict(torch.load(args.model).state_dict())
	response = model.annotate(sac)
	picks = model.classify(sac, P_threshold=.41, S_threshold=.36)
	sac.plot()
	response.plot()
#	# log the results
#	if not os.path.exists(args.result_dir):
#		os.makedirs(args.result_dir)
#	file=os.path.join(args.result_dir, 'picks.csv')
#	with open(file, 'w') as f:
#		f.write(','.join(('fname', 'itp', 'prob_p', 'its', 'prob_s', 'true_p',
#						'true_s'))+'\n')
#	for i,sample in enumerate(test_loader):
#		input = Variable(sample['input'].type(dtype)).to(device) # channel first
#		output = model(input)
#		input = input.cpu().detach().numpy()
#		output = output.cpu().detach().numpy()
#		extract_picks(output,
#				file=file,
#				mph_p=.3, mph_s=.3,
#				label=[sample['p'], sample['s']],
#				fnames=sample['fname'])
