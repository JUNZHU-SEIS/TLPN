# Purpose:	Phase picking using the transfer-learned PhaseNet (tlstead)
# Date:		7 Mar 2022
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

def stdo(array, format):
	if format=="int":
		t = ' '.join(["%d"%x for x in array])
	else:
		t = ' '.join(["%.3f"%x for x in array])
	return "["+t+"]"

def extract_picks(pred, mph_p=.5, mph_s=.5, mpd=50,
		file = os.path.join('output', 'TLpick.csv'),
		label=[-1,-1],
		fnames=['']):
	for i,m in enumerate(pred):
		fname = fnames[i]
		peaks1, prob1 = detect_peaks(m[0], mph=mph_p, mpd=mpd)
		peaks2, prob2 = detect_peaks(m[1], mph=mph_s, mpd=mpd)
		true1, true2 = str(label[0][i].item()), str(label[1][i].item())
		with open(file, 'a') as f:
			f.write(','.join([fname,
						stdo(peaks1, 'int'),
						stdo(prob1, 'float'),
						stdo(peaks2, 'int'),
						stdo(prob2, 'float'),
						true1, true2])+'\n')
	return


if __name__ == "__main__":
	args = read_args()
	# create the test dataloader
	test_loader = DataLoader(Dataset(args.data_list, args.data_dir, mode='test'),
			batch_size=batch_size, shuffle=False, num_workers=args.num_workers)
	# choose a model to do phase picking
	model = torch.load(args.model, map_location=device)
	# log the results
	if not os.path.exists(args.result_dir):
		os.makedirs(args.result_dir)
	file=os.path.join(args.result_dir, 'segmentedpicks.csv')
	with open(file, 'w') as f:
		f.write(','.join(('fname', 'itp', 'prob_p', 'its', 'prob_s', 'true_p',
						'true_s'))+'\n')
	for i,sample in enumerate(test_loader):
		input = Variable(sample['input'].type(dtype)).to(device) # channel first
		output = model(input)
		input = input.cpu().detach().numpy()
		output = output.cpu().detach().numpy()
		extract_picks(output,
				file=file,
				mph_p=.3, mph_s=.3,
				label=[sample['p'], sample['s']],
				fnames=sample['fname'])
