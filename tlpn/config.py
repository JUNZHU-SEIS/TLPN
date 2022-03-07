# Purpose:	Configuration of the Transfer Learning for PhaseNet 
# Date:		7 Mar 2022
# Author:	Jun ZHU
# Email:	Jun__Zhu@outlook.com

import os


batch_size = 20
# root directory
root_dir = os.path.join('..', 'dataset')
# predict
flist = os.path.join(root_dir, 'waveform.csv')
# waveform directory
data_dir = os.path.join(root_dir, 'waveform_pred')
# model
model_dir = os.path.join('..', 'model')
retrain_path = os.path.join(model_dir, 'retrain.pt')
tl_scedc_path = os.path.join(model_dir, 'tlscedc.pt')
tl_stead_path = os.path.join(model_dir, 'tlstead.pt')
scedc_path = os.path.join(model_dir, 'scedc.pt')
stead_path = os.path.join(model_dir, 'stead.pt')
# result directory
result_dir = os.path.join('..', 'results')
