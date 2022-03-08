import torch
import seisbench.models as sbm
from obspy import read
import glob
st = read('/home/lilab/jzhu/data/tlPhasenet/xichang/20151006.279.154848.840/YN.YOS*')
print(st)
model = sbm.PhaseNet(phases='PSN')
x = torch.load('tlscedc.pt')
model.load_state_dict(x.state_dict())
re = model.annotate(st)
picks = model.classify(st, P_threshold=.1, S_threshold=.1)
print(re)
for x in picks:
	print(x)
re.plot()
st.plot()
