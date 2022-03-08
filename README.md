<p align="right">Last updated: 7 Mar 2022</p>

## 1. Install [Anaconda](https://www.anaconda.com/) and requirements
* Download TLPN repository
```bash
git clone https://github.com/JUNZHU-SEIS/TLPN.git
cd TLPN
```
* Create TLPN environment (for Chinese users, [the anaconda mirror maintained by tuna](https://mirror.tuna.tsinghua.edu.cn/help/anaconda/) is recommended to download pytorch)
```bash
conda env create -f env.yml
conda activate TLPN
```
## 2. Pre-trained model
Located in directory: **model/tlstead.pt**
## 3. Batch prediction
See details in the notebook: [example_batch_prediction.ipynb](docs/example_batch_prediction.ipynb)
