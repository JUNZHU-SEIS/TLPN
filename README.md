<p align="right">Last updated: 7 Mar 2022</p>

## 1. Install [Anaconda](https://www.anaconda.com/) and requirements
* Download CEB repository
```bash
git clone https://github.com/JUNZHU-SEIS/TLPN.git
cd TLPN
```
* Create TLPN environment
```bash
conda create -n TLPN
conda activate TLPN
conda install pip
pip install seisbench
```
## 2. Pre-trained model
Located in directory: **model/socal_classifier.py**
## 3. Batch prediction
See details in the notebook: [example_batch_prediction.ipynb](docs/example_batch_prediction.ipynb)
