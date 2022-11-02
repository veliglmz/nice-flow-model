# Python Implementation of NICE: Nonlinear Independent Components Estimation
This code is an unofficial Python implementation of [NICE: Nonlinear Independent Components Estimation](https://arxiv.org/abs/1410.8516). I trained a simple model with a generated spiral train data. I tested it using Gaussian distribution.

## Installation
* Clone the repository.
* Create a virtual environment.
* Install pip packages.
If you have trouble with torch, please install it according to [PyTorch](https://pytorch.org/).

```bash
git clone https://github.com/veliglmz/nice-flow-model.git
cd nice-flow-model
python3.8 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage
```bash
python main.py
```