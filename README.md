# CHB-MIT scalp EEG Dataset

My goal with this project is to become more acquainted/familiar with time-series data, digital signal processing, preprocessing pipelines, and ML/AI solutions that utilize these type of data. The (CHB-MIT scalp EEG dataset)[https://physionet.org/content/chbmit/1.0.0/] contains real time-series data of pediactric subjects. I anticipate this project will involve learning quite a bit of neuroscience vernacular and techniques. 

### Technology Used

python 3.11, mne (EEG/MEG library), jupyter lab

### Data Acquisition

Anyone can access the data, download via wget in terminal. 
```bash
wget -r -N -c -np https://physionet.org/files/chbmit/1.0.0/

# The initial download creates many subfolders, move them up as many levels

```

### Setup

```
# create virtual environment
virtualenv -p=/usr/bin/python3.11 ./venv

# install packages
pip install -r requirements.txt
```

### Running Jupyter lab on a remote server

```
# run in background
jupyter lab --no-browser --ip=100.109.213.114 --port=8080 > logs/jupyter.log 2>&1 &
```

