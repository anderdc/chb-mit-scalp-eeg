# CHB-MIT Scalp EEG Dataset

My goal is to become more acquainted/familiar with time-series data, digital signal processing, preprocessing pipelines, and ML/AI solutions that utilize these type of data. The [CHB-MIT scalp EEG dataset](https://physionet.org/content/chbmit/1.0.0/) contains real time-series data of pediatric subjects. The focus of this project is to create a classifier that can distinguish seizure and non-seizure states.

### Technology Used

python 3.11, mne (EEG/MEG library), jupyter lab

### Data Acquisition

Anyone can access the data, download via wget in terminal. 
```bash
wget -r -N -c -np https://physionet.org/files/chbmit/1.0.0/

# or with s2 (this is way quicker)
aws s3 sync --no-sign-request s3://physionet-open/chbmit/1.0.0/ DESTINATION

# Note: The initial download creates many subfolders, move them up as many levels as you'd like
```

### Setup

```
# create virtual environment
virtualenv -p=/usr/bin/python3.11 ./venv

# install packages
pip install -r requirements.txt
```

#### Running Jupyter lab on a remote server

```bash
source venv/bin/activate

# run in background
jupyter lab --no-browser --ip=<your_server_ip> --port=8080 > logs/jupyter.log 2>&1 &
```

## Inspiration

[Professor Millan](https://www.ece.utexas.edu/people/faculty/jose-del-r-millan) was the instructor for one of my last undergraduate classes at UT. His class, neural engineering, covered many advanced topics: cochlear implants, robotic prosthesis, TMS/TACS, BCI's, Deep brain stimulation, etc.. Every lecture was eye opening, inspiring, and interesting. My favorite part involved a homework where we created an ML model (SVM) to predict whether a rat's sciatic nerve was experiencing a pinch, flexion, or at rest. Multiclass classification. In order to whet that interest, I decided (as an ML/AI hobbyist) to take a real EEG dataset and try my hand at building a model, so binary classification of seizure states seemed most similar and doable.

### Path

So far, this is the general path I've followed throughout the project

- exploratory data analyis

basic understanding of how much data, which patients experience the most individual seizures, most/least seizure time, total interictal time vs total ictal time, basic data visualization

- processing the summary files

This dataset had summary files per patient data file that detailed the onsets and end times of each seizure. These files had to be parsed via python script and prepared for easy access later for data preprocessing and labeling.

- selecting one patient's file

    I found the patient with the highest seizure time and decided to start an entire 'workflow' based around that patient. To make it even easier on me, I selected one file, at random, from that patient and started the workflow.

  - Loading and preprocessing data

    Preprocessing data looked like: dropping 'dead' channels that had no signal, loading up seizure onsets/ends for the file, annotating the data with the seizure onsets/duration, segmenting the data with a specific WINDOWSIZE and OVERLAP, and creating a label (target) vector that was |segments| x 1 large for future training/visualization.

  - Feature Extraction
 
    I started with extracting time-domain features such as: mean, VAR, MAV, Skewness.
 
    I then moved on to extracting frequency domain features: Absolute Band Power, Relative Band Power, Spectral Entropy, Peak Frequency

- Generalizing the data pipeline for all patient files 






