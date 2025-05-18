# CHB-MIT Scalp EEG Dataset

My goal is to become more acquainted/familiar with time-series data, digital signal processing, preprocessing pipelines, and ML/AI solutions that utilize these type of data. The [CHB-MIT scalp EEG dataset](https://physionet.org/content/chbmit/1.0.0/) contains real time-series data of pediatric subjects and is the focus of this project.

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

### Running Jupyter lab on a remote server

```bash
# run in background
jupyter lab --no-browser --ip=100.109.213.114 --port=8080 > logs/jupyter.log 2>&1 &

# Note: the ip used was my tailscale ip so I could access through other machines on my tailscale network
```


## Inspiration & Goal

[Professor Millan](https://www.ece.utexas.edu/people/faculty/jose-del-r-millan) was the instructor for one of my last undergraduate classes at UT. His class, neural engineering, covered many advanced topics: cochlear implants, robotic prosthesis, TMS/TACS, BCI's, Deep brain stimulation, etc.. Every lecture was eye opening, inspiring, and interesting. My favorite part involved a homework where we created an ML model (SVM) to predict whether a rat's sciatic nerve was experiencing a pinch, flexion, or at rest. Multiclass classification. In order to whet that interest, I decided (as an ML/AI hobbyist) to take a real EEG dataset and try my hand at building a model, so binary classification of seizure states seem most interesting and doable.

The goal is to build an ML model (most likely SVM) to predict whether a segment of data is ictal (seizure) or interictal (non-seizure). Then, to build an AI model (RNN? Transformer?) that will do the same thing.

*Note: Stretch goal is once you have a functioning model, productionize the model and create a full end to end ML/AI pipeline?*

*Note: Perhaps when everything is all said and done, do another model that does multiclass classification, and the classes are: ictal (seizure), pre-ictal (just before a seizure), interictal (nonseizure)*

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

  - Feature Selection
 


  - Data Filtering, and Channel Selection








