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

```bash
# install dependencies and run any project command
uv sync

# or just run directly (uv sync is implicit)
uv run jupyter lab
```

#### Running Jupyter lab on a remote server

```bash
uv run jupyter lab --no-browser --ip=<your_server_ip> --port=8080 > logs/jupyter.log 2>&1 &
```

## Inspiration

[Professor Millan](https://www.ece.utexas.edu/people/faculty/jose-del-r-millan) was the instructor for one of my last undergraduate classes at UT. His class, neural engineering, covered many advanced topics: cochlear implants, robotic prosthesis, TMS/TACS, BCI's, Deep brain stimulation, etc.. Every lecture was eye opening, inspiring, and interesting. My favorite part involved a homework where we created an ML model (SVM) to predict whether a rat's sciatic nerve was experiencing a pinch, flexion, or at rest. Multiclass classification. In order to whet that interest, I decided (as an ML/AI hobbyist) to take a real EEG dataset and try my hand at building a model, so binary classification of seizure states seemed most similar and doable.

### Progress

1. **Exploratory data analysis** — understanding data volume, which patients experience the most seizures, most/least seizure time, total interictal vs ictal time, basic visualization
2. **Parsing summary files** — each patient's data files came with summary files detailing seizure onsets and end times; wrote a parser to extract these for downstream labeling
3. **Single-patient workflow** — selected the patient with the highest seizure time and one of their files to develop an end-to-end workflow:
   - **Preprocessing** — dropping dead channels, annotating seizure onsets/duration, segmenting data with configurable window size and overlap, creating a label vector for training
   - **Feature extraction** — time-domain (mean, variance, MAV, skewness) and frequency-domain (absolute/relative band power, spectral entropy, peak frequency)
4. **Generalizing the pipeline** — extended the single-file workflow to process all patient files
5. **Model experiments** — SVM and neural network classifiers (see `models/`)
   - SVM failed to predict ictal states due to extreme class imbalance (~1,144:1 non-ictal to ictal)
   - Deep NN with class-weighted cross-entropy loss had the same problem — 0% recall on ictal class
   - Added resampling (random oversampling of ictal + undersampling of non-ictal) to the pipeline, which got the model to finally produce ictal predictions
6. **Current challenge: feature quality** — the model predicts ictal states now but with very low recall. Visual inspection of feature distributions shows that separation between classes is highly channel-dependent (e.g., beta/theta band power on C3-P3 shows clear signal, while other channels show none). Next steps:
   - Feature selection — drop low-signal features and channels that add noise
   - Focus on the most discriminative features (band power features, especially beta and theta)
   - Potentially reduce from 345 features to a smaller, higher-quality set
