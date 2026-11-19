# Heartbeat-Based Deepfake Detection

An experimental project investigating whether **remote
photoplethysmography (rPPG)** features extracted from facial videos can
help distinguish authentic videos from manipulated ones.

The first completed experiment uses whole-face RGB intensity traces, the
Plane-Orthogonal-to-Skin (POS) algorithm, signal-processing features,
and a Random Forest classifier on FaceForensics++ C23.

> **Status:** The whole-face feature-extraction and Random Forest
> baseline were run. The reported validation results show that the
> current features do not reliably distinguish REAL from FAKE.
> Region-specific extraction (forehead and cheeks) is a proposed
> follow-up, not part of the results reported here.

------------------------------------------------------------------------

## 1. Research question

Can subtle physiological information in a person's face provide useful
evidence for detecting manipulated videos?

Blood-volume changes during the cardiac cycle cause very small changes
in the light reflected from facial skin. A normal camera records these
as subtle variations in pixel colour and intensity. Remote
photoplethysmography attempts to recover a pulse-related signal from
these variations without physical contact.

The hypothesis is that face manipulation may alter, weaken, or make
inconsistent the spatial and temporal colour patterns associated with a
real person's pulse. If so, signal characteristics from authentic and
manipulated videos may differ.

This is an investigation, not an assumption that every real video
contains a clean pulse signal or that every deepfake lacks one.
Lighting, motion, compression, camera settings, skin visibility, and
manipulation method all affect the extracted signal.

## 2. Dataset and labels

**Dataset:** FaceForensics++ (C23).

The experiment used C23-compressed videos and metadata CSV files. The
metadata identifies the video path and its authenticity/manipulation
category.

The first source-pair grouping experiment considered:

-   `original`
-   `Deepfakes`
-   `Face2Face`
-   `FaceShifter`
-   `FaceSwap`
-   `NeuralTextures`

`DeepFakeDetection` metadata was also present, but its filenames did not
follow the same simple pair-ID pattern. It was not included in the first
pair-based grouping experiment because its source relationships had not
been mapped using the same method.

  Class     Numeric label
  ------- ---------------
  REAL                  0
  FAKE                  1

The manifest is an index of video paths and labels; it is not the video
data itself.

### Why source grouping matters

A manipulated video can be associated with one or more source
identities/videos. If related originals and manipulations are randomly
distributed across training and test sets, a model may learn
source-specific characteristics instead of general manipulation cues.

For pair-named files such as `000_003.mp4`, the two IDs were connected.
Connected IDs and their associated originals/manipulations were assigned
to the same source group. Groups were then split so that a group could
not appear in more than one split.

This reduces one important form of leakage, but does not guarantee that
every possible dataset bias or source relationship has been eliminated.

## 3. End-to-end methodology

``` text
FaceForensics++ C23 videos
          |
          v
Video manifest and REAL/FAKE labels
          |
          v
Source-group train / validation / test split
          |
          v
Face detection in sampled frames
          |
          v
Whole-face RGB intensity extraction
          |
          v
Three temporal RGB traces: R, G, B
          |
          v
POS pulse-related waveform estimation
          |
          v
Signal filtering and frequency analysis
          |
          v
Numerical feature extraction
          |
          v
Feature tables (CSV)
          |
          v
Random Forest classifier
          |
          v
Validation and held-out test evaluation
```

### 3.1 Face detection and frame sampling

The video is read frame by frame, with frames sampled at a fixed
interval to reduce processing cost. A face detector locates the face in
each sampled frame.

In the original whole-face implementation, the detected face bounding
box is used as the region of interest (ROI). It is not a precise skin
segmentation mask: depending on the detection, it can include eyes,
eyebrows, hair, mouth, and other non-skin pixels.

### 3.2 RGB signal extraction

For each sampled frame, average pixel intensity is calculated
independently for the red, green, and blue channels within the detected
face crop. This produces three time series:

-   Red intensity over time
-   Green intensity over time
-   Blue intensity over time

These are pixel-intensity measurements, not heartbeat measurements. The
idea is that tiny pulse-related colour changes may be embedded in the
traces alongside larger effects from lighting, movement, expression, and
compression.

### 3.3 POS pulse-related signal estimation

The three RGB traces are passed to the **Plane-Orthogonal-to-Skin
(POS)** method.

At a high level, POS: 1. Processes temporally normalized RGB values in
short overlapping windows. 2. Projects colour variations into a plane
designed to reduce skin-tone/intensity variation. 3. Combines projected
signals using a weighting factor based on their relative variation. 4.
Overlap-adds the window outputs to form a continuous pulse-related
waveform.

The output oscillates around zero. Positive and negative values
represent changes in the estimated signal relative to its local
baseline; they are not direct blood-pressure or pixel-intensity values.

POS attempts to isolate pulse-related colour variation. It does not
prove that the resulting waveform is a clean physiological pulse.

### 3.4 Filtering and frequency analysis

The POS waveform is filtered to focus on a frequency band associated
with plausible pulse rates. Welch's power spectral density estimate is
used to examine how signal power is distributed across frequencies. The
frequency with the largest power in the selected band is used as the
dominant frequency.

Estimated BPM is calculated as:

\[ `\text{Estimated BPM}`{=tex} = 60
`\times `{=tex}`\text{Dominant Frequency (Hz)}`{=tex} \]

This is an estimate from the extracted signal, not a clinically
validated heart-rate measurement.

### 3.5 Feature engineering

Each processed video is converted into one row of numerical features.

  -----------------------------------------------------------------------
  Feature                             Meaning
  ----------------------------------- -----------------------------------
  `mean`                              Average processed-signal value

  `std`                               Standard deviation; signal spread

  `variance`                          Squared measure of signal spread

  `skewness`                          Asymmetry of the signal-value
                                      distribution

  `kurtosis`                          Tail heaviness/peakedness of the
                                      distribution

  `rms`                               Root mean square signal magnitude

  `dominant_frequency`                Frequency with the largest
                                      estimated spectral power in the
                                      selected band

  `estimated_bpm`                     Dominant frequency converted to
                                      beats per minute

  `total_power`                       Sum of spectral power in the
                                      selected band

  `peak_power`                        Largest spectral power value in the
                                      selected band

  `spectral_entropy`                  How spread out or concentrated
                                      spectral power is

  `spectral_flatness`                 How noise-like or tone-like the
                                      spectrum is

  `peak_count`                        Number of detected peaks in the
                                      waveform

  `peak_rate`                         Peak count divided by signal
                                      duration

  `max_autocorrelation`               Maximum normalized autocorrelation
                                      over the selected non-zero lag
                                      range
  -----------------------------------------------------------------------

These features summarize the signal and do not preserve every detail of
the original waveform. Some are correlated: estimated BPM is directly
derived from dominant frequency, so those two columns are not
independent evidence.

## 4. Feature tables and preparation

Feature extraction is performed separately for training, validation, and
test splits. The outputs include:

-   `train_features.csv`
-   `val_features.csv`
-   `test_features.csv`
-   `video_manifest.csv`

Each feature-table row represents a successfully processed video, with
numerical features and its label. File paths and source-group
identifiers are useful for tracking but must not be used as model
inputs.

If the notebook kernel restarts, the saved CSVs should be reloaded
before training. In-memory variables can be overwritten by pilot
experiments, so verify saved row counts and label distributions before
model fitting.

### Reported feature-table counts

  Split                     REAL    FAKE   Total
  ----------------------- ------ ------- -------
  Training                   683   3,419   4,102
  Validation                 148     743     891
  Training + validation      831   4,162   4,993

The available run output did not establish the final test-set row count,
so it is not stated here.

The classes are imbalanced, with substantially more FAKE examples than
REAL examples. Ordinary accuracy is therefore insufficient as a
standalone metric.

### Missing values

A median imputer was used: - `fit_transform` on training features -
`transform` on validation features - `transform` on test features

Fitting the imputer only on training data prevents validation/test data
from influencing preprocessing statistics.

## 5. Random Forest baseline

A Random Forest classifier was used as the first machine-learning
baseline. It is an ensemble of decision trees. Each tree learns decision
rules from the training data, and the ensemble combines their
predictions.

The model used class weighting to account for unequal class counts. Only
numerical signal features were passed as model inputs; labels and
metadata were excluded.

The model's `.fit()` method is the actual training step. Feature
extraction is not model training: it creates the numerical examples the
classifier learns from.

Random Forest was selected because it is a practical baseline for
tabular numerical features, can model non-linear relationships, does not
require feature standardization, and provides exploratory
feature-importance scores. This baseline is heartbeat-only; it does not
use a CNN or learn directly from raw video frames.

## 6. Experimental results

### Validation result

The reported validation set contained 891 videos: 148 REAL and 743 FAKE.

  Metric                              Result
  ------------------- ----------------------
  Accuracy                            83.28%
  ROC-AUC                             0.5735
  Balanced accuracy     approximately 50.20%
  REAL recall            approximately 0.68%
  FAKE recall           approximately 99.73%

With REAL = 0 and FAKE = 1, the confusion matrix was:

``` text
                 Predicted
                 REAL   FAKE
Actual REAL        1     147
Actual FAKE        2     741
```

The model identified nearly all FAKE videos but correctly identified
only one REAL video.

### Why 83.28% accuracy is misleading

Because the validation set contains 743 FAKE videos out of 891, a
classifier that predicts FAKE for every video would achieve:

\[ `\frac{743}{891}`{=tex} `\times 100`{=tex} `\approx 83.39`{=tex}% \]

The trained model's 83.28% accuracy is slightly below that
majority-class baseline.

Balanced accuracy averages recall across the two classes and exposes the
failure to identify REAL videos. ROC-AUC measures how well the model
ranks positive examples above negative examples across thresholds; a
value near 0.5 indicates weak separation.

### Threshold experiment

The FAKE probability threshold was varied:

    FAKE threshold   REAL recall   FAKE recall   Balanced accuracy
  ---------------- ------------- ------------- -------------------
               0.3         0.00%       100.00%              50.00%
               0.4         0.00%       100.00%              50.00%
               0.5         0.68%        99.73%              50.20%
               0.6         4.73%        98.25%              51.49%
               0.7        20.95%        88.83%              54.89%

Increasing the threshold made the classifier more willing to predict
REAL, but also reduced FAKE recall. Even at 0.7, balanced accuracy
remained low. Threshold adjustment did not solve the underlying
class-separation problem.

### Feature-distribution analysis

Histograms were plotted for dominant frequency, estimated BPM, total
power, spectral entropy, peak count, and maximum autocorrelation.

The REAL and FAKE distributions overlapped substantially:

-   Dominant frequency and estimated BPM had similar shapes. BPM is
    derived from dominant frequency, so these are closely related
    features.
-   Total power was concentrated near zero for both classes and had a
    long right tail.
-   Spectral entropy showed considerable overlap.
-   Peak count showed some distribution differences but also substantial
    overlap.
-   Maximum autocorrelation distributions were broadly similar.

This is consistent with the weak validation performance. It does not
prove that no combination of features can be useful, but suggests that
the current whole-face summary features provide limited class
separation.

## 7. Interpretation and limitations

The experiment successfully implemented a baseline pipeline: videos were
indexed and labelled; face regions were located; RGB traces were
extracted; POS was used to estimate pulse-related waveforms; features
were calculated; and a Random Forest was trained and evaluated.

However, the validation results do **not** support claiming that this
baseline is an effective deepfake detector. Its accuracy is explained by
class imbalance, while balanced accuracy and REAL recall show that it
fails to distinguish the two classes reliably.

Possible explanations to investigate include:

-   Whole-face averaging combines skin and non-skin pixels, potentially
    diluting subtle pulse-related information.
-   Facial movement, expression, lighting variation, and compression may
    dominate the small colour changes of interest.
-   Short or noisy signals may produce unstable dominant-frequency and
    peak measurements.
-   The selected statistical features may discard spatial and temporal
    relationships that matter.
-   Different manipulation methods may preserve or alter physiological
    traces differently.

These are plausible explanations, not causes proven by this experiment.

## 8. Next research direction: region-specific rPPG

The next experiment is to compare whole-face averaging with
region-specific extraction. Instead of averaging the entire detected
face into one RGB trace, the face can be divided into smaller
skin-focused regions:

-   Forehead
-   Left cheek
-   Right cheek

Each region can produce its own RGB traces, POS waveform, and feature
set. Regional signals can then be compared for consistency and
potentially combined.

This is a proposed improvement and is not guaranteed to improve
performance. Precise facial landmarks or skin segmentation may be
preferable to approximate rectangular crops, because a crop can include
eyes, hair, mouth, or other non-skin areas.

A fair comparison should use the same source-group split for the
whole-face and regional experiments. Report extraction success rates as
well as classification metrics.

### Further experiments

1.  Improve ROI selection using facial landmarks or skin masks.
2.  Assess signal quality before feature extraction.
3.  Add temporal features such as window-to-window consistency,
    pulse-rate variability, waveform similarity, and cross-region
    synchronization.
4.  Reduce redundant features; dominant frequency and estimated BPM are
    mathematically linked.
5.  Evaluate manipulation categories separately.
6.  Test generalization on videos from different sources and identities.
7.  Explore a hybrid model combining rPPG features with visual features
    learned by a CNN, after establishing a reliable physiological
    baseline.

## 9. Research conclusion

This project investigated remote photoplethysmography as a potential
physiological cue for deepfake detection. Facial RGB intensity traces
were extracted from FaceForensics++ C23 videos, processed using the POS
algorithm to estimate pulse-related signals, and converted into
statistical and frequency-domain features. A Random Forest classifier
was trained to distinguish authentic and manipulated videos using these
features alone.

The initial whole-face baseline achieved 83.28% validation accuracy, but
this result was misleading because of the imbalanced validation set. The
model correctly identified only 1 of 148 REAL videos, achieved
approximately 50.20% balanced accuracy, and produced a ROC-AUC of
approximately 0.573. Feature-distribution plots also showed substantial
overlap between REAL and FAKE examples.

Therefore, the experiment does not demonstrate that the current
whole-face rPPG features are sufficient for reliable deepfake
classification. Instead, it establishes a baseline and highlights the
challenges of recovering weak physiological information from compressed,
moving facial video. The next stage will investigate whether
region-specific extraction and richer temporal features preserve more
useful information. A later hybrid approach may combine physiological
signals with visual analysis, but its value must be demonstrated through
controlled experiments and leakage-aware evaluation.

The central research takeaway is that **a physiological signal can be
measurable without necessarily being discriminative**. The quality,
reliability, and consistency of the extracted rPPG signal matter just as
much as the classifier used to interpret it.

## 10. Key terminology

  -----------------------------------------------------------------------
  Term                                Meaning
  ----------------------------------- -----------------------------------
  Deepfake                            A video in which a person's face or
                                      appearance has been synthetically
                                      manipulated

  rPPG                                Remote estimation of pulse-related
                                      information from subtle changes in
                                      skin colour

  RGB trace                           Time series of average red, green,
                                      and blue pixel intensities

  ROI                                 Region of interest selected from a
                                      video frame

  POS                                 Plane-Orthogonal-to-Skin method for
                                      estimating a pulse-related signal
                                      from normalized RGB traces

  BVP                                 Blood-volume pulse; the
                                      physiological signal rPPG methods
                                      aim to estimate

  Bandpass filter                     Filter that retains a chosen
                                      frequency interval

  Welch PSD                           Method for estimating how signal
                                      power is distributed across
                                      frequencies

  Dominant frequency                  Frequency with the largest
                                      estimated spectral power in the
                                      selected band

  Feature extraction                  Converting a signal into numerical
                                      measurements for a machine-learning
                                      model

  Random Forest                       Ensemble of decision trees used for
                                      classification

  Class imbalance                     Unequal numbers of examples in the
                                      REAL and FAKE classes

  Balanced accuracy                   Average of the recall values for
                                      each class

  ROC-AUC                             Threshold-independent measure of
                                      how well a model ranks the two
                                      classes
  -----------------------------------------------------------------------

## 11. References and further reading

1.  Wang, W., den Brinker, A. C., Stuijk, S., and de Haan, G.
    **"Algorithmic Principles of Remote-PPG."** *IEEE Transactions on
    Biomedical Engineering*, 2017. Explains the optical and
    signal-processing principles behind rPPG and introduces POS.
    https://doi.org/10.1109/TBME.2016.2609282

2.  **"pyVHR: a Python framework for remote photoplethysmography."**
    Describes an rPPG framework and discusses a FaceForensics++
    deepfake-detection experiment using estimated BVP signals.
    https://pmc.ncbi.nlm.nih.gov/articles/PMC9044207/

3.  **"Local attention and long-distance interaction of rPPG for
    deepfake detection."** Explores spatial and temporal patterns across
    multiple facial regions for rPPG-based deepfake detection.
    https://pmc.ncbi.nlm.nih.gov/articles/PMC10052279/

4.  **FaceForensics++ dataset.** Dataset used as the source of real and
    manipulated video examples. https://github.com/ondyari/FaceForensics

## 12. Reproducibility checklist

-   Record the exact dataset version and file list used.
-   Keep the train/validation/test source-group split fixed when
    comparing feature-extraction approaches.
-   Save the manifest, split assignments, extracted feature CSVs,
    imputer, model, and feature-column list.
-   Record sampling interval, effective FPS, POS window size, filter
    cut-offs, and feature-extraction settings.
-   Report failed face detections and failed signal extractions rather
    than silently excluding them.
-   Do not tune model settings using the held-out test set.
-   Report balanced accuracy, class-wise precision/recall, macro F1,
    ROC-AUC, and confusion matrix---not accuracy alone.
-   Treat rPPG output as an estimated pulse-related signal, not a
    medical measurement or proof of authenticity.
