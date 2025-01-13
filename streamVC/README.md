# End-to-end streaming model for low-latency speech anonymization

### Waris Quamer, Ricardo Gutierrez-Osuna

In our [paper](https://arxiv.org/pdf/2406.09277), 
we proposed a lightweight alternative for speech synthesis to perform real-time speech anonymization.<br/>
We provide our implementation and pretrained models in this repository.

**Abstract :**
Speaker anonymization aims to conceal cues to speaker identity while preserving linguistic content. Current machine
learning based approaches require substantial computational resources, hindering real-time streaming applications. To
address these concerns, we propose a streaming model that achieves speaker anonymization with low latency. The system 
is trained in an end-to-end autoencoder fashion using a lightweight content encoder that extracts HuBERT-like information, 
a pretrained speaker encoder that extract speaker identity, and a variance encoder that injects pitch and energy information. 
These three disentangled representations are fed to a decoder that re-synthesizes the speech signal. We present 
evaluation results from two implementations of our system, a full model that achieves a latency of 230ms, 
and a lite version (0.1x in size) that further reduces latency to 66ms while maintaining state-of-the-art performance in 
naturalness, intelligibility, and privacy preservation.

Visit our [demo website](https://warisqr007.github.io/demos/stream-anonymization/) for audio samples.


## Pre-requisites
1. Python >= 3.10
2. Clone this repository.
3. Install python requirements. Please refer [requirements.txt](requirements.txt)
4. Download and extract the [LibriTTS dataset](https://openslr.org/60/).
And move all wav files to `data` folder
5. Convert all files to waveform and rearrange the data folder to look like
```
    data
    ├── metadata
    └── speakers                 # Folder containing all speakers
        ├── spkr1                   
        ├── spkr2
        │    └── wav             # Folder containing all wav files
        │        ├── file1.wav
        │        ├── file2.wav
        │        └── file3.wav
        ├── spkr3
        └── spkr4
```
6. Build similar structure separately for validation or test data.
7. Download the `Hubert Base` checkpoint from [here](https://github.com/facebookresearch/fairseq/blob/main/examples/hubert/README.md) and place it under `pretrained_models/hubert` folder.


## Data Preprocessing
To preprocess data, run
```
./data_preprocess.sh
```
Edit `data_preprocess.sh` file accordingly to preprocess data at a different path, specifically change `data` argument to point to the path of your data folder. Need to run separately for train and validation folders.

After running the data preprocessing, your `data` directory should look like
```
    data
    ├── metadata
    └── speakers                 
        ├── spkr1                   
        ├── spkr2
        │    ├── code               
        │    │   ├── file1.km
        │    │   └── file2.km
        │    ├── energy             
        │    │   ├── file1.eng.npy
        │    │   └── file2.eng.npy
        │    ├── pitch           
        │    │   ├── file1.pit.npy
        │    │   └── file2.pit.npy
        │    ├── spkr             
        │    │   ├── file1.spk.npy
        │    │   └── file2.spk.npy
        │    └── wav             
        │        ├── file1.wav
        │        └── file2.wav
        ├── spkr3
        └── spkr4
```

## Training
1. Pretrain only the `Encoder`
```
python train_encoder.py -p experiments/base/encoder -c experiments/base/config.json
```

2. Train `Encoder` and `Decoder` together
```
python train.py -p experiments/base -c experiments/base/config.json
```

## Pretrained Model
You can also use pretrained models we provide.<br/>
[Download pretrained models](https://drive.google.com/drive/folders/1auLt2tjZ7qQE5Nn2I9tqNPXsRhVIoikv?usp=sharing)<br/> 

Place the pretrained `base` and `lite` checkpoints under `experiments/base` and `experiments/lite` respectively.


## Inference
See `inference.ipynb`.


## Acknowledgements
We referred to [HiFiGAN](https://github.com/jik876/hifi-gan), [Fairseq](https://github.com/facebookresearch/fairseq) 
and [Speechbrain](https://github.com/speechbrain/speechbrain) to implement this.