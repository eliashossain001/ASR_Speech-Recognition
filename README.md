# ASR_Speech-Recognition
# speech2phoneme+grapheme_bangla
Uses a self supervised model (Wav2Vec2) developed by Facebook to recognize speech from Bangla speech corpus. We incorporated both grapheme and phoneme as an input features. However, development is still going on.

# Dataset
The dataset to train the model should be under the trainable_dataset folder or any folder you can create to keep data. Inside the trainable_dataset folder we should have a train folder and valid folder. Inside these folders there should be audio files and rheir corresponding phoneme sequence inside a .txt file

# Mlflow
The training experiments are tracked using Mlflow.
Run the mlflow server using the command mlflow server -h URL -p portnumber the code was written for the dev server on port 5000. So if you want to integrate the Mlflow, it would be better to setup your own URL. 
