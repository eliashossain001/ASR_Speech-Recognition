# ASR_Speech-Recognition
# speech2phoneme+grapheme_bangla
This repository is developed for the purpose of Bangla speech recognition using state-of-the-art NLP tecnique and model. We used a self supervised model (Wav2Vec2) developed by Facebook AI team to recognize speech from Bangla speech corpus. I incorporated both grapheme and phoneme as an input features. However, development is still going on. Wav2Vec 2.0 is one of the current state-of-the-art models for Automatic Speech Recognition due to a self-supervised training which is quite a new concept in this field. I haven't used any open access dataset, I utilized custom dataset. If you want to use Common Voice/Bengali ASR data, you may change the data adapter, e.g., create a data loader and split your dataset into two/three folds: train, test and validation. However, I have experimented through train and validation data. 


# MlOps
I integrated the mlflow tool to track the training history. MLflow is an MLOps tool that enables data scientist to quickly productionize their Machine Learning projects. To achieve this, MLFlow has four major components which are Tracking, Projects, Models, and Registry. MLflow lets you train, reuse, and deploy models with any library and package them into reproducible steps.

# How to use?

I have tested the code through VS code script file and notebook as well. It's better to use the VS code and you should have a high computing support other than that, you are gonna fail to run this model. To be more specific, this is a powerful speech recognization model. If you want to run on phoneme level, the check out the Notebooks folder and open the 'STT_Wav2vec2_Base_Phoneme.ipynb' file. On the other hand, to run trough sentence level, use the Wav2Vec2_ASR_Bangla_V1.

# Essential libraries
-Pytorch, -Librosa, -Pandas, -Wav2Vec2 transformer,-CTC tokenizer
