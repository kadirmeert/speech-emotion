import librosa
import soundfile
import os, glob, pickle
import numpy as np
import glob
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from google.colab import drive
import speech_recognition as sr
import pandas as pd
from google.colab import drive
drive.mount('/content/drive')
#Data Link : https://zenodo.org/record/1188976#.XrBhw_IzY5k
data_folder = "/content/drive/My Drive/speech-emotion-recognition-ravdess-data"
folders = os.listdir(data_folder)
emotions = {
  '01':'neutral',
  '02':'calm',
  '03':'happy',
  '04':'sad',
  '05':'angry',
  '06':'fearful',
  '07':'disgust',
  '08':'surprised'
}
classes = {
  'calm':'calm',
  'happy':'happy',
  'sad':'sad',
  'angry':'angry',
  'fearful':'fearful',
  'surprised':'surprised'
}

unobserved_emotions = ['neutral', 'disgust']
def noise(data):
    noise_amp = 0.05*np.random.uniform()*np.amax(data)   # more noise reduce the value to 0.5
    data = data.astype('float64') + noise_amp * np.random.normal(size=data.shape[0])
    return data
def extract_feature(file_name, mfcc, mel):
  with soundfile.SoundFile(file_name) as sound_file:
    X = sound_file.read(dtype = "float32")          # read sound as float
    X = noise(X)
    sample_rate = sound_file.samplerate
    result = np.array([])
    if mfcc:
        mfccs = np.mean(librosa.feature.mfcc(y = X, sr = sample_rate, n_mfcc = 30).T, axis = 0)
        result = np.hstack((result, mfccs))     # create matrix
    
    if mel:
        mel=np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
        result=np.hstack((result, mel))
  return result
def load_data(test_size = 0.2):
  x,y = [],[]
  for file in glob.glob(data_folder+"C:\\Users\serge\Desktop\speech emotion\Actor_04\\*.wav"):
      file_name = os.path.basename(file)
      emotion = emotions[file_name.split("-")[2]]
      if emotion in unobserved_emotions:
          continue
      feature = extract_feature(file, mfcc = True, mel= False)
      x.append(feature)
      y.append(emotion)

x_train,x_test,y_train,y_test = load_data(test_size = 0.20)
print((x_train.shape[0], x_test.shape[0]))
print(f'Features extracted: {x_train.shape[1]}')
from sklearn.neural_network import MLPClassifier
MLP = MLPClassifier(alpha = 0.01, batch_size = 32, hidden_layer_sizes = (64,32,32,16,8), learning_rate = 'adaptive', max_iter = 215)
from xgboost import XGBClassifier
xgb = XGBClassifier(colsample_bytree=0.2, gamma=0.0468, 
                             learning_rate=0.1, max_depth=4, 
                             min_child_weight=1.7817, n_estimators=100,
                             reg_alpha=0.4640, reg_lambda=1.25,
                             subsample=0.2, silent=0,
                             random_state =7, nthread = -1)

MLP.fit(x_train, y_train)
train_pred = MLP.predict(x_test)
# evaluate predictions
accuracy = accuracy_score(y_test, train_pred)
print("accuracy: %.2f%%" % (accuracy))
#get results
x_axis = range(0, MLP.n_iter_)
fig, ax = plt.subplots()
ax.plot(x_axis, MLP.loss_curve_, label='Train')
ax.legend()
plt.ylabel('Classification Error')
plt.title('MLP Classification Error')
plt.show()

