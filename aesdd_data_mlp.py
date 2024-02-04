import os, glob
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import torch
from fairseq.models.wav2vec import Wav2VecModel
from sklearn.metrics import accuracy_score

# Emotions in the RAVDESS dataset
emotions = {
    'N': 'Neutral',
    'L': 'Boredom',
    'E': 'Disgust',
    'A': 'Fearful',
    'F': 'Happy',
    'T': 'Sad',
    'W': 'Angry',
    '01': 'Neutral',
    '02': 'Calm',
    '03': 'Happy',
    '04': 'Sad',
    '05': 'Angry',
    '06': 'Fearful',
    '07': 'Disgust',
    '08': 'Surprised',
    'sad.wav': 'Sad',
    'ps.wav':'Surprised',
    'angry.wav':'Angry',
    'fear.wav':'Fearful',
    'disgust.wav':'Disgust',
    'happy.wav':'Happy',
    'neutral.wav':'Neutral',
    'a':'Angry',
    'f':'Fearful',
    'h':'Happy',
    's':'Sad',
    'd':'Disgust'
}
# Emotions to observe   'Neutral', 'Calm',, 'Surprised' ,'Boredom'
observed_emotions = ['Happy', 'Sad', 'Angry', 'Fearful','Disgust']
#observed_emotions = ['Neutral', 'Calm','Happy', 'Sad', 'Angry', 'Fearful',
#                     'Disgust', 'Surprised' ,'Boredom']


#map_location=torch.device('cpu')
cp = torch.load('/Users/fatmagumus/Desktop/wav2vec_large.pt',map_location=torch.device('cpu'))
model_vec = Wav2VecModel.build_model(cp['args'], task=None)
model_vec.load_state_dict(cp['model'])
model_vec.eval()

def extract_wav_2_vec(sample,model_vec):
    y, sr = librosa.load(sample, sr=16000)
    b = torch.from_numpy(y).unsqueeze(0)
    z = model_vec.feature_extractor(b)
    c = model_vec.feature_aggregator(z)#.detach().numpy()
    z = torch.mean(c, 2).squeeze(0)
    return z



def load_data():
    x, y,features = [], [], []
    for file in glob.glob('your_path'):
        file_name = os.path.basename(file)
        emotion = emotions[file_name.split("-")[2]]
        if emotion not in observed_emotions:
            continue
        feature = extract_wav_2_vec(file,model_vec)
        feature = feature.detach().numpy()
        x.append(feature)
        y.append(emotion)
    for file in glob.glob('your_path'):
        file_name = os.path.basename(file)
        emotion = emotions[file_name[5]]
        if emotion not in observed_emotions:
            continue
        feature = extract_wav_2_vec(file,model_vec)
        feature = feature.detach().numpy()
        x.append(feature)
        y.append(emotion)
    for file in glob.glob('your_path'):
        file_name = os.path.basename(file)
        emotion = emotions[file_name.split("_")[2]]
        if emotion not in observed_emotions:
           continue
        x.append(feature)
        y.append(emotion)
    return np.array(x, dtype="object"),y

x,y=load_data()
print('y:', y)
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(y)
y=le.transform(y)
print(y)
# Split the dataset for train&test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.10, stratify=y, random_state=42)

#arranging datasets for cnn model
#Y_train = np.array(y_train)
#Y_test = np.array(y_test)

#creating model
model = MLPClassifier(alpha=0.0001, batch_size=256, epsilon=1e-08,hidden_layer_sizes=(300,), max_iter=500, learning_rate='adaptive'
                      )#learning_rate='invscaling',power_t=0.005


#alpha=0.1, batch_size=256, epsilon=1e-08,hidden_layer_sizes=(300,),learning_rate='adaptive', max_iter=500)
# Train the model
model.fit(x_train, y_train)


#predictions = model.predict(x_test)
#print('tahminler:' , predictions)

#from sklearn.metrics import classification_report
#report = classification_report(y_test, predictions)
#print('modelreport:')
#print(report)

def load_aesdd_data():
    x, y,features = [], [], []
    for file in glob.glob('your_path'):
        file_name = os.path.basename(file)
        emotion = emotions[file_name[0]]
        if emotion not in observed_emotions:
            continue
#        duration = librosa.get_duration(filename=file)
#        if duration > 4:
        features = extract_wav_2_vec(file,model_vec)
        features = features.detach().numpy()
        x.append(features)
        y.append(emotion)
    return np.array(x, dtype="object"),y

x_aes,y_aes=load_aesdd_data()
print('y_aes:', y_aes)
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(y_aes)
y_aes=le.transform(y_aes)
print(y_aes)
# Split the dataset for train&test
x_trainaes, x_testaes, y_trainaes, y_testaes = train_test_split(x_aes, y_aes, test_size=0.25, random_state=42)

#arranging datasets for cnn model
#Y_trainaes = np.array(y_trainaes)
#Y_testaes = np.array(y_testaes)


predictionsaesdd = model.predict(x_aes)
print('predicted aesdd:' , predictionsaesdd)

from sklearn.metrics import classification_report
reportaesdd = classification_report(y_aes, predictionsaesdd)
print("reportaesdd",reportaesdd)
