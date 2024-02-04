import os, glob
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import torch
from fairseq.models.wav2vec import Wav2VecModel
from sklearn.metrics import accuracy_score
import torch.optim as optim


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
    'neutral.wav':'Neutral'
}
# Emotions to observe
observed_emotions = ['Neutral', 'Calm','Happy', 'Sad', 'Angry', 'Fearful',
                     'Disgust', 'Boredom','Surprised']


#map_location=torch.device('cpu')
cp = torch.load('/Users/fatmagumus/Desktop/vq-wav2vec_kmeans.pt',map_location=torch.device('cpu'))#vq-wav2vec_kmeans.pt  Accuracy: 91.33%
model = Wav2VecModel.build_model(cp['args'], task=None)
model.load_state_dict(cp['model'])
model.eval()

#for name,param in model.named_parameters():
#    if param.requires_grad and 'feature_extractor.conv_layers.5.0.weight' in name:
#        param.requires_grad=False
#    if param.requires_grad and  'feature_extractor.conv_layers.5.2.weight'in name:
#        param.requires_grad=False
#    if param.requires_grad and  'feature_extractor.conv_layers.5.2.bias' in name:
#        param.requires_grad = False
#    if param.requires_grad and 'feature_extractor.conv_layers.4.0.weight' in name:
#        param.requires_grad=False
#    if param.requires_grad and  'feature_extractor.conv_layers.4.2.weight'in name:
#        param.requires_grad=False
#    if param.requires_grad and  'feature_extractor.conv_layers.4.2.bias' in name:
#        param.requires_grad = False
#    if param.requires_grad and 'feature_extractor.conv_layers.3.0.weight' in name:
#        param.requires_grad=False
#    if param.requires_grad and  'feature_extractor.conv_layers.3.2.weight'in name:
#       param.requires_grad=False
#    if param.requires_grad and  'feature_extractor.conv_layers.3.2.bias' in name:
#       param.requires_grad = False
#    if param.requires_grad and 'feature_extractor.conv_layers.2.0.weight' in name:
#        param.requires_grad=False
#    if param.requires_grad and  'feature_extractor.conv_layers.2.2.weight'in name:
#        param.requires_grad=False
#    if param.requires_grad and  'feature_extractor.conv_layers.2.2.bias' in name:
#        param.requires_grad = False
#    if param.requires_grad and 'feature_extractor.conv_layers.1.0.weight' in name:
#        param.requires_grad=False
#    if param.requires_grad and  'feature_extractor.conv_layers.1.2.weight'in name:
#        param.requires_grad=False
#    if param.requires_grad and  'feature_extractor.conv_layers.1.2.bias' in name:
#        param.requires_grad = False
#    if param.requires_grad and 'feature_extractor.conv_layers.0.0.weight' in name:
#        param.requires_grad=False
#    if param.requires_grad and  'feature_extractor.conv_layers.0.2.weight'in name:
#        param.requires_grad=False
#   if param.requires_grad and  'feature_extractor.conv_layers.0.0.bias' in name:
#        param.requires_grad = False
#optimizer=optim.SGD(filter(lambda p: p.requires_grad,model.parameters()),lr=0.001)


def extract_wav_2_vec(sample,model):
    y, sr = librosa.load(sample, sr=16000)
    b = torch.from_numpy(y).unsqueeze(0)
    z = model.feature_extractor(b)
#    c = model.feature_aggregator(z).detach().numpy()
    z = torch.mean(z, 2).squeeze(0)
    return z



def load_data():
    x, y,features = [], [], []
    for file in glob.glob('your_path'):
        file_name = os.path.basename(file)
        emotion = emotions[file_name.split("-")[2]]
        if emotion not in observed_emotions:
            continue
        feature = extract_wav_2_vec(file,model)
        feature = feature.detach().numpy()
        x.append(feature)
        y.append(emotion)
    for file in glob.glob('your_path'):
        file_name = os.path.basename(file)
        emotion = emotions[file_name[5]]
        if emotion not in observed_emotions:
            continue
        feature = extract_wav_2_vec(file,model)
        feature = feature.detach().numpy()
        x.append(feature)
        y.append(emotion)
    for file in glob.glob('your_path'):
            file_name = os.path.basename(file)
        emotion = emotions[file_name.split("_")[2]]
        if emotion not in observed_emotions:
            continue
        feature = extract_wav_2_vec(file,model)
        feature = feature.detach().numpy()
        x.append(feature)
        y.append(emotion)
    return np.array(x, dtype="object"),y

x,y=load_data()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)


# Get the shape of the training and testing datasets
print((x_train.shape[0], x_test.shape[0]))  #(1079, 360)

# Get the number of features extracted
print(f'Features extracted: {x_train.shape[1]}')

print('unique:', np.unique(y_train,return_counts=True))

#creating model
modelml = MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08,
                      hidden_layer_sizes=(300,),
                    learning_rate='adaptive', max_iter=1000)



# Train the model
modelml.fit(x_train, y_train)

# Predict for the test set
y_pred = modelml.predict(x_test)

from sklearn.metrics import confusion_matrix
matrix = confusion_matrix(y_pred,y_test)
print ('matrix',matrix)

#Calculate the accuracy of our model
accuracy=accuracy_score(y_true=y_test, y_pred=y_pred)
 #Print the accuracy
print("Accuracy: {:.2f}%".format(accuracy*100))

from sklearn.metrics import classification_report
report = classification_report(y_test, y_pred)
print(report)

