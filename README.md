# speech_emotion_recognition
 Our voice conveys both linguistic and paralinguistic messages in the course of speaking. The paralinguistic part, for example, rhythm and pitch, provides emotional cues to the speaker. Emotions consist of cognitive, physiological and behavioural changes and all these phenomena are interrelated. Generally, an emotion is a state that affects the thoughts and is capable of determining behaviour. Emotion also creates physical and psychological changes. Speech Emotion Recognition topic examines the question ‘How is it said?’ and an algorithm detects the emotional state of the speaker from an audio record.

 The main aim in this study is obtaining the most distinctive emotional features. For this purpose, in order to compare audio features based on different domains Root Mean Square Energy (RMSE), Zero Crossing Rate (ZCR), Chroma and Mel Frequency Cepstral Coefficients (MFCC) features are examined for emotion recognition. A pre-trained model namely wav2vec Large which has been developed more recently is used to create the inputs also. Support Vector Machine, Multi-Layer Perceptron and Convolutional Neural Network techniques are utilized for developing learning models for comparing traditional features and the pre-trained model representations. In this paper emotions namely, Happy, Calm, Angry, Boredom, Disgust, Fear, Neutral, Sad and Surprise are classified, and furthermore, the models are trained and tested with English and German speech datasets.

When the classification results are examined, it is concluded that the most successful predictions are obtained with the pre-trained representations. The weighted accuracy ratio is 91% for both Convolutional Neural Network and Multilayer Perceptrons models while this ratio is 87% for the Support Vector Machine models. Among the emotional states, Fear has the highest recognition ratio with 95% f-score with Convolutional Neural Network technique which uses a pre-trained model

Components of the study:


<img width="606" alt="image" src="https://github.com/fatma010/speech_emotion_recognition/assets/73019838/cc281a1f-2e94-47cf-8b1c-a52e83a734f9">

Waveforms are given in figures and the X-axis shows Time (sec) and the Y-axis shows amplitude. Figures in ‘a’ to ‘c’ belong to the emotion of neutral. In (a) record is taken from speaker 1 from RAVDESS which represents “Kids are talking by the door” utterances. In (b) record is taken from OAF actress from the TESS which represents “Say the word dog” utterance. In (c) record is taken from speaker 13 from EMODB for “Das will sie am Mittwoch abgeben” utterance which means “She will hand it in on Wednesday”.

a:


<img width="378" alt="image" src="https://github.com/fatma010/speech_emotion_recognition/assets/73019838/f2f46d10-dcc8-4d31-9245-0f777910da6a">

b:


<img width="378" alt="image" src="https://github.com/fatma010/speech_emotion_recognition/assets/73019838/7b0d83c5-cb74-4a7c-b18a-bed661c3fa72">

c:

<img width="378" alt="image" src="https://github.com/fatma010/speech_emotion_recognition/assets/73019838/ba676424-d9a4-4232-91c2-f1be9c32c3ff">

