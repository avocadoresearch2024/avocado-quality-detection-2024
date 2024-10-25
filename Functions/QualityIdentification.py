import tensorflow as tf
from keras.utils import img_to_array
import numpy as np
import joblib
from skimage.transform import resize
from skimage.io import imread
import sklearn
print(sklearn.__version__)

modelPath = 'TrainedModels/AvocadoQualityClassificationModel.joblib'
qualityModel = joblib.load(modelPath)
classes = ['Local Market Quality',"can't use"]

def qualityIdentification(imagePaths) -> np.ndarray:
    predictionsArr = []
    for image in imagePaths:
        imageRead = imread(image)
        resizedImage =resize(imageRead,(150,150,3))
        flattenImage=[resizedImage.flatten()]

        prediction = qualityModel.predict_proba(flattenImage)
        probabilities = prediction[0]
        maxProbability = np.amax(probabilities)
        predictedClass = classes[qualityModel.predict(flattenImage)[0]]

        prediction = {
            'predicted_class' : predictedClass,
            'probability':maxProbability,
        }

        predictionsArr.append(prediction)
         
    return predictionsArr
   
        
    
