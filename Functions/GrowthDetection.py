import tensorflow as tf
from keras.utils import img_to_array
import numpy as np

modelPath = 'TrainedModels/AvocadoPlantGrowthIdentificationModel.h5'
growthDetectionModel = tf.keras.models.load_model(modelPath, compile=False)
classes = ['3 months', '6 months']

def growthDetection(image, month) -> np.ndarray:
    imageRead = image.read()
    decodedImage = tf.image.decode_image(imageRead)
    resizedImage = tf.image.resize(decodedImage, [256, 192])
    imgArray = img_to_array(resizedImage)
    imgArray = tf.expand_dims(imgArray, 0)

    prediction = growthDetectionModel.predict(imgArray)

    probabilities = prediction[0]
    predictedClass = np.argmax(probabilities)
    predictedLabel = classes[predictedClass]

    maximumProbability = np.amax(probabilities)
    print(predictedLabel)

    predictedMonth = 0

    if('3' in predictedLabel):
         predictedMonth = 3
    else:
         predictedMonth = 6

    if (month in predictedLabel):
            prediction = {
            'predicted_class' : predictedLabel,
            'predicted_probabilities':str(max(probabilities))
        }

    elif(int(predictedMonth) < int(month)):
        prediction = {
            'predicted_class' : 'Undergrown'
        }

    else:
        prediction = {
            'predicted_class' : 'Overgrown'
        }
     
    return prediction
   
        
    
