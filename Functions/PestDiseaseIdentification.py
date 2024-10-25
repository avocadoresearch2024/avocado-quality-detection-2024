import tensorflow as tf
from keras.utils import img_to_array
import numpy as np

modelPath = 'TrainedModels/AvocadoPlantPestIdentificationModel.h5'
pestDiseaseIdentificationModel = tf.keras.models.load_model(modelPath, compile=False)
classes = ['Dots', 'good', 'pest']

def pestDiseaseIdentification(imageFile) -> np.ndarray:
    imageRead = imageFile.read()
    decodedImage = tf.image.decode_image(imageRead)
    resizedImage = tf.image.resize(decodedImage, [256, 192])
    imgArray = img_to_array(resizedImage)
    imgArray = tf.expand_dims(imgArray, 0)

    prediction = pestDiseaseIdentificationModel.predict(imgArray)

    probabilities = prediction[0]
    predictedClass = np.argmax(probabilities)
    predictedLabel = classes[predictedClass]

    maximumProbability = np.amax(probabilities)

    prediction = {
        'predicted_class' : predictedLabel,
        'probability':str(maximumProbability),
    }
         
    return prediction
   
        
    
