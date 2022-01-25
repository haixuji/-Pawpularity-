# -Pawpularity-
Data source:
https://www.kaggle.com/c/petfinder-pawpularity-score/data
How to run:
●	Machine-learning Method + Xeption:
  ○	The saved model is name “HJ_model_2_xception.h5”, can be directly import and use to predict.
  ○	Using TensorFlow API ‘tf.keras.models.load_model’ to load the model.
  ○	Using TensorFlow function ‘model.evaluate’ to do the evaluation.
  ○	If you want to run the Xception python code, define the data load path first.
  ○	Then, run the notebook, which named “HJ_Model_Final” line by line.
●	MobileNet:
  ○	The best model of MobileNetV2 has been saved with the name of ‘’ in the code folder.
  ○	Using TensorFlow API ‘tf.keras.models.load_model’ to load the model.
  ○	Using TensorFlow function ‘model.evaluate’ to do the evaluation.
  ○	If you want to see the whole experiment, define the data load path first.
  ○	Then, run the notebook line by line.
●	EfficientNet Feature Extraction + MLP
  ○	The transfer learning EfficientNet model has been saved as “EfficientNetB4_Final.h5”
  ○	The Feature Extracted MLP model has been saved as “EfficientNetB4_MLP_Final.h5”
  ○	In order to use the final MLP model, need to load the intermediate model to transfer the image data into shape Nx128 features“EfficientNetB4_Intermidiate_Final.h5” and concatenate it with metadata (detailed usage is in “Feature_Extract_MLP.ipynb”)
  ○	Using TensorFlow API ‘tf.keras.models.load_model’ to load the model (need to add custom_objects={ 'my_rmse': my_rmse} when loading). My rmse is a function defined in notebook named “Feature_Extract_MLP.ipynb”
  ○	If you want to train the model from scratch, run “EfficientNet.ipynb” to train the base model first, “EfficientNet_fine_tune.ipynb” is for fine tuning the model. “Feature_Extract_MLP.ipynb” is for training the MLP model.
●	VGG-16 Feature Extraction with Multilayer Perceptron Model
  ○	The saved model is name “vgg16.h5”, can be directly import and use to predict.
  ○	Using TensorFlow API ‘tf.keras.models.load_model’ to load the model.
  ○	Using TensorFlow function ‘model.evaluate’ to do the evaluation.
  ○	define the data load path first at variable ‘data_dir’in ‘VGG16_fea_reduc_mlp’.ipynb.
  ○	Then, run the notebook, ‘VGG16_fea_reduc.ipynb’ line by line.
