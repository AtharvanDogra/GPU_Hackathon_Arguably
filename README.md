# GPU_Hackathon_Arguably
Contains the submission files for the team Arguably
# Readme

The team approach to solve the problem can bve divided internally into 2 phases

1. Usage of Pretrained models
2. Using novel CNN model for classification purposes

After thorough analysis of the given data and understanding the basic attributes the team trained the classification models on the data and tested the results as listed below:

### **Accuracy Score**
Model | Accuracy
----- | --------
InceptionV3 | .92
InceptionResnetV2 | .90
VGG19 | .8738
CNNV1 | .97
CNNV2 | 0.9897



From table 1 the model CNNV2 was selected as the best mdel since it had the best accuracy among all the tested model and nearly approached ~99%

The preformance of the network can be attributed to the use of (Atharvan)

```python
cnn.add(Dense(1, kernel_regularizer=tf.keras.regularizers.l2(0.01),activation ='linear'))
 ```
This part, setting activation `linear` is for applying linear SVM for binary classification in the model

```python
cnn.compile(optimizer = 'adam', loss = 'hinge', metrics = ['accuracy'])
```
Secondly, this loss function `hinge` is compiling the model for SVM. 


CNNv1.ipynb -> This is the notebook corresponding to the first trial on the test data using CNN

CNNv2.ipynb -> This is the notebook corresponding to the second trial on the test data using CNN.[The final model selected by the team]

pretrained_approach.ipynb -> Corresponds to the use of pretrained networks for classification
