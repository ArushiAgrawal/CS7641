# Recognition of Facial Expressions
 
## Introduction 

Facial expression is the most universal, natural, and powerful way for human beings to convey their thoughts. Emotions don’t have a concrete definition, yet they drive every other decision made in our lives. 

Our project focuses on recognition of facial expressions. We are using [Facial Image Recognition Dataset (FER2013)](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data) – an open-source dataset containing approximately 30 thousand labelled grayscale images of faces. This dataset was published during the International Conference on Machine Learning (ICML). The emotions in the images belong to the seven categories - anger, disgust, fear, happy, sad, surprise, and neutral.

<p align = "center"> <img width="762" alt="image" src="https://user-images.githubusercontent.com/41327028/161588858-6e6f14e4-3d7c-4883-9104-f516f378c094.png"> </p>
<p align = "center"> Fig.1 Sample images </p>
 
## Problem Definition 

Market research has proven that predicting sentiments correctly can be a huge source of growth for businesses, as it could help to gauge customer mood towards their brand or product. In addition to marketing and advertising, recognizing facial emotions is also important in various other fields – surveillance and law enforcement, video game testing, driving safety in cars, etc.  

We aim to create deep-learning based models that can classify human emotions.

## Literature Review

The depth of representations is of central importance for many visual recognition tasks. However, training Deep neural networks is difficult. The two widely used models to overcome these limitations are ResNets and DenseNet. ResNets use skip connections from initial layers to later ones to reformulate the layers as learning residual functions with reference to the layer inputs and thus ease the training of substantially deep networks<sup>[2]</sup>. Dense Convolutional Network (DenseNet), connects each layer to every other layer in a feed-forward fashion and help alleviate the vanishing-gradient problem, strengthen feature propagation, encourage feature reuse, and substantially reduce the number of parameters<sup>[1]</sup>. Different CNN architectures and CNN-LSTMs for accurate detection of human emotions have been explored in literature.<sup>[3]</sup>

## Methods and Results

### EDA (Exploratory Data Analysis) and PCA (Principal Component Analysis): 

The data obtained from [Facial Image Recognition Dataset (FER2013)](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data) consists of 48x48 pixel grayscale images of faces. The face is alomst centered and occupies about the same amount of space in each image. Furthermore, all the images in the training dataset are clearly labeled with the distinct emotions they represent. 

We are using around 30K images for training our models, while 3.5K images for both validating and testing the results. On studing the distribution of images into the corresponding classes, we observed that the data is unbalanced. Out of the seven classes we have, 'Happy' emotion makes upto 25% while 'Disgust' is just 1.5% of the entire dataset. However, this distribution is similar across training, validation and test datasets. 

<p align = "center">
<img width="1004" alt="image" src="https://user-images.githubusercontent.com/41327028/161564709-5decd3f8-61a1-4d6d-a55d-d6a08f87e0d0.png"> </p>
<p align = "center"> Fig.2: Class Distribution </p>

The mean face of all 30K faces in our dataset looks as below - 
<p align = "center"><img width="300" alt="image" src="https://user-images.githubusercontent.com/41327028/161585948-19087721-11be-4e67-80c0-bf8bfdd74150.png"> </p>
<p align = "center"> Fig.3: Average face over all images</p>


For understanding the difference between the images belonging to different emotions, we checked the <b> Average face </b> for each emotion. In this case, the pixel values are just averaged over different emotions and plotted. We can clearly see that the images obtained are well alligned with the emotion classes they represent. 
<p align = "center">
<img width="863" alt="image" src="https://user-images.githubusercontent.com/41327028/161566642-bcf66ec3-e684-4ebd-a289-87a3b000efaa.png"> </p>
<p align = "center"> Fig.4: Average face by emotion </p>


To visualize the variation in the given collection of images and compare them in a holistic manner, we explored <b> Eigenfaces</b>. The eigenfaces are the principal components of a distribution of faces, basically, the eigenvectors of the covariance matrix of the set of face images. We have considered the first eigen vector (corresponding to the maximum eigen value) for plotting the below images. 
<p align = "center"> <img width="857" alt="image" src="https://user-images.githubusercontent.com/41327028/161566805-44d6e31c-e7ef-404d-aee9-8cd27cf4b1a5.png"> </p>
<p align = "center"> Fig.5: Eigenface by emotion </p>

### Modeling:

#### Baseline Classification Models explored:

In order to truly understand the performance of proposed CNN models, we decided to first explore baseline traditional classification algorithms like Logistic Regression, Linear SVM, SVM with RBF and Random Forest. We conducted PCA of the given dataset and observed that 103 components are required in order to explain 90% of the variance in the data. The validation and test set accuracies of the four models were obtained as shown in the table below:




#### ResNet18

Residual networks have shortcut connections (skip-connections) which turn a deep network into its counterpart residual version.  This enables gradients to flow directly through the skip connections backwards from later layers to initial layers and thus helps overcome the vanishing gradient problem to a great extent. ResNet consists on one convolution and pooling layer followed by repetition of this layers. We have evaluated ResNet18 CNN architecture to perform this multiclass classification. We generated the accuracy curves for the training and the validation set as the model kept learning. We got the validation and test set accuracies as 62.7% and 61.1% respectively. We see that the model is overfitting, so our next step is to improve its generalization accuracy using implicit or explicit regularization techniques like image augmentation, dropout layers etc.

In order to investigate the model further, we also plot the Top-3 accuracies curve which represents how well the model performed in order to predict a correct image class, within the top 3 classes having the highest softmax probabilities. We also generated the confusion matrix of the test dataset to understand which emotion classes are easy to classify and which emotion pairs are confusing. We found that the model performed really well with the "Happy" class, whereas the "Neutral" class was often confused with the "Sad" class and vice versa. Further, a good proportion of images in "Fear" class also got mis-classified as "Sad".

<p align="center">
  <img img width="600" src="https://user-images.githubusercontent.com/29612754/161646159-2e76498c-1349-445d-b0a5-593f3e73441b.png">
</p>
<p align = "center"> Fig.6 Model Performance </p>

<p align="center">
  <img img width="300" src="https://user-images.githubusercontent.com/29612754/161646267-5ab242df-a785-4cc5-9274-0d51582346d5.png">
</p>
<p align = "center"> Fig.7 Confusion matrix for test images </p>


#### ResNet18 evaluation using t-SNE

<b> t-SNE </b> is a nonlinear dimensionality reduction technique, well suited for embedding high dimension data into lower dimensional data (2D or 3D) for data visualization. After training our model (Resnet18 architecture), we collected the features from the last convolutional layer and visualized this feature vector for every image in the training dataset. This embedding feature vectors for the dataset is of the dimension N X 512. 

We get the following visualization for the training set embeddings - 
<p align = "center"> <img width="500" alt="image" src="https://user-images.githubusercontent.com/12424401/161621666-f802f29e-9155-4bfa-9602-c03dec5be01b.png"> </p>
<p align = "center"> Fig.8: t-SNE plot</p>
Form the above visalization, we can observe that the current model is susceptible to confuse between 'Sad' and 'Fear' as there is not a lot of seperation between the two classes. 

## Future Work and Discussion

We plan to consider a few more candidate models: 
* Vanilla Convolutional Neural Network
* ResNet variants: 34, 50, 101 
* DenseNet 

In addition to the different Neural network architectures, we also plan to perform:
* Image Augmentation: To improve generalization accuracies of the models  
* Learning Rate Finding & Scheduling: To get optimal learning rate for different stages of training the models 
* Transfer Learning: To utilize the feature representation from the same models trained on larger datasets 
* Visualization of activations and kernels: To understand feature representations of different emotions and the learned kernels.

Metrics to be considered:
* Accuracy, Precision, Recall, F-Beta Score
* Evaluation of models’ performances using Top-k Accuracy & confusion matrix.  
* Evaluation of incorrect predictions: To analyze which emotion classes are hard to tell apart by the models 

Challenges in emotion recognition:
* Different people can interpret emotions in different ways, and hence the training data may not be 100% reliable. 
* It is very tough to detect all possible cues for an emotion, and some cues can be common among different emotions. e.g., Visual cues like furrowed eyebrows can mean something aside from anger, and other non-so-obvious facial cues may be subtle hints of anger. 
* Bias based on race, gender, and age. 

## Timeline and Responsibilities

<p align = "center"> 
 <img width="886" alt="image" src="https://user-images.githubusercontent.com/41327028/161758365-988186d9-68ea-415c-b4d4-98881e8c113d.png">
</p>

## References

1) [Huang, Gao, Zhuang Liu, Laurens Van Der Maaten, and Kilian Q. Weinberger. "Densely connected convolutional networks." In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 4700-4708. 2017](https://arxiv.org/abs/1608.06993)

2) [He, Kaiming, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. "Deep residual learning for image recognition." In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 770-778. 2016](https://ieeexplore.ieee.org/document/7780459)

3) [Wafa Mellouk, Wahida Handouzi. “Facial emotion recognition using deep learning: review and insights.” Procedia Computer Science, Volume 175, pp. 689-694, 2020](https://www.sciencedirect.com/science/article/pii/S1877050920318019)
