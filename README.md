# Recognition of Facial Expressions
 
## Introduction 

Facial expressions are the most universal, natural, and powerful way for human beings to convey their thoughts. Emotions don’t have a concrete definition, yet they drive every other decision made in our lives. 

Our project focuses on recognition of emotions from facial expressions. We are using [Facial Image Recognition Dataset (FER2013)](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data) – an open-source dataset containing approximately 30 thousand labelled grayscale images of faces. This dataset was published during the International Conference on Machine Learning (ICML). The emotions in the images belong to the seven categories - anger, disgust, fear, happy, sad, surprise, and neutral.

<p align = "center"> <img width="762" alt="image" src="https://user-images.githubusercontent.com/41327028/161588858-6e6f14e4-3d7c-4883-9104-f516f378c094.png"> </p>
<p align = "center"> Fig.1: Sample images </p>
 
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

<p align="center">
  <img width="417" alt="image" src="https://user-images.githubusercontent.com/41327028/164942857-7b3835de-defb-4449-8b6a-01503cd13e1b.png">
</p>
<p align = "center"> Fig.6: Baseline Model Performance </p>


#### Convolutional Neural Network 

CNNs are the reguarized versions of the multilayer perceptrons. CNNs take advantage of the hierarchical pattern in the data and assemble patterns of increasing complexity using smaller and simpler patterns embossed in the filters. As compared to the 'fully connected' layers, CNNs are at a lower extreme in comparision to the connections. 

While defining the convolutional layers, we have used six convolution layers using a kernel of size (3,3) and 'ReLU' activation function. Each convolution layer is followed by a maxpooling layer. After the convolution layers, the neurons are flattened with the addition of a dense layer. At last, softmax activation function is used to calculate the probability for each class. 

<p align="center">
  <img width="526" alt="image" src="https://user-images.githubusercontent.com/41327028/164942790-9015a3b8-9496-403d-991e-dfd5d3e0a6c0.png">
</p>
<p align = "center"> Fig.7: CNN architecture </p>

In order to investigate the model further, we also plot the Top-3 accuracies curve which represents how well the model performed in order to predict a correct image class, within the top 3 classes having the highest softmax probabilities. We also generated the confusion matrix of the test dataset to understand which emotion classes are easy to classify and which emotion pairs are confusing. We found that the model performed really well with the "Happy" class, whereas the "Neutral" class was often confused with the "Sad" class and vice versa. Further, a good proportion of images in "Fear" class also got mis-classified as "Sad".

<p align="center">
  <img width="624" alt="image" src="https://user-images.githubusercontent.com/41327028/164945939-dcbec7c8-bb1a-45b4-82b6-90cb80001d92.png">
</p>
<p align = "center"> Fig.8: Model Performance </p>

<p align="center">
  <img width="399" alt="image" src="https://user-images.githubusercontent.com/41327028/164946094-5a0ae07a-d008-4c51-b12c-9bc2a031e9ba.png">
</p>
<p align = "center"> Fig.9: Model Performance </p>

Similar to the Resnet18 plot, we can see that the agent confuses Fear with Sad in the confusion matrix. This is also in line  with our t-SNE based visualization results (discussed later).

##### Image Augmentation 

Image Augmentation is a technique used to artifically create new images using the existing images. It creates variation in the training dataset which 
can improve the performance and ability of the model to generailze. These images are created using transformations that include a range of operations like shifts, flips, zooms, random rotation, gaussian blur, random perspective, random sharpness adjusts and normalization. The following picture shows some images from the training set after image augmentation techniques were applied. 

<p align="center">
  <img width="1039" alt="image" src="https://user-images.githubusercontent.com/29612754/165002826-6388ba60-fb57-4d0d-abc6-ef61629a5e65.png">
</p>
<p align = "center"> Fig.10: Augmented images from the Training-set </p>

We used image augmentation and then tried fitting the same nueral network as stated above. However there wasn't much improvement in the train and validation accuracies in the vanilla CNN network. The deeper ResNets seemed to have benefitted from this process quite a lot as the overfitting problem vanished when used with early-stopping while getting appreciable training and test accuracies.


#### ResNets

Residual networks have shortcut connections (skip-connections) which turn a deep network into its counterpart residual version.  This enables gradients to flow directly through the skip connections backwards from later layers to initial layers and thus helps overcome the vanishing gradient problem to a great extent. ResNet consists on one convolution and pooling layer followed by repetition of this layers. Starting with the basic version, we have evaluated ResNet18 CNN architecture to perform this multiclass classification. We generated the accuracy curves for the training and the validation set as the model kept learning. We got the validation and test set accuracies as 62.7% and 61.1% respectively. We see that the model is overfitting, so our next step is to improve its generalization accuracy using implicit or explicit regularization techniques like image augmentation, dropout layers etc.

Top-3 accuracies along with the confusion matrix of the test dataset has been plotted below. We found that the model performed really well with the "Happy" class, whereas the "Neutral" class was often confused with the "Sad" class and vice versa. Further, a good proportion of images in "Fear" class also got mis-classified as "Sad".

<p align="center">
  <img img width="600" src="https://user-images.githubusercontent.com/29612754/161646159-2e76498c-1349-445d-b0a5-593f3e73441b.png">
</p>
<p align = "center"> Fig.11: Model Performance </p>

<p align="center">
  <img width="400" alt="image" src="https://user-images.githubusercontent.com/41327028/164944226-f49b4691-70f7-4046-a30e-78829cd56a15.png">
</p>
<p align = "center"> Fig.12: Confusion matrix for test images </p>

To overcome low validation and test accuracies, we employed deeper networks and image augmentation techniques. We trained ResNet 18 (again), ResNet34 and ResNet101 and experimented with batch sizes & number of epochs to improve the accuracies. After implementing image augmentation in the training images, we were able to overcome the ovefitting problem as shown in the accuracy curves. We also witnessed the peculiar case of double descent for the ResNet101 model, which is expected of very deep neural network architectures.

<p align="center">
<img width="452" alt="image" src="https://user-images.githubusercontent.com/29612754/165004026-7f505c30-4d0e-4f74-ad88-4e7fcc640bf0.png">
</p>
<p align = "center"> Fig.13: Model Performance comparison across models </p>

Also, the test accuracies have improved quite a lot after regularizing the models. The best test accuracy that we could get was **66.4%** with ResNet34, which places us in **top-5 on the leaderboard** of the associated Kaggle competition. The following table summarizes the model performances.

| Model | Training Accuracy | Validation Accuracy  | Test Accuracy | 
| ------------- | ------------- | ------------- | ------------- |
| ResNet18 (No Image Augmentation)  | 99.8%  | 62.7%  | 61.1%  |
| ResNet18 (Image Augmentation)  | 75.9%  | 67%  | 65.7%  |
| ResNet34 (Image Augmentation) | 72.8%  | 66.48%  | 66.4% |
| ResNet101 (Image Augmentation)  | 71.97%  | 65.61%  | 65.28%  |

We generated the confusion matrix of the test dataset again with all the new models. We found similar confusion pairs amongst different emotion classes as the base ResNet18 model, the number of instances classified correctly to their true classes has significantly improved. 

<p align="center">
<img width="756" alt="image" src="https://user-images.githubusercontent.com/29612754/165005359-52790a0c-4a0b-4a51-a5cd-dfdae994ddda.png">
 </p>
 <p align = "center"> Fig.14: Comparison of confusion matrice for different models (test-set) </p>

#### ResNet18 evaluation using t-SNE

<b> t-SNE </b> is a nonlinear dimensionality reduction technique, well suited for embedding high dimension data into lower dimensional data (2D or 3D) for data visualization. After training our model (Resnet18 architecture), we collected the features from the last convolutional layer and visualized this feature vector for every image in the training dataset. This embedding feature vectors for the dataset is of the dimension N X 512. 

We get the following visualization for the training set embeddings - 
<p align = "center"> <img width="400" alt="image" src="https://user-images.githubusercontent.com/12424401/161621666-f802f29e-9155-4bfa-9602-c03dec5be01b.png"> </p>
<p align = "center"> Fig.15: t-SNE plot for training data embeddings</p>
Form the above visalization, we can observe that the current model is susceptible to confuse between 'Sad' and 'Fear' as there is not a lot of seperation between the two classes. We have also visualized the validation data -
<p align = "center"> <img width="400" alt="image" src="https://user-images.githubusercontent.com/12424401/161788108-92583917-87b7-4845-b793-d10db78b3ae0.png"> </p>
<p align = "center"> Fig.16: t-SNE plot for validation data embeddings</p>
We can clearly see that only 'Happy' and 'Surprise' classes have been clustered to some extent which is in line with our confusion matrix results. There is no clear separation between other classes. The model suffers from high variance and improving generalization will be our next step.

#### ResNet34 evaluation using t-SNE
For the final project submission we tried different models and ended up using Resnet34 with augmented images which gave us the best test accuracy of 66.4%. This is pretty good for reference as the highest accuracy in Kaggle leaderboard is 71%. For the above model we visualized the embeddings for the dataset using t-SNE. The embeddings are of the size N X 512.

<p align = "center"> <img width="400" alt="image" src="https://user-images.githubusercontent.com/12424401/165203749-d0adfe26-4c96-4deb-9c02-5def45bc06b7.png"> </p>
<p align = "center"> Fig.17: t-SNE plot for training data embeddings</p>

<p align = "center"> <img width="400" alt="image" src="https://user-images.githubusercontent.com/12424401/165204099-dcd30bf6-76e6-4aa3-82e9-c39f24915a76.png"> </p>
<p align = "center"> Fig.18: t-SNE plot for validation data embeddings</p>

We have similar results as Resnet18, where the model confused Fear(yellow) with Sad(green) as seen in the confusion matrix. We can infer the same thing from the t-SNE visualization of training data where the data points of Fear(yellow) are spread out especially near the Sad(green) datapoints. We see similar visualizations(as Resnet18) as both the models are Resnet based and both have similar accuracies.

#### Why Poor Test Accuracies?
This dataset has been found to have low test accuracies, with the top leaderboard accuracy being just 71%. We decided to explore further and understand what was limiting the test accuracies to the relatively low values. We exploited the last layer's class wise scores for all the mis-classified images in the test-set. We sorted (decreasing order) the mis-classified images based on the values of wrong class scores. For example, a "happy" image mis-classified as "neutral" having the neutral class score=0.7 would be placed before a "sad" image mis-classified as "surprised", with the "surprised" score being 0.6. This analysis gave us an idea of the "worst" mis-classified images. To our surprise, we discovered that most of these mis-classifications were happening because of wrong labeling of the test-set and not because the model was confusing between two classes. We show here top 24 images with decreasing order of mis-classification. For example, the first four images are definitely in the "happy" class as a human would see it, which the model learns as well. However, they seem to be having wrong labels as neutral, neutral, sad and neutral.

<p align="center">
<img width="1231" alt="image" src="https://user-images.githubusercontent.com/29612754/165006124-a9a40025-863d-4b7d-85ab-5b17425c446b.png">
 </p>
 <p align = "center"> Fig.19: The worst mis-classified images tend to be wrongly labeled.</p>

## Conclusion

We obtained a maximum accuracy of 66.4% on the Kaggle private test set with ResNet34, which places us in top-5 on the leaderboard of the associated Kaggle competition. This model also has a ~93% top-3-accuracy on the validation set.

ResNet34 model also tends to predict images labelled "fear" as "sad".

This dataset has been found to have low test accuracies. Upon investigation, we found that some of the mis-classified images in the test set were wrongly labeled. As our model predicted them correctly, we obtained relatively low accuracy score. 

Challenges in emotion recognition:
* Different people can interpret emotions in different ways, and hence the training data may not be 100% reliable. 
* It is very tough to detect all possible cues for an emotion, and some cues can be common among different emotions. e.g., Visual cues like furrowed eyebrows can mean something aside from anger, and other non-so-obvious facial cues may be subtle hints of anger. 
* Bias based on race, gender, and age. 

## References

1) [Huang, Gao, Zhuang Liu, Laurens Van Der Maaten, and Kilian Q. Weinberger. "Densely connected convolutional networks." In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 4700-4708. 2017](https://arxiv.org/abs/1608.06993)

2) [He, Kaiming, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. "Deep residual learning for image recognition." In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 770-778. 2016](https://ieeexplore.ieee.org/document/7780459)

3) [Wafa Mellouk, Wahida Handouzi. “Facial emotion recognition using deep learning: review and insights.” Procedia Computer Science, Volume 175, pp. 689-694, 2020](https://www.sciencedirect.com/science/article/pii/S1877050920318019)
