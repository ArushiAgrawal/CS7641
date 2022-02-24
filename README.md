
## Introduction/Background

Facial expression is the most universal, natural, and powerful signal for human beings to convey their thoughts. Emotions don’t have a concrete definition, yet they drive every other decision made in our lives. Market research has proven that predicting sentiments correctly can be a huge source of growth for businesses, as it could help to gauge customer mood towards their brand or product. 

A lot of research has been done in the area. *Lit review*

For our project, we are using Kaggle’s fer2013 dataset – an open-source dataset containing around 30 thousand 48x48 pixel grayscale images of faces.  This dataset was published during the International Conference on Machine Learning (ICML). The emotions in the images belong to the seven categories - anger, disgust, fear, happy, sad, surprise, and neutral.


## Problem definition
Recognizing facial emotions is important in various fields – marketing and advertising, surveillance and law enforcement, video game testing, safe and personalized cars, etc. 
We aim to create neural network models that could identify human emotions. 


## Methods

Nueral network models - 
* Simple 2-3 layer Convnet
* ResNet variants: 18, 34, 50 (most common), 101..., 
* DenseNet

Image Augmentation Augmentation, Learning Rate Finder/Scheduler, Transfer Learning, activations and kernel visualizations, Top-k Accuracy, confusion matrix, 
Evaluate most incorrectly predicted images: predicted vs true classes to analyze confusing emotions

## Potential results and Discussion

### Metrics to be considered - 

### Challenges in emotion recognition – 
* Different people can interpret emotions in different ways, and hence the training of the model (training data) can not be 100% accurate. 
* It’s very tough to detect all possible cues for an emotion, and some cues can be common among different emotions. Eg: Visual cues like furrowed eyebrows can mean something aside from anger, and other non-so-obvious facial cues may be subtle hints of anger.
* Bias based on race, gender, age, or color 

## References - 
* Huang, Gao, Zhuang Liu, Laurens Van Der Maaten, and Kilian Q. Weinberger. "Densely connected convolutional networks." In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 4700-4708. 2017.
* He, Kaiming, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. "Deep residual learning for image recognition." In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 770-778. 2016.
