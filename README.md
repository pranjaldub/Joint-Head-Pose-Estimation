# Joint-Head-Pose-Estimation
Facial keypoints prediction and head pose estimation using MTCNN , Face-alignment and Dlib on masked/unmasked faces with 5 facial key points on masked faces.




<h2>Abstract</h2>—Numerous Governing authorities/organizations expect people to utilize the services only if they wear masks, effectively masking both their nose and mouth, according to the rules from the World Health Organization (WHO). Manual screening and distinguishing proof of individuals following/not following this arrangement is an enormous assignment in public places.Keeping in mind these challenges, the ideal methodology is to utilize innovations in Artificial Intelligence and Deep Learning; to be utilized as to make this undertaking straightforward, which is anything but difficult to utilize and robotized. In this paper, we propose "DeepFaceMask", which is a high-precision and efficient face mask classifier. The presented DeepFaceMask is a one-stage identifier, which consists of a Deep Convolutional Neural Network (DCNN) to combine significant level semantic data with different element/feature maps. Other than this, we additionally investigate the chance of actualizing DeepFace-Mask with a light-weighted neural organization MobileNet for cell phones. MTCNN, utilizes the inalienable connection among's recognition and alignment to help boost their performance. Specifically, our frame work uses a cascaded architecture with three phases of diligently planned DCNN to predict the face and its key points or landmarks in a coarse-to-fine way.
Keywords: Computer Vision, Face detection, Image Recognition, Image Classification, Object Detection and  Deep Learning Algorithm.
<br>
<h2>I. INTRODUCTION  </h2>
To viably stop the spread of COVID-19 pandemic, everyone is required to wear a mask in public places.This nearly makes regular facial recognition techniques ineffective, for example, public access control, face access control, facial recognition, facial security checks at train stations, and so forth.. The science around the utilization of masks by the overall population to prevent COVID-19 transmission is progressing quickly. Policymakers need guidance on how masks should be utilized by everybody to battle the 
recognition is to recognize a specific class of objects, for example face. Uses of object and face recognition can be found in numerous territories, for example, self driving vehicles,education,surveillance, etc. Customary object locators are based on handmade feature extractors. 

<h2>II. PROBLEM STATEMENT</h2> 
The objective of this project is to prepare 'Object Detection Models' fit for distinguishing facial keypoints for 'Face Recognition' and 'Attention Detection' and the location of Masked and Unmasked faces in static as well as moving(video) images. The detection technique should be robust to the occlusion present in the images for better predictability Preferably, they should be sufficiently quick to function admirably for certifiable programs would like to zero in on in our future executions. 
<br>
<h2>III. VISION </h2>
This undertaking was made with the vision of building up a "Real-Time Mask Detection System'' accessible for public use, to help general wellbeing authorities and little to huge foundations everywhere on the world viably battle this COVID19 pandemic. We trust that the models created here by the little exploration AI/ML people group empower engineers around the planet to have the option to utilize and convey the equivalent to construct systems that would be fit for withstanding the requests of a real-time, real-world use case. Specifically, it would assist manufacturing plants with guaranteeing mask consistence is followed, help guarantee security for guests in control zones or public spots where it is vital for such measures to be taken, etc. The applications are endless and are of earnest need in this crucial time. 
<br>
<h2>IV. DATASETS </h2>
COVID-19 pandemic. Furthermore, masks should be worn effectively on the face with the end goal that it masks the 
The dataset we will be using primarily is the MaskPascalVOC zip file taken from the website: 
nose and mouth totally, which is frequently not being followed. Consequently, it is dire to improve the recognition capabilities of the current face/mask recognition technology. Face mask identification alludes to distinguish if an individual is using mask and amount of area covered, which 
https://makeml.app/datasets/mask The dataset contains 853 images of the following classes: With mask, Without mask, and Mask weared incorrect. It is labeled with bounding box annotations for object detection. But the number of images 
we identify by including the facial keypoints too. The issue is firmly identified with general object identification to distinguish the classes of items (here we manage primarily 
belonging to the class of mask worn incorrectly are too less in quantity compared to the other two classes in the dataset, which was creating class imbalance so, we collected data 
three classes specifically: wearing mask accurately, wearing 
from additional sources having the class name as None for 
mask erroneously, and not wearing mask) and face 
people not wearing masks correctly, which we have
combined separately and uploaded on the web, So, finally our combined dataset overall has the following four labels namely: with_mask, without_mask, mask_weared_incorrect and none. This is divided finally into 3 classes, first one having the label “with_mask” which we signify later by a green colour bounding box on the face with a text label over it as “Correctly Masked”, second having the label “without_mask” which we signify later by a red colour bounding box on the face with a text label over it as “Unmasked”, and the third one having either of the two labels, “mask_weared_incorrect” or “none” which we signify later by a blue colour bounding box on the face with a text label over it as “Incorrectly Masked''. . 3 Additionally, we are also implementing the main facial keypoints inside the bounding box while detecting the face and the dataset used to train this model is taken from the website: https://ibug.doc.ic.ac.uk/download/annotations/ xm2vts.zip/ The data is in the format of a CSV (Comma Separated Values) file where there are sixty- eight key points of images representing x, y coordinates. This data is being fed into a deep CNN or ConvNet model with the final layer having 68*2=136 dimensions output predicting the X and Y coordinates of those sixty eight key points. Smooth L1 loss and MSE (Mean Squared Error) loss metrics resulted in the best accuracy outputs, we choose Smooth L1 
loss metric for our final model as it performed better in real-time comparatively. 



<h2>V. RELATED WORK</h2> 
A. OBJECT DETECTION 
The face detection technique used here is MTCNN (Multi-task Cascaded Convolutional Networks). Humanface classification and arrangement in unconstrained climate Ongoing investigations show that profound learning approaches can accomplish great execution on these two errands. In this paper, we have utilized a Deep Cascaded perform various tasks system which abuses the inalienable relationship among discovery and arrangement to help up their exhibition. Specifically, this casing work uses a fell engineering with three phases of painstakingly planned Deep Convolutional Neural Networks to anticipate face and milestone area in a coarse-to-fine way. What's more, it proposes another online hard example mining technique that further improves the presentation practically speaking.. 

Real Time analysis of MTCN ( its workflow which is being followed by all the three sequential models which are the P , R and O model respectively) :

N-face and keypoints detection:		MTCNN is a technique comprising of three stages, 								which can predict basic facial keypoints and perform basic face 
alignment . To avoid detection errors , it uses a technique 
called Non Max Suppression . 

The MTCNN framework / Architecture uses three separate 
networks: 
● “P” – Network 
● “R” – Network 
● “O” – Network 

• Structure of P-Net: 
P-Net predicts bounding box using sliding a 12*12 size 
kernel/filter across the image.







• Structure of R-Net: 
R-Net has similar structure, but uses more layer, thus predicting more accurate bounding box coordinates. 




• Structure of O-Net: 
O-Net takes the output of R-Net and predicts three sets of data namely - the probability of face being in the box, bounding box, and facial keypoints. 

B. IMAGE CLASSIFICATION 
Image classification refers to extracting specific desired features from a static or a real time image and classifying it to solve a specific problem at hand. This objective was accomplished by using a transfer learning approach. The ResNet-50 pre-trained model was used as a feature extractor connected with a custom fully connected layer for robust and efficient image classification. The model was trained on a dataset consisting three classes, masked, not masked, not properly masked respectively. The problem with the dataset was that it didn't represent the same amount of each class i.e. it was an imbalance of data, so the model was trained on two datasets combined. To achieve more robust results, custom image augment- ation techniques were implemented during the training process. The convolutional layers of ResNet-50 were used as feature extractor (last convolutional layers), rest all were frozen during training. Thus, fine tuning the model gave much better results from traditional state-of-the-art architectures. It also helped in tackling vanishing gradients problem by leveraging the use of skip connections and strong robust feature extractor proved to be efficient enough to extract features from a relatively small dataset. ResNet-50 layers were connected to linear layers before end-to-end result prediction.
Recently DNN community started experimenting with deeper networks because they were able to achieve high accuracy values. All in all, the underlying layers of the organization won't adapt successfully. Thus, profound organization preparing won't combine and precision will either begin to corrupt or immerse at a specific worth. In spite of the fact that the disappearing angle issue tended to utilizing the standardized instatement of loads, further organization exactness was as yet not expanding. Profound Residual Network is practically like the organizations which have convolution, pool-ing, activation and completely associated layers stacked one over the other. Skip connections used by ResNet-50. 
Key Features of ResNet: 
• Resnet utilizes the layer called Batch 
normalization which has a sole purpose of adjusting the input of the next layer hence increasing the performance. 
The problem of covariate shift is mitigated. 
• Resnet uses skip connection to overcome the gradient diminishing problems. 
C. HEAD POSE ESTIMATION : 
Alignment of any object suggests its general direction and position with respect to a camera. We can change the posture by either moving the thing regarding the camera, or the camera concerning the article. 
The posture estimation issue portrayed in this paper is often alluded to as Perspective-n-Point issue or PNP . In this issue the objective is to determine the inclination or posture of an article as for the camera , and we know the coordinates of n 3D points on the item and the corresponding 2D projections in the picture. 
Motions performed by a third dimensional rigid object : 1. Translation : Change in the pixel values such that there is a motion caused to the image in either x axis or the y axis. 
2. Rotation : In this type of movement the image is translated with respect to a single pivot point . 
So, estimating the pose of a 3D object means finding 6 numbers — three for translation and three for rotation. To calculate the 3D pose of an object in an image you need the following information 
1. 2D coordinates of a couple of points : You 
need the 2D (x,y) locations of a couple of 
points in the picture. For the situation of a 
face, you could pick the corners of the 
eyes, the tip of the nose, corners of the 
mouth and so on . 
2. 3D locations of the same points : We need 
the 3D coordinates of the 2D feature 
points. Primary 3d coordinates refer to : 
Nose tip , Chin , right corner of mouth , left 
corner of mouth , left eye , right eye. 
OpenCV solvePnP 
The capacity solvePnP and solvePnPRansac can be utilized to gauge pose. 
solvePnP actualizes a few calculations for pose estimation which can be chosen utilizing the boundary flag. As a matter of course it utilizes the check solve pnp iteration to true and its basically distributed ledger technology arrangement trailed by LM algorithm. Solve pnp p3p function utilizes just three focuses ascertaining the alignment and it must be utilized just when utilizing solve pnp pransac. 
<br>
<h2>VI. TRAINING </h2>
After preprocessing the data, our combined dataset consists a total number of 4198 images. Number of images labelled 1 i.e. wearing mask correctly are
3232, number of images labelled 2 i.e. not wearing mask are 717, number of images labelled 3 i.e. Wearing a mask incorrectly is 249. 
The dataset was then divided into training data, 	validation data and test data. It was split into 	8:1:1 ratio i.e. train set size is 3358, 	validation set size is 419 and test set size is 	421. The difference in the validation and test size, despite the same ratio, is because test set size was calculated after calculating the train set size and validation set size and their summation was subtracted from the total number of images. Also, images were randomly shuffled for no imbalance of class and robust performance of model, in batch size of 64 for faster computation. We trained the model using cross-entropy loss and Adam 
optimizer (an upgrade to stochastic gradient descent with momentum capabilities). In addition, the learning rate was set as 10-3 i.e. 0.001 and the number of epochs as 20, post this the model stopped learning based on earlier observations during training. 
Challenges faced during the training process was that single data source wasn’t enough to provide sufficient number of images belonging to each class. So, many data sources were considered and a robust, 
balanced, sufficiently large dataset was created that would provide enough data for the model to adapt to variances in data. GPU was used in training the model due to the large data. Training on GPU proved to be about 3x times faster than training on the CPU. GPU model used while training: NVIDIA GeForce GTX 1050 2GB GDDR5. Lighting and camera settings play a major role in model performance. Thus, we used MTCNN, which easily tackles such problems. Total params: 24,558,146 
Trainable params: 16,014,850 
Non-trainable params: 8,543,296 
<br>
<h2>VII. RESULTS</h2> 
The best model saved during training resulted in a validation loss of 0.9591 and validation accuracy of 0.9689 which was 



<h2>VIII. REAL-TIME APPLICATIONS:</h2> 
• Mall security checks / Super markets • Offices spaces / Schools 
• Hospitals 
• Mobile applications for alerts 

<h2>IX. FURTHER IMPLEMENTATIONS</h2>
It is evident that one of our biggest obstacles during the COVID-19 pandemic is to make sure people follow the safety regulations especially in public places for his/her own safety and the safety of others around. Our DeepFaceMask model will thus detect if people are wearing masks or not, correctly, when deployed to the CCTVs in the public places and can alert the admin as and when people are not wearing masks or wearing masks incorrectly. Additionally, it can be used in head pose estimation, attention detection in classrooms and lectures on masked faces, drowsiness detection on masked faces using facial keypoints tracking the driver’s eyes, and so on. 

<h2>X. REFERENCES</h2>  
[1] P. Viola and M. J. Jones, "Robust real-time face detection", Int. J. Comput. Vision, vol. 57, no. 2, pp. 137-154, May 2004. 
[2] Z. A. Memish, A. I. Zumla, R. F. Al-Hakeem, A. A. Al-Rabeeah, and G. M. Stephens, “Family cluster of middle east respiratory syndrome coronavirus infections,” New England Journal of Medicine, vol. 368, no. 26, pp. 2487–2494, 2013. 
[3] T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollár, “Focal loss for dense object detection,” 2017.
 [4] A. Shrivastava, A. Gupta, and R. Girshick, “Training region-based object detectors with online hard example mining,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2016, pp. 761–769.
[5] S. Ge, J. Li, Q. Ye, and Z. Luo, “Detecting masked faces in the wild with lle-cnns,” in Proceedings of the IEEE.

[6] A. Paszke, S. Gross, F. Massa, A. Lerer, J. Bradbury, G.Chanan, T. Killeen, Z. Lin, N. Gimelshein, L. Antiga et al.,“Pytorch: An imperative style, high-performance deep learning library,” in Advances in Neural Information Processing Systems, 2019, pp. 8024–8035.
[7] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual
learning for image recognition,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2016, pp. 770–778
[8] A. G. Howard, M. Zhu, B. Chen, D. Kalenichenko, W.Wang, T. Weyand, M. Andreetto, and H. Adam,
“Mobilenets: Efficient convolutional neural networks for mobile vision applications,” arXiv preprint arXiv:1704.04861, 2017.
[9] J. Deng, J. Guo, Y. Zhou, J. Yu, I. Kotsia, and
S.Zafeiriou, “Retinaface: Single-stage dense face localization in the wild,” arXiv preprint arXiv:1905.00641, 2019.
[10] R. Girshick, J. Donahue, T. Darrell, and J. Malik, “Rich feature hierarchies for accurate object detection and semantic segmentation,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2014, pp. 580–587.
[11] Krizhevsky A, Sutskever I, Hinton GE (2012)
ImageNet classification with deep convolutional neural
networks. Adv Neural Inf Process Syst 25.
[12] S. Yang, P. Luo, C.-C. Loy, and X. Tang, “Wider face:A face detection benchmark,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2016, pp. 5525–5533.
[13] R. Girshick, “Fast r-cnn,” in Proceedings of the IEEE international conference on computer vision, 2015, pp.1440–1448.
[14] J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L.
Fei-Fei, “Imagenet: A large-scale hierarchical image
database,” in 2009 IEEE conference on computer vision and pattern recognition. Ieee, 2009, pp. 248–255.
[15] J. Redmon, S. Divvala, R. Girshick, and A. Farhadi, “You only look once: Unified, real-time object detection,” in Proceedings of the IEEE conference on computer vision, 2016, pp. 779–788.
[16] A. Krizhevsky, I. Sutskever and G. E. Hinton,
"Imagenet classification with deep convolutional neural
networks", Advances in Neural Information Processing
Systems 25, pp. 1097-1105, 2012.
[17] K. Simonyan and A. Zisserman, "Very deep convolutional networks for large-scale image recognition",CoRR, 2014.
[18] C. Szegedy, W. Liu, Y. Jia, P. Sermanet, S. Reed, D.
Anguelov, et al., "Going deeper with convolutions", 2015.
[19] K. He, X. Zhang, S. Ren and J. Sun, "Deep residual
learning for image recognition", 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 770-778, 2016.
[20] P. Viola and M. J. Jones, "Robust real-time face
detection", Int. J. Comput. Vision, vol. 57, no. 2, pp.
137-154, May 2004



<h1>Detailed Analysis </h1>
Table of Contents


S.No	Particulars	Page No
1	Introduction	1-4
2	Requirements, feasibility and scope	5
3	Analysis, Activity time Schedule	6
4	Design	7
5	Implementation and testing	8-9
6	Limitations and future scope for Project	10
7	Conclusion	11
8	References	12

Introduction

Deep Learning
 Deep is an immensely increasing subset of Machine Learning, which tries to mimic the multiple parts of the human brain that sends signals to recognize stuff or making a conclusion based on sensor data, whereas in deep learning the models are trained to perform a specific task in certain way based on the models including multiple different layers working over different logical units on human brain over mathematical computation. A neural network takes in inputs, which are then processed in hidden layers using weights that are adjusted during training. Then the model spits out a prediction.






A single layer without any activation function is considered nothing more than a linear model of Machine Learning Algorithms, these activation are introduced to add more irregularity to input values for making it as far as possible from the Linear Models for better learning processes
Convolution Neural Network

A Convolutional Neural Network is a Deep Learning algorithm which can take in an input image, assign multiple learnable weights and biases to various aspects of the image and be able to differentiate one from the other after passing through multiple layers to output either the class of the image to reduce the input complexity for the model based on edge detection made through multiple ConvLayers of the model done through the matrix multiplication.

Working mechanism for CNN.


Application of CNN

1. Object Detection
The face detection technique used here is MTCNN
(Multi-task Cascaded Convolutional Networks). Face
detection and alignment in unconstrained environment
are challenging due to various poses, illuminations and
occlusions. Recent studies show that deep learning
approaches can achieve impressive performance on
these two tasks. In this paper, we have used a Deep
Cascaded multi-task framework which exploits the
inherent correlation between detection and alignment to
boost up their performance. In particular, this framework
leverages a cascaded architecture with three
stages of carefully designed Deep Convolutional Neural
Networks to predict face and landmark location in a
coarse-to-fine manner. In addition, it proposes a new
online hard sample mining strategy that further
improves the performance in practice. This method
achieves superior accuracy over the state-of-the-art
techniques on the challenging FDBB (Face Detection
Data Set and Benchmark) and Wider Face benchmarks
for face detection, and AFLW (Annotated Facial
Landmarks in the Wild) benchmark for face alignment,
while keeps real time performance.

Here we tried face detection using two networks :

1)Haar Cascade : First published by Paul Viola and Michael Jones in their 2001 paper, Rapid Object Detection using a Boosted Cascade of Simple Features, this original work has become one of the most cited papers in computer vision literature.

●In their paper, Viola and Jones propose an algorithm that is capable of detecting objects in images, regardless of their location and scale in an image. Furthermore, this algorithm can run in real-time, making it possible to detect objects in video streams.

●Specifically, Viola and Jones focus on detecting faces in images. Still, the framework can be used to train detectors for arbitrary “objects,” such as cars, buildings, kitchen utensils, and even bananas.

●While the Viola-Jones framework certainly opened the door to object detection, it is now far surpassed by other methods, such as using Histogram of Oriented Gradients (HOG) + Linear SVM and deep learning. We need to respect this algorithm and at least have a high-level understanding of what’s going on underneath the hood.

●Though , this technique worked quite well but we face a major issue while testing it in realtime on unseen data . It was predicting multiple bounding box for a single face because internally it dont use non max suppression .

●The second major problem we faced was that it was utilizing max CPU , thus giving around 2 frames per second which can’t be considered for real time inference.




2)MTCNN : one of the more popular approaches is called the “Multi-Task Cascaded Convolutional Neural Network,” or MTCNN for short, described by Kaipeng Zhang, et al. in the 2016 paper titled “Joint Face Detection and Alignment Using Multitask Cascaded Convolutional Networks.”

●The MTCNN is popular because it achieved state-of-the-art results on a range of benchmark datasets, and because it is capable of also recognizing other facial features such as eyes and mouth, called landmark detection.

●The network uses a cascade structure with three networks; first the image is rescaled to a range of different sizes (called an image pyramid), then the first model (Proposal Network or P-Net) proposes candidate facial regions, the second model (Refine Network or R-Net) filters the bounding boxes, and the third model (Output Network or O-Net) 

●This network worked best in this case . It uses non max suppression behind the scenes and this predicts the most accurate bounding box.

●It gave 7 fps on CPU inference and using this did not make the cpu reach its bottleneck.



Visual representation of three architectures that are stack on top of each other to form MTCNN :


















Samples of face detection using MTCNN :



Multiple face detection using MTCNN:











2. Image Classification
Image classification refers to extracting specific
desired features from a static or a real time image and
classifying it to solve a specific problem at hand. This objective was accomplished by using transfer learning approach. ResNet-50 pre-trained model was used as a feature extractor connected with a custom fully connected layer for robust and efficient image
classification. The model was trained on a dataset
consisting three classes, masked, not masked, not
properly masked respectively. The problem with the
dataset was that it didn't represent the same amount
of each class i.e. it was an imbalance of data, so the
model was trained on two datasets combined. To
achieve more robust results, custom image augmentation
techniques were implemented during the training
process. The convolutional layers of ResNet-50 were
used as feature extractor (last convolutional layers),
rest all were frozen during training. Thus, fine tuning
the model gave much better results from traditional
state-of-the-art architectures. It also helped in tackling
vanishing gradients problem by leveraging the use of
skip connections and strong robust feature extractor
proved to be efficient enough to extract features from a
relatively small dataset. ResNet-50 layers were
connected to linear layers before end-to-end result
prediction.

Image classification category used in this project is multiclass cklassification where the output was ann array of three numbers represnenting the probability of each class .

The classes were :
Class 0 : Not masked
Class 1 : Masked
Class 2 : Incorrectly masked
Here we also tried two approaches :

1)State of the art  : We built our own neural network architecture in pytorch .
Due to the imbalanced small dataset and current gpu capacity , this model was did not performed as well as pretrained models . Pretrained models are trained on millions and billions of parameters thus they are more robust .Though it was able to achieve a validation accuracy of ~97 percent.

State of the art model architecture :


from torch import nn
import torch.nn.functional as F
class state(nn.Module):
    def __init__(self):
        super(state , self).__init__()
            
        self.cnn1 = nn.Conv2d(in_channels=3 , out_channels=8 , kernel_size = 3 , stride = 1 , padding  = 1)
        self.batchnorm1 = nn.BatchNorm2d(8)
        self.relu = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.cnn2 = nn.Conv2d(in_channels=8 , out_channels=16 , kernel_size=3 , stride = 1 , padding = 1)
        self.batchnorm2 = nn.BatchNorm2d(16)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(in_features=16*56*56 , out_features=4000)
        self.dropout = nn.Dropout(0.55)
        self.fc2 = nn.Linear(in_features=4000 , out_features=2000)
        self.dropout = nn.Dropout(0.55)
        self.fc3 = nn.Linear(in_features=2000 , out_features=512)
        self.dropout = nn.Dropout(0.45)
        self.fc4 = nn.Linear(in_features=512 , out_features=2)
        self.final_act = nn.LogSoftmax(dim=1)
    def forward(self , x):
        out = self.cnn1(x)
        out = self.batchnorm1(out)
        out = self.relu(out)
        out = self.maxpool1(out)
        out = self.cnn2(out)
        out = self.batchnorm2(out)
        out = self.relu(out)
        out = self.maxpool2(out)
      #  print(out.shape)
        out = out.view(-1,16*56*56)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc4(out)
        out = self.final_act(out)
        return out



2)Resnet-50 : This was the second approach we tried for an image classification task .
This gave an accuracy of 98% thus outperforming the state of the art model . 
ResNet-50 is a convolutional neural network that is 50 layers deep. You can load a pre-trained version of the network trained on more than a million images from the ImageNet database . The pretrained network can classify images into 1000 object categories, such as keyboard, mouse, pencil, and many animals. As a result, the network has learned rich feature representations for a wide range of images. The network has an image input size of 224-by-224.
we also tried variants of resnet :
Resnet-18 and resnet-34: the former consists of 18 layers and the latter has 34 layers . Having less layers with respect to imbalance and small data resulted in underfitting and inaccurate reaktime inference .
Resnet-101 and resnet 152 : The former consist of 101 layers and the layer consists of 152 layers. The networks were too deep for a small dataset and we experienced a little gradient diminishing problem and at some time the models were stuck at local minima and the loss was not decreasing anymore .We got an outstanding accuracy on train data but it was surely overfilled as it performed worse of validation data / real time inference.

Resnet-50 : This was the best architecture which was quite balanced and well suited for the dataset .It also tackled the gradient diminishing problem smoothly when compared to all its variants.



Resnet-50 tackles the problem of diminishing gradients by using skip connections:
The core idea is to backpropagate through the identity function, by just using a vector addition. Then the gradient would simply be multiplied by one and its value will be maintained in the earlier layers. This is the main idea behind Residual Networks (ResNets): they stack these skip residual blocks together. We use an identity function to preserve the gradient.


Implementation of Resnet-50  architecture :

# Loading the pre-trained ResNet-50 model for image classification on cropped images
model = models.resnet50(pretrained=True)
for layer, param in model.named_parameters():
    print(layer)
    if 'layer4' not in layer:
# setting requires grad = False, as we do not want to backpropagate and change the weights and gradients,
# we will freeze this layer for feature extraction and connect it to our fully connected trainable layers
        param.requires_grad = False 
# Adding our fully connected layer to the pre-trained ResNet-50 block
model.fc = torch.nn.Sequential(torch.nn.Linear(2048, 512),
                               torch.nn.ReLU(),
                               torch.nn.Dropout(0.2),
                               torch.nn.Linear(512, 3),
                               torch.nn.LogSoftmax(dim=1))


Here all the layers are pre trained so they are freezed , that means they will not be trained during the training process and will be used just for feature extraction and the layers below layer 4 (last convolutional  layer of resnet 50 ) will be trained . In pytorch ,  setting requiredGrad as False will disable the gradient calculation , hence the weights will not be altered  So basically we are using resnet 50 weights and layers for feature extraction and combining with our own model of fully connected layer for desired probabilistic output .



3.Head Pose Estimation : The main objective of this task is to find the relative orientation (and position) of the human’s head with respect to the camera.So, the head pose estimates can provide information on which direction the human head is facing. Despite the head pose estimation task may seem to be easily solved, achieving acceptable quality on it has become possible only with recent advances in Deep Learning.Challenging conditions like extreme pose, bad lighting, occlusions and other faces in the frame make it difficult for data scientists to detect and estimate head poses.
This task was done using OpenCV solvepnp function by giving the camera calibration parameters and facial keypoints as input . We require both 2D and 3D object points. The 2D object points are of (x,y) format and the 3D object points are of (x,y,z) type. For example, if OpenCV solvepnp is used for face estimation, we need 2D and 3D points for facial characteristics such as eyes, mouth, nose, and chin.
For facial keypoints or face landmarks we used MTCNN in case of unmasked face , where we got 5 facial keypoints . In total we need six facial keypoints to predict the pose or face alignment , so we used another pretrained network known as face-alignment which uses DLib under the hood to predict a total of 64 facial keypoints . From that 64 key points we took one keypoint representing the lower part of the face and added it to the MTCNN key points array ,  thus we got 6 facial keypoints .
In the case of the masked face  ,  the prediction was having large errors with respect to the ground truth as the face is hidden under the mask . So ,  it was the better option to predict as much less facial keypoints for the face as possible to avoid large summation of errors.S , in this case we used MTCNN as the main  keypoint predictor . Now, we got the 5 facial keypoints and left with a missing keypoint which represent the chin area .Here we used a small trick by calculating the midpoint using two key points on either side of the lips  and translating it down to negative x axis by some constant value .This is not the ideal solution but in case of masked faces it world quite well . There is much room for improvement but still we were able to predict pose on masked faces using only 5 key points up to some extent .


Facial keypoints prediction :


Head Pose estimation from facial keypoints :

Requirements, feasibility and Scope

Required Tools
●Python
●Pytorch API
●OpenCV
●Jupyter Notebook
●Dlib

Feasibility Study and Scope
A real life solution that can be performed by computer vision can be face attention  and drowsiness detection . In this pandemic , it's difficult to perform such tasks while the person is wearing a mask . So we have to apply a two step process instead of one step process . The problem statement is that , we have to detect whether a person is attentive while driving or is he feeling drowsy , so a proper warning should be displayed to avoid any accidents. The main problem is that earlier various models have been made but they were made to work on unmasked faces . 

So ,the approach should be as follows :
There will be several models having their respective tasks :

MODEL 1 :
This model will be made for detecting the presence of faces and masks on the faces . This model has great importance as we have to detect the face otherwise all other components in the pipeline wont work and it will be beneficial to know which type of data i.e. in  which category the input frame lies .

MODEL2 :
The second model will have the responsibility of predicting the facial landmarks or keypoints. This model will be trained on a dataset having various key points of face . Instead of categorical loss , it will have a regression loss at the end as its a regression problem .Apart from previous face key points prediction models , this will generate 5 key points , as the traditional six points keypoints are difficult to perform on masked faces.

MODEL 3:
The third model will be used to predict the face alignment angle or face pose estimation for the key points obtained from the above model . This is the most challenging task as many techniques come into play such as camera calibration , 2-dimension to 3-dimension points conversion and 3d pose estimation.

Implementation and Testing

Implementation Modules

I.Image Detection.py 
Face detection is one of the important tasks of object detection. Typically detection is the first stage of pattern recognition and identity authentication. In recent years, deep learning-based algorithms in object detection have grown rapidly. These algorithms can be generally divided into two categories, i.e., two-stage detector like Faster R-CNN and one-stage detector like MTCNN. Although MTCNN and its varieties are not so good as two-stage detectors in terms of accuracy, they outperform the counterparts by a large margin in speed. MTCNN performs well when facing normal size objects, but is incapable of detecting small objects. The accuracy decreases notably when dealing with objects that have large-scale changing like faces. Aimed to solve the detection problem of varying face scales, we propose a face detector named MTCNN-face based on MTCNNv3 to improve the performance for face detection. The present approach includes using anchor boxes more appropriate for face detection and a more precise regression loss function. The improved detector significantly increased accuracy while remaining fast detection speed. Experiments on the WIDER FACE and the FDDB datasets show that our improved algorithm outperforms MTCNN and its varieties.


II.Image Classification.py
Image classification is a complex process that may be affected by many factors. Because classification results are the basis for many environmental and socioeconomic applications, scientists and practitioners have made great efforts in developing advanced classification approaches and techniques for improving classification accuracy. Image classification is used in a lot in basic fields like medicine, education and security. Correct classification has vital importance, especially in medicine. Therefore, improved methods are needed in this field. The proposed deep CNNs are an often-used architecture for deep learning and have been widely used in computer vision and audio recognition. In the literature, different values of factors used for the CNNs are considered. From the results of the experiments on the CIFAR dataset, we argue that the network depth is of the first priority for improving the accuracy. It can not only improve the accuracy, but also achieve the same high accuracy with less complexity compared to increasing the network width.


Modules Interaction
In order to classify a set of data into different classes or categories, the relationship between the data and the classes into which they are classified must be well understood. Generally, classification is done by a computer, so, to achieve classification by a computer, the computer must be trained. Sometimes it never gets sufficient accuracy with the results obtained, so training is a key to the success of classification. To improve the classification accuracy, inspired by the ImageNet challenge, the proposed work considers classification of multiple images into the different categories (classes) with more accuracy in classification, reduction in cost and in a shorter time by applying parallelism using a deep neural network model.

The image classification problem requires determining the category (class) that an image belongs to. The problem is considerably complicated by the growth of categories' count, if several objects of different classes are present in the image and if the semantic class hierarchy is of interest, because an image can belong to several categories simultaneously. Fuzzy classes present another difficulty for probabilistic categories' assignment. Moreover, a combination of different classification approaches has shown to be helpful for the improvement of classification accuracy.
In our case the classes will be masked or unmasked images of the detected images from the above module.


Testing

Phase 1 testing:
In the initial stages each modules will be tested individually with different distribution of data to obtain the best analysis of testing TensorBoard will be integrated into the testing command, this will provide the best graphical representation so to obtain a detailed views over the faults into the working is either due to parameter or due to poor data optimization.

This phase will allow us to measure the performance for individual models used into the play for project performance, once a certain performance is measured into individuals it be moved to phase 2 of testing.


Phase 2 Testing:
Under phase 2 of testing the final input data and output data is to be prepared to make a performance measure after the modules interaction with each other, this measurement will be supporting for fine interaction between the modules implemented and tested in phase 1, if this result out to be fine then final phase will be tested or models will be fine tuned or outputs will be engineered to build a better predictions through the model.

Limitation and Future Scope.
Limitations:

I.Large amounts of approaches have been proposed for face detection. The early research on face detection mainly focused on the design of handcraft feature and used traditional machine learning algorithms to train effective classifiers for detection and recognition. Such approaches are limited in that the efficient feature design is complex and the detection accuracy is relatively low, but the model MTCNN model is better in performance but the time factor for detection depends upon the area covered and the number of face detected individually and then individually running over two steps:
1. Detection for masked faces.
2.Tracing Data points over the mask.
Under multiple model processing it may sometimes takes high time if many peoples are to be detected over the camera for a while time.

II.Datapoints detection could be erupted if two faces overlaps one another, in the image or if the camera angle is not good based on the data is trained over this could be due to different distribution encounter than the training dataset used for learning the model in different stages of the project.


Future Scope
There are number of aspects we are planning to work on shortly:
1.Current the model takes .32nsec to obtain a prediction over the speed of CPU. So, we could optimize the data size adding transfer learning to optimize the complexity of input data to the model and obtain a prediction better than before
2.The use of Machine Learning in the field of IOT device development is rising rapidly. Hence, we plan to port our model to their respective version of tensorflow lite.
3.As the distribution of real data is not yet tested we are planning to build it over the web development for testing using model implementation over tensorflow javascript of the model.
4.Future model improvement could be done using parameter tuning or replacing models with transfer learning.
Conclusion
The best model saved during training resulted in a
validation loss of 0.9591 and validation accuracy of
0.9689 which was used in testing in real-time. Also,
the model resulted in 98% accuracy on the test data.




Possible application areas where this Model can be deployed:
1.Mall security checks
2.Offices spaces
3.Super market entrance
4.Hospitals
5.Schools and parks
6.Mobile applications for alerts
