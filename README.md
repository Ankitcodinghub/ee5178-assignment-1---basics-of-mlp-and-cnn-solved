# ee5178-assignment-1---basics-of-mlp-and-cnn-solved
**TO GET THIS SOLUTION VISIT:** [EE5178 Assignment 1 – Basics of MLP and CNN Solved](https://www.ankitcodinghub.com/product/ee5178-modern-computer-vision-solved/)


---

📩 **If you need this solution or have special requests:** **Email:** ankitcoding@gmail.com  
📱 **WhatsApp:** +1 419 877 7882  
📄 **Get a quote instantly using this form:** [Ask Homework Questions](https://www.ankitcodinghub.com/services/ask-homework-questions/)

*We deliver fast, professional, and affordable academic help.*

---

<h2>Description</h2>



<div class="kk-star-ratings kksr-auto kksr-align-center kksr-valign-top" data-payload="{&quot;align&quot;:&quot;center&quot;,&quot;id&quot;:&quot;110004&quot;,&quot;slug&quot;:&quot;default&quot;,&quot;valign&quot;:&quot;top&quot;,&quot;ignore&quot;:&quot;&quot;,&quot;reference&quot;:&quot;auto&quot;,&quot;class&quot;:&quot;&quot;,&quot;count&quot;:&quot;1&quot;,&quot;legendonly&quot;:&quot;&quot;,&quot;readonly&quot;:&quot;&quot;,&quot;score&quot;:&quot;5&quot;,&quot;starsonly&quot;:&quot;&quot;,&quot;best&quot;:&quot;5&quot;,&quot;gap&quot;:&quot;4&quot;,&quot;greet&quot;:&quot;Rate this product&quot;,&quot;legend&quot;:&quot;5\/5 - (1 vote)&quot;,&quot;size&quot;:&quot;24&quot;,&quot;title&quot;:&quot;EE5178&nbsp;Assignment 1 - Basics of MLP and CNN Solved&quot;,&quot;width&quot;:&quot;138&quot;,&quot;_legend&quot;:&quot;{score}\/{best} - ({count} {votes})&quot;,&quot;font_factor&quot;:&quot;1.25&quot;}">

<div class="kksr-stars">

<div class="kksr-stars-inactive">
            <div class="kksr-star" data-star="1" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="2" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="3" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="4" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="5" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>

<div class="kksr-stars-active" style="width: 138px;">
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>
</div>


<div class="kksr-legend" style="font-size: 19.2px;">
            5/5 - (1 vote)    </div>
    </div>
&nbsp;

Notes:

1. Please use moodle discussion threads for posting your doubts.

2. Before posting any question, check if the same question has been asked earlier.

3. Submit a single zip file in the moodle named as PA1 Rollno.zip containing the report and folders containing corresponding codes.

4. Read the problem fully to understand the whole procedure.

5. Comment your code generously.

6. Put titles to all of the figures shown.

7. You are supposed to use Python and Pytorch for this assignment.

Dataset: The aim of this assignment is to implement an image classifier on the popular CIFAR-10 dataset. This dataset contains 60000 32 × 32 color images in 10 different classes, with 6000 images per class. The 10 different classes represent airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, and trucks. It is a subset of a larger 80 million tiny images dataset.

Note: Pytorch provides CIFAR-10 dataset in the torchvision.datasets module, which you can use directly to load the dataset.

1 MLP

For this part of the assignment, we will implement a simple MLP baseline for image classification on CIFAR-10 dataset.

1.1 Architecture and Training

Input to the network is an image (32×32×3) flattened as 3072 dimensional vector xi. Input is followed by layers h1(500), h2(250), h3(100). The number of units are mentioned beside the corresponding hidden layer. Use ReLU activation for all these hidden layers. The output unit consists of 10 units, one for each class. Let the output activation be linear. Since you want to do classification, use softmax to get probabilities of image belonging to 10 classes. This will be a 10 dimensional vector ˆyi. Now use cross-entropy loss between ˆyi and yi, where yi is the one hot representation of the ground-truth label.

1.2 Deliverables

1. For the experiment above, you are required to show the plot of training error, validationerror and prediction accuracy as the training progresses.

2. At the end of training, report the average prediction accuracy for the whole test set of10000 images.

3. You should also plot randomly selected test images showing the true class label as wellas predicted class label.

4. Report the confusion matrix that shows the kind of errors that your classifier makes.In this problem, your confusion matrix is a 10 × 10 matrix, where the rows represent the true label of a test sample and the columns represent the predicted labels of the classifier.

5. Use batch-normalization. Does it improve the test accuracy? Does it affect trainingtime?

2 CNN

For this part of the assignment, we will implement VGG11 architecture for image classification from scratch. The architecture of VGG11 is described in the below image. Note that max-pooling layers reduce the image/feature size by half. All the experiments should be performed on CIFAR-10 datasets. Please follow the link https://www.binarystudy.com/

Figure 1: VGG11 architecture https://arxiv.org/pdf/1409.1556.pdf

2.1 Experimental Settings

Since the dataset contains 10 classes, modify (or add an extra layer) the last layer of VGG11. You are free to experiment with different settings of learning rates(such as 0.001,0.0001) and batch size for the training and report the set of hyperparameters, which resulted in the best performance. Use CrossEntropy loss and SGD optimizer for all of the experiments.

Report the result with

1. Training setting (such as learning rate and training epoch used). Report this for allthe different settings that you experimented with.

2. Any bottleneck/challenges you faced during training or testing.

3. Training loss/accuracy plot (loss/accuracy versus epoch)

4. Report the accuracy of the test dataset.

5. Show visual results for 5 images for each category of test classes.

6. Now download at least 5 random images (or capture images from your smartphone,camera, or any imaging device) of appropriate classes and test those images on the trained network. What is your observation?
