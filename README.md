# Distance-transform-skeletonization

## Introduction
  It is difficult to detect human bodies using a 2D image since human body has a complex structure and people wear clothes with different fit and texture. This project aims to create a skeletonized image of the human body to increase the accuracy of Convolutional Neural Networks (CNN) by using simpler inputs.
  
## Methods
### Image Processing
Background of the images were removed using the Python tool ”rembg” which uses U2-Net, an architecture designed for salient object detection (SOD). Result image is converted to a binary image by setting the background color as black and every pixel belonging to foreground object as white.
### Distance Transform
####Euclidean Distance Transform (EDT) 
EDT is applied to the binary image. EDT simply measures the distance of white pixels to the closest black (edge) pixel using the equation below:

$D_{Euclid} = \sqrt{(x_2-x_1)^2+(y_2-y_1)^2}$

#### Local Maximum Points
In the distance transform image, each pixel and neighboring pixels are inspected. If the value of center pixel is larger than all neightboring pixels, that pixel is marked as LMP.
#### Critical Points (CP)
Calculating the gradient of the distance transform (∆DT ) results in even thinner skeleton image with minimum values in vertical and horizontal parts of the skeleton. Finding the local minimum points with similar method to the LMP, and taking the LMP points where ∆DT is the local minimum gives the Critical Points of the skeleton.

### Classification
Image classification is done using a CNN model. Architecture of the model can be seen in the PDF file in the repository.

### Results
CNN model trained with original images had higher training accuracy than validation accuracy, while the model trained with skeleton images had these two values very close to each other (Higher training accuracy compared to the validation accuracy can indicate the risk of overfitting. Since the skeleton model had closer values, it can be less prone to this problem.

### Issues & Possible Solutions
Currently, the skeleton image obtained from the critical points is just some points scattered around the place where the skeleton should be. In future works, these points can be combined to create a complete stick figure. Angle and position data of the stick figure can be turned into a feature matrix and machine learning models other than CNN can be used to classify human images.

## References
[1] *Chen, H. S., Chen, H. T., Chen, Y. W., Lee, S. Y. (2006, October). Human action recognition using star skeleton. In Proceedings of the 4th ACM international workshop on Video surveillance and sensor networks (pp. 171-178).*<br>
[2] *Fujiyoshi, H., Lipton, A. J., Kanade, T. (2004). Real-time human motion analysis by image skeletonization. IEICE TRANSACTIONS on Information and Systems, 87(1), 113-120.<br>
[3] Schwarz, L. A., Mkhitaryan, A., Mateus, D., Navab, N. (2012). Human skeleton tracking from depth data using geodesic distances and optical flow. Image and Vision Computing, 30(3), 217-226*<br>
[4] *Pavlakos, G., Choutas, V., Ghorbani, N., Bolkart, T., Osman, A. A., Tzionas, D., Black, M. J. (2019). Expressive body capture: 3d hands, face, and body from a single image. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition (pp. 10975-
10985).*<br>
[5] *Ding, J., Wang, Y., Yu, L. (2010, March). Extraction of human body skeleton based on silhouette images. In 2010 Second International Workshop on Education Technology and Computer Science (Vol. 1, pp. 71-74). IEEE.*<br>
[6] *Niblack, C. W., Gibbons, P. B., Capson, D. W. (1992). Generating skeletons and centerlines from the distance transform. CVGIP: Graphical Models and image processing, 54(5), 420-437.<br>
[7] Yoga Poses Dataset. (2023). Retrieved 13 April 2023, from https://www.kaggle.com/datasets/niharika41298/yoga-poses-dataset*<br>
[8] *Qin, X., Zhang, Z., Huang, C., Dehghan, M., Zaiane, O., Jager- sand, M. (2020). U2-Net: Going deeper with nested U-structure for salient object detection. Pattern Recognition, 106, 107404. doi: 10.1016/j.patcog.2020.107404*<br>
[9] *Distance Transform of a Binary Image - MATLAB amp; Simulink. Available at: https://www.mathworks.com/help/images/distance-transform-of-a-binary-image.html (Accessed: April 13, 2023)*.
