# Visual-Inertial-Odometry
In this project we implement the pipeline for Visual Odometry (VO) from scratch on the Oxford dataset given, and compare it with the implementation using OpenCV built-in functions.


## DEPENDENCIES: 
- Python Version: 3.x
- Matplotlib.pyplot 
- pandas
- xlrd
- OpenCV 3.4 (For the purpose of using SIFT)

## Authors

- [Arjun Srinivasan](https://github.com/aarjunsrinivasan)
- [Arun Kumar](https://github.com/akdhandy)
- [Praveen](https://github.com/Praveen1098)

To run the codes:


1.)Run the python files in the current directory which contains all the codes.

2.)Two python scripts namely:

cvvom - which uses inbuilt CV functions to solve the Visual Odometry problem

myvom - which uses our own implementation of functions to solve. 

3.)Place the relative path of the images and the model in both the python scripts CV_VOM and myVOM and run:
pathimage="/home/arjun/Desktop/VOM/Oxford_dataset/stereo/centre/"

pathmodel="/home/arjun/Desktop/VOM/Oxford_dataset/model/".

In order to get more information visit the below GitHub link: 
https://github.com/akdhandy/Visual-Odometry-ENPM-673.git

