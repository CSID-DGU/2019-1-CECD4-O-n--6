# 2019-1-CECD4-O-n--6

## Table of Contents

- [About](#about)
- [Getting Started](#getting_started)
- [Usage](#usage)

## About <a name = "about"></a>

This project is about developing integrated system for deploying generative model. The project is composed of three components.
##
Server  - https://github.com/JuhunC/2019-1-CECD4-O-n--6_Server
## 
Model   - https://github.com/CSID-DGU/2019-1-CECD4-O-n--6
##
Android - https://github.com/JuhunC/CECD_Android

## Getting Started - Model<a name = "getting_started"></a>


These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See [deployment](#deployment) for notes on how to deploy the project on a live system.

### Prerequisites

Please use python 3.7 or higher for configuration. Also, make sure to install fowllowing prerequisite libraries for smooth run.

```
conda install pytorch
pip3 install tqdm
pip3 install pillow
pip3 install pandas
pip3 install tensorflow-gpu==1.13.2
pip3 install pyyaml
pip3 install git+https://github.com/JiahuiYu/neuralgym
pip3 install -U -r requirements.txt
```
Andorid Application IP setting.
```
* NetworkClient.java
    change BASE_URL!!!
* network_security_config.xml
    change <domain includeSubdomains="true">127.0.0.1</domain>
```

## Usage <a name = "usage"></a>

These are the pictures of using Android Application.
Depending on the GPU, the maximum size of image is limited. Our test is based on GTX 1070 GPU where 1024x1024 image is the maximum.
##
Click on 'Select Image' button for image selection.
##
![image_png](https://github.com/CSID-DGU/2019-1-CECD4-O-n--6/blob/master/usage_images/1.png?raw=true)
##
Press Upload Image for activation
##
![image_png](https://github.com/CSID-DGU/2019-1-CECD4-O-n--6/blob/master/usage_images/2.png?raw=true)
##
Choose Objects for Regeneration and click 'Regenerate' button.
##
![image_png](https://github.com/CSID-DGU/2019-1-CECD4-O-n--6/blob/master/usage_images/3.png?raw=true)
![image_png](https://github.com/CSID-DGU/2019-1-CECD4-O-n--6/blob/master/usage_images/4.png?raw=true)
##
The regenerated image is automatically saved onto your device and Enjoy!!
##
![image_png](https://github.com/CSID-DGU/2019-1-CECD4-O-n--6/blob/master/usage_images/5.png?raw=true)
