### **Face Detector using OpenVINO™ Toolkit**
----
### **Introduction**

The repository contains the code that was used to create a live face detector using the OpenVINO™ Toolkit. The tool is largely focused around optimizing neural network inference and is open source. Developed by Intel®,  helps support fast inference across Intel® CPUs, GPUs, FPGAs and Neural Compute Stick with a common API. 

OpenVINO™ can take models built with multiple different frameworks, like TensorFlow or Caffe, and use its Model Optimizer to optimize for inference. This optimized model can then be used with the Inference Engine, which helps speed inference on the related hardware. It also has a wide variety of Pre-Trained Models already put through Model Optimizer.

The model used for performing inference in this project is [**face-detection-adas-0001**](https://docs.openvinotoolkit.org/2018_R5/_docs_Transportation_object_detection_face_pruned_mobilenet_reduced_ssd_shared_weights_caffe_desc_face_detection_adas_0001.html )



![demo](https://github.com/lucastakara/Face_Detector/blob/master/Images/test.png?raw=true)


### **Getting Started**

The following items are required in order to visualize the project:

#### **Prerequisites / Dependencies:**

- OpenVino Toolkit

- Python 3

- CV2 library

- Numpy


### **Instructions**
---------------
- Download / clone the repository:

`$ git clone https://github.com/lucastakara/Face_Detector.git`

- Open terminal and go to the Face Detector folder

- Run the face_detector.py

`$ python3 -m face-detection-adas-0001`


### **Explanation for face_detection.py parameters**
----

m_desc = "The location of the model XML file" (**Required**)

d_desc = "The device name, if not CPU" (Optional)

c_desc = "The color of the bounding boxes to draw. RED, GREEN or BLUE" (Optional)

ct_desc = "The confidence threshold to use with the bounding boxes" (Optional)

t_desc = "Text which you may want to write above the bounding boxes" (Optional)


### Prequisites Installation
---
The following links contain the instructions for the installion of the required tools.

- **OpenVino Tool Kit:**

1. [Linux](https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_linux.html)
2. [OSX](https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_macos.html)
3. [Windows 10](https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_windows.html)
4. [Raspbian* OS](https://docs.openvinotoolkit.org/latest/_docs_install_guides_installing_openvino_raspbian.html)

- [OpenCV](https://pypi.org/project/opencv-python/)

- [Python 3](https://www.python.org/downloads/)

- [Numpy](https://pypi.org/project/numpy/)

### How It Works
----
1. The application reads command-line parameters and loads up to five networks depending on -m... options family to the Inference Engine.

2. The application gets a frame from the OpenCV VideoCapture.

3. The application performs inference on the Face Detection network.

4. The application performs four simultaneous inferences, Head Pose, Facial Landmarks detection networks if they are specified in the command line.

5. The application displays the results.


