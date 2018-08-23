# MNIST_withTensorRT
Fast inference on GPU with NVIDIA TensorRT (MNIST handwritten digits dataset)


The file "new_cnn.py" creates and train a Convolutional Neural Network with TensorFlow (with GPU support), saves the model as a
.meta file with a .data file, to store both the variables and the weights after the training phase. The folders
Training_FileWriter and Validation_FileWriter include the event files. They are obtained in order to check the trend of both
the loss function and the accuracy with TensorBoard, along with the graph.

The next step is to freeze the graph, which means to merge the variables and the weights in a single .pb file, storing the full
model. This can be done by "freeze.py".

Since TensoRT C++ API cannot import .pb files, a conversion is needed for the file obtained in the previous step.
You need to run the command as described in "ConvertToUff_forTensorRT.txt" file, on the shell, to obtain a .uff file (Universal
Framework Format), which can be imported and optimized by TensorRT.

Finally, the file "inferenceTRT.cpp" is a C++ code which imports the model in the .uff format, optimizes it, and the uses it to
make fast inference on GPU on a single image read by means of a function defined in TensorRT "common.h" library. The
instructions to compile the program are described in the file "compile.txt".


The project runs has been developped in a virtual environment on vinavx2 machine at CERN.
Reference for TensorFlow: https://www.tensorflow.org/install/install_linux
Reference for TensorRT: https://docs.nvidia.com/deeplearning/sdk/tensorrt-install-guide/index.html
                        https://docs.nvidia.com/deeplearning/sdk/tensorrt-developer-guide/index.html

TensorFlow version: tensorflow-gpu 1.10.0
TensorRT version: tensorrt 4.0.1.6
Cuda version: 9.0.176



