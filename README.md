# Network of Neural Networks (NoNN) framework
This is a project with the NoNN model compression technique mentioned [Memory- and Communication-Aware Model Compression for Distributed Deep Learning Inference on IoT](https://arxiv.org/pdf/1907.11804.pdf)
which was publicated at the International Conference on Hardware/Software Codesign and System Synthesis (CODES+ISSS) in 2019.

The framework was created to facilitate flexibility and to help the Machine Learning community deploy efficiently models 
on edge devices for distributed inference. It was presented at the Low-Power Computer Vision workshop from CVPR 2020 as 
[A Hardware Prototype Targeting Distributed Deep Learning for On-device Inference](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w28/Farcas_A_Hardware_Prototype_Targeting_Distributed_Deep_Learning_for_On-Device_Inference_CVPRW_2020_paper.pdf)


The main workflow replicated by this framework is the following:
* A large teacher network desired for compression is selected and trained (if not already pre-trained).
* The teacher network is partitioned using network science and community detection to detect disjoint communities in teacher's
last convolutional layer feature maps.
* A student model is selected.
* The number of students, *N*, is selected and *N* students are concatenated with a Fully Connected layer appended at the end to create 
a trainable network from them, called Unified Student Model.
* The Unified Student Model is trained using Knowledge Distillation [[Bucila, C.](http://www.cs.cornell.edu/~caruana/compression.kdd06.pdf), 
[Hinton, G.](https://arxiv.org/pdf/1503.02531.pdf)]. Each student model learns the knowledge from one partition of the teacher.
* After training, each student model is deployed on an edge device, and the Fully Connected layer from the Unified Student Model
runs either locally or on another edge device. The system is either wired or wireless.

For more information about Knowledge Distillation check this [link](https://github.com/dkozlov/awesome-knowledge-distillation)

# How it works?

**WARNING!** Currently only works for 2 students (i.e. for 2 partitions of the teacher).

## Example 
Example with default values is available in `default_test.py` and contains the following:
```
# Instantiate NoNN class with default configs
nonn = NoNN(config_file="configs/default.config")

# Create the teacher
nonn.create_teacher()

# Execute the mode specified for the teacher (default = "train_and_partition")
nonn.teacher.execute()

# Create the unified student model
nonn.create_unified_student()

# Create the student
nonn.create_student()

# Train the student
nonn.student.train_model()

# Deploy student
nonn.student.deploy()
```

## NoNN class
The first step is to initialize your NoNN class after importing it. Usually, a configuration file (found inside the `configs` folder)
is used refactor all the important aspects of your network and easily provide multiple configurations for the used networks.

The configuration file **must have** the following content:
```
{
  "dataset_configs": {
    "dataset_root": "dataset/<YOUR DATASET FOLDER HERE>",
    "train_val_batch_size": <BATCH SIZE FOR TRAIN AND VAL>,
    "test_batch_size": <BATCH SIZE FOR TEST>,
    "val_split": <VALIDATION SPLIT>,
    "num_workers": <HOW MANY PROCESSES TO PREPARE THE DATASET>,
    "download_train": <true/false>,
    "download_val": <true/false>,
    "download_test": <true/false>
  },

  "base": {
    "num_students": <NUMBER OF STUDENTS USED>,
    "activation_network_path": "filter_activation_network/<ACTIVATION FILENAME>.pkl",
    "student_model_path": "models/student/<STUDENT WEIGHTS FILENAME>.pt7",
    "teacher_model_path": "models/teacher/<TEACHER WEIGHTS FILENAME>.pt7",
    "num_classes": <NUMBER OF CLASSES>,
    "deploy_ips": <LIST OF IP STRINGS CORRESPONDING TO EACH STUDENT>,
    "deploy_ports": <LIST OF PORT INTEGER CORRESPONDING TO EACH STUDENT>
  },

  "teacher": {
    "gpu_id": <GPU ON WHICH YOU WILL TRAIN THE TEACHER>,
    "epochs": <NUMBER OF EPOCHS>,
    "parallel": <TRAIN IN PARALLEL ON MULTIPLE GPUs? true/false>,
    "resume": <RESUME TRAINING FROM PREVIOUSLY SAVED FILE? true/false>,
    "mode": <EXECUTION MODE. CHOOSE FROM: "train", "partition" or "train_and_partition">,
    "log_file": "<FILENAME TO OUTPUT THE TRAINING ACCURACIES AND LOSSES OBTAINED>.csv"
  },

  "student": {
    "gpu_id": <GPU ON WHICH YOU WILL TRAIN THE STUDENT>,
    "epochs": <NUMBER OF EPOCHS>,
    "parallel": <TRAIN IN PARALLEL ON MULTIPLE GPUs? true/false>,
    "resume": <RESUME TRAINING FROM PREVIOUSLY SAVED FILE? true/false>,
    "alpha": <ALPHA PARAMETER FOR KNOWLEDGE DISTILLATION>,
    "temperature": <TEMPERATURE PARAMETER FOR KNOWLEDGE DISTILLATION>,
    "beta": <BETA PARAMETER FOR KNOWLEDGE DISTILLATION>,
    "mode": <CHOOSE FROM: "train" OR "deploy">,
    "log_file": "<FILENAME TO OUTPUT THE TRAINING ACCURACIES AND LOSSES OBTAINED>.csv",
    "inference_log_file": "<FILENAME TO OUTPUT THE INFERENCE TIMINGS>.csv"
  }
}
```
Your code should start with NoNN instance:
```
nonn = NoNN(config_file="configs/<YOUR CONFIG FILE>")
```

## Teacher Network
If you want to play with models, optimizers, loss functions and learning rate schedulers, the NoNN framework allows you to do that easily.
In the following example we use the flexibility of NoNN framework to tune our experiment.
```
my_teacher = VGG(depth=19, num_classes=10)
my_loss_function = nn.MSELoss()
my_optimizer = torch.optim.Adam(my_teacher.parameters())
my_lr_scheduler = torch.optim.lr_scheduler.StepLR(my_optimizer, step_size=5, gamma=0.9)

nonn.create_teacher(model=my_teacher, loss=my_loss_function, optimizer=my_optimizer, scheduler=my_lr_scheduler)
```

## Student network
We can do the same changes for the student as well, but we cannot modify the loss function, as Knowledge Distillation is the 
main technique for student learning.
We can change the student model that is replicated as many times as the `num_students` parameter from the config file suggests
and then we can change the optimizer and learning rate scheduler for the resulted unified student network that will actually be trained.
```
my_student = VGG(depth=11, num_classes=10)
my_unified_student = nonn.create_unified_student(model=my_student)
my_optimizer = torch.optim.Adam(my_unified_student.parameters())
my_lr_scheduler = torch.optim.lr_scheduler.StepLR(my_optimizer, step_size=5, gamma=0.9)

nonn.create_student(model=my_unified_student, optimizer=my_optimizer, scheduler=my_lr_scheduler)
```

# Preparing the environment Ubuntu 18.04
## Conda
* Go [here](https://www.anaconda.com/distribution/) and download Anaconda.sh
* `cd ~/Downloads/ ; bash Anaconda3-2020.02-Linux-x86_64.sh` (Or whatever the name of the downloaded file is
* **Restart terminal now**
* ```
  $ sudo apt-get update 
  $ sudo apt-get install -y python3 python3-dev python3-setuptools gcc libtinfo-dev zlib1g-dev build-essential cmake libedit-dev libxml2-dev
  ```
* ```
  $ conda create --name ml python=3.8
  $ conda activate ml
  $ conda install -c conda-forge tqdm python-louvain numpy matplotlib 
  $ conda install -c anaconda networkx
  ```
* ```
  $ conda create -n ml_pytorch --clone ml
  $ conda activate ml_pytorch
  $ conda install pytorch torchvision -c pytorch
  ```

## Installing TVM
* Make sure you have installed vim (or run `sudo apt install vim`)
* Go [here](https://releases.llvm.org/download.html) and download a **pre-built** LLVM version for your system. 
  We used LLVM 10.0.0.
* We installed the code in the home directory `~` but you can modify this if you want. You will need those files for 
  installation and later usage when running TVM applications, so make sure you install it somewhere safe.

  ```
  $ cd ~
  $ git clone --recursive https://github.com/apache/incubator-tvm.git tvm
  $ cd tvm/ 
  $ mkdir build
  $ cp cmake/config.cmake build
  $ cd build
  $ vim config.cmake
  ```
* Modify the opened file in vim by finding the line that starts as `set(USE_LLVM` and paste the following 
  `set(USE_LLVM /home/clang+llvm-9.0.0-x86_64-linux-gnu-ubuntu-18.04/bin/llvm-config)` 
Do not forget the `/bin/llvm-config` after inserting the name of the file previously downloaded. To exit vim press 
  **ESC**, type `:wq` and hit **ENTER**
* Build and compile tvm: 
  ```
  $ cmake .. 
  $ make -j8
  $ cd ../python/ 
  $ python setup.py install 
  $ cd ../nnvm/
  $ make -j4
  ```

**WARNING** Make sure to be in the PyTorch environment created in order to install TVM inside it


# Preparing RaspberryPi 3B+
* Properly install TVM Runtime on the device 
  ```
  $ cd ~ ;
  $ mkdir NoNN_framework
  $ cd NoNN_framework
  $ git clone --recursive https://github.com/apache/incubator-tvm.git tvm
  $ cd tvm
  $ mkdir build
  $ cp cmake/config.cmake build
  $ cd build
  $ cmake ..
  $ make runtime -j4
  ```
* Everytime you turn on the device make sure to execute the following: 

  ```
  $ export PYTHONPATH=$PYTHONPATH:~/NoNN_framework/tvm/python
  $ source ~/.bashrc
  ```
* Run the RPC server 
  ```
  $ python -m tvm.exec.rpc_server --host [Raspberry IP Address] --port=9090
  ```