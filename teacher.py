import torch
import torch.nn as nn
from filter_activation_network.activation_network import ActivationNetwork
import copy
from tqdm import tqdm
import time
import os
from torch import randn


class Teacher:
    """
    Class for the  teacher network
    """


    def __init__(self, teacher_configs, base_configs,
                 model=None, loss_function=None, optimizer=None, lr_scheduler=None,
                 dataset=None, print_model=False, print_params=False):
        """
        Constructor for teacher network. Loads configurations for the teacher network.
        Params
        ------
        - teacher_configs: teacher configurations
            - gpu_id: graphics processing unit ID
            - epochs: total number of epochs
            - parallel: whether parallel processing is used
            - resume: whether to resume training on an existing/previously trained model
            - mode: what mode the teacher is in: train, partition, or train_and_partition
            - log_file: csv log file for teacher model
        - base_configs: general NoNN configurations
            - num_students: number of students
            - activation_network_path: path directory to the filter activation network
            - teacher_model_path: path directory to the teacher model
            - num_classes: number of total classes
        - model: teacher model
        - loss_function: function to measure how far the prediction is from the true value
        - optimizer: function to optimize the loss function (minimize loss)
        - lr_scheduler: learning rate scheduler for scheduling the learning rate during epochs
        - dataset: training, validation, and testing data set
        - print_model: whether to print the teacher model
        - print_params: whether to print the parameters
       """

        # Load base configurations
        self.model_save_name = base_configs["teacher_model_path"].split('/')[-1]  # wrn_40_4.pt7
        self.model_save_path = base_configs["teacher_model_path"][:-len(self.model_save_name)]  # models/teacher/
        self.model_path = base_configs["teacher_model_path"]  # models/teacher/wrn_40_4.pt7
        self.activation_network_path = base_configs["activation_network_path"]
        self.num_students = base_configs["num_students"]

        # Load teacher configurations
        self.gpu_id = teacher_configs["gpu_id"]
        self.epochs = teacher_configs["epochs"]
        self.parallel = teacher_configs["parallel"]
        self.resume = teacher_configs["resume"]
        self.mode = teacher_configs["mode"]
        self.log_file = teacher_configs["log_file"]

        # Handle dataset
        self.data_loaders, self.dataset_sizes, self.num_classes = dataset
        self.model = model

        # Manage the device
        if torch.cuda.device_count() > 1 and self.parallel:
            print(f"USING {torch.cuda.device_count()} GPUs!")
            self.model = nn.DataParallel(self.model)
        self.device = torch.device(f"cuda:{self.gpu_id}" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        # Print the model
        if print_model:
            print("======== MODEL INFO =========")
            print(self.model)
            print("=" * 40)

        # Print the number of parameters
        if print_params:
            n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f"TOTAL NUMBER OF PARAMETERS = {n_parameters}")
            print("-" * 40)

        # Handle loss
        self.loss_function = loss_function

        # Handle optimizer
        self.optimizer = optimizer

        # Handle LR Scheduler
        self.exp_lr_scheduler = lr_scheduler

        print("Loaded configs for teacher")

    def execute(self):
        """
        Function for executing the teacher network based on the
        teacher's mode: partition, train_and_partition, or train.
        """
        # Handle Filter Activation Network
        # Partition mode
        if self.mode == "partition":
            self.create_activation_network = True
            self.partition_activation_network = True
            self.eval_on_val_set = True
            print(f"Loading model from the file at {self.model_path}")
            self.model.load_state_dict(torch.load(self.model_path))
            self.partition_model()

        # Train and Partition mode
        elif self.mode == "train_and_partition":
            # Training part
            # Load or not the model from file
            if self.resume:
                print(f"Loading model from the file at {self.model_path}")
                self.model.load_state_dict(torch.load(self.model_path))
            self.train_model()
            # Partitioning part
            self.create_activation_network = True
            self.partition_activation_network = True
            self.eval_on_val_set = True
            print(f"Loading model from the file at {self.model_path}")
            self.model.load_state_dict(torch.load(self.model_path))
            self.partition_model()

        # Train mode
        elif self.mode == "train":
            # Load or not the model from file
            if self.resume:
                print(f"Loading model from the file at {self.model_path}")
                self.model.load_state_dict(torch.load(self.model_path))
            self.train_model()

    def train_model(self):
        """
        Function for training the teacher network.
        """
        since = time.time()
        best_acc = -1.0
        results = []
        print("Training model... ")
        for epoch in range(self.epochs):

            # Each epoch has a training and validation phase
            phase_list = ['train', 'val', 'test']

            # Print learning rate at every epoch
            print('Epoch {}/{} learning_rate={}'.format(epoch, self.epochs - 1, [param_group["lr"] for param_group in
                                                                                 self.optimizer.param_groups]))
            print('-' * 80)
            lists = [epoch + 1]

            # Iterate through each phase: training, validation, and testing
            for phase in phase_list:
                if phase == 'train':
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0

                # Iterate over data.
                for images, labels in tqdm(self.data_loaders[phase]):
                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    # Zero the parameter gradients
                    self.optimizer.zero_grad()

                    # Forward
                    with torch.set_grad_enabled(phase == 'train' or phase == 'val'):
                        outputs, _ = self.model(images)
                        _, preds = torch.max(outputs, 1)
                        loss = self.loss_function(outputs, labels)

                        # Backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                    # Statistics
                    running_loss += loss.item() * images.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / self.dataset_sizes[phase]
                epoch_acc = float(running_corrects) / self.dataset_sizes[phase]

                # Print statistics: phase, epoch loss, and epoch accuracy
                print('{} Loss: {:.4f} Acc: {:.2f}'.format(phase, epoch_loss, epoch_acc * 100))
                lists += [phase, epoch_loss, epoch_acc]

                # Update "best_model" if current epoch accuracy is greater than best model accuracy
                # Deep copy the new best_model
                if phase == 'test' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    if self.parallel:
                        best_model = copy.deepcopy(self.model.module.state_dict())
                    else:
                        best_model = copy.deepcopy(self.model.state_dict())
                    print(f"Saving the best model with accuracy {best_acc * 100} on test set")
                    torch.save(best_model, os.path.join(self.model_save_path, self.model_save_name))
            results += [lists]
            self.exp_lr_scheduler.step()

            # If you want to always (every epoch) save your model please uncomment the next block
            # if phase == 'train':
            #     # Save the best val model
            #     # WARNING: Always save state_dict() even if trained with multiple GPUs which adds .module
            #     if self.parallel:
            #         torch.save(self.model.module.state_dict(),
            #                    os.path.join(self.model_save_path, self.model_save_name))
            #     else:
            #         torch.save(self.model.state_dict(), os.path.join(self.model_save_path, self.model_save_name))

        # Print training results
        with open(f"train/{self.log_file}", "w") as f:
            f.write(
                "Epoch;Training loss;Training accuracy;Validation loss;Validation accuracy;Test loss;Test accuracy;\n")
            for lst in results:
                epoch, _, tr_l, tr_a, _, val_l, val_a, _, test_l, test_a = lst
                f.write(f"{epoch};{tr_l};{tr_a};{val_l};{val_a};{test_l};{test_a};\n")
        print("Training finished!")
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    def partition_model(self):
        """
        Function for partitioning the teacher network.
        """

        print("Partitioning filter activation network...")

        # Set model to evaluate mode
        self.model.eval()

        with torch.set_grad_enabled(True):
            y = randn(128, 3, 32, 32).to(torch.device('cpu'))
            another = copy.deepcopy(self.model) # TODO: rename "another" with a better name (not sure whats happening here)
            another.to(torch.device('cpu'))
            _, final_conv = another.forward(y)

        # Get activation network
        act_net = ActivationNetwork(num_classes=self.num_classes,
                                    num_students=self.num_students,
                                    c_final_conv=final_conv.shape[1],
                                    percentile_value=99,
                                    top_k_creation=50)

        # Iterate over data
        for images, labels in tqdm(self.data_loaders['val']):
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Zero the parameter gradients
            self.optimizer.zero_grad()

            # Forward
            with torch.set_grad_enabled(True):
                outputs, act_net.interm_out_gr = self.model(images)
                _, preds = torch.max(outputs, 1)

                if self.create_activation_network:
                    act_net.labels = copy.deepcopy(labels)
                    act_net.preds = copy.deepcopy(preds)
            act_net.create_activation_network(batchSize=len(labels))
        act_net.dump_activation_network(self.activation_network_path)

        print("Partitioning filter activation network...")
        filename = self.activation_network_path.split('/')[-1]
        path = self.activation_network_path[:-len(filename)]
        act_net.partition_activation_network(filename=self.activation_network_path,
                                             filename_cluster=path + "cluster_" + filename)
        print("Partitioning done!")
