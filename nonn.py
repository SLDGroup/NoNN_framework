from teacher import Teacher
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
import copy
import json
from models.CIFAR10_models.wide_resnet import Wide_ResNet
from unified_student_model import UnifiedStudentModel
from dataset import dataset
from student import Student
import pickle


class NoNN:
    def __init__(self, config_file=""):

        if config_file == "":
            config_file = "configs/twrn_swrn.config"

        # Read config file
        with open(config_file) as cfg:
            configs = json.load(cfg)

        # Load configs
        self.dataset_config = configs["dataset_configs"]
        self.base_configs = configs["base"]
        self.teacher_configs = configs["teacher"]
        self.student_configs = configs["student"]

        self.data_loaders, self.dataset_sizes, self.num_classes = \
            dataset.prepare_datasets(data_dir=self.dataset_config["dataset_root"],
                                     train_val_batch_size=self.dataset_config["train_val_batch_size"],
                                     test_batch_size=self.dataset_config["test_batch_size"],
                                     val_split=self.dataset_config["val_split"],
                                     num_workers=self.dataset_config["num_workers"],
                                     download_train=self.dataset_config["download_train"],
                                     download_val=self.dataset_config["download_val"],
                                     download_test=self.dataset_config["download_test"])

    def create_teacher(self, model=None, loss=None, optimizer=None, scheduler=None,
                       print_model=False, print_params=False):

        if model is None:
            self.t_model = Wide_ResNet(depth=16, widen_factor=4, dropout=0, num_classes=10)
        else:
            self.t_model = model
        _, self.t_filter_shape = self.t_model(torch.randn(128, 3, 32, 32))

        if loss is None:
            self.t_loss_function = nn.CrossEntropyLoss()
        else:
            self.t_loss_function = loss

        if optimizer is None:
            optimizer_configs = self.teacher_configs["optim_config"]
            self.t_optimizer = optim.SGD(self.t_model.parameters(),
                                         lr=optimizer_configs["lr"],
                                         weight_decay=optimizer_configs["weight_decay"],
                                         momentum=optimizer_configs["momentum"])
        else:
            self.t_optimizer = optimizer

        if scheduler is None:
            scheduler_configs = self.teacher_configs["lr_scheduler"]
            self.t_exp_lr_scheduler = lr_scheduler.StepLR(optimizer=self.t_optimizer,
                                                          step_size=scheduler_configs["step_size"],
                                                          gamma=scheduler_configs["gamma"])
        else:
            self.t_exp_lr_scheduler = scheduler

        self.teacher = Teacher(teacher_configs=self.teacher_configs,
                               base_configs=self.base_configs,
                               model=self.t_model,
                               loss_function=self.t_loss_function,
                               optimizer=self.t_optimizer,
                               lr_scheduler=self.t_exp_lr_scheduler,
                               dataset=(self.data_loaders, self.dataset_sizes, self.num_classes),
                               print_model=print_model,
                               print_params=print_params)

    def create_unified_student(self, model=None):
        num_students = self.base_configs["num_students"]

        # Handle Filter Activation Network
        self.activation_network_path = self.base_configs["activation_network_path"]
        filename = self.activation_network_path.split('/')[-1]
        path = self.activation_network_path[:-len(filename)]
        self.filename_cluster = "cluster_" + filename

        with open(path+self.filename_cluster, 'rb') as f:
            print(f"Reading filters from {path + self.filename_cluster}")
            self.filter_cluster, self.filter_cluster_sizes, *_ = pickle.load(f)

        for i in range(num_students):
            print('Student {}: {} Channels'.format(i, self.filter_cluster_sizes[i]))

        if model is None:
            self.one_model = Wide_ResNet(depth=16, widen_factor=1, dropout=0, num_classes=10)
            self.split_index = -2
            self.mode = "train"
        else:
            self.one_model = model
            self.split_index = self.student_configs["split_index"]
            self.mode = self.student_configs["mode"]

        self.s_model = UnifiedStudentModel(num_students=num_students,
                                           filter_cluster_sizes=self.filter_cluster_sizes,
                                           num_classes=self.num_classes,
                                           mode=self.mode,
                                           model=self.one_model,
                                           split_index=self.split_index,
                                           teacher_filter_shape=self.t_filter_shape.shape[2:])
        return self.s_model

    def create_student(self, optimizer=None, scheduler=None):

        if self.s_model is None:
            # Handle Filter Activation Network
            self.activation_network_path = self.base_configs["activation_network_path"]
            filename = self.activation_network_path.split('/')[-1]
            path = self.activation_network_path[:-len(filename)]
            self.filename_cluster = "cluster_" + filename

            with open(path + self.filename_cluster, 'rb') as f:
                print(f"Reading filters from {path + self.filename_cluster}")
                self.filter_cluster, self.filter_cluster_sizes, *_ = pickle.load(f)

            for i in range(2):
                print('Student {}: {} Channels'.format(i, self.filter_cluster_sizes[i]))
            self.s_model = UnifiedStudentModel(num_students=2,
                                               filter_cluster_sizes=self.filter_cluster_sizes,
                                               num_classes=10,
                                               mode="train",
                                               model=Wide_ResNet(depth=16, widen_factor=1, dropout=0, num_classes=10),
                                               split_index=-2,
                                               teacher_filter_shape=self.t_filter_shape.shape[2:])

        if optimizer is None:
            s_optim_configs = self.student_configs["optim_config"]
            self.s_optimizer = optim.SGD(self.s_model.parameters(),
                                         lr=s_optim_configs["lr"],
                                         weight_decay=s_optim_configs["weight_decay"],
                                         momentum=s_optim_configs["momentum"])
        else:
            self.s_optimizer = optimizer

        if scheduler is None:
            s_scheduler_configs = self.student_configs["lr_scheduler"]
            self.s_exp_lr_scheduler = lr_scheduler.StepLR(optimizer=self.s_optimizer,
                                                          step_size=s_scheduler_configs["step_size"],
                                                          gamma=s_scheduler_configs["gamma"])
        else:
            self.s_exp_lr_scheduler = scheduler

        self.student = Student(student_configs=self.student_configs,
                               base_configs=self.base_configs,
                               teacher_model=copy.deepcopy(self.t_model),
                               student_model=self.s_model,
                               optimizer=self.s_optimizer,
                               lr_scheduler=self.s_exp_lr_scheduler,
                               dataset=(self.data_loaders, self.dataset_sizes, self.num_classes),
                               filter_cluster=self.filter_cluster,
                               print_model=True,
                               print_params=True)


if __name__ == "__main__":
    m = NoNN()
    m.create_teacher()
    m.create_student()
    m.student.train_model()

