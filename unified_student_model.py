import torch.nn as nn
from torch import cat
from torch import randn
from torch import Size
from copy import deepcopy


class UnifiedStudentModel(nn.Module):
    def __init__(self, num_students=2, filter_cluster_sizes=None, num_classes=10, mode="train", model=None,
                 split_index=None, teacher_filter_shape=Size([1, 1]), final_size=Size([1, 1])):
        """"
        Constructor for the unified student model class.

        Params
        ------
        - num_students: number of students in NoNN
        - filter_cluster_sizes: size of each cluster
        - num_classes: total number of classes/labels
        - mode: what mode the network is in, training, validation, testing
        - model: unified student model
        - split_index: where to split the architecture
        - teacher_filter_shape: shape of teacher's last layer
        - final_size: final size of student layer

        """
        super(UnifiedStudentModel, self).__init__()
        self.mode = mode
        self.has_pool = False
        self.teacher_filter_shape = teacher_filter_shape
        if filter_cluster_sizes is None:
            # Default size for students
            filter_cluster_sizes = [10, 10]

        self.num_students = num_students
        self.s_list = nn.ModuleList()

        s_model = deepcopy(model)
        y = randn(128, 3, 32, 32)
        _, input_for_conv1x1 = s_model.forward(y)

        for i in range(num_students):
            bridge = [nn.Conv2d(input_for_conv1x1.shape[1], filter_cluster_sizes[i], kernel_size=1, bias=False),
                      nn.BatchNorm2d(filter_cluster_sizes[i]), nn.ReLU(inplace=True)]
            if split_index is None:
                student = nn.Sequential(*(list(s_model.children()) + bridge))
            else:
                student = nn.Sequential(*(list(s_model.children())[:split_index] + bridge))

            self.s_list.append(student)

        aux = []
        for i in range(self.num_students):
            aux += [self.s_list[i](y)]
        self.before_avg = cat(aux, dim=1).shape

        if self.before_avg[-1] < self.teacher_filter_shape[-1]:
            scale = self.teacher_filter_shape[0] / self.before_avg[-1]
            up = nn.Upsample(scale_factor=scale, mode='bilinear')
            for i in range(num_students):
                self.s_list[i] = nn.Sequential(*(list(self.s_list[i].children())) + [up])
        elif self.before_avg[-1] > self.teacher_filter_shape[-1]:
            scale = int(self.before_avg[2:][0] - self.teacher_filter_shape[-1] + 1)
            up = nn.AvgPool2d([scale, scale], stride=1)
            for i in range(num_students):
                self.s_list[i] = nn.Sequential(*(list(self.s_list[i].children())) + [up])
        aux = []
        for i in range(self.num_students):
            aux += [self.s_list[i](y)]
        self.before_avg = cat(aux, dim=1).shape

        filters = 0
        for i in range(self.num_students):
            filters += filter_cluster_sizes[i]
        if self.before_avg[2:] == final_size:
            self.has_pool = False
            mul = self.before_avg[2] * self.before_avg[3]
            self.fc = nn.Linear(filters*mul, num_classes)
        else:
            self.has_pool = True
            self.pool = nn.AvgPool2d(self.before_avg[2:][0], stride=1)
            self.fc = nn.Linear(filters, num_classes)   # fc = linear = dense layer = fully connected (FC)

    def forward(self, x):
        """
        Function for flattening the unified student model and forming a fully connected layer

        Params
        ------
        - x: randomly generated tensor

        Returns
        -------
        - x: fully connected layer
        - layer_out: last convolutional layer
        """

        # Get list of students
        s_out = []
        for i in range(self.num_students):
            # s_out += [self.s_list[i](x)]
            s_out.append(self.s_list[i](x))

        # One dimension of all student partitions
        layer_out = cat(s_out, dim=1)

        # If has_pool is true, then max pool. Else, only flatten.
        if self.has_pool:
            x = self.pool(layer_out)
            x = x.view(x.size(0), -1)
        else:
            x = layer_out.view(layer_out.size(0), -1)
        x = self.fc(x)

        # If student mode is train
        if self.mode == "train":
            return x, layer_out
        elif self.mode == "deploy":
            return x
        else:
            print("ERROR: Wrong student model mode. Choose form [train, deploy]")
            exit(-1)
