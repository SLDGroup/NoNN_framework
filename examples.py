from nonn import NoNN
from models.CIFAR10_models.vggnet import VGG
from models.CIFAR10_models.wide_resnet import Wide_ResNet
from models.CIFAR10_models.mobilenet_v2 import MobileNetV2
import torch

if __name__ == "__main__":
    # Example 1
    nonn = NoNN("configs/twrn_swrn161.config")
    my_teacher = Wide_ResNet(depth=nonn.teacher_configs["depth"],
                             widen_factor=nonn.teacher_configs["widen_factor"],
                             dropout=nonn.teacher_configs["dropout"],
                             num_classes=nonn.base_configs["num_classes"])

    nonn.create_teacher(model=my_teacher)
    nonn.teacher.execute()
    my_student = Wide_ResNet(depth=nonn.student_configs["depth"],
                             widen_factor=nonn.student_configs["widen_factor"],
                             dropout=nonn.student_configs["dropout"],
                             num_classes=nonn.base_configs["num_classes"])
    nonn.create_unified_student(model=my_student)
    nonn.create_student()
    nonn.student.train_model()
    nonn.student.deploy()

    # Example 2
    # nonn = NoNN("configs/twrn_smbnv2.config")
    # my_teacher = Wide_ResNet(depth=nonn.teacher_configs["depth"],
    #                          widen_factor=nonn.teacher_configs["widen_factor"],
    #                          dropout=nonn.teacher_configs["dropout"],
    #                          num_classes=nonn.base_configs["num_classes"])
    # nonn.create_teacher(model=my_teacher)
    # my_student = MobileNetV2(num_classes=nonn.base_configs["num_classes"])
    # nonn.create_unified_student(model=my_student)
    # nonn.create_student()
    # nonn.student.train_model()
    # nonn.student.deploy()


    # Example 3
    # nonn = NoNN("configs/twrn_svgg.config")
    # my_teacher = Wide_ResNet(depth=nonn.teacher_configs["depth"],
    #                          widen_factor=nonn.teacher_configs["widen_factor"],
    #                          dropout=nonn.teacher_configs["dropout"],
    #                          num_classes=nonn.base_configs["num_classes"])
    # nonn.create_teacher(model=my_teacher)
    # my_student = VGG(depth=nonn.student_configs["depth"],
    #                  num_classes=nonn.base_configs["num_classes"])
    # nonn.create_unified_student(model=my_student)
    # nonn.create_student()
    # nonn.student.train_model()
    # nonn.student.deploy()
