import time
from copy import deepcopy
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from torch.autograd import Variable
import os
import tvm
from tvm import relay
from tvm import rpc
from tvm.contrib import graph_runtime as runtime
from PIL import Image
from torchvision import transforms
import numpy as np
from torch import jit
from tvm.contrib import graph_executor


class Student:
    def __init__(self, student_configs, base_configs,
                 teacher_model=None, student_model=None, optimizer=None, lr_scheduler=None,
                 dataset=None, filter_cluster=None, print_model=False, print_params=False):
        self.gpu_id = student_configs["gpu_id"]
        self.epochs = student_configs["epochs"]
        self.parallel = student_configs["parallel"]
        self.resume = student_configs["resume"]
        self.model_save_name = base_configs["student_model_path"].split('/')[-1]  # twrn404_swrn161.pt7
        self.model_save_path = base_configs["student_model_path"][:-len(self.model_save_name)]  # models/student/
        self.student_model_path = base_configs["student_model_path"]  # models/student/twrn404_swrn161.pt7
        self.teacher_model_path = base_configs["teacher_model_path"]  # models/teacher/wrn_40_4.pt7
        self.deploy_dir_path = base_configs["deploy_dir_path"]
        self.deploy_target = base_configs["deploy_target"]
        self.alpha = student_configs["alpha"]
        self.beta = student_configs["beta"]
        self.temperature = student_configs["temperature"]
        self.log_file = student_configs["log_file"]
        self.inference_log_file = student_configs["inference_log_file"]

        self.num_students = base_configs["num_students"]

        # Handle dataset
        self.data_loaders, self.dataset_sizes, self.num_classes = dataset

        # TEACHER MODEL
        self.teacher_model = teacher_model

        self.device_teacher = torch.device(f"cuda:{self.gpu_id}" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() > 1 and self.parallel:
            print(f"USING {torch.cuda.device_count()} GPUs!")
            self.teacher_model = nn.DataParallel(self.teacher_model)
        self.teacher_model.load_state_dict(torch.load(self.teacher_model_path))
        self.teacher_model = self.teacher_model.to(self.device_teacher)
        if print_model:
            print('======== TEACHER INFO =========')
            print(self.teacher_model)
            print('=============================')
        if print_params:
            n_teacher_parameters = sum(p.numel() for p in self.teacher_model.parameters() if p.requires_grad)
            print('TOTAL NUMBER OF PARAMETERS TEACHER = {}'.format(n_teacher_parameters))
            print('-----------------------------')

        # STUDENT MODEL
        self.student_model = student_model
        self.student_deploy_model = deepcopy(student_model)
        self.student_deploy_model.mode = "deploy"

        self.device_student = torch.device(f"cuda:{self.gpu_id}" if torch.cuda.is_available() else "cpu")
        if torch.cuda.device_count() > 1 and self.parallel:
            print(f"USING {torch.cuda.device_count()} GPUs!")
            self.student_model = nn.DataParallel(self.student_model)

        self.student_model = self.student_model.to(self.device_student)
        if print_model:
            print('======== STUDENT INFO =========')
            print(self.student_model)
            print('=============================')
        if print_params:
            n_student_parameters = sum(p.numel() for p in self.student_model.parameters() if p.requires_grad)
            print('TOTAL NUMBER OF PARAMETERS STUDENT = {}'.format(n_student_parameters))
            print('-----------------------------')

        if self.resume:
            self.student_model.load_state_dict(torch.load(self.student_model_path))
            print(f"Loaded student model from: {self.student_model_path}")

        # Handle optimizer
        self.optimizer = optimizer

        # Handle LR Scheduler
        self.exp_lr_scheduler = lr_scheduler

        self.filter_cluster = filter_cluster

        print("Loaded configs for student")

        self.ips = base_configs["deploy_ips"]
        self.ports = base_configs["deploy_ports"]

        print("Loaded configs for deployment")

    def compile_part(self, model_part, input_data, name="", target='llvm'):
        scripted_model = jit.trace(model_part, input_data).eval()

        tvm_target = None
        if target == 'llvm':
            tvm_target = tvm.target.Target(target, host=target)
        elif target == 'rasp3b':
            tvm_target = tvm.target.arm_cpu(target)
        elif target == 'cuda':
            tvm_target = tvm.target.cuda(target)

        input_name = "any_input_name"  # The input name can be arbitrary
        shape_list = [(input_name, input_data.shape)]
        mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)

        # Compile the graph to llvm target with given input specification.
        with tvm.transform.PassContext(opt_level=3):
            lib = relay.build(mod, target=tvm_target, params=params)

        path_lib = os.path.join(self.deploy_dir_path, f"lib_{name}.tar")
        if not os.path.exists(self.deploy_dir_path):
            print("Create Deployment Dir")
            os.makedirs(self.deploy_dir_path)
        lib.export_library(path_lib)

        return input_name

    def compile_and_load(self, name, part, input_data, host=None, port=None, target='llvm'):
        path_lib = os.path.join(self.deploy_dir_path, f"lib_{name}.tar")
        compiled_name = self.compile_part(part, input_data, name, target=target)
        # create the remote runtime module
        if target == 'llvm':
            dev = tvm.cpu(0)
            loaded_lib = tvm.runtime.load_module(path_lib)
        else:
            remote = rpc.connect(host, port)
            # upload the library to remote device and load it
            remote.upload(path_lib)
            loaded_lib = remote.load_module(f"lib_{name}.tar")
            dev = remote.cpu(0)

        runtm = graph_executor.GraphModule(loaded_lib["default"](dev))
        return compiled_name, runtm

    def deploy(self, print_model=False, print_params=False, print_stats=False):

        self.student_deploy_model.load_state_dict(torch.load(self.student_model_path, map_location=torch.device('cpu')))
        # Print the model
        if print_model:
            print(f"======== {self.num_students} STUDENTS MODEL =========")
            print(self.student_deploy_model)
            print("=" * 40)

        # Print the number of parameters
        if print_params:
            n_parameters = sum(p.numel() for p in self.student_deploy_model.parameters() if p.requires_grad)
            print(f"TOTAL NUMBER OF PARAMETERS = {n_parameters}")
            print("-" * 40)

        # Print the model
        if print_model:
            print("======== 1 STUDENT MODEL =========")
            print(self.student_deploy_model.s_list[0])
            print("=" * 40)

        # Print the number of parameters
        if print_params:
            n_parameters = sum(p.numel() for p in self.student_deploy_model.s_list[0].parameters() if p.requires_grad)
            print(f"TOTAL NUMBER OF PARAMETERS = {n_parameters}")
            print("-" * 40)

        for param in self.student_deploy_model.parameters():
            param.requires_grad = False

        input_shape = [1, 3, 32, 32]
        input_data = [torch.randn(input_shape)] * self.num_students

        if self.student_deploy_model.has_pool:
            avg_shape = [1] + list(self.student_deploy_model.before_avg)[1:]
            input_data += [torch.randn(avg_shape)]

        fc_shape = [1, list(self.student_deploy_model.before_avg)[1]]
        input_data += [torch.randn(fc_shape)]

        students = []
        for i in range(self.num_students):
            compile_name, runtm = self.compile_and_load(name=f"s{i}",
                                                        part=self.student_deploy_model.s_list[i].eval(),
                                                        input_data=input_data[i],
                                                        host=self.ips[i],
                                                        port=self.ports[i],
                                                        target=self.deploy_target)
            students += [(compile_name, runtm)]

        if self.student_deploy_model.has_pool:
            pool_compile_name, pool_runtm = self.compile_and_load(name="avg",
                                                                  part=self.student_deploy_model.pool.eval(),
                                                                  input_data=input_data[self.num_students],
                                                                  target='llvm')

        fc_compile_name, fc_runtm = self.compile_and_load(name="fc",
                                                          part=self.student_deploy_model.fc.eval(),
                                                          input_data=input_data[-1],
                                                          target='llvm')

        label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        my_preprocess = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                 std=[0.2023, 0.1994, 0.2010])
        ])
        with open(os.path.join(self.deploy_dir_path, f"{self.inference_log_file}"), "a") as f:
            f.write("Run")
            for i in range(self.num_students):
                f.write(f";s{i}_input;s{i}_run")
            if self.student_deploy_model.has_pool:
                f.write(";pool_input;pool_run")
            f.write("fc_input;fc_run;total_sum;Energy [mWh];Charge [mAh];Power [mW];Power/image [us];Energy [J];\n")
        while True:
            total_sum = 0
            cnt = 0.0
            s_inputs = np.zeros([self.num_students])
            s_runs = np.zeros([self.num_students])
            s_outputs = {}
            fc_input = 0
            fc_run = 0
            if self.student_deploy_model.has_pool:
                avg_input = 0
                avg_run = 0
            input("Press Enter to start inference")
            for _ in tqdm(range(10)):
                for filename in tqdm(os.listdir("dataset/small_test/")):
                    with Image.open(os.path.join("dataset/small_test/", filename)).resize((32, 32)) as img:
                        cnt += 1.0
                        start_total = time.time()
                        img = my_preprocess(img)
                        img = np.expand_dims(img, 0)

                        for i in range(self.num_students):
                            name, s = students[i]
                            start = time.time()
                            s.set_input(name, tvm.nd.array(img.astype('float32')))
                            aux = time.time() - start
                            s_inputs[i] = aux

                            start = time.time()
                            s.run()
                            aux = time.time() - start
                            s_runs[i] = aux

                            s_output = s.get_output(0)
                            s_output = s_output.asnumpy().astype('float32')
                            s_outputs[i] = s_output
                        for_cat = []
                        for i in range(self.num_students):
                            for_cat += [s_outputs[i]]

                        if self.student_deploy_model.has_pool:
                            avg_in = np.concatenate(for_cat, axis=1)

                            start = time.time()
                            pool_runtm.set_input(pool_compile_name, tvm.nd.array(avg_in.astype('float32')))
                            aux = time.time() - start
                            avg_input += aux
                            start = time.time()
                            pool_runtm.run()
                            aux = time.time() - start
                            avg_run += aux
                            avg_out = pool_runtm.get_output(0)

                            avg_out = avg_out.asnumpy()
                            avg_out = np.reshape(avg_out, [1, -1])
                            start = time.time()
                            fc_runtm.set_input(fc_compile_name, tvm.nd.array(avg_out.astype('float32')))
                            aux = time.time() - start
                            fc_input += aux
                        else:
                            fc_in = np.concatenate(for_cat, axis=1)
                            fc_in = np.reshape(fc_in, [1, -1])

                            start = time.time()
                            fc_runtm.set_input(fc_compile_name, tvm.nd.array(fc_in.astype('float32')))
                            aux = time.time() - start
                            fc_input += aux

                        start = time.time()
                        fc_runtm.run()
                        aux = time.time() - start
                        fc_run += aux
                        tvm_output = fc_runtm.get_output(0)

                        top1_tvm = np.argmax(tvm_output.asnumpy()[0])
                        tvm_class_key = label_names[top1_tvm]

                        aux = time.time() - start_total
                        total_sum += aux
                        if print_stats:
                            print(filename)
                            print('Relay top-1 id: {}, class name: {}'.format(top1_tvm, tvm_class_key))
                            print(
                                f"{filename} Relay top-1 id: {top1_tvm}, class name: {tvm_class_key} TIME Req: {time.time() - start_total}")
                            print()

            with open(os.path.join(self.deploy_dir_path, f"{self.inference_log_file}"), "a") as f:
                for i in range(self.num_students):
                    f.write(f";{s_inputs[i] / cnt * 1000};{s_runs[i] / cnt * 1000}")
                if self.student_deploy_model.has_pool:
                    f.write(f";{avg_input / cnt * 1000};{avg_run / cnt * 1000}")
                f.write(f";{fc_input / cnt * 1000};{fc_run / cnt * 1000};{total_sum / cnt * 1000};;;;;;\n")

    def train_model(self):
        since = time.time()
        best_acc_s = 0.0
        results = []

        for epoch in range(self.epochs):
            print('Epoch {}/{}'.format(epoch, self.epochs - 1))
            print('-' * 80)

            # Each epoch has a training and validation phase
            phase_list = ['train', 'val', 'test']
            lists = [epoch + 1]
            for phase in phase_list:
                if phase == 'train':
                    self.student_model.train()  # Set student model to training mode
                else:
                    self.student_model.eval()  # Set student model to evaluate mode

                # Set teacher model to evaluate mode
                self.teacher_model.eval()

                # Student's accuracy and loss metrics
                running_loss = 0.0

                running_corrects = 0

                # Iterate over data.
                for inputs, labels in tqdm(self.data_loaders[phase]):

                    with torch.set_grad_enabled(False):  # is_train = False for the teacher network
                        inputs = inputs.to(self.device_teacher)
                        labels = labels.to(self.device_teacher)

                        outputs_T, t_filt_out = self.teacher_model(inputs)
                        _, preds_T = torch.max(outputs_T, 1)

                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                    # forward
                    with torch.set_grad_enabled(phase == 'train'):
                        inputs = inputs.to(self.device_student)
                        labels = labels.to(self.device_student)

                        outputs_S, s_filt_out = self.student_model(inputs)
                        _, preds_S = torch.max(outputs_S, 1)

                        outputs_T = outputs_T.to(self.device_student)
                        t_filt_out = t_filt_out.to(self.device_student)
                        soft_target_loss = self.distillation(student_scores=outputs_S,
                                                             teacher_scores=outputs_T,
                                                             labels=labels,
                                                             temperature=self.temperature,
                                                             alpha=self.alpha)
                        t_cluster_list = []
                        for i in range(len(self.filter_cluster)):
                            params = torch.Tensor(self.filter_cluster[i])
                            dtype = 'long'
                            if isinstance(params, dict):
                                print("We have a dictionary here!")
                                exit(0)
                            else:
                                to_var = getattr(params.cuda(), dtype)()
                            g2_idx = Variable(to_var)
                            t_cluster_list.append(torch.index_select(t_filt_out, dim=1, index=g2_idx))

                        g_t = torch.cat(t_cluster_list, dim=1)

                        at_loss = [self.actTransfer_loss(x, y) for x, y in zip([s_filt_out], [g_t])]

                        actT_Loss2 = self.beta * sum(at_loss)

                        loss = actT_Loss2 + soft_target_loss

                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)

                    running_corrects += torch.sum(preds_S == labels.data)

                epoch_loss = running_loss / self.dataset_sizes[phase]

                epoch_acc = float(running_corrects) / self.dataset_sizes[phase]

                print('{} Loss: {:.4f} Acc: {:.2f}'.format(phase, epoch_loss, epoch_acc * 100))
                lists += [phase, epoch_loss, epoch_acc]

                # deep copy the student model with best val accuracy
                if phase == 'test' and epoch_acc > best_acc_s:
                    best_acc_s = epoch_acc
                    best_model_S = deepcopy(self.student_model.state_dict())
                    print(f"Saving the best model with accuracy {best_acc_s * 100} on test set")
                    torch.save(best_model_S, os.path.join(self.model_save_path, self.model_save_name))
            results += [lists]
            self.exp_lr_scheduler.step()
            # If you want to always (every epoch) save your model please uncomment the next block
            # if phase == 'train':
            #     # Save the best val model
            #     # WARNING: Always save state_dict() even if trained with multiple GPUs which adds .module
            #     if self.parallel:
            #         torch.save(self.student_model.module.state_dict(),
            #                    os.path.join(self.model_save_path, self.student_model_save_name))
            #     else:
            #         torch.save(self.student_model.state_dict(),
            #                    os.path.join(self.model_save_path, self.model_save_name))


        with open(f"train/{self.log_file}", "w") as f:
            f.write(
                "Epoch;Training loss;Training accuracy;Validation loss;Validation accuracy;Test loss;Test accuracy;\n")
            for lst in results:
                epoch, _, tr_l, tr_a, _, val_l, val_a, _, test_l, test_a = lst
                f.write(f"{epoch};{tr_l};{tr_a};{val_l};{val_a};{test_l};{test_a};\n")
        print("Training finished!")
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    def distillation(self, student_scores, teacher_scores, labels, temperature, alpha):
        p_t = F.softmax(teacher_scores / temperature)
        p_s = F.log_softmax(student_scores / temperature)
        h = F.cross_entropy(student_scores, labels)
        return F.kl_div(p_s, p_t) * (temperature * temperature * 2. * alpha) + h * (1. - alpha)

    def actTransfer_loss(self, x, y, normalize_acts=True):
        if normalize_acts:
            return (F.normalize(x.view(x.size(0), -1)) - F.normalize(y.view(y.size(0), -1))).pow(2).mean()
        else:
            return (x.view(x.size(0), -1) - y.view(y.size(0), -1)).pow(2).mean()
