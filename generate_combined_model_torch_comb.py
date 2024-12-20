import torch
import torch.nn as nn
import torch.onnx

from training_exp import LyapunovNetworkV, TwoDimDocking
from attempt_conversion import LearnedController

def combined_model(file_1,file_2,file_3, file_4): 
    V = torch.load(file_1)
    #V = V.to(device="cpu")

    controller = torch.load(file_2)
    #controller = controller.to(device="cpu")
    prev_V = torch.load(file_3)
    # Create the combined network with stacked A and two copies of B and one C
    class CombinedNetwork(nn.Module):
        def __init__(self, model_A, model_B, model_C):
            super(CombinedNetwork, self).__init__()
            self.model_A = model_A
            self.model_B1 = model_B
            self.model_B2 = model_B
            self.model_C1 = model_C
            self.model_C2 = model_C

        def forward(self, x, y):
            input_A = x
            input_B1 = x
            input_B2 = y
            input_C1 = x
            input_C2 = y

            output_A = self.model_A(input_A)
            output_B1 = self.model_B1(input_B1)
            output_B2 = self.model_B2(input_B2)
            output_C1 = self.model_C1(input_C1)
            output_C2 = self.model_C2(input_C2)

            return output_A, output_B1, output_B2, output_C1, output_C2


    # Create an instance of the combined network
    combined_network = CombinedNetwork(controller,V,prev_V)

    # Test the combined network
    input_data_one = torch.randn(10, 4)  
    input_data_two = torch.randn(10, 4)

    input_data_one = torch.Tensor([[1,1,1,1]])
    input_data_two =  torch.Tensor([[1,1,1,1]])

    print(input_data_one)
    print(input_data_two)
    output_one, output_two, output_three, output_four, output_five = combined_network(input_data_one,input_data_two)
    print(output_one)
    print(output_two)
    print(output_three)
    print(output_four)
    print(output_five)

    x = torch.randn(1,4,requires_grad=True)
    y = torch.randn(1,4,requires_grad=True)

    torch.onnx.export(combined_network,(x,y),file_4,export_params=True,opset_version=10,do_constant_folding=True,input_names = ['input_1','input_2'],output_names = ['output_1','output_2','output_3','output_4','output_5'])

def combine_prev_cur(file_1, file_2, file_3):
    V = torch.load(file_1)
    prev_V = torch.load(file_2)
    class CombinedNetwork(nn.Module):
        def __init__(self, model_A, model_B):
            super(CombinedNetwork, self).__init__()
            self.model_A = model_A
            self.model_B = model_B

        def forward(self, x):
            input_A = x
            input_B = x

            output_A = self.model_A(input_A)
            output_B = self.model_B(input_B)

            return output_A, output_B


    # Create an instance of the combined network
    combined_network = CombinedNetwork(V,prev_V)

    # Test the combined network
    input_data_one = torch.randn(10, 4)  

    input_data_one = torch.Tensor([[1,1,1,1]])

    print(input_data_one)
    output_one, output_two = combined_network(input_data_one)
    print(output_one)
    print(output_two)

    x = torch.randn(1,4,requires_grad=True)

    torch.onnx.export(combined_network,(x),file_3,export_params=True,opset_version=10,do_constant_folding=True,input_names = ['input_1'],output_names = ['output_1','output_2'])