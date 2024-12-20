import torch
import torch.nn as nn
import torch.onnx

from training_exp import LyapunovNetworkV, TwoDimDocking

def single_model(file_1, file_2):
	x = torch.randn(1,4,requires_grad=True)

	model = torch.load(file_1, map_location=torch.device('cpu'))
	torch.onnx.export(model, x, file_2,export_params=True,opset_version=10,do_constant_folding=True,input_names = ['input'],output_names = ['output'])