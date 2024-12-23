import torch
import torch.nn as nn
import torch.onnx


def single_model(V_net, output_file, state_dims):
	"""
	Convert single V_net to ONNX model
	
	Args:
		V_net: Vector Lyapunov network
		output_file: Path to save ONNX model
	"""
	class SingleNetwork(nn.Module):
		def __init__(self, V_net, state_dims):
			super(SingleNetwork, self).__init__()
			self.V_net = V_net
			self.state_dims = state_dims

		def forward(self, x):
			x_stars = torch.tensor([[20.0, 15.0]] * self.state_dims[1], device=x.device)  # [num_vehicles, 2]
			x_stars = x_stars.unsqueeze(0).expand(x.shape[0], -1, -1)  # [batch_size, num_vehicles, 2]
			return self.V_net(x, x_stars)

	# Create and export model
	single_network = SingleNetwork(V_net, state_dims)

	# Create dummy input based on V_net's input dimension
	dummy_input = torch.randn(1, state_dims[1], state_dims[0])
	
	torch.onnx.export(
		single_network,
		dummy_input,
		output_file,
		input_names=['input'],
		output_names=['output'],
		dynamic_axes={'input': {0: 'batch_size'}}
	)