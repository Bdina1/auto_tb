import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torchvision
import inspect
from PIL import Image

class Flatten(nn.Module):
    def __init__(self, dim):
        super(Flatten, self).__init__()
        self.start_dim = dim
    def forward(self, x):
        return torch.flatten(x, start_dim=self.start_dim)

class Concat(nn.Module):
    def __init__(self, dim):
        super(Concat, self).__init__()
        self.dim = dim

    def forward(self, *inputs):
        return torch.cat(inputs, dim=self.dim)
    
    
class TB_object:
    def __init__(self, writer, config):
        self.writer = writer
        self.config = config
        self.operations = {
            'add_scalar': self.writer.add_scalar,
            'add_histogram': self.writer.add_histogram,
            'add_image': self.writer.add_image,
            'add_weights': self.writer.add_image,
        }
        
    def __call__(self, prev_layer_output=None, layer_index=None):
        operation_name = self.config[0]
        operation = self.operations.get(operation_name)
        if operation is None:
            raise ValueError(f'Unknown operation: {operation_name}')
        if operation_name == "add_image":
            img_name = self.config[1]
            if len(prev_layer_output.shape) == 4:
                # 4D tensor - handle filter outputs
                if prev_layer_output.shape[1] > 3:
                    # Many filters - log each filter separately
                    for i in range(prev_layer_output.shape[1]):
                        filter_img = prev_layer_output[:, i, :, :]
                        filter_img = filter_img.unsqueeze(1)
                        grid = torchvision.utils.make_grid(filter_img)
                        operation(f'{img_name}_filter{i}', grid, 0)

                # else:
                #     # Few filters - stack them in grid
                #     grid = prev_layer_output
                #     grid = grid.permute(1, 0, 2, 3) 
                #     grid = grid.reshape(grid.shape[0], -1, grid.shape[2])
                #     operation(img_name, grid, 0)

                else:
                    # 3D tensor - add as image  
                    grid = torchvision.utils.make_grid(prev_layer_output)
                    operation(f'{img_name}_layer{layer_index}', grid, 0)
                
        else:
            # Now config should contain the parameters for the operation in the correct order
            operation(*self.config[1:], prev_layer_output)

        return prev_layer_output



class CustomModel(nn.Module):
    def __init__(self, config, writer):
        super(CustomModel, self).__init__()

        self.network = nn.ModuleDict()
        self.log_layers = {}
        self.connections = {}
        layer_config = config['model']

        for i, layer in enumerate(layer_config):
            if layer[2] == 'TB_object':
                if i == 0:
                    self.log_layers[i-1] = TB_object(writer, layer[3])
                else:
                    self.log_layers[i] = TB_object(writer, layer[3])
            elif isinstance(layer[2], str):
                if hasattr(nn, layer[2]):
                    layer_name = f'{layer[2]}_{i}'
                    self.network[layer_name] = getattr(nn, layer[2])(*layer[3])
                elif layer[2] == "Concat":
                    self.network[f'{layer[2]}_{i}'] = Concat(*layer[3])
                    self.connections[i] = layer[0]
                else:
                    raise ValueError(f"Unknown layer: {layer[2]}")
                
                
    def forward(self, x):
        layer_outputs = []
        
        # Check if input should be logged
        if -1 in self.log_layers:
            self.log_layers[-1](x)

        for i, (name, layer) in enumerate(self.network.items()):
            if i in self.connections:
                x = layer(*[layer_outputs[prev_layer] for prev_layer in self.connections[i]])
            else:
                x = layer(x)
            layer_outputs.append(x)
            if (i+1) in self.log_layers:
                self.log_layers[i+1](x)
        return x
