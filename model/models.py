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
        }
        
    def __call__(self, prev_layer_output=None, layer_index=None):
        operation_name = self.config[0]
        operation = self.operations.get(operation_name)
        if operation is None:
            raise ValueError(f'Unknown operation: {operation_name}')

        if operation_name == "add_image":
            img_name = self.config[1]

            # Check if output is a 4D tensor (batch_size, channels, height, width)
            if len(prev_layer_output.shape) == 4:
                if prev_layer_output.shape[1] > 3:
                    channel_images = []
                    for i in range(prev_layer_output.shape[1]):
                        single_channel = prev_layer_output[:, i, :, :].unsqueeze(1)
                        channel_images.append(single_channel)
                        
                    grid = torchvision.utils.make_grid(torch.cat(channel_images, dim=1), nrow=3)
                    # Convert grid tensor to numpy array with data type uint8
                    grid_np = (grid * 255).byte().permute(1, 2, 0).numpy()

                    # Create PIL Image from numpy array
                    image = Image.fromarray(grid_np)

                    # Convert PIL Image back to tensor
                    grid_tensor = torchvision.transforms.ToTensor()(image)

                    operation(img_name, grid_tensor, 0)
                else:
                    grid = torchvision.utils.make_grid(prev_layer_output)
                    operation(f'{img_name}_layer{layer_index}', grid, 0)
            else:
                raise ValueError(f'Cannot create image from layer output: {prev_layer_output.shape}')
        # elif operation_name == "add_histogram":
        #     hist_name = self.config[1]
            
        #     hist_data = prev_layer_output.detach().cpu().numpy()  # Convert tensor to numpy array
        #     if hist_data[1] > 1:
        #         pass
        #     operation(hist_name, hist_data, bins='auto')  # Use 'auto' to determine the number of bins automatically
            
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
