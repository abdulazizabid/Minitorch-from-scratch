"""
Module system for organizing neural network components and parameters.

This provides the foundation for building complex neural networks
with proper parameter management and hierarchy.
"""

from typing import Dict, List, Tuple, Any, Sequence


class Parameter:
    """A trainable parameter in a neural network."""
    
    def __init__(self, value: Any):
        self.value = value
        
    @property
    def shape(self):
        """Return parameter shape."""
        if hasattr(self.value, 'shape'):
            return self.value.shape
        return ()
    
    def __repr__(self):
        return f"Parameter(value={self.value})"


class Module:
    """Base class for all neural network modules."""
    
    def __init__(self):
        self._parameters={}
        self._modules={}
        self.training=True
    
    def modules(self) -> Sequence["Module"]:
        """Return all sub-modules recursively (depth-first)."""
        results=[]
        for module in self._modules.values():
            results.append(module)
            results.extend(module.modules())
        return results
    
    def train(self):
        """Set training mode for this module and all sub-modules."""
        self.training=True
        for module in self.modules():
            module.training=True
    
    def eval(self):
        """Set evaluation mode."""
        self.training=False
        for module in self.modules():
            module.training=False
    
    def named_parameters(self) -> Sequence[Tuple[str, Parameter]]:
        """Return all parameters with names."""
        result=[]
        for name, param in self._parameters.items():
            result.append((name,param))

        for module_name,module in self._modules.items():
            for name,param in module.named_parameters():
                full_name=f"{module_name}.{name}"
                result.append((full_name,param))
        return result

    
    def parameters(self) -> Sequence[Parameter]:
        """Return all parameters."""
        return [param for _,param in self.named_parameters()]
    
    def add_parameter(self, name: str, value: Any) -> Parameter:
        """Add a parameter."""
        if isinstance(value,Parameter):
            param=value
        else:
            param=Parameter(value)
        self._parameters[name]=param
        return param

    
    def __setattr__(self, key: str, value: Any):
        """Custom attribute setter."""
        if isinstance(value,Parameter):
            self._parameters[key]=value
        elif isinstance(value,Module):
            self._modules[key]=value
        super().__setattr__(key,value)

# For testing

# class Network(Module):
#     def __init__(self):
#         super().__init__()
#         self.layer1 = Module()
#         self.layer1.weight = Parameter([[1, 2], [3, 4]])
#         self.layer1.bias = Parameter([0.1, 0.2])
#         self.layer2 = Module()
#         self.layer2.add_parameter("weight",[0.5])

# net = Network()
# print(net.named_parameters())
