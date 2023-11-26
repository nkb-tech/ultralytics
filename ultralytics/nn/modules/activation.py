import torch
import torch.nn as nn

__all__ = ['MemoryEfficientMish', 'MemoryEfficientSwish', 'Squareplus', 'HSigmoid']


# A memory-efficient implementation of Swish function
class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_tensors[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)
    

# A memory-efficient implementation of Mish function
class MishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.tanh(torch.nn.functional.softplus(i))
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
  
        v = 1. + i.exp()
        h = v.log() 
        grad_gh = 1. / h.cosh().pow_(2) 

        # Note that grad_hv * grad_vx = sigmoid(x)
        #grad_hv = 1./v  
        #grad_vx = i.exp()
        
        grad_hx = i.sigmoid()

        grad_gx = grad_gh * grad_hx  # grad_hv * grad_vx 
        
        grad_f = torch.tanh(torch.nn.functional.softplus(i)) + i * grad_gx 
        
        return grad_output * grad_f 


class MemoryEfficientMish(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        pass
    def forward(self, input_tensor):
        return MishImplementation.apply(input_tensor)
    

class Squareplus(nn.Module):
    """Squareplus activation presented https://arxiv.org/pdf/2112.11687.pdf.
    This function produces stable results when inputs is high enough.
    Faster than Softplus.
    """

    __constants__ = ['beta', 'shift']
    beta: float
    shift: float

    def __init__(
        self,
        beta: float = 1.0,
        shift: float = 4,
    ) -> None:
        super().__init__()
        self.beta = beta
        self.shift = shift

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Call forward and returns and processed tensor."""
        _input = input * self.beta
        return 1 / (2 * self.beta) * (_input + torch.sqrt(_input * _input + self.shift))
    
    def extra_repr(self) -> str:
        return f'beta={self.beta}, shift={self.shift}'

    
class HSigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(HSigmoid, self).__init__()
        self.inplace = inplace
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6
    
    def extra_repr(self) -> str:
        return f'inplace={self.inplace}'
