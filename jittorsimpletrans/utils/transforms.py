import jittor as jt
from jittor import Module

# Add helper functions to replace einops's rearrange and repeat
def rearrange_b_n_h_d(x, h):
    """Replace rearrange(x, 'b n (h d) -> b h n d', h=h)"""
    b, n, d_all = x.shape
    d = d_all // h
    return x.reshape(b, n, h, d).permute(0, 2, 1, 3)  # b, h, n, d

def rearrange_b_h_n_d(x):
    """Replace rearrange(x, 'b h n d -> b n (h d)')"""
    b, h, n, d = x.shape
    return x.permute(0, 2, 1, 3).reshape(b, n, h*d)  # b, n, h*d

def repeat(tensor, pattern, **axes_lengths):
    """Simplified version of repeat, only handling '1 1 d -> b 1 d' format"""
    if pattern == "1 1 d -> b 1 d" and "b" in axes_lengths:
        b = axes_lengths["b"]
        # Repeat tensor with shape [1, 1, d] to [b, 1, d]
        return tensor.repeat(b, 1, 1)

# Create a custom class to replace einops.Rearrange
class Rearrange(Module):
    def __init__(self, pattern, **axes_lengths):
        super().__init__()
        self.pattern = pattern
        self.axes_lengths = axes_lengths
        
    def execute(self, x):
        # Specifically handle the pattern used in ViT: 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)'
        if self.pattern == "b c (h p1) (w p2) -> b (h w) (p1 p2 c)":
            p1 = self.axes_lengths.get('p1')
            p2 = self.axes_lengths.get('p2')
            
            # Ensure p1 and p2 are provided
            assert p1 is not None and p2 is not None, "p1 and p2 must be provided"
            
            # Get input tensor shape
            b, c, h_full, w_full = x.shape
            
            # Calculate h and w
            h = h_full // p1
            w = w_full // p2
            
            # Ensure dimensions are divisible by patch size
            assert h_full % p1 == 0 and w_full % p2 == 0, "Image dimensions must be divisible by patch size"
            
            # Rearrange to (b, h, p1, w, p2, c)
            x = x.reshape(b, c, h, p1, w, p2)
            
            # Transpose to (b, h, w, p1, p2, c)
            x = x.permute(0, 2, 4, 3, 5, 1)
            
            # Final shape (b, h*w, p1*p2*c)
            x = x.reshape(b, h * w, p1 * p2 * c)
            
            return x
            
        # Raise error for other patterns
        raise NotImplementedError(f"Pattern {self.pattern} is not implemented")

def pair(t):
    """Convert a single value to a pair of identical values, or return the pair as is"""
    return t if isinstance(t, tuple) else (t, t)
