import jittor as jt
from jittorsimpletrans.models import ViT, SimplTrans

def basic_vit_example():
    # Create a standard ViT model for MNIST
    vit = ViT(
        image_size=28,       # input image size
        patch_size=4,        # patch size
        num_classes=10,      # number of classes
        dim=48,              # embedding dimension
        depth=12,            # transformer depth
        heads=3,             # number of attention heads
        mlp_dim=192,         # MLP hidden dimension
        channels=3,          # number of input channels
        dim_head=16,         # dimension per head
        dropout=0.1,         # dropout rate
        emb_dropout=0.1      # embedding dropout rate
    )
    
    # Create a sample input batch (8 images of size 28x28 with 3 channels)
    sample_input = jt.randn(8, 3, 28, 28)
    
    # Forward pass
    output = vit(sample_input)
    
    # Output shape should be [8, 10] (batch_size, num_classes)
    print(f"Input shape: {sample_input.shape}")
    print(f"Output shape: {output.shape}")
    # Avoid printing tensors directly in f-string
    print("Output sample: (printing sample values only)")
    return output

def modified_vit_example():
    # Create a SimplTrans model (with identity attention mode)
    simpltrans = SimplTrans(
        image_size=28,
        patch_size=4,
        num_classes=10,
        dim=48,
        depth=12,
        heads=3,
        mlp_dim=192,
        channels=3,
        dim_head=16,
        dropout=0.1,
        emb_dropout=0.1,
        mode="identity"  # can also be "random" or "diagonal"
    )
    
    # Create a sample input batch
    sample_input = jt.randn(8, 3, 28, 28)
    
    # Forward pass
    output = simpltrans(sample_input)
    
    print(f"SimplTrans Input shape: {sample_input.shape}")
    print(f"SimplTrans Output shape: {output.shape}")
    # Avoid printing tensors directly in f-string
    print("SimplTrans Output sample: (printing sample values only)")
    return output

if __name__ == "__main__":
    # Initialize Jittor - enable CPU mode if CUDA not available
    if not jt.has_cuda:
        jt.flags.use_cuda = 0
    print("===== Basic ViT Example =====")
    basic_vit_example()
    
    print("\n===== Modified ViT Example =====")
    modified_vit_example()
