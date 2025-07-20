CODING = dict()
CODING["file"] = "coding.html"
CODING["title"] = "Coding and Engineering"
CODING["description"] = "Tools used to build AI systems."
CODING["categories"] = ["torch & torchvision"]
CODING["papers"] = [
# {
# "title": "",
# "author": "",
# "organization": "",
# "date": "",
# "venue": "",
# "pdf_url": "",
# "code_url": "",
# "name": "",
# "comment": "",
# "category": "",
# "jupyter_notes": "",
# "summary": """""",
# "details": 
# """
# <pre>
# <code class="language-python">
# </code>
# </pre>
# """,
# },
{
"title": "Module",
"author": "",
"organization": "",
"date": "20240629"
""
"",
"venue": "docs",
"pdf_url": "https://docs.pytorch.org/docs/stable/nn.html",
"code_url": "",
"name": "module",
"comment": "",
"category": "torch & torchvision",
"jupyter_notes": "",
"summary": """It includes tools to build neural networks: <a href="https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#conv2d">Conv2d</a>, <a href="https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv3d.html#conv3d">Conv3d</a>.""",
"details": 
"""
<pre>
<code class="language-python">
import torch
from torch.nn import Conv2d, Conv3d

## --------------------------------------------------------------------------------
## Conv2d
## --------------------------------------------------------------------------------
in_channels = /  # *** int. Number of channels in the input
out_channels = /  # *** int. Number of channels in the output
kernel_size = /  # *** int, tuple. Size of convolving kernel
stride = 1  # *** int, tuple. Stride of convolution
padding = 0  # int, tuple, str. Padding added to all four sides of the input
dilation = 1  # int, tuple. Spacing between kernel elements
groups = 1  # int. Number of blocked connections from input channels to output
bias = True  # bool. If True, add a learnable bias to the output
padding_mode = "zeros"  # str. "zeros", "reflect", "replicate", or "circular"
device = None # str, torch.device. 
dtype = None # torch.dtype.

## Weight. Shape: [out_channels, in_channels/groups, k_size[0], k_size[1]]
## Bias. Shape: [out_channels,]
conv2d = Conv2d(
    in_channels, out_channels, kernel_size, stride, padding, dilation, groups, 
    bias, padding_mode, device, dtype
)
## [B, C, H_in, W_in] => [B, C, H_out, W_out]
## H_out = [(H_in + 2*padding[0] - dilation[0]*(kernel[0]-1)-1) / stride[0] + 1]
## W_out = [(W_in + 2*padding[1] - dilation[1]*(kernel[1]-1)-1) / stride[1] + 1]
y = conv2d(x)  # Tensor => Tensor
## --------------------------------------------------------------------------------

## --------------------------------------------------------------------------------
## Conv3d
## --------------------------------------------------------------------------------
in_channels = /  # *** int. Number of channels in the input
out_channels = /  # *** int. Number of channels in the output
kernel_size = /  # *** int, tuple. Size of convolving kernel
stride = 1  # *** int, tuple. Stride of convolution
padding = 0  # int, tuple, str. Padding added to all six sides of the input
dilation = 1  # int, tuple. Spacing between kernel elements
groups = 1  # int. Number of blocked connections from input channels to output
bias = True  # bool. If True, add a learnable bias to the output
padding_mode = "zeros"  # str. "zeros", "reflect", "replicate", or "circular"
device = None # str, torch.device. 
dtype = None # torch.dtype.

## Weight. Shape: [out_channels, in_channels/groups, k_size[0], k_size[1], k_size[2]]
## Bias. Shape: [out_channels,]
conv2d = Conv2d(
    in_channels, out_channels, kernel_size, stride, padding, dilation, groups, 
    bias, padding_mode, device, dtype
)
## [B, C, D_in, H_in, W_in] => [B, C, D_out, H_out, W_out]
## D_out = [(D_in + 2*padding[0] - dilation[0]*(kernel[0]-1)-1) / stride[0] + 1]
## H_out = [(H_in + 2*padding[1] - dilation[1]*(kernel[1]-1)-1) / stride[1] + 1]
## W_out = [(W_in + 2*padding[2] - dilation[2]*(kernel[2]-1)-2) / stride[2] + 1]
y = conv2d(x)  # Tensor => Tensor
## --------------------------------------------------------------------------------
</code>
</pre>
""",
},
{
"title": "Optimizer",
"author": "",
"organization": "",
"date": "20240628",
"venue": "docs",
"pdf_url": "https://docs.pytorch.org/docs/stable/optim.html",
"code_url": "",
"name": "optimizer",
"comment": "",
"category": "torch & torchvision",
"jupyter_notes": "",
"summary": """It includes tools for building optimization algorithms: <a href="https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html#torch.optim.AdamW.step">AdamW</a>.""",
"details": 
"""
<pre>
<code class="language-python">
from torch.optim import AdamW

## --------------------------------------------------------------------------------
## AdamW
## --------------------------------------------------------------------------------
params = /  # *** iterable. Parameters / named_parameters / parameter groups to optimize
lr = 0.001  # *** float, Tensor. Learning rate
betas = (0.9, 0.999)  # tuple. For computing running averages of gradients & squares
eps = 1e-08  # float. Added to denominator to improve numerical stability
weight_decay = 0.01  # float. Weight decay coefficient
amsgrad = False # bool. Whether to use AMSGrad 
maximize = False  # bool. Maximize the objective with respect to params, not minimize
foreach = None  # bool. Whether foreach implementation of optimizer is used.
capturable = False  # bool. Pass True can impair ungraphed performance
differentiable = False  # bool. Whether has gradient
fused = None  # bool. Whether use the fused implementation

adam_optim = AdamW(
    params, lr, betas, eps, weight_decay, amsgrad, maximize, 
    foreach, capturable, differentiable, fused
)
## --------------------------------------------------------------------------------
</code>
</pre>
""",
},
{
"title": "Data Loader",
"author": "",
"organization": "",
"date": "20240630",
"venue": "docs",
"pdf_url": "https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader",
"code_url": "",
"name": "data loader",
"comment": "",
"category": "torch & torchvision",
"jupyter_notes": "",
"summary": """It includes tools for data loading: <a href="https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader">DataLoader</a>.""",
"details": 
"""
<pre>
<code class="language-python">
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

## --------------------------------------------------------------------------------
## DataLoader
## --------------------------------------------------------------------------------
dataset = /  # *** Dataset
batch_size = 1  # *** int. Number of samples per batch.
shuffle = False  # *** bool. If True, have the data shuffled at every epoch
sampler = None  # Sampler or Iterable. Define how to draw samples
batch_sampler = None  # Sampler or Iterable. customize sampling by giving indices
num_workers = 0  # *** int. Number of subprocesses to use for data loading
collate_fn = None  # Callable. Merge a list of samples to form a batch of tensors.
pin_memory = False  # *** bool. If True, copy Tensors into CUDA pinned memory.
drop_last = False  # *** bool. If True, drop the last incomplete batch
timeout = 0  # numeric. If positive, set timeout for collecting a batch from workers.
worker_init_fn = None  # Callable. If not None, it will be called (worker id as input) 
multiprocessing_context = None  # str or multiprocessing.context.BaseContext.
generator = None  # torch.Generator. If not None, it will be used by sampler & workers
prefetch_factor = None  # int. Default = None if num_workers == 0 else 2
persistent_workers = False  # bool. If True, workers will not shut down after an epoch.
pin_memory_device = ""  # str. The device to pin memory
in_order = True  # bool. If False, it will not enforce batches to return in order

data_loader = DataLoader(
    dataset, batch_size, shuffle, sampler, batch_sampler, num_workers, collate_fn, 
    pin_memory, drop_last, timeout, worker_init_fn, multiprocessing_context, 
    generator, prefetch_factor, persistent_workers, pin_memory_device, in_order
)
## --------------------------------------------------------------------------------
</code>
</pre>
""",
},
{
"title": "Data Transforms",
"author": "",
"organization": "",
"date": "20240701",
"venue": "docs",
"pdf_url": "https://docs.pytorch.org/vision/stable/transforms.html",
"code_url": "",
"name": "data transforms",
"comment": "",
"category": "torch & torchvision",
"jupyter_notes": "",
"summary": """It includes tools to transform and augment data: <a href="https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.Resize.html?highlight=transforms+resize#torchvision.transforms.Resize">Resize</a>, <a href="https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.RandomHorizontalFlip.html#torchvision.transforms.RandomHorizontalFlip">RandomHorizontalFlip</a>, <a href="https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.ToTensor.html?highlight=totensor#torchvision.transforms.ToTensor">ToTensor</a>, <a href="https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.Compose.html#torchvision.transforms.Compose">Compose</a>, <a href="https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.Normalize.html#torchvision.transforms.Normalize">Normalize</a>.""",
"details": 
"""
<pre>
<code class="language-python">
from torchvision import transforms
from torchvision.transforms.InterpolationMode import BILINEAR, NEAREST, BICUBIC 

## --------------------------------------------------------------------------------
## Geometry: Resize
## --------------------------------------------------------------------------------
size = /  # *** sequence or int. For example (512, 768)
interpolation = BILINEAR  # InterpolationMode
max_size = None  # int. Maximum allowed for the longer edge, supported if `size` is int
antialias = True  # bool. Apply antialiasing, only under bilinear or bicubic modes

trans = transforms.Resize(size, interpolation, max_size, antialias)
image_trans = trans(image)  # PIL Image => PIL Image or Tensor => Tensor
## --------------------------------------------------------------------------------

## --------------------------------------------------------------------------------
## Geometry: RandomHorizontalFlip
## --------------------------------------------------------------------------------
p = 0.5  # *** float. Probability to flip image

trans = <b>transforms.RandomHorizontalFlip</b>(p)
image_trans = trans(image)  # PIL Image => PIL Image or Tensor => Tensor
## --------------------------------------------------------------------------------

## --------------------------------------------------------------------------------
## Conversion: ToTensor
## --------------------------------------------------------------------------------
## Input: PIL Image / numpy.ndarray (np.uint8) of shape (HxWxC) in the range [0, 255]
## Output: torch.FloatTensor of shape (CxHxW) in the range (0.0, 1.0)
## Other inputs: only apply type transform
trans = <b>transforms.ToTensor</b>()
image_trans = trans(image)  # PIL Image / ndarray => Tensor
## --------------------------------------------------------------------------------

## --------------------------------------------------------------------------------
## Composition: Compose
## --------------------------------------------------------------------------------
transforms = /  # *** list of Transform objects

trans = <b>transforms.Compose</b>(transforms)
image_trans = trans(image)  # PIL Image / ndarray / Tensor => Tensor
## --------------------------------------------------------------------------------

## --------------------------------------------------------------------------------
## Miscellaneous: Normalize
## --------------------------------------------------------------------------------
mean = /  # *** sequence. Means for each channel.
std = /  # *** sequence. Standard deviations for each channel.
inplace = False  # bool. Bool to make this operation in-place.

trans = <b>transforms.Normalize</b>(mean, std, inplace)
image_trans = trans(image)  # Tensor => Tensor
## --------------------------------------------------------------------------------
</code>
</pre>
""",
},
]