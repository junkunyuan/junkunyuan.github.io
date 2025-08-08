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
# "info": "",
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
"info": "",
"summary": """It includes tools to transform and augment data: <a href="https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.Resize.html?highlight=transforms+resize#torchvision.transforms.Resize">Resize</a>, <a href="https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.RandomHorizontalFlip.html#torchvision.transforms.RandomHorizontalFlip">RandomHorizontalFlip</a>, <a href="https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.ToTensor.html?highlight=totensor#torchvision.transforms.ToTensor">ToTensor</a>, <a href="https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.Compose.html#torchvision.transforms.Compose">Compose</a>, <a href="https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.Normalize.html#torchvision.transforms.Normalize">Normalize</a>.""",
"details": 
"""
<pre>
<code class="language-python">
from torchvision import transforms
from torchvision.transforms.InterpolationMode import BILINEAR, NEAREST, BICUBIC 

## --------------------------------------------------------------------------------
## Resize
## --------------------------------------------------------------------------------
size = /  # *** sequence or int. For example (512, 768)
interpolation = BILINEAR  # InterpolationMode
max_size = None  # int. Maximum allowed for the longer edge, supported if `size` is int
antialias = True  # bool. Apply antialiasing, only under bilinear or bicubic modes
trans = transforms.Resize(size, interpolation, max_size, antialias)
image_trans = trans(image)  # PIL Image => PIL Image or Tensor => Tensor
## --------------------------------------------------------------------------------

## --------------------------------------------------------------------------------
## RandomHorizontalFlip
## --------------------------------------------------------------------------------
p = 0.5  # *** float. Probability to flip image
trans = <b>transforms.RandomHorizontalFlip</b>(p)
image_trans = trans(image)  # PIL Image => PIL Image or Tensor => Tensor
## --------------------------------------------------------------------------------

## --------------------------------------------------------------------------------
## ToTensor
## --------------------------------------------------------------------------------
## Input: PIL Image / numpy.ndarray (np.uint8) of shape (HxWxC) in the range [0, 255]
## Output: torch.FloatTensor of shape (CxHxW) in the range (0.0, 1.0)
## Other inputs: only apply type transform
trans = <b>transforms.ToTensor</b>()
image_trans = trans(image)  # PIL Image / ndarray => Tensor
## --------------------------------------------------------------------------------

## --------------------------------------------------------------------------------
## Compose
## --------------------------------------------------------------------------------
transforms = /  # *** list of Transform objects
trans = <b>transforms.Compose</b>(transforms)
image_trans = trans(image)  # PIL Image / ndarray / Tensor => Tensor
## --------------------------------------------------------------------------------

## --------------------------------------------------------------------------------
## Normalize
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
"info": "",
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
num_workers = 0  # *** int. Number of subprocesses to use for data loading
collate_fn = None  # Callable. Merge a list of samples to form a batch of tensors
pin_memory = False  # *** bool. If True, copy Tensors into CUDA pinned memory
drop_last = False  # *** bool. If True, drop the last incomplete batch
timeout = 0  # numeric. If positive, set timeout for collecting a batch from workers
prefetch_factor = None  # int. Default = None if num_workers == 0 else 2
# ...
data_loader = DataLoader(
    dataset, batch_size, shuffle, sampler, num_workers, collate_fn, 
    pin_memory, drop_last, timeout, prefetch_factor
)
## --------------------------------------------------------------------------------
</code>
</pre>
""",
},
{
"title": "Operation",
"author": "",
"organization": "",
"date": "20230630",
"venue": "docs",
"pdf_url": "https://docs.pytorch.org/docs/stable/index.html",
"code_url": "",
"name": "operation",
"comment": "",
"category": "torch & torchvision",
"jupyter_notes": "",
"info": "",
"summary": 
"""It includes operations: 
<a href="#operations">operations</a>,
<a href="#data generation">data generation</a>, 
<a href="#size & reshape">size & reshape</a>.""",
"details": 
"""
<p class="larger" id="operations"><b>Operations:</b> 
<a href="https://docs.pytorch.org/docs/stable/torch.html">basic operations</a>,
<a href="https://docs.pytorch.org/docs/stable/generated/torch.mean.html">mean</a>,
<a href="https://docs.pytorch.org/docs/stable/generated/torch.var.html">var</a>,
<a href="https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html">softmax</a>,
<a href="https://docs.pytorch.org/docs/stable/generated/torch.matmul.html">matmul</a>,
<a href="https://docs.pytorch.org/docs/stable/generated/torch.einsum.html">einsum</a>.</p>
<pre>
<code class="language-python">
## --------------------------------------------------------------------------------
## Operations
## --------------------------------------------------------------------------------
import torch

## Basic operations
## function: exp, sin, cos, sqrt
y = torch.function(x)

## Mean and var
dim = /  # *** int or tuple of ints. The dims to reduce
keepdim = False # *** bool. If True, return tensor with the same dims
mean = x.mean(dim, keepdim)
## In version>=2.0, correction=1 == unbiased=True, correction=0 == unbiased=False 
correction = 1  # *** int. 
var = x.var(dim, keepdim, correction)

## Softmax
dim = None  # *** input
y = x.softmax(dim)

## Matrix multiplication
other = /  # *** tensor
y = x.matmul(other)

## Einsum
equation = /  # *** str. The subscript for the Einstein summation
operands = /  # *** list of tensor. The tensor to be computed
## torch.einsum("ii", tensor)  # trace
## torch.einsum("ii->i", tensor)  # diagonal
## torch.einsum("i,j->ij", tensor1, tensor2)  # outer product
## torch.einsum("bij,bjk->bik", tensor1, tensor2)  # batch matrix multiplication
## torch.einsum("...ij->...jk", tensor)  # batch permute
y = torch.einsum(equation, operands)
## --------------------------------------------------------------------------------
</code>
</pre>

<p class="larger" id="data generation"><b>Data generation:</b> 
<a href="https://docs.pytorch.org/docs/stable/generated/torch.zeros.html">zeros</a>, 
<a href="https://docs.pytorch.org/docs/stable/generated/torch.ones.html">ones</a>, 
<a href="https://docs.pytorch.org/docs/stable/generated/torch.rand.html">uniform distribution</a>,
<a href="https://docs.pytorch.org/docs/stable/generated/torch.arange.html">arange</a>.</p>
<pre>
<code class="language-python">
## --------------------------------------------------------------------------------
## Data generation
## --------------------------------------------------------------------------------
## Zeros, Ones
size = /  # *** sequence of int. The shape of output
y = torch.ones(size)
y = torch.zeros(size)

## Random number from a uniform distribution [0, 1)
size = /  # *** sequence of int. The shape of output
y = torch.rand(size)

## Arange
start = 0  # *** number. The starting value
end = /  # *** number. The ending value
step = 1  # *** number. The gap between adjacent points
arange = torch.arange(start, end, step)
## --------------------------------------------------------------------------------
</code>
</pre>

<p class="larger" id="size & reshape"><b>Resize & reshape:</b> 
<a href="https://docs.pytorch.org/docs/stable/generated/torch.Tensor.size.html">size</a>,
<a href="https://docs.pytorch.org/docs/stable/generated/torch.reshape.html">reshape & view</a>,
<a href="https://docs.pytorch.org/docs/stable/generated/torch.flatten.html">flatten</a>,
<a href="https://docs.pytorch.org/docs/stable/generated/torch.tranpose.html">tranpose</a>,
<a href="https://docs.pytorch.org/docs/stable/generated/torch.permute.html">permute</a>,
<a href="https://docs.pytorch.org/docs/stable/generated/torch.permute.html">permute</a>,
<a href="https://docs.pytorch.org/docs/stable/generated/torch.unsqueeze.html">unsqueeze</a>,
<a href="https://docs.pytorch.org/docs/stable/generated/torch.cat.html">cat</a>,
<a href="https://docs.pytorch.org/docs/stable/generated/torch.squeeze.html">squeeze</a>,
<a href="https://docs.pytorch.org/docs/stable/generated/torch.unbind.html">unbind</a>,
<a href="https://docs.pytorch.org/docs/stable/generated/torch.chunk.html">chunk</a>,
<a href="https://docs.pytorch.org/docs/stable/generated/torch.split.html">split</a>,
<a href="https://docs.pytorch.org/docs/stable/generated/torch.where.html">where</a>.</p>

<pre>
<code class="language-python">
## --------------------------------------------------------------------------------
## Size & reshape
## --------------------------------------------------------------------------------
## Get size
dim = None  # int. The dimension to retrieve the size
size = x.size(dim)  # => torch.Size or int
size = x.shape  #  => torch.Size

## Reshape & View
shape = /  # sequence of int. The new shape. Note: a single dimension could be -1
y = x.reshape(shape)  # recommend since it could call .contiguous() if needed
y = x.view(shape)

## Flatten: flatten along the given dimensions
start_dim = 0  # *** int. The first dimension to flatten
end_dim = -1  # *** int. The last dimension to flatten
## Note: it could be either copying or viewing
y = x.flatten(start_dim, end_dim)

## Transpose: swap two dimensions
dim0 = /  # *** int. The first dimension to be tranposed
dim1 = /  # *** int. The second dimension to be tranposed
## Note: the result tensor shares the strorage
y = x.tranpose(dim0, dim1)

## Permute
dims = /  # *** sequence of int. The desired ordering of dims
y = x.permute(dims)

## Unsqueeze: insert one dimension
dim = /  # *** int. The index at which to insert the singleton dim
y = x.unsqueeze(dim)

## Concat: concatenate some tensors along a dimension
tensors = /  # *** tuple of tensors. Tensors with the same shape except in the cat dim
dim = 0  # *** int. The concatenation dim
y = torch.cat(tensors, dim)

## Squeeze: remove all dim with size 1
dim = None  # *** int or tuple of ints. If given, only the dim will be squeezed
y = x.squeeze(dim)

## Unbind
dim = 0  # *** int. Dim to remove
y = x.unbind(0)

## Chunk: split a tensor into the spicific number of chunks
chunks = /  # *** int
dim = 0  # *** int
## If the given dim is divisible by chunks, all returned chunks will be the same size
## If the given dim is not divisible by chunks, the last one will not be the same size
## If such division is not possible, it returns fewer than the specified number of chunks
y = x.chunk(chunks, dim)

## Split
indices_or_sections = /  # *** tensor, int, list, tuple of ints
dim = 0  # *** int. Dimension along which to split the tensor
## If split_size_or_sections is an integer type, split into equally sized chunks
## If split_size_or_sections is a list, split into len(split_size_or_sections) chunks
y = x.split(indices_or_sections, dim)

## Where: select elements
condition = /  # *** bool. When True, yield input, otherwise yield other
input = /  # *** tensor or scalar
other = /  # *** tensor or scalar
y = torch.where(condition, input, output)
## --------------------------------------------------------------------------------
</code>
</pre>
""",
},
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
"info": "",
"summary": 
"""
It includes tools to build neural networks: 
<a href="https://docs.pytorch.org/docs/stable/generated/torch.nn.parameter.Parameter.html">Parameter</a>,
<a href="https://docs.pytorch.org/docs/stable/generated/torch.nn.parameter.Buffer.html">Buffer</a>,
<a href="https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html">Linear</a>,
<a href="https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#conv2d">Conv2d</a>, 
<a href="https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv3d.html#conv3d">Conv3d</a>,
<a href="https://docs.pytorch.org/docs/stable/generated/torch.nn.Dropout.html">Dropout</a>.
""",
"details": 
"""
<pre>
<code class="language-python">
import torch
from torch import nn 

## --------------------------------------------------------------------------------
## Parameter
## --------------------------------------------------------------------------------
data = /  # *** tensor. Parameter tensor
requires_grad = True
gamma = torch.Parameter(data, requires_grad)
## --------------------------------------------------------------------------------

## --------------------------------------------------------------------------------
## Buffer
## --------------------------------------------------------------------------------
data = /  # *** tensor. Buffer tensor
persistent = True  # whether the buffer is part of the module's state_dict
gamma = self.register_buffer(data, persistent)  # used in the module class
gamma = nn.paramter.Buffer(data, persistent)  # not usually used
## --------------------------------------------------------------------------------

## --------------------------------------------------------------------------------
## Linear
## --------------------------------------------------------------------------------
in_features = /  # *** int. Size of each input sample
out_features = /  # *** int. Size of each output sample
bias = True  # *** bool. If set to False, the layer will not learn an additive bias
device = None  # torch.device or int 
dtype = None  # torch.dtype
## [..., H_in] => [..., H_out]
linear = Linear(in_features, out_features, bias, device, dtype)
## --------------------------------------------------------------------------------

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
device = None  # torch.device or int
dtype = None  # torch.dtype
## Weight. Shape: [out_channels, in_channels/groups, k_size[0], k_size[1]]
## Bias. Shape: [out_channels,]
conv2d = nn.Conv2d(
    in_channels, out_channels, kernel_size, stride, padding, dilation, groups, 
    bias, padding_mode, device, dtype
)
## [B, C, H_in, W_in] => [B, C, H_out, W_out]
## H_out = [(H_in + 2*padding[0] - dilation[0]*(kernel[0]-1)-1) / stride[0] + 1]
## W_out = [(W_in + 2*padding[1] - dilation[1]*(kernel[1]-1)-1) / stride[1] + 1]
y = conv2d(x)
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
device = None  # torch.device or int
dtype = None  # torch.dtype
## Weight. Shape: [out_channels, in_channels/groups, k_size[0], k_size[1], k_size[2]]
## Bias. Shape: [out_channels,]
conv3d = nn.Conv3d(
    in_channels, out_channels, kernel_size, stride, padding, dilation, groups, 
    bias, padding_mode, device, dtype
)
## [B, C, D_in, H_in, W_in] => [B, C, D_out, H_out, W_out]
## D_out = [(D_in + 2*padding[0] - dilation[0]*(kernel[0]-1)-1) / stride[0] + 1]
## H_out = [(H_in + 2*padding[1] - dilation[1]*(kernel[1]-1)-1) / stride[1] + 1]
## W_out = [(W_in + 2*padding[2] - dilation[2]*(kernel[2]-1)-2) / stride[2] + 1]
y = conv3d(x)
## --------------------------------------------------------------------------------

## --------------------------------------------------------------------------------
## Dropout
## --------------------------------------------------------------------------------
p = 0.5  # *** float. Probability of an element to be zeroed
inplace = False  # bool
dropout = nn.Dropout(p, inplace)
y = dropout(x)
## --------------------------------------------------------------------------------
</code>
</pre>
""",
},
{
"title": "Activation Function",
"author": "",
"organization": "",
"date": "20240628",
"venue": "docs",
"pdf_url": "https://docs.pytorch.org/docs/stable/nn.html",
"code_url": "",
"name": "activation function",
"comment": "",
"category": "torch & torchvision",
"jupyter_notes": "",
"info": "",
"summary": 
"""
It includes activation functions: 
<a href="https://docs.pytorch.org/docs/stable/generated/torch.nn.GELU.html">GeLU</a>,
<a href="https://docs.pytorch.org/docs/stable/generated/torch.nn.SiLU.html">SiLU / swish</a>.
""",
"details": 
"""
<pre>
<code class="language-python">
from torch import nn

## --------------------------------------------------------------------------------
## GeLU (Gaussian Error Linear Units)
## --------------------------------------------------------------------------------
## GeLU(x) = x * phi(x)
gelu = nn.GeLU()
y = gelu(x)
## --------------------------------------------------------------------------------

## --------------------------------------------------------------------------------
## SiLU / swish (Sigmoid Linear Unit)
## --------------------------------------------------------------------------------
inplace = False
## silu(x) = x * sigmoid(x)
silu = nn.SiLU(inplace)
y = silu(x)
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
"info": "",
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
weight_decay = 0.01  # float. Weight decay coefficient
# ...
adam_optim = AdamW(params, lr, betas, weight_decay)
## --------------------------------------------------------------------------------
</code>
</pre>
""",
},
{
"title": "huggingface",
"author": "",
"organization": "",
"date": "20230530",
"venue": "docs",
"pdf_url": "https://huggingface.co/docs",
"code_url": "",
"name": "huggingface",
"comment": "",
"category": "torch & torchvision",
"jupyter_notes": "",
"info": "",
"summary": """It includes tools from huggingface: <a href="https://huggingface.co/docs/huggingface_hub/main/en/package_reference/file_download#huggingface_hub.snapshot_download">snapshot_download</a>""",
"details": 
"""
<pre>
<code class="language-python">
from huggingface_hub import snapshot_download

## --------------------------------------------------------------------------------
## Download checkpoints from huggingface
## --------------------------------------------------------------------------------
repo_id = /  # *** str. A user name and a repo name, e.g., "Qwen/Qwen-VL-Chat"
repo_type = None  # *** str. "dataset", "space", or "model"
local_dir = None  # *** str or Path. If provided, directory to place the downloaded files
token = None  # str, bool. User token
max_workers = 8  # int. Number of concurrent threads to download files
# ...

snapshot_download(repo_id, repo_type, local_dir, token, max_workers)
</code>
</pre>
""",
},
]