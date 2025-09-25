LIST = dict()
LIST["file"] = "coding.html"
LIST["title"] = "Coding and Engineering"
LIST["description"] = "Tools used to build AI systems."
LIST["categories"] = ["PyTorch", "Distributed Training", "Tools"]
LIST["papers"] = [
# {
# "title": "",
# "date": "",
# "venue": "",
# "name": "",
# "category": "",
# "summary": 
# """
# """,
# "details": 
# """
# <table class="center">
# <tr>
# <th>category</th>
# <th>tool (alphabetical)</th>
# </tr>
# <tr>
# <td>default</td>
# <td><a href="#"></a> &nbsp;&nbsp;</td>
# </tr>
# </table>
# <p class="larger" id="">
# <b><a href=""></a>: </b>
# .
# <pre>
# <code class="language-python" style="font-size: 14px;">
# </code>
# </pre>
# """,
# },
{
"title": "Git",
"date": "20250923",
"venue": "git tools",
"name": "git",
"category": "Tools",
"jupyter_notes": "",
"info": "",
"summary": """Git is a <b>distributed version control system</b> that allows you to track changes in your code and collaborate with others.""",
"details": 
f"""
<table class="center">
<tr>
<th>category</th>
<th>tool (alphabetical)</th>
</tr>
<tr>
<td>Setup and configure</td>
<td>
<a href="#git-config">config</a> &nbsp;&nbsp;
<a href="#ssh keys">ssh keys</a> &nbsp;&nbsp;</td>
</tr>
<tr>
<td>Get and create projects</td>
<td>
<a href="#git-clone">clone</a> &nbsp;&nbsp;
</tr>
</table>

<p class="larger" id="git-clone">
<b><a href="https://git-scm.com/docs/git-clone">clone</a>: </b>
clone a repository into a new directory.

<pre>
<code class="language-bash" style="font-size: 14px;">
git clone git@github.com:[user name]/[repo name].git
</code>
</pre>

<p class="larger" id="git-config">
<b><a href="https://git-scm.com/docs/git-config">config</a>: </b>
get and set repository or global options.

<pre>
<code class="language-bash" style="font-size: 14px;">
git config --list  # list all config
git config user.name [your name] && git config user.email [your email]  # repo config; cd to repo and execute
git config --global user.name [your name] && git config --global user.email [your email]  # global config
</code>
</pre>

<p class="larger" id="ssh keys">
<b><a href="https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent">ssh keys</a>: </b>
generate a new SSH key to use for authentication.
<pre>
<code class="language-bash" style="font-size: 14px;">
## 1. Check if SSH key exists: *.pub
ls -al ~/.ssh
## 2. If not, generate a new SSH key
ssh-keygen -t rsa -b 4096 -C [your GitHub email]
## 3. Copy SSH key
cat ~/.ssh/[your key name].pub
## 4. Open GitHub -> Settings -> SSH and GPG keys -> New SSH key -> paste the SSH key
## 5. Test
ssh -T git@github.com  # to see if it prints "Hi *! You've successfully ..."
</code>
</pre>
""",
},
{
"title": "Docker",
"date": "20250923",
"venue": "docker tools",
"name": "docker",
"category": "Tools",
"summary": 
"""
Docker is a <b>containerization tool</b> that allows you to <b>package your application with all its dependencies</b> into a container.
""",
"details": 
"""
<table class="center">
<tr>
<th>category</th>
<th>tool (alphabetical)</th>
</tr>
<tr>
<td>docker</td>
<td>
<a href="#start & restart & stop">start & restart & stop</a> &nbsp;&nbsp;
</td>
</tr>
<tr>
<td>image</td>
<td><a href="#docker-ls & pull & rm">ls & pull & rm</a> &nbsp;&nbsp;</td>
</tr>
<tr>
<td>container</td>
<td>
<a href="#docker-ps & run & enter & stop">ps & run & enter & stop</a> &nbsp;&nbsp;
</td>
</tr>
</table>

<p class="larger" id="docker-ps & run & enter & stop">
<b><a href="">ps & run & enter & stop</a>: </b>
list containers, run a container, enter a container, stop a container.
<pre>
<code class="language-bash" style="font-size: 14px;">
docker ps -a  # list all containers
## --------------------------------------------------------------------------------
## Recommended params: -i: interactive mode; -d: detached mode; -t: allocate a pseudo-TTY
## If need mapping: add "-v [local path, e.g., /home/user]:/root"
## If run GPUs: add "--gpu all", "--ipc host"
## If run container without executing any commands, append "tail -f /dev/null"
docker run [params, e.g., -dit] --name [container name] [image name]  # run a container  # run a container
## --------------------------------------------------------------------------------
docker exec -it [container name or container id] /bin/bash  # enter a container
docker stop [container name or container id]  # stop a container
docker restart [container name or container id]  # restart a container
docker rm [container name or container id]  # remove a container
</code>
</pre>

<p class="larger" id="docker-ls & pull & rm">
<b><a href="">ls & pull & rm</a>: </b>
list, pull, and remove images.
<pre>
<code class="language-bash" style="font-size: 14px;">
docker images  # or docker image ls; list all images
docker pull [image name]  # pull an image from a registry
docker rmi [image name or image id]  # remove an image
</code>
</pre>

<p class="larger" id="start & restart & stop">
<b><a href="">start & restart & stop</a>: </b>
start, restart, stop docker.
<pre>
<code class="language-bash" style="font-size: 14px;">
sudo service docker start  # start docker
sudo service docker restart  # restart docker
sudo service docker stop  # stop docker
</code>
</pre>
""",
},
{
"title": "DeepSpeed",
"author": "",
"organization": "",
"date": "20250821",
"venue": "deepspeed framework",
"pdf_url": "",
"code_url": "",
"name": "deepspeed",
"comment": "",
"category": "Distributed Training",
"jupyter_notes": "",
"info": "",
"summary": """DeepSpeed is an open-sourced deep learning optimization library developed by <b>Microsoft Research</b>, designed to simplify and accelerate the training and deployment of <b>large-scale</b> deep learning models.""",
"details": 
"""
<table class="center">
    <tr>
        <th>stage</th>
        <th>partition</th>
        <th>memory saving</th>
        <th>complexity</th>
    </tr>
    <tr>
        <td>stage 1</td>
        <td>optimizer states</td>
        <td>~40% - 60% (for Adam)</td>
        <td>low</td>
    </tr>
    <tr>
        <td>stage 2</td>
        <td>optimizer states & gradients</td>
        <td> additional ~15% - 25%</td>
        <td>medium</td>
    </tr>
    <tr>
        <td>stage 3</td>
        <td>optimizer states & gradients & model parameters</td>
        <td> up to 80% - 90%</td>
        <td>high</td>
    </tr>
</table>
""",
},
{
"title": "Data Transforms",
"author": "",
"organization": "",
"date": "20240701",
"venue": "transform and augment data",
"pdf_url": "",
"code_url": "",
"name": "data transforms",
"comment": "",
"category": "PyTorch",
"jupyter_notes": "",
"info": "",
"summary": """""",
"details": 
"""
<table class="center">
<tr>
<th>category</th>
<th>class / function (alphabetical)</th>
</tr>

<tr><td>geometry</td><td>
<a href="#RandomHorizontalFlip">RandomHorizontalFlip</a>
</td></tr>

<tr><td>resizing</td><td>
<a href="#Resize">Resize</a>
</td></tr>

<tr><td>conversion</td><td>
<a href="#Normalize">Normalize</a> &nbsp;&nbsp;
<a href="#ToTensor">ToTensor</a> &nbsp;&nbsp;
</td></tr>

<tr><td>else</td><td>
<a href="#Compose">Compose</a> &nbsp;&nbsp;
</td></tr>

</table>

<pre>
<code class="language-python" style="font-size: 14px;">
from torchvision import transforms
from torchvision.transforms import InterpolationMode BILINEAR, NEAREST, BICUBIC 
</code>
</pre>

<p class="larger" id="RandomHorizontalFlip">
<b><a href="https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.RandomHorizontalFlip.html">RandomHorizontalFlip</a>: </b>
horizontally flip the image randomly with a probability. 
<pre>
<code class="language-python" style="font-size: 14px;">
p = 0.5  # *** float. Probability to flip
trans = <b>transforms.RandomHorizontalFlip</b>(p)
image_trans = trans(image)  # PIL Image => PIL Image, or Tensor => Tensor
</code>
</pre>

<p class="larger" id="Resize">
<b><a href="https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.Resize.html">Resize</a>: </b>
resize the image to a size.
<pre>
<code class="language-python" style="font-size: 14px;">
## When `size` is int, the image shorter size will be resized to `size` with aspect ratio fixed
## When `size` is tuple, the image size will be resized to `size` with aspect ratio changed
size = /  # *** tuple or int
## NEAREST: fastest; lowest quality, jagged
## BILINEAR: fast; low quality, blur
## (recommend) BICUBIC: slow; good quality
## (recommend) LANCOZ: slowest; best quality
interpolation = InterpolationMode.BILINEAR  # *** InterpolationMode
## The shorter size may lower than `size` if longer size exceeds `max_size` after resizing
max_size = None  # int. Maximum allowed for the longer edge, supported if `size` is int
trans = transforms.Resize(size, interpolation, max_size)
image_trans = trans(image)  # PIL Image => PIL Image, or Tensor => Tensor
</code>
</pre>

<p class="larger" id="ToTensor">
<b><a href="https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.ToTensor.html">ToTensor</a>: </b>
convert a PIL Image or ndarray to tensor and scale the values accordingly.
<pre>
<code class="language-python" style="font-size: 14px;">
## Input: PIL Image / numpy.ndarray (np.uint8) of shape (HxWxC) in the range [0, 255]
## Output: torch.FloatTensor of shape (CxHxW) in the range (0.0, 1.0)
## Other inputs: only apply type transform
trans = <b>transforms.ToTensor</b>()
image_trans = trans(image)
</code>
</pre>

<p class="larger" id="Compose">
<b><a href="https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.Compose.html">Compose</a>: </b>
compose several transforms.
<pre>
<code class="language-python" style="font-size: 14px;">
transforms = /  # *** list of Transform objects
trans = <b>transforms.Compose</b>(transforms)
image_trans = trans(image)  # PIL Image / ndarray / Tensor => Tensor
</code>
</pre>

<p class="larger" id="Normalize">
<b><a href="https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.Normalize.html">Normalize</a>: </b>
normalize a tensor image with mean and standard deviation.
<pre>
<code class="language-python" style="font-size: 14px;">
mean = /  # *** sequence. Means for each channel
std = /  # *** sequence. Standard deviations for each channel
inplace = False  # bool. Bool to make this operation in-place
trans = <b>transforms.Normalize</b>(mean, std, inplace)
image_trans = trans(image)  # Tensor => Tensor
</code>
</pre>
""",
},
{
"title": "Data Loader",
"author": "",
"organization": "",
"date": "20240630",
"venue": "load data",
"pdf_url": "",
"code_url": "",
"name": "data loader",
"comment": "",
"category": "PyTorch",
"jupyter_notes": "",
"info": "",
"summary": """""",
"details": 
"""
<pre>
<code class="language-python" style="font-size: 14px;">
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

dataset = /  # *** Dataset
batch_size = 1  # *** int. Number of samples per batch
shuffle = False  # *** bool. If True, have the data shuffled at every epoch
sampler = None  # Sampler or Iterable. Define how to draw samples
num_workers = 0  # *** int. Number of subprocesses to use for data loading
collate_fn = None  # Callable. Merge a list of samples to form a batch
pin_memory = False  # *** bool. If True, copy Tensors into CUDA pinned memory
drop_last = False  # *** bool. If True, drop the last incomplete batch
timeout = 0  # numeric. If positive, set timeout for collecting a batch from workers
prefetch_factor = None  # int. Default = None if num_workers == 0 else 2
# ...
data_loader = DataLoader(dataset, batch_size, shuffle, sampler, num_workers, 
    collate_fn, pin_memory, drop_last, timeout, prefetch_factor)
</code>
</pre>
""",
},
{
"title": "Operation",
"author": "",
"organization": "",
"date": "20230630",
"venue": "tensor operations",
"pdf_url": "",
"code_url": "",
"name": "operation",
"comment": "",
"category": "PyTorch",
"jupyter_notes": "",
"info": "",
"summary": 
"""""",
"details": 
"""
<table class="center">
<tr>
<th>category</th>
<th>class / function (alphabetical)</th>
</tr>

<tr>
<td>operations</td>
<td>
<a href="#basic operations">basic operations</a> &nbsp;&nbsp;
<a href="#einsum">einsum</a> &nbsp;&nbsp;
<a href="#isclose & allclose">isclose & allclose</a> &nbsp;&nbsp;
<a href="#matmul">matmul</a> &nbsp;&nbsp;
<a href="#mean & var">mean & var</a> &nbsp;&nbsp;
<a href="#softmax">softmax</a> &nbsp;&nbsp;
</td>
</tr>

<tr>
<td>data generation</td>
<td>
<a href="#arange">arange</a> &nbsp;&nbsp;
<a href="#uniform & normal">uniform & normal</a> &nbsp;&nbsp;
<a href="#zeros & ones">zeros & ones</a> &nbsp;&nbsp;
</td>
</tr>

<tr><td>size</td><td>
<a href="#cat">cat</a> &nbsp;&nbsp;
<a href="#chunk & split">chunk & split</a> &nbsp;&nbsp;
<a href="#flatten">flatten</a> &nbsp;&nbsp;
<a href="#permute">permute</a> &nbsp;&nbsp;
<a href="#reshape & view">reshape & view</a> &nbsp;&nbsp;
<a href="#size & shape">size & shape</a> &nbsp;&nbsp;
<a href="#squeeze & unsqueeze">squeeze & unsqueeze</a> &nbsp;&nbsp;
<a href="#tranpose">tranpose</a> &nbsp;&nbsp;
<a href="#unbind">unbind</a> &nbsp;&nbsp;
<a href="#unsqueeze">unsqueeze</a> &nbsp;&nbsp;
</td></tr>

<tr><td>else</td><td>
<a href="#where">where</a> &nbsp;&nbsp;
</td></tr>

</table>

<pre>
<code class="language-python" style="font-size: 14px;">
import torch
</code>
</pre>

<p class="larger" id="basic operations">
<b><a href="https://docs.pytorch.org/docs/stable/torch.html">basic operations</a>:</b> 
exp, sin, cos, sqrt.
<pre>
<code class="language-python" style="font-size: 14px;">
y = torch.function(x)  # function: exp, sin, cos, sqrt
</code>
</pre>

<p class="larger" id="mean & var">
<b><a href="https://docs.pytorch.org/docs/stable/mean.html">mean & var</a></b>. 
<pre>
<code class="language-python" style="font-size: 14px;">
dim = /  # *** int or tuple of ints. Dims to reduce
keepdim = False # *** bool. If True, return tensor with the same dims
mean = x.mean(dim, keepdim)

## In version>=2.0, `correction=1` equals to `unbiased=True`, `correction=0` equals to `unbiased=False` 
correction = 1  # *** int. 
var = x.var(dim, keepdim, correction)
</code>
</pre>

<p class="larger" id="softmax">
<b><a href="https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html">softmax</a></b>. 
<pre>
<code class="language-python" style="font-size: 14px;">
dim = None  # *** int. Dim to apply softmax
y = x.softmax(dim)
</code>
</pre>

<p class="larger" id="matmul">
<b><a href="https://docs.pytorch.org/docs/stable/generated/torch.matmul.html">matmul</a>:</b> 
matrix multiplication.
<pre>
<code class="language-python" style="font-size: 14px;">
other = /  # *** tensor
y = x.matmul(other)
</code>
</pre>

<p class="larger" id="einsum">
<b><a href="https://docs.pytorch.org/docs/stable/generated/torch.einsum.html">einsum</a>:</b> 
Einstein summation convention. 
<pre>
<code class="language-python" style="font-size: 14px;">
equation = /  # *** str. The subscript for the Einstein summation
operands = /  # *** list of tensor. The tensor to be computed
## torch.einsum("ii", tensor)  # trace
## torch.einsum("ii->i", tensor)  # diagonal
## torch.einsum("i,j->ij", tensor1, tensor2)  # outer product
## torch.einsum("bij,bjk->bik", tensor1, tensor2)  # batch matrix multiplication
## torch.einsum("...ij->...jk", tensor)  # batch permute
y = torch.einsum(equation, operands)
</code>
</pre>

<p class="larger" id="isclose & allclose">
<b><a href="https://docs.pytorch.org/docs/stable/generated/torch.isclose.html">isclose & allclose</a>:</b> 
check whether two tensors are close. 
<pre>
<code class="language-python" style="font-size: 14px;">
other = /  # *** tensor. The second tensor to compare
rtol = 1e-5  # float. Relative tolerance
atol = 1e-8  # float. Absolute tolerance
equal_nan = False  # bool. If True, then two NaN will be considered equal
## Check if elements satisfy: |input - other| <= atol + rtol * other
x.isclose(other, rtol, atol, equal_nan)  # return a tensor of bool
x.allclose(other, rtol, atol, equal_nan)  # return True or False
## --------------------------------------------------------------------------------
</code>
</pre>

<p class="larger" id="zeros & ones">
<b>
<a href="https://docs.pytorch.org/docs/stable/generated/torch.zeros.html">zeros</a> & 
<a href="https://docs.pytorch.org/docs/stable/generated/torch.ones.html">ones</a>:</b> 
fill a tensor with a given value. 
<pre>
<code class="language-python" style="font-size: 14px;">
size = /  # *** sequence of int. The shape of output
y = torch.zeros(size)
y = torch.ones(size)
</code>
</pre>


<p class="larger" id="uniform & normal">
<b>
<a href="https://docs.pytorch.org/docs/stable/generated/torch.rand.html">uniform</a> & 
<a href="https://docs.pytorch.org/docs/stable/generated/torch.randn.html">normal</a>:</b> 
fill a tensor with a given value. 
<pre>
<code class="language-python" style="font-size: 14px;">
size = /  # *** sequence of int. The shape of output
generator = None  # torch.Generator. A pseudorandom number generator for sampling
requires_grad = False  # bool. If use autograd
dtype = None  # torch.dtype. The desired data type
device = None  # torch.device. The desired device 
y = torch.rand(size, generator, requires_grad, dytpe, device)  # uniform distribution U(0, 1)
y = torch.randn(size, generator, requires_grad, dytpe, device)  # standard normal distribution N(0, 1)
</code>
</pre>

<p class="larger" id="arange">
<b><a href="https://docs.pytorch.org/docs/stable/generated/torch.arange.html">arange</a>:</b> 
a sequence in order. 
<pre>
<code class="language-python" style="font-size: 14px;">
start = 0  # *** number. The starting value
end = /  # *** number. The ending value
step = 1  # *** number. The gap between adjacent points
arange = torch.arange(start, end, step)
</code>
</pre>

<p class="larger" id="size & shape">
<b><a href="https://docs.pytorch.org/docs/stable/generated/torch.Tensor.size.html">size</a> & 
<a href="https://docs.pytorch.org/docs/stable/generated/torch.Tensor.shape.html">shape</a>:</b> 
get tensor size.
<pre>
<code class="language-python" style="font-size: 14px;">
dim = None  # int. Dim to retrieve the size
size = x.size(dim)  # => torch.Size or int
size = x.shape  #  => torch.Size
</code>
</pre>

<p class="larger" id="reshape & view">
<b><a href="https://docs.pytorch.org/docs/stable/generated/torch.reshape.html">reshape</a> & 
<a href="https://docs.pytorch.org/docs/stable/generated/torch.Tensor.view.html">view</a>:</b> 
reshape a tensor with the given shape. 
<pre>
<code class="language-python" style="font-size: 14px;">
shape = /  # sequence of int. The new shape. A single dim could be -1
y = x.reshape(shape)  # recommend since it could call .contiguous() if needed
y = x.view(shape)
</code>
</pre>

<p class="larger" id="flatten">
<b><a href="https://docs.pytorch.org/docs/stable/generated/torch.flatten.html">flatten</a>:</b> 
flatten along the given dimensions. 
<pre>
<code class="language-python" style="font-size: 14px;">
start_dim = 0  # *** int. The first dimension to flatten
end_dim = -1  # *** int. The last dimension to flatten
y = x.flatten(start_dim, end_dim)
</code>
</pre>

<p class="larger" id="tranpose">
<b><a href="https://docs.pytorch.org/docs/stable/generated/torch.tranpose.html">tranpose</a>:</b> 
swap two dimensions. 
<pre>
<code class="language-python" style="font-size: 14px;">
dim0 = /  # *** int. The first dim to be tranposed
dim1 = /  # *** int. The second dim to be tranposed
y = x.tranpose(dim0, dim1)
</code>
</pre>

<p class="larger" id="permute">
<b><a href="https://docs.pytorch.org/docs/stable/generated/torch.permute.html">permute</a>:</b> 
permute dimensions of a tensor. 
<pre>
<code class="language-python" style="font-size: 14px;">
dims = /  # *** sequence of int. The desired ordering of dims
y = x.permute(dims)
</code>
</pre>

<p class="larger" id="unsqueeze">
<b><a href="https://docs.pytorch.org/docs/stable/generated/torch.squeeze.html">squeeze</a> & 
<a href="https://docs.pytorch.org/docs/stable/generated/torch.unsqueeze.html">unsqueeze</a>:</b> 
insert and remove dimensions. 
<pre>
<code class="language-python" style="font-size: 14px;">
dim = None  # *** int or tuple of ints. If given, only the dim will be squeezed
y = x.squeeze(dim)

dim = /  # *** int. The index at which to insert the singleton dim
y = x.unsqueeze(dim)  # Eqaul to y = x[:, :, None, :] when dim = 2
</code>
</pre>

<p class="larger" id="cat">
<b><a href="https://docs.pytorch.org/docs/stable/generated/torch.cat.html">cat</a>:</b> 
concatenate tensors along a dimension. 
<pre>
<code class="language-python" style="font-size: 14px;">
tensors = /  # *** tuple of tensors. Tensors with the same shape except in the cat dim
dim = 0  # *** int. The concatenation dim
y = torch.cat(tensors, dim)
</code>
</pre>

<p class="larger" id="unbind">
<b><a href="https://docs.pytorch.org/docs/stable/generated/torch.unbind.html">unbind</a>:</b> 
remove a dimension by splitting it. 
<pre>
<code class="language-python" style="font-size: 14px;">
dim = 0  # *** int. Dim to remove
y = x.unbind(dim)
</code>
</pre>

<p class="larger" id="arange">
<b><a href="https://docs.pytorch.org/docs/stable/generated/torch.chunk.html">chunk</a> & 
<a href="https://docs.pytorch.org/docs/stable/generated/torch.split.html">split</a>:</b> 
split a tensor with chunk numbers or split sizes.
<pre>
<code class="language-python" style="font-size: 14px;">
chunks = /  # *** int
dim = 0  # *** int
## If the given dim is divisible by chunks, all returned chunks will be the same size
## If the given dim is not divisible by chunks, the last one will not be the same size
## If such division is not possible, it returns fewer than the specified number of chunks
y = x.chunk(chunks, dim)

indices_or_sections = /  # *** tensor, int, list, tuple of ints
dim = 0  # *** int. Dim along which to split the tensor
## If split_size_or_sections is an integer type, split into equally sized chunks
## If split_size_or_sections is a list, split into len(split_size_or_sections) chunks
y = x.split(indices_or_sections, dim)
</code>
</pre>

<p class="larger" id="where">
<b><a href="https://docs.pytorch.org/docs/stable/generated/torch.where.html">where</a>:</b> 
select elements from a tensor.
<pre>
<code class="language-python" style="font-size: 14px;">
condition = /  # *** bool. When True, yield input, otherwise yield other
input = /  # *** tensor or scalar
other = /  # *** tensor or scalar
y = torch.where(condition, input, output)
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
"venue": "modules to build models",
"pdf_url": "",
"code_url": "",
"name": "module",
"comment": "",
"category": "PyTorch",
"jupyter_notes": "",
"info": "",
"summary": 
"""
""",
"details": 
"""
<table class="center">
<tr>
<th>category</th>
<th>tool (alphabetical)</th>
</tr>

<tr>
<td>parameter</td>
<td>
<a href="#Parameter & Buffer">Parameter & Buffer</a> &nbsp;&nbsp;
</td>
</tr>

<tr>
<td>convolution</td>
<td>
<a href="#Conv2d">Conv2d</a> &nbsp;&nbsp;
<a href="#Conv3d">Conv3d</a> &nbsp;&nbsp;
</td>
</tr>

<tr>
<td>other module</td>
<td>
<a href="#Linear">Linear</a> &nbsp;&nbsp;
</td>
</tr>

<tr>
<td>else</td>
<td>
<a href="#Dropout">Dropout</a> &nbsp;&nbsp;
</td>
</tr>

</td>
</tr>

</table>

<pre>
<code class="language-python" style="font-size: 14px;">
import torch
from torch import nn 
</code>
</pre>

<p class="larger" id="Parameter & Buffer">
<b><a href="https://docs.pytorch.org/docs/stable/generated/torch.nn.parameter.Parameter.html">Parameter</a> & 
<a href="https://docs.pytorch.org/docs/stable/generated/torch.nn.parameter.Buffer.html">Buffer</a></b>.

<pre>
<code class="language-python" style="font-size: 14px;">
data = /  # *** tensor. Parameter tensor
requires_grad = True
gamma = torch.Parameter(data, requires_grad)

persistent = True  # whether the buffer is part of the module's state_dict
gamma = self.register_buffer(data, persistent)  # usually used in __init__
gamma = nn.paramter.Buffer(data, persistent)  # not usually used
</code>
</pre>

<p class="larger" id="Linear">
<b><a href="https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html">Linear</a>:</b> affine linear transformation.
<pre>
<code class="language-python" style="font-size: 14px;">
in_features = /  # *** int. Input features
out_features = /  # *** int. Output features
bias = True  # *** bool. If True, learn an additive bias
device = None  # torch.device or int 
dtype = None  # torch.dtype
linear = Linear(in_features, out_features, bias, device, dtype)  
y = linear(x)  # [..., H_in] => [..., H_out]
</code>
</pre>

<p class="larger" id="Conv2d">
<b><a href="https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html">Conv2d</a>:</b> 2D convolution.
<pre>
<code class="language-python" style="font-size: 14px;">
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
conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, 
    padding, dilation, groups, bias, padding_mode, device, dtype)
## H_out = [(H_in + 2*padding[0] - dilation[0]*(kernel[0]-1)-1) / stride[0] + 1]
## W_out = [(W_in + 2*padding[1] - dilation[1]*(kernel[1]-1)-1) / stride[1] + 1]
y = conv2d(x)  # [B, C, H_in, W_in] => [B, C, H_out, W_out]
</code>
</pre>

<p class="larger" id="Conv3d">
<b><a href="https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv3d.html">Conv3d</a>:</b> 3D convolution.
<pre>
<code class="language-python" style="font-size: 14px;">
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
conv3d = nn.Conv3d(in_channels, out_channels, kernel_size, stride, 
    padding, dilation, groups, bias, padding_mode, device, dtype)
## D_out = [(D_in + 2*padding[0] - dilation[0]*(kernel[0]-1)-1) / stride[0] + 1]
## H_out = [(H_in + 2*padding[1] - dilation[1]*(kernel[1]-1)-1) / stride[1] + 1]
## W_out = [(W_in + 2*padding[2] - dilation[2]*(kernel[2]-1)-2) / stride[2] + 1]
y = conv3d(x)  # [B, C, D_in, H_in, W_in] => [B, C, D_out, H_out, W_out]
</code>
</pre>

<p class="larger" id="Dropout">
<b><a href="https://docs.pytorch.org/docs/stable/generated/torch.nn.Dropout.html">Dropout</a></b>.
<pre>
<code class="language-python" style="font-size: 14px;">
p = 0.5  # *** float. Probability of an element to be zeroed
inplace = False  # bool
dropout = nn.Dropout(p, inplace)
y = dropout(x)
</code>
</pre>
""",
},
{
"title": "Activation Function",
"author": "",
"organization": "",
"date": "20240628",
"venue": "activation functions",
"pdf_url": "",
"code_url": "",
"name": "activation function",
"comment": "",
"category": "PyTorch",
"jupyter_notes": "",
"info": "",
"summary": 
"""
""",
"details": 
"""
<table class="center">
<tr>
<th>tool (alphabetical)</th>
<th>popular applications</th>
</tr>

<tr>
<td>
<a href="#GeLU">GeLU</a> &nbsp;&nbsp;
</td>
<td>
/
</td>
</tr>

<tr>
<td>
<a href="#SiLU/swish">SiLU/swish</a> &nbsp;&nbsp;
</td>
<td>
/
</td>
</tr>

</table>

<pre>
<code class="language-python" style="font-size: 14px;">
from torch import nn
</code>
</pre>

<p class="larger" id="GeLU">
<b><a href="https://docs.pytorch.org/docs/stable/generated/torch.nn.GELU.html">GeLU</a>:</b> Gaussian Error Linear Units function, \(\mathrm{GeLU}(x)=x*\phi(x)\), where \(\phi\) is cumulative distribution function.
<pre>
<code class="language-python" style="font-size: 14px;">
gelu = nn.GeLU()
y = gelu(x)
</code>
</pre>

<p class="larger" id="SiLU/swish">
<b><a href="https://docs.pytorch.org/docs/stable/generated/torch.nn.SiLU.html">SiLU/swish</a>:</b>  Sigmoid Linear Unit function, \(\mathrm{SiLU}(x)=x*\sigma(x)\), where \(\sigma\) is logistic function.

<pre>
<code class="language-python" style="font-size: 14px;">
inplace = False
silu = nn.SiLU(inplace)
y = silu(x)
</code>
</pre>
""",
},
{
"title": "Optimizer",
"author": "",
"organization": "",
"date": "20240628",
"venue": "optimizers",
"pdf_url": "",
"code_url": "",
"name": "optimizer",
"comment": "",
"category": "PyTorch",
"jupyter_notes": "",
"info": "",
"summary": """It includes tools for building optimization algorithms: <a href="https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html#torch.optim.AdamW.step">AdamW</a>.""",
"details": 
"""
<pre>
<code class="language-python" style="font-size: 14px;">
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
"venue": "huggingface tools",
"pdf_url": "",
"code_url": "",
"name": "huggingface",
"comment": "",
"category": "PyTorch",
"jupyter_notes": "",
"info": "",
"summary": """It includes tools from huggingface: <a href="https://huggingface.co/docs/huggingface_hub/main/en/package_reference/file_download#huggingface_hub.snapshot_download">snapshot_download</a>""",
"details": 
"""
<pre>
<code class="language-python" style="font-size: 14px;">
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