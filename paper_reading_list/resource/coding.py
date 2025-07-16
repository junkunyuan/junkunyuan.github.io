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
# <ul>
#     <li>
# </ul>
# """,
# },
{
"title": "new paper Data Loader",
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
"summary": """""",
"details": 
"""
<pre>
<code class="language-python">
from torch.utils.data import DataLoader, Dataset, Sampler

dataset = /  # Dataset
batch_size = 1  # int. Number of samples per batch.
shuffle = False  # bool. If True, have the data shuffled at every epoch
sampler = None  # Sampler or Iterable. Define how to draw samples
batch_sampler = None  # Sampler or Iterable. customize sampling by giving indices
num_workers = 0  # int. Number of subprocesses to use for data loading
collate_fn = None  # Callable. Merge a list of samples to form a batch of tensors.
pin_memory = False  # bool. If True, copy Tensors into device/CUDA pinned memory before ruturning.
drop_last = False  # bool. If True, drop the last incomplete batch
timeout = 0  # numeric. If positive, set the timeout value for collecting a batch from workers.
worker_init_fn = None  # Callable. If not None, this will be called with the worker id as input
multiprocessing_context = None  # str or multiprocessing.context.BaseContext. If None, use the default multiprocessing context  If None, use the default multiprocessing context


data_loader = DataLoader(
    dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler, batch_sampler=batch_sampler, 
    num_workers=num_workers, collate_fn=collate_fn, pin_memory=pin_memory, drop_last=drop_last, 
    timeout=timeout, worker_init_fn=worker_init_fn, multiprocessing_context=multiprocessing_context, 
    generator=None, *, prefetch_factor=None, persistent_workers=False, pin_memory_device='', in_order=True)
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
"summary": """It includes tools to transform and augment data.""",
"details": 
"""
<pre>
<code class="language-python">
from torchvision import transforms
from torchvision.transforms.InterpolationMode import BILINEAR, NEAREST, NEAREST_EXACT, BILINEAR, BICUBIC 

## --------------------------------------------------------------------------------
## <a class="no_dec a_black" href="https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.Resize.html?highlight=transforms+resize#torchvision.transforms.Resize"><b>Geometry: Resize</b></a>
## --------------------------------------------------------------------------------
size = /  # sequence or int. For example (512, 768)
interpolation = BILINEAR  # InterpolationMode
max_size = /  # int. Maximum allowed for the longer image edge, only supported if `size` is an int
antialias = /  # bool. Apply antialiasing, only under bilinear or bicubic modes

trans = <b>transforms.Resize</b>(size, interpolation=interpolation, max_size=max_size, antialias=antialias)
image_trans = trans(image)  # PIL Image => PIL Image or Tensor => Tensor
## --------------------------------------------------------------------------------

## --------------------------------------------------------------------------------
## <a class="no_dec a_black" href="https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.RandomHorizontalFlip.html#torchvision.transforms.RandomHorizontalFlip"><b>Geometry: RandomHorizontalFlip</b></a>
## --------------------------------------------------------------------------------
p = 0.5  # float. Probability to flip image

trans = <b>transforms.RandomHorizontalFlip</b>(p=p)
image_trans = trans(image)  # PIL Image => PIL Image or Tensor => Tensor
## --------------------------------------------------------------------------------

## --------------------------------------------------------------------------------
## <a class="no_dec a_black" href="https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.ToTensor.html?highlight=totensor#torchvision.transforms.ToTensor"><b>Conversion: ToTensor</b></a>
## --------------------------------------------------------------------------------
## Input: PIL Image / numpy.ndarray (np.uint8) of shape (HxWxC) in the range [0, 255]
## Output: torch.FloatTensor of shape (CxHxW) in the range (0.0, 1.0)
## Other inputs: only apply type transform
trans = <b>transforms.ToTensor</b>()
image_trans = trans(image)  # PIL Image / ndarray => Tensor
## --------------------------------------------------------------------------------

## --------------------------------------------------------------------------------
## <a class="no_dec a_black" href="https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.Compose.html#torchvision.transforms.Compose"><b>Composition: Compose</b></a>
## --------------------------------------------------------------------------------
transforms = /  # list of Transform objects

trans = <b>transforms.Compose</b>(transforms)
image_trans = trans(image)  # PIL Image / ndarray / Tensor => Tensor
## --------------------------------------------------------------------------------

## --------------------------------------------------------------------------------
## <a class="no_dec a_black" href="https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.Normalize.html#torchvision.transforms.Normalize"><b>Miscellaneous: Normalize</b></a>
## --------------------------------------------------------------------------------
mean = /  # sequence. Means for each channel.
std = /  # sequence. Standard deviations for each channel.
inplace = False  # bool. Bool to make this operation in-place.

trans = <b>transforms.Normalize</b>(mean, std, inplace=inplace)
image_trans = trans(image)  # Tensor => Tensor
## --------------------------------------------------------------------------------
</code>
</pre>
""",
},
]