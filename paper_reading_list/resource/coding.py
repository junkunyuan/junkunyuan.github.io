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
"title": "Data Transforms",
"author": "",
"organization": "",
"date": "20240101",
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
from torchvision import transforms
from torchvision.transforms.InterpolationMode import BILINEAR, NEAREST, NEAREST_EXACT, BILINEAR, BICUBIC 

## --------------------------------------------------------------------------------
## <a class="no_dec a_black" href="https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.Resize.html?highlight=transforms+resize#torchvision.transforms.Resize"><b>Geometry: Resize</b></a>
## --------------------------------------------------------------------------------
size = /  # sequence or int. For example (512, 768)
interpolation = BILINEAR  # InterpolationMode
max_size = /  # int, optional. Maximum allowed for the longer edge of the resized image, only supported if `size` is an int
antialias = /  # bool, optional. Apply antialiasing, only affects tensors with bilinear or bicubic modes

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
inplace = False  # bool, optional. Bool to make this operation in-place.

trans = <b>transforms.Normalize</b>(mean, std, inplace=inplace)
image_trans = trans(image)  # Tensor => Tensor
## --------------------------------------------------------------------------------
</pre>
""",
},
]