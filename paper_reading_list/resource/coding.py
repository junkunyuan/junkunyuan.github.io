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
"title": "new paper Data Transforms",
"author": "",
"organization": "",
"date": "20240816",
"venue": "docs",
"pdf_url": "https://docs.pytorch.org/vision/stable/transforms.html",
"code_url": "",
"name": "data transforms",
"comment": "",
"category": "torch & torchvision",
"jupyter_notes": "",
"summary": """Resize""",
"details": 
"""
<pre>
from torchvision import transforms
from torchvision.transforms.InterpolationMode import BILINEAR, NEAREST, NEAREST_EXACT, BILINEAR, BICUBIC 

## --------------------------------------------------------------------------------
## <a class="no_dec" href="https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.Resize.html?highlight=transforms+resize#torchvision.transforms.Resize">Resize</a>
## --------------------------------------------------------------------------------
size = /  # sequence or int. For example (512, 768)
interpolation = BILINEAR  # InterpolationMode
max_size = /  # int optional. Maximum allowed for the longer edge of the resized image, only supported if `size` is an int
antialias = /  # bool optional. Apply antialiasing, only affects tensors with bilinear or bicubic modes

resize_trans = <b>transforms.Resize</b>(size, interpolation=interpolation, max_size=max_size, antialias=antialias)
image_transformed = resize_trans(image)  # PIL Image => PIL Image or Tensor => Tensor
## --------------------------------------------------------------------------------

## --------------------------------------------------------------------------------
## <a class="no_dec" href="https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.RandomHorizontalFlip.html#torchvision.transforms.RandomHorizontalFlip">RandomHorizontalFlip</a>
## --------------------------------------------------------------------------------
p = 0.5  # float. Probability to flip image

resize_trans = <b>transforms.RandomHorizontalFlip</b>(p=p)
image_transformed = resize_trans(image)  # PIL Image => PIL Image or Tensor => Tensor
## --------------------------------------------------------------------------------

## --------------------------------------------------------------------------------
## <a class="no_dec" href="https://docs.pytorch.org/vision/stable/generated/torchvision.transforms.ToTensor.html?highlight=totensor#torchvision.transforms.ToTensor">ToTensor</a>
## --------------------------------------------------------------------------------
## Input: PIL Image / numpy.ndarray (np.uint8) of shape (HxWxC) in the range [0, 255]
## Output: torch.FloatTensor of shape (CxHxW) in the range (0.0, 1.0)
## Other inputs: only apply type transform
resize_trans = <b>transforms.ToTensor</b>()
image_transformed = resize_trans(image)  # PIL Image / ndarray => Tensor
## --------------------------------------------------------------------------------
</pre>
""",
},
]