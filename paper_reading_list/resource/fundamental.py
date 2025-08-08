FUNDAMENTAL_COMPONENT = dict()
FUNDAMENTAL_COMPONENT["file"] = "fundamental_component.html"
FUNDAMENTAL_COMPONENT["title"] = "Deep Learning Fundamental Components"
FUNDAMENTAL_COMPONENT["description"] = "Fundamental components to build deep learning systems."
FUNDAMENTAL_COMPONENT["categories"] = ["Normalization"]
FUNDAMENTAL_COMPONENT["papers"] = [
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
# "summary": 
# """
# """,
# "details": 
# """
# <ul>
#     <li>
# </ul>
# """,
# },
{
"title": "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift",
"author": "Sergey Ioffe, Christian Szegedy",
"organization": "Google",
"date": "20150211",
"venue": "ICML 2015",
"pdf_url": "https://arxiv.org/pdf/1502.03167",
"code_url": "",
"name": "Batch Normalization",
"comment": "It normalizes the activations of each layer within a batch, improving training speed, stability, and generalization. It has over 60,000 citations (as of Aug 2025).",
"category": "Normalization",
"jupyter_notes": "",
"info": "**",
"summary": 
"""
It <b>normalizes layer inputs along channels</b> such that higher lr and saturating nonlinearities can be applied, careful param intialization is not needed.
""",
"details": 
"""
<ul>
    <li> <b>Reason of instability.</b> The inputs to each layer are affected by the parameters of all preceding layers, so that small changes to the network amplify as the network becomes deeper. Besides, the input and output distributions of each layer changes hinder the training of the layer.
    <li> <b>Previous solutions to instability.</b> non-saturating nonlinearities like ReLU, careful parameter intialization, small learning rate, dropout.
    <li> <b>Batch normalization.</b> It normalizes each channel by the mean and standard error to stablize the input distribution of each layer.
    <li> <b>Performance.</b> It applies to the best performing ImageNet classification network and matches its performance using only 7% of the training steps.
</ul>
fig: fig1.png 300 fig2.png 300
cap: <b>(left) Algorithm of Batch Normalization.</b> The gamma and beta is employed to make it can represent identity transformation. <b> (right) Algorithm of training and inference with Batch Normalization.</b>
<pre>
<code class="language-python">
import torch
from torch import nn

## --------------------------------------------------------------------------------
## Build customized Batch Normalization (2D)
## --------------------------------------------------------------------------------
class MyBatchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.weight = torch.nn.Parameter(torch.ones(num_features))
        self.bias = torch.nn.Parameter(torch.zeros(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        if self.training:
            mean = x.mean(dim=(0, 2, 3), keepdim=True)
            var = x.var(dim=(0, 2, 3), keepdim=True, unbiased=False)
            x_hat = (x - mean) / torch.sqrt(var + self.eps)

            # Update running stats
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean.view(-1)
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var.view(-1)
        else:
            mean = self.running_mean.view(1, -1, 1, 1)
            var = self.running_var.view(1, -1, 1, 1)
            x_hat = (x - mean) / torch.sqrt(var + self.eps)

        return self.weight.view(1, -1, 1, 1) * x_hat + self.bias.view(1, -1, 1, 1)
## --------------------------------------------------------------------------------

## --------------------------------------------------------------------------------
## Test the customized Batch Normalization
## --------------------------------------------------------------------------------
# Input
x = torch.randn(8, 3, 32, 32)  # BCHW

# Instantiate both modules
bn_ref = torch.nn.BatchNorm2d(3)
bn_custom = MyBatchNorm2d(3)

# Sync initial parameters
bn_custom.weight.data.copy_(bn_ref.weight.data)
bn_custom.bias.data.copy_(bn_ref.bias.data)
bn_custom.running_mean.copy_(bn_ref.running_mean)
bn_custom.running_var.copy_(bn_ref.running_var)

# --- Training mode ---
bn_ref.train()
bn_custom.train()
y_ref_train = bn_ref(x)
y_custom_train = bn_custom(x)
print("Train diff:", torch.norm(y_ref_train - y_custom_train).item())

# --- Inference mode ---
bn_ref.eval()
bn_custom.eval()
y_ref_eval = bn_ref(x)
y_custom_eval = bn_custom(x)
print("Eval diff:", torch.norm(y_ref_eval - y_custom_eval).item())
## --------------------------------------------------------------------------------
</code>
</pre>
""",
},
]