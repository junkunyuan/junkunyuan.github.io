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
"title": "new paper Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift",
"author": "Sergey Ioffe, Christian Szegedy",
"organization": "Google",
"date": "20150211",
"venue": "ICML 2015",
"pdf_url": "https://arxiv.org/pdf/1502.03167",
"code_url": "",
"name": "Batch Normalization",
"comment": "",
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
""",
},
]