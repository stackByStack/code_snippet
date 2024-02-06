# Understanding **ONNX**: The Open Neural Network Exchange Format

## Introduction
With the growing interest in **Deep Learning** and **AI**, the need to explore and experiment with different **machine learning frameworks** becomes essential. However, the community often faces the challenge of interoperability between these frameworks. To address this, the **Open Neural Network Exchange (ONNX)** format comes into the picture.

## What is ONNX?
**ONNX** is an open standard format for AI and machine learning models that provides **interoperability** between different ML frameworks. It was created by **Facebook** and **Microsoft** in collaboration with other prominent companies like **IBM**, **Huawei**, **Intel**, and **AMD**. The ONNX format encodes **model architecture** and **learned parameters** in a manner that permits models to be transferred between frameworks.

## Benefits of ONNX
The universal character of ONNX offers several benefits:
- **Interoperability**: Models trained in one framework, like **TensorFlow** or **PyTorch**, can be exported to another for inference. It breaks down the barriers that often segregate teams working with different ML tools.
- **Portability**: ONNX model files can be deployed on various platforms, whether it's **cloud**, **edge devices**, or local machines with ease.
- **Optimized Inference**: ONNX also provides access to the **ONNX Runtime**, a performance-focused engine for deploying ML models.

## Creating an ONNX Model
Creating an ONNX model is as simple as exporting trained models from supported ML frameworks. Here's a simple example considering **PyTorch**:
```python
import torch
import torchvision

model = torchvision.models.resnet50(pretrained=True)
model.eval()
example = torch.rand(1, 3, 224, 224)
torch_out = model(example)
torch.onnx.export(model, example, "model.onnx")
```
This code snippet fetches a pretrained **ResNet-50** model from **torchvision**, sets it to inference mode, defines an example input tensor, and exports the model to an ONNX file labeled "model.onnx".

## Using ONNX Models
ONNX models can be consumed and used for inference using a variety of tools. One popular option is with the **ONNX Runtime**. The example below demonstrates loading a model and running inference:
```python
import onnxruntime

ort_session = onnxruntime.InferenceSession("model.onnx")
inputs = {ort_session.get_inputs()[0].name: example.numpy()}
predictions = ort_session.run(None, inputs)
```

## Conclusion
**ONNX** provides a robust and broadly embraced format for model sharing between different **AI frameworks**, granting the capacity to reuse models regardless of the training tools used. **ONNX** has become the go-to solution for those looking to mitigate the clash of different ML frameworks and maintain **interoperability** in their projects.


## Example Code
### export_onnx.py
```Python
import torch
import timm

def export_onnx(model, output_dir:str):
    # Create a dummy input to the model
    dummy_input = torch.randn(1, 3, 224, 224)

    # Export the model to ONNX format
    torch.onnx.export(model, dummy_input, f"{output_dir}/model1.onnx")
    

# param parser
import argparse
parser = argparse.ArgumentParser(description='Export a model to ONNX format')
parser.add_argument('--ckpt_path', type=str, help='Path to the model checkpoint')
parser.add_argument('--model', type=str, help='Name of the model to load from timm')
# parser.add_argument('--num_classes', type=int, help='Number of classes in the model')
parser.add_argument('--output_dir', type=str, help='Directory to save the ONNX model')
# whether is timm
# parser.add_argument('--is_timm', type=int, help='Whether the model is from timm, 1 is true, 0 is false')
args = parser.parse_args()

from timm import create_model
import model
import utils

model = create_model(args.model, distillation=True)
if args.ckpt_path:
    model.load_state_dict(torch.load(args.ckpt_path)['model'])
    
# utils.replace_batchnorm(model)
# model.eval()
    
export_onnx(model, args.output_dir)
```

**How to use**
```Bash
python export_onnx.py --ckpt_path pretrain/repvit_m0_9_distill_300e.pth --output_dir onnx_output --model repvit_m0_9
```

### onnx_inference.py (GPU)
```Python
import onnxruntime
import torch

# Initialize the CUDA device if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using device:', device)

# Load the ONNX model onto the GPU
ort_session = onnxruntime.InferenceSession("onnx_output/model.onnx", providers=['CUDAExecutionProvider'])

# Initialize example input with 1*3*224*224 on the GPU
example = torch.rand(1, 3, 224, 224).to(device)

for i in range(100):
    # Compute ONNX Runtime output prediction on the GPU
    ort_inputs = {ort_session.get_inputs()[0].name: example.detach().cpu().numpy()}  # Move the input back to the CPU
    ort_outs = ort_session.run(None, ort_inputs)
    print(ort_outs[0])

```