# This config file is needed for convert models to onnx runtime.
# Use it for extract pth files from pretrained hugginface model
# Clone BiRefNet_demo locally in order to have a local models copy:
# git clone https://huggingface.co/spaces/ZhengPeng7/BiRefNet_demo
import os

import torch
import transformers

# Fix the HF space permission error when using from_pretrained(..., trust_remote_code=True)
os.environ["HF_MODULES_CACHE"] = os.path.join("/tmp/hf_cache", "modules")


usage_to_weights_file = {
    'General': 'BiRefNet',
    'General-HR': 'BiRefNet_HR',
    'Matting-HR': 'BiRefNet_HR-matting',
    'Matting': 'BiRefNet-matting',
    'Portrait': 'BiRefNet-portrait',
    'General-reso_512': 'BiRefNet_512x512',
    'General-Lite': 'BiRefNet_lite',
    'General-Lite-2K': 'BiRefNet_lite-2K',
    # 'Anime-Lite': 'BiRefNet_lite-Anime',
    'DIS': 'BiRefNet-DIS5K',
    'HRSOD': 'BiRefNet-HRSOD',
    'COD': 'BiRefNet-COD',
    'DIS-TR_TEs': 'BiRefNet-DIS5K-TR_TEs',
    'General-legacy': 'BiRefNet-legacy',
    'General-dynamic': 'BiRefNet_dynamic',
    'Matting-dynamic': 'BiRefNet_dynamic-matting',
}

for k in usage_to_weights_file:
    print(usage_to_weights_file[k])
    birefnet = transformers.AutoModelForImageSegmentation.from_pretrained('/'.join(('zhengpeng7', usage_to_weights_file[k])), trust_remote_code=True)
    model_state_dict = birefnet.state_dict()
    output_path = f"checkpoints/pth/{usage_to_weights_file[k]}.pth"
    torch.save(model_state_dict, output_path)
    print(f"Model state dictionary saved to {output_path}")
