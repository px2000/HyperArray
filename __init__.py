import torch
from .UM_SRNet import UM_SRNet

def model_generator(method, pretrained_model_path=None, diffusion=None):
    if method == 'UM_SRNet':
        model = UM_SRNet(in_channels=1, out_channels=61, dim=32, deep_stage=3, num_blocks=[1, 1, 1, 1], num_heads=[1, 2, 4, 8]).cuda()

    else:
        raise ValueError(f'Method {method} is not defined!')
    
    if pretrained_model_path is not None:
        print(f'Loading model from {pretrained_model_path}')
        checkpoint = torch.load(pretrained_model_path, weights_only=False)

        if 'state_dict' in checkpoint:
            model.load_state_dict(
                {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()},
                strict=True
            )
        elif 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'], strict=True)
        else:
            try:
                model.load_state_dict(checkpoint, strict=True)
            except Exception as e:
                print(f"Error loading model: {str(e)}")
                print("Attempting to load model weights directly...")

                model.load_state_dict(checkpoint, strict=False)

    return model