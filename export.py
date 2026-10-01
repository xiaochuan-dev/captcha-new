import torch
import onnx
from model import CaptchaCNNTransformer
from config import *

def export_to_onnx(model_path='./checkpoints/best.pth', output_path='./checkpoints/model.onnx'):

    model = CaptchaCNNTransformer(
        dim=MODEL_DIM,
        depth=MODEL_DEPTH,
        heads=MODEL_HEADS,
        num_classes=NUM_CLASSES,
        channels=CHANNELS,
        dropout=MODEL_DROPOUT,
    )
    
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    dummy_input = torch.randn(1, 1, 32, 128)  # (batch, channels, height, width)
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        dynamo=False,
        opset_version=18,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'},
        },
    )

    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)

    print(f'模型已导出为: {output_path}')
    print('ONNX 模型检查通过')

if __name__ == '__main__':
    export_to_onnx()