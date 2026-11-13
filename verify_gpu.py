"""Run real CUDA kernels and a YOLO11n forward/backward pass before training."""
import json
import os
from pathlib import Path
import torch
import torchvision

os.environ.setdefault('YOLO_CONFIG_DIR', str(Path(__file__).with_name('.ultralytics')))


def verify():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. Use the project .venv and CUDA PyTorch build.')
    device = torch.device('cuda:0')
    a = torch.randn(256, 256, device=device, requires_grad=True)
    a.square().mean().backward()
    boxes = torch.tensor([[0., 0., 10., 10.], [1., 1., 9., 9.]], device=device)
    torchvision.ops.nms(boxes, torch.tensor([.9, .8], device=device), .5)
    del a, boxes
    from ultralytics import YOLO
    model = YOLO('yolo11n.yaml').model.to(device).train()
    x = torch.randn(1, 3, 320, 320, device=device)
    outputs = model(x)
    def tensors(value):
        if isinstance(value, torch.Tensor):
            yield value
        elif isinstance(value, dict):
            for item in value.values():
                yield from tensors(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                yield from tensors(item)
    loss = sum(t.float().square().mean() for t in tensors(outputs))
    loss.backward()
    torch.cuda.synchronize()
    if not torch.isfinite(loss):
        raise RuntimeError('Nonfinite GPU smoke-test loss')
    import ultralytics
    report = dict(torch=torch.__version__, torchvision=torchvision.__version__,
                  ultralytics=ultralytics.__version__,
                  cuda_runtime=torch.version.cuda, gpu=torch.cuda.get_device_name(0),
                  capability=torch.cuda.get_device_capability(0),
                  compiled_architectures=torch.cuda.get_arch_list(),
                  total_vram_gib=round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2),
                  peak_allocated_mib=round(torch.cuda.max_memory_allocated() / 2**20, 1),
                  cuda_kernels='PASS', torchvision_cuda_nms='PASS',
                  yolo11n_forward_backward='PASS', test_image_size=320)
    Path(__file__).with_name('gpu-verification.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    del model, x, outputs, loss
    torch.cuda.empty_cache()
    return report


if __name__ == '__main__':
    verify()
