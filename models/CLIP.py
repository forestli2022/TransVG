import torch
import clip
from PIL import Image


def build_clip(args):
    if args.backbone not in clip.available_models():
        raise RuntimeError(f"Model {args.backbone} not found; available models = {clip.available_models()}")
    else:
        model, preprocess = clip.load(args.backbone, torch.device(args.device))
        return model, preprocess