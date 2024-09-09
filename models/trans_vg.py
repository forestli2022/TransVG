import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_pretrained_bert.modeling import BertModel
from .clip.clip import load
from .vl_transformer import build_vl_transformer
from utils.box_utils import xywh2xyxy


class TransVG(nn.Module):
    def __init__(self, args):
        super(TransVG, self).__init__()
        self.args = args
        divisor = get_clip_divisor(args.backbone)

        self.modified_clip, _ = load(args.backbone, args, device=args.device)
        for param in self.modified_clip.parameters():
            param.requires_grad = False

        hidden_dim = self.modified_clip.visual.output_dim
        self.modified_clip = self.modified_clip.float().to(args.device)
        self.visumodel = self.modified_clip.visual

        # MODIFIED: drop prompts at Fusion Encoder
        num_visu_token = int((args.imsize / divisor) ** 2) + 1 + (args.visual_prompt_length if not args.drop_prompt else 0)
        num_text_token = args.max_query_len + (args.text_prompt_length if not args.drop_prompt else 0)
        num_total = num_visu_token + num_text_token + 1
        self.drop_prompt = args.drop_prompt
        self.vl_pos_embed = nn.Embedding(num_total, hidden_dim)
        self.reg_token = nn.Embedding(1, hidden_dim)

        self.vl_transformer = build_vl_transformer(args, vl_hidden_dim=hidden_dim)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

        # prompt
        self.text_prompt = nn.parameter.Parameter(torch.randn(1, args.text_prompt_length, hidden_dim))
        self.visual_prompt = nn.parameter.Parameter(torch.randn(1, args.visual_prompt_length, self.visumodel.transformer.width))

    def preprocess(self, img_data):
        # preprocess img_data as NestedTensor
        return F.interpolate(img_data.tensors, size=(224, 224), mode='bilinear', align_corners=False)


    def forward(self, img_data, text_data):
        bs = img_data.tensors.shape[0]
        preprocessed_img_data = self.preprocess(img_data)

        # visual encoder:
        # [batch_size, w, h, d_model]
        visu_src = self.visumodel(preprocessed_img_data, self.visual_prompt)
        visu_src = visu_src.permute(1, 0, 2)

        # language encoder:
        # [batch_size, n_ctx, d_model]
        text_src = self.modified_clip.encode_text(text_data.tensors, self.text_prompt)
        text_src = text_src.permute(1, 0, 2)

        # drop prompt after encoding
        if self.drop_prompt:
            text_src = text_src[self.args.text_prompt_length:]
            visu_src = visu_src[self.args.visual_prompt_length:]

        # target regression token
        tgt_src = self.reg_token.weight.unsqueeze(1).repeat(1, bs, 1)

        vl_src = torch.cat([tgt_src, text_src, visu_src], dim=0)
        vl_pos = self.vl_pos_embed.weight.unsqueeze(1).repeat(1, bs, 1)

        vg_hs = self.vl_transformer(vl_src, None, vl_pos) # (1+L+N)xBxC
        vg_hs = vg_hs[0]

        pred_box = self.bbox_embed(vg_hs).sigmoid()


        return pred_box


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
    

# new: automatically get the divisor from clip models
def get_clip_divisor(backbone: str):
    divisor_dict = {
    "ViT-B/32": 32,
    "ViT-B/16": 16,
    "ViT-L/14": 14,
    "ViT-L/14@336px": 14,
    }
    if backbone not in divisor_dict.keys():
        return 1
    return divisor_dict[backbone]