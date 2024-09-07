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
        divisor = get_clip_divisor(args.backbone)
        self.num_visu_token = int((args.imsize / divisor) ** 2) + 1
        self.num_text_token = args.max_query_len

        self.modified_clip, _ = load(args.backbone, args.device)
        hidden_dim = self.modified_clip.visual.output_dim
        print(hidden_dim)
        self.modified_clip = self.modified_clip.float().to(args.device)
        self.visumodel = self.modified_clip.visual

        num_total = self.num_visu_token + self.num_text_token + 1
        self.vl_pos_embed = nn.Embedding(num_total, hidden_dim)
        self.reg_token = nn.Embedding(1, hidden_dim)

        self.vl_transformer = build_vl_transformer(args, vl_hidden_dim=hidden_dim)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

        # prompt
        self.prompt = torch.randn(1, args.prompt_length, hidden_dim, requires_grad=True).to(args.device)

    def preprocess(self, img_data):
        # preprocess img_data as NestedTensor
        return F.interpolate(img_data.tensors, size=(224, 224), mode='bilinear', align_corners=False)


    def forward(self, img_data, text_data):
        bs = img_data.tensors.shape[0]
        preprocessed_img_data = self.preprocess(img_data)

        # visual encoder
        with torch.no_grad():
            visu_src = self.visumodel(preprocessed_img_data)
        visu_src = visu_src.permute(1, 0, 2)

        # add prompts to tokens
        tokens = self.modified_clip.token_embedding(text_data.tensors).type(self.modified_clip.dtype)
        expanded_prompt = self.prompt.expand(bs, -1, -1)
        tokens = torch.cat([expanded_prompt, tokens], dim=1)
        tokens = tokens[:, :self.num_text_token, :]

        # language encoder
        with torch.no_grad():
            text_src = self.modified_clip.encode_token(tokens)
        text_src = text_src.permute(1, 0, 2)

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