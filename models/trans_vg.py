import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_pretrained_bert.modeling import BertModel
from .CLIP import build_clip
from .vl_transformer import build_vl_transformer
from utils.box_utils import xywh2xyxy


class TransVG(nn.Module):
    def __init__(self, args):
        super(TransVG, self).__init__()
        hidden_dim = args.vl_hidden_dim
        divisor = 16 if args.dilation else 32
        self.num_visu_token = int((args.imsize / divisor) ** 2)
        self.num_text_token = args.max_query_len

        self.clip, _ = build_clip(args)
        self.clip = self.clip.float().to(args.device)
        self.visumodel = self.clip.visual
        # self.visumodel = build_detr(args)
        # self.textmodel = build_bert(args)

        num_total = self.num_visu_token + self.num_text_token + 1
        self.vl_pos_embed = nn.Embedding(num_total, hidden_dim)
        self.reg_token = nn.Embedding(1, hidden_dim)

        self.visu_proj = nn.Linear(self.visumodel.proj.shape[1], hidden_dim)
        self.text_proj = nn.Linear(self.visumodel.proj.shape[1], hidden_dim)

        self.vl_transformer = build_vl_transformer(args)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)

    def preprocess(self, img_data):
        # preprocess img_data as NestedTensor
        return F.interpolate(img_data.tensors, size=(224, 224), mode='bilinear', align_corners=False)


    def forward(self, img_data, text_data):
        bs = img_data.tensors.shape[0]
        preprocessed_img_data = self.preprocess(img_data)

        # visual encoder
        visu_src = self.visumodel(preprocessed_img_data)
        visu_src = self.visu_proj(visu_src)

        # language encoderd
        text_src = self.clip.encode_text(text_data.tensors)
        text_src = self.text_proj(text_src)

        # target regression token
        tgt_src = self.reg_token.weight.unsqueeze(1).repeat(1, bs, 1)

        print(tgt_src.shape, text_src.shape, visu_src.shape)
        
        vl_src = torch.cat([tgt_src, text_src, visu_src], dim=0)
        vl_pos = self.vl_pos_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        print(vl_pos.shape)

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