# Copyright (c) Jinyang Li. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from [ViLD](https://github.com/tensorflow/tpu/tree/master/models/official/detection/projects/vild)
# ------------------------------------------------------------------------
import os

import torch
import torch.nn as nn

from .coco_categories import COCO_CATEGORIES
from .lvis_v1_categories import LVIS_CATEGORIES
import torch.nn.functional as F


def _get_clip_backend():
    try:
        from clip import clip as clip_backend

        return "openai", clip_backend
    except ImportError:
        try:
            import open_clip

            return "open_clip", open_clip
        except ImportError as exc:
            raise ImportError(
                "CLIP is only needed to regenerate text/image embeddings. "
                "Install `open-clip-torch` or OpenAI CLIP if you need that workflow."
            ) from exc

def article(name):
    return "an" if name[0] in "aeiou" else "a"


def processed_name(name, rm_dot=False):
    # _ for lvis
    # / for obj365
    res = name.replace("_", " ").replace("/", " or ").lower()
    if rm_dot:
        res = res.rstrip(".")
    return res


single_template = ["a photo of a {}."]

multiple_templates = [
    "There is {article} {} in the scene.",
    "There is the {} in the scene.",
    "a photo of {article} {} in the scene.",
    "a photo of the {} in the scene.",
    "a photo of one {} in the scene.",
    "itap of {article} {}.",
    "itap of my {}.",  # itap: I took a picture of
    "itap of the {}.",
    "a photo of {article} {}.",
    "a photo of my {}.",
    "a photo of the {}.",
    "a photo of one {}.",
    "a photo of many {}.",
    "a good photo of {article} {}.",
    "a good photo of the {}.",
    "a bad photo of {article} {}.",
    "a bad photo of the {}.",
    "a photo of a nice {}.",
    "a photo of the nice {}.",
    "a photo of a cool {}.",
    "a photo of the cool {}.",
    "a photo of a weird {}.",
    "a photo of the weird {}.",
    "a photo of a small {}.",
    "a photo of the small {}.",
    "a photo of a large {}.",
    "a photo of the large {}.",
    "a photo of a clean {}.",
    "a photo of the clean {}.",
    "a photo of a dirty {}.",
    "a photo of the dirty {}.",
    "a bright photo of {article} {}.",
    "a bright photo of the {}.",
    "a dark photo of {article} {}.",
    "a dark photo of the {}.",
    "a photo of a hard to see {}.",
    "a photo of the hard to see {}.",
    "a low resolution photo of {article} {}.",
    "a low resolution photo of the {}.",
    "a cropped photo of {article} {}.",
    "a cropped photo of the {}.",
    "a close-up photo of {article} {}.",
    "a close-up photo of the {}.",
    "a jpeg corrupted photo of {article} {}.",
    "a jpeg corrupted photo of the {}.",
    "a blurry photo of {article} {}.",
    "a blurry photo of the {}.",
    "a pixelated photo of {article} {}.",
    "a pixelated photo of the {}.",
    "a black and white photo of the {}.",
    "a black and white photo of {article} {}.",
    "a plastic {}.",
    "the plastic {}.",
    "a toy {}.",
    "the toy {}.",
    "a plushie {}.",
    "the plushie {}.",
    "a cartoon {}.",
    "the cartoon {}.",
    "an embroidered {}.",
    "the embroidered {}.",
    "a painting of the {}.",
    "a painting of a {}.",
]


def load_clip_to_cpu(visual_backbone):
    backend_name, backend = _get_clip_backend()
    if backend_name == "openai":
        backbone_name = visual_backbone
        url = backend._MODELS[backbone_name]
        model_path = backend._download(url, os.path.expanduser("~/.cache/clip"))

        try:
            model = torch.jit.load(model_path, map_location="cpu").eval()
            state_dict = None
        except RuntimeError:
            state_dict = torch.load(model_path, map_location="cpu", weights_only=False)

        return backend.build_model(state_dict or model.state_dict())

    model_name = visual_backbone.replace("/", "-")
    return backend.create_model(model_name, pretrained="openai").cpu().eval()


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
        self.token_embedding = clip_model.token_embedding

    def forward(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x


def load_embeddings(Clip_text_embeddings, Clip_image_embeddings):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text_embedding = torch.load(
        Clip_text_embeddings,
        map_location="cpu",
        weights_only=False,
    )
    image_embedding = torch.load(
        Clip_image_embeddings,
        map_location="cpu",
        weights_only=False,
    )
    return text_embedding.to(device), image_embedding.to(device)


def build_text_embedding_lvis():
    categories = LVIS_CATEGORIES
    backend_name, backend = _get_clip_backend()
    if backend_name == "openai":
        model, _ = backend.load("ViT-B/32")
        tokenize = backend.tokenize
    else:
        model = backend.create_model("ViT-B-32", pretrained="openai")
        tokenize = backend.get_tokenizer("ViT-B-32")
    templates = multiple_templates

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    with torch.no_grad():
        all_text_embeddings = []
        for category in categories:
            texts = [
                template.format(
                    processed_name(category["name"], rm_dot=True), article=article(category["name"])
                )
                for template in templates
            ]
            texts = [
                "This is " + text if text.startswith("a") or text.startswith("the") else text
                for text in texts
            ]
            texts = tokenize(texts).to(device)
            text_embeddings = model.encode_text(texts)
            text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
            text_embedding = text_embeddings.mean(dim=0)
            text_embedding /= text_embedding.norm()
            all_text_embeddings.append(text_embedding)
        all_text_embeddings = torch.stack(all_text_embeddings, dim=1)
        all_text_embeddings = all_text_embeddings.to(device)

    all_text_embeddings = all_text_embeddings.t()
    return all_text_embeddings
