import os
import torch
import logging
import torch.nn as nn
import numpy as np
from typing import Union

from model.model import build_model

import torch.nn.functional as F


class HouseHolder(nn.Module):
    """Custom rotation layer using cayley transform to parametrize"""

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.weights = nn.Parameter(torch.eye(dim))
        self.pad = (1 << (dim - 1).bit_length()) - dim
        self.p_dim = self.dim + self.pad
        self.log2dim = (1 << (dim - 1).bit_length()).bit_length() - 1

    def pad_X(self, X):
        return nn.functional.pad(X, (0, 0, 0, self.pad), "constant", 0)

    def unpad_X(self, X):
        if self.pad:
            return X[: -self.pad, :]
        else:
            return X

    def get_V(self):
        V = nn.functional.pad(
            nn.functional.normalize(self.weights.clone(), dim=0), (0, self.pad, 0, self.pad), "constant", 0
        )
        if self.pad:
            V[-self.pad :, -self.pad :] += torch.eye(self.pad).to(V.device)
        return V

    def fasthpp(self, X):
        V = self.get_V()
        Y_ = V.clone().T
        W_ = -2 * Y_.clone()

        k = 1
        for _ in range(self.log2dim):
            k_2 = k
            k *= 2

            W_view = W_.view(self.p_dim // k_2, k_2, self.p_dim).clone()
            m1_ = Y_.view(self.p_dim // k_2, k_2, self.p_dim)[0::2] @ torch.transpose(W_view[1::2], 1, 2)
            m2_ = torch.transpose(W_view[0::2], 1, 2) @ m1_

            W_ = W_.view(self.p_dim // k_2, k_2, self.p_dim)
            W_[1::2] += torch.transpose(m2_, 1, 2)
            W_ = W_.view(self.p_dim, self.p_dim)

        return X + self.unpad_X(W_.T @ (Y_ @ self.pad_X(X)))

    def forward(self, X):
        return self.fasthpp(X)

class LinearHash(nn.Module):

    def __init__(self, inputDim=2048, outputDim=64):
        super(LinearHash, self).__init__()
        self.fc = nn.Linear(inputDim, outputDim)
        self.drop_out = nn.Dropout(p=0.2)

    def forward(self, data):
        result = self.fc(data)
        return torch.tanh(self.drop_out(result))


class LAM(nn.Module):

    def __init__(self,
                 num_class=80,
                 hash_bit=64,
                 clipPath="./ViT-B-32.pt",
                 writer=None,
                 saveDir="./result/log",
                 n_samples=100):
        super(LAM, self).__init__()
        os.makedirs(saveDir, exist_ok=True)

        self.embedDim, self.clip = self.load_clip(clipPath)

        self.image_hash = LinearHash(inputDim=self.embedDim, outputDim=hash_bit)
        self.text_hash = LinearHash(inputDim=self.embedDim, outputDim=hash_bit)

    def freezen(self):
        for name, param in self.clip.named_parameters():
            # print(name)
            if name.find("ln_final.") == 0 or name.find("text_projection") == 0 or name.find("logit_scale") == 0 \
                    or name.find("visual.ln_post.") == 0 or name.find("visual.proj") == 0:
                # print("1")
                continue
            elif name.find("visual.transformer.resblocks.") == 0 or name.find("transformer.resblocks.") == 0:
                layer_num = int(name.split(".resblocks.")[1].split(".")[0])
                if layer_num >= 12:
                    # print("2")
                    continue
            if name.find("conv2.") == 0:
                # print("3")
                continue
            else:
                # paramenters which < freeze_layer_num will be freezed
                param.requires_grad = False

    def load_clip(self, clipPath: str) -> tuple:
        try:
            model = torch.jit.load(clipPath, map_location="cpu").eval()
            state_dict = model.state_dict()
        except RuntimeError:
            state_dict = torch.load(clipPath, map_location="cpu")

        return state_dict["text_projection"].shape[1], build_model(state_dict)

    def hashing(self, image, text):
        x_i = self.image_hash(image)
        x_t = self.text_hash(text)
        # x_i = F.normalize(x_i, dim=-1)
        # x_t = F.normalize(x_t, dim=-1)
        return x_i, x_t

    def encode(self, image, text):
        x_i = self.clip.encode_image(image)
        x_t = self.clip.encode_text(text)
        x_i, x_t = self.hashing(x_i, x_t)
        return x_i, x_t

    def eval(self):
        self.image_hash.eval()
        self.text_hash.eval()
        # self.training = False

    def train(self):
        self.image_hash.train()
        self.text_hash.train()
        # self.training = True

    def forward(self, image, text):
        image_embed, text_embed = self.encode(image, text)
        return image_embed, text_embed
