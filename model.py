import timm
import torch
import torch.nn as nn
from monai.networks.nets import DiffusionModelUNet

class ViTFeatureExtractor(nn.Module):
    def __init__(self, model_name="vit_base_patch16_224", use_cls_token=False):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=False, img_size=256)
        self.model.reset_classifier(0)
        self.out_dim = self.model.num_features
        self.use_cls_token = use_cls_token

    def forward(self, x):  # [B, 1, H, W]
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        feats = self.model.forward_features(x)
        if isinstance(feats, tuple):
            feats = feats[0]
        if self.use_cls_token:
            return feats[:, 0:1, :]
        else:
            return feats[:, 1:, :].mean(dim=1, keepdim=True)


class ASLDM(nn.Module):
    def __init__(
        self,
        unet_config: dict,
        num_modalities: int,
        embedding_dim: int = 16,
        context_dim: int = 768,
        freeze_vit: bool = True,
    ):
        super().__init__()
        self.unet = DiffusionModelUNet(**unet_config)
        self.vit = ViTFeatureExtractor()  # 共用一个冻结ViT
        self.missing_embed = nn.Embedding(2, embedding_dim)
        self.fc = nn.Linear(
            2 * self.vit.out_dim + num_modalities * embedding_dim, context_dim
        )

        if freeze_vit:
            # 冻结ViT权重
            self.vit.eval()
            for p in self.vit.parameters():
                p.requires_grad = False

    def forward(self, x, timesteps, seg, sketch, missing_code):
        """
        x:            [B, C, H, W]       - noised source latents
        timesteps:    [B]                - diffusion timestep
        seg:          [B, 1, H, W]       - segmentation map
        sketch:       [B, 1, H, W]       - sketch map
        missing_code: [B, N]             - modality presence code (1 or 0)
        """

        # ViT 提取两份独立特征
        feat_seg = self.vit(seg)  # [B, 1, 768]
        feat_sketch = self.vit(sketch)  # [B, 1, 768]

        # 模态缺失编码
        missing_embed = self.missing_embed(missing_code)  # [B, N, embed_dim]
        missing_feat = missing_embed.view(missing_embed.size(0), -1).unsqueeze(
            1
        )  # [B, 1, N*embed_dim]

        # 融合成 context
        context = self.fc(
            torch.cat([feat_seg, feat_sketch, missing_feat], dim=-1)
        )  # [B, 1, context_dim]

        # UNet 条件扩散预测
        return self.unet(x=x, timesteps=timesteps, context=context)
