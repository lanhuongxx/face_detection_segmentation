"""Module 3: U-Net phân vùng mặt. Định nghĩa model KHỚP CHÍNH XÁC với notebook train
(tên lớp/thuộc tính giống nhau để nạp được state_dict từ models/unet_best.pt)."""
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchvision.models import resnet34


def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
    )


class UNetSmall(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = conv_block(3, 16)
        self.enc2 = conv_block(16, 32)
        self.center = conv_block(32, 64)
        self.pool = nn.MaxPool2d(2)
        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec2 = conv_block(64, 32)
        self.up1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec1 = conv_block(32, 16)
        self.head = nn.Conv2d(16, 1, 1)

    def forward(self, image):
        first = self.enc1(image)
        second = self.enc2(self.pool(first))
        center = self.center(self.pool(second))
        second_up = self.dec2(torch.cat((self.up2(center), second), dim=1))
        first_up = self.dec1(torch.cat((self.up1(second_up), first), dim=1))
        return self.head(first_up)


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.block = conv_block(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = nn.functional.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.block(torch.cat((x, skip), dim=1))


class ResNetUNet(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = resnet34(weights=None)          # trọng số lấy từ checkpoint, không tải ImageNet
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.pool = backbone.maxpool
        self.l1, self.l2, self.l3, self.l4 = (backbone.layer1, backbone.layer2,
                                              backbone.layer3, backbone.layer4)
        self.up4 = UpBlock(512, 256, 256)
        self.up3 = UpBlock(256, 128, 128)
        self.up2 = UpBlock(128, 64, 64)
        self.up1 = UpBlock(64, 64, 32)
        self.head = nn.Conv2d(32, 1, 1)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        size = x.shape[-2:]
        x = (x - self.mean) / self.std
        s0 = self.stem(x)
        s1 = self.l1(self.pool(s0))
        s2 = self.l2(s1)
        s3 = self.l3(s2)
        s4 = self.l4(s3)
        d = self.up4(s4, s3)
        d = self.up3(d, s2)
        d = self.up2(d, s1)
        d = self.up1(d, s0)
        d = nn.functional.interpolate(d, size=size, mode="bilinear", align_corners=False)
        return self.head(d)


class FaceSegmenter:
    """Nạp trọng số và dự đoán xác suất mặt cho một lô crop."""
    def __init__(self, weights="models/unet_best.pt", device="cuda", batch_size=16):
        weights = Path(weights)
        if not weights.is_file():
            raise FileNotFoundError(f"Không thấy trọng số U-Net: {weights}. Chạy CELL 13 của notebook để tạo models/unet_best.pt.")
        info_path = weights.with_name("unet_best_info.json")        # CELL 13 ghi file này
        name = "resnet_unet"
        if info_path.is_file():
            name = json.loads(info_path.read_text(encoding="utf-8")).get("model", name)
        self.model = {"resnet_unet": ResNetUNet, "unet_small": UNetSmall}[name]()
        self.model.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
        self.model.to(device).eval()
        self.device = device
        self.batch_size = batch_size
        self.model_name = name

    def predict(self, crops):
        """crops: (N,S,S,3) uint8 RGB. Trả về xác suất (N,S,S) float32 trong [0,1]."""
        outputs = []
        with torch.inference_mode():
            for start in range(0, len(crops), self.batch_size):
                batch = torch.from_numpy(np.ascontiguousarray(crops[start:start + self.batch_size]))
                batch = batch.to(self.device).permute(0, 3, 1, 2).float() / 255   # đúng như Dataset lúc train
                outputs.append(self.model(batch).sigmoid()[:, 0].float().cpu().numpy())
        return np.concatenate(outputs) if outputs else np.zeros((0,) + crops.shape[1:3], np.float32)
