import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    """LayerNorm for NCHW tensor over channel dimension."""

    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight[:, None, None] + self.bias[:, None, None]


class MDTA(nn.Module):
    """
    Multi-DConv Head Transposed Attention style block.

    This is implemented from scratch and does not use pretrained weights.
    """

    def __init__(self, channels, num_heads=4):
        super().__init__()
        assert channels % num_heads == 0

        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.norm = LayerNorm2d(channels)

        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.qkv_dwconv = nn.Conv2d(
            channels * 3,
            channels * 3,
            kernel_size=3,
            padding=1,
            groups=channels * 3,
        )

        self.project_out = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        residual = x

        x = self.norm(x)
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        head_dim = c // self.num_heads

        q = q.reshape(b, self.num_heads, head_dim, h * w)
        k = k.reshape(b, self.num_heads, head_dim, h * w)
        v = v.reshape(b, self.num_heads, head_dim, h * w)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = torch.matmul(q, k.transpose(-2, -1))
        attn = attn * self.temperature
        attn = torch.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)
        out = out.reshape(b, c, h, w)
        out = self.project_out(out)

        return residual + out


class GDFN(nn.Module):
    """Gated-DConv Feed-Forward Network."""

    def __init__(self, channels, expansion=2):
        super().__init__()

        hidden = channels * expansion

        self.norm = LayerNorm2d(channels)
        self.project_in = nn.Conv2d(channels, hidden * 2, kernel_size=1)
        self.dwconv = nn.Conv2d(
            hidden * 2,
            hidden * 2,
            kernel_size=3,
            padding=1,
            groups=hidden * 2,
        )
        self.project_out = nn.Conv2d(hidden, channels, kernel_size=1)

    def forward(self, x):
        residual = x

        x = self.norm(x)
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)

        return residual + x


class TransformerBlock(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.attn = MDTA(channels, num_heads=num_heads)
        self.ffn = GDFN(channels)

    def forward(self, x):
        x = self.attn(x)
        x = self.ffn(x)
        return x


class PromptGenBlock(nn.Module):
    """
    PromptIR-style prompt generation block.

    Learns prompt tensors and adaptively combines them based on input features.
    """

    def __init__(self, channels, num_prompts=8, prompt_size=16):
        super().__init__()

        self.num_prompts = num_prompts

        self.prompts = nn.Parameter(
            torch.randn(num_prompts, channels, prompt_size, prompt_size) * 0.02
        )

        hidden = max(channels // 4, 1)

        self.weight_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, num_prompts, kernel_size=1),
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x):
        b, _, h, w = x.shape

        weights = self.weight_predictor(x).view(b, self.num_prompts)
        weights = torch.softmax(weights, dim=1)

        prompt = torch.einsum("bn,nchw->bchw", weights, self.prompts)
        prompt = F.interpolate(
            prompt,
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        )

        x = torch.cat([x, prompt], dim=1)
        x = self.fuse(x)

        return x


class PromptIR(nn.Module):
    """
    PromptIR-like model for this homework.

    - Single model for rain and snow.
    - Trained from scratch.
    - No pretrained weights.
    - No external data.
    - Uses PromptIR-style prompt generation.
    """

    def __init__(self, dim=48, num_blocks=(4, 6, 6, 8), heads=(1, 2, 4, 8)):
        super().__init__()

        self.intro = nn.Conv2d(3, dim, kernel_size=3, padding=1)

        self.encoder1 = nn.Sequential(
            *[
                TransformerBlock(dim, num_heads=heads[0])
                for _ in range(num_blocks[0])
            ]
        )
        self.down1 = nn.Conv2d(dim, dim * 2, kernel_size=4, stride=2, padding=1)

        self.encoder2 = nn.Sequential(
            *[
                TransformerBlock(dim * 2, num_heads=heads[1])
                for _ in range(num_blocks[1])
            ]
        )
        self.down2 = nn.Conv2d(dim * 2, dim * 4, kernel_size=4, stride=2, padding=1)

        self.encoder3 = nn.Sequential(
            *[
                TransformerBlock(dim * 4, num_heads=heads[2])
                for _ in range(num_blocks[2])
            ]
        )
        self.down3 = nn.Conv2d(dim * 4, dim * 8, kernel_size=4, stride=2, padding=1)

        self.latent = nn.Sequential(
            *[
                TransformerBlock(dim * 8, num_heads=heads[3])
                for _ in range(num_blocks[3])
            ]
        )

        self.prompt_latent = PromptGenBlock(
            dim * 8,
            num_prompts=8,
            prompt_size=16,
        )

        self.up3 = nn.ConvTranspose2d(dim * 8, dim * 4, kernel_size=2, stride=2)
        self.reduce3 = nn.Conv2d(dim * 8, dim * 4, kernel_size=1)
        self.decoder3 = nn.Sequential(
            PromptGenBlock(dim * 4, num_prompts=8, prompt_size=16),
            *[
                TransformerBlock(dim * 4, num_heads=heads[2])
                for _ in range(num_blocks[2])
            ],
        )

        self.up2 = nn.ConvTranspose2d(dim * 4, dim * 2, kernel_size=2, stride=2)
        self.reduce2 = nn.Conv2d(dim * 4, dim * 2, kernel_size=1)
        self.decoder2 = nn.Sequential(
            PromptGenBlock(dim * 2, num_prompts=6, prompt_size=16),
            *[
                TransformerBlock(dim * 2, num_heads=heads[1])
                for _ in range(num_blocks[1])
            ],
        )

        self.up1 = nn.ConvTranspose2d(dim * 2, dim, kernel_size=2, stride=2)
        self.reduce1 = nn.Conv2d(dim * 2, dim, kernel_size=1)
        self.decoder1 = nn.Sequential(
            PromptGenBlock(dim, num_prompts=6, prompt_size=16),
            *[
                TransformerBlock(dim, num_heads=heads[0])
                for _ in range(num_blocks[0])
            ],
        )

        self.outro = nn.Conv2d(dim, 3, kernel_size=3, padding=1)

    def forward(self, x):
        inp = x

        x = self.intro(x)

        e1 = self.encoder1(x)

        x = self.down1(e1)
        e2 = self.encoder2(x)

        x = self.down2(e2)
        e3 = self.encoder3(x)

        x = self.down3(e3)
        x = self.latent(x)
        x = self.prompt_latent(x)

        x = self.up3(x)
        if x.shape[-2:] != e3.shape[-2:]:
            x = F.interpolate(
                x,
                size=e3.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        x = torch.cat([x, e3], dim=1)
        x = self.reduce3(x)
        x = self.decoder3(x)

        x = self.up2(x)
        if x.shape[-2:] != e2.shape[-2:]:
            x = F.interpolate(
                x,
                size=e2.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        x = torch.cat([x, e2], dim=1)
        x = self.reduce2(x)
        x = self.decoder2(x)

        x = self.up1(x)
        if x.shape[-2:] != e1.shape[-2:]:
            x = F.interpolate(
                x,
                size=e1.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        x = torch.cat([x, e1], dim=1)
        x = self.reduce1(x)
        x = self.decoder1(x)

        residual = self.outro(x)

        # Do not clamp during training.
        return inp + residual
