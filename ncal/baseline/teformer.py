from einops import rearrange
import torch
import torch.nn as nn
from spikingjelly.clock_driven.neuron import MultiStepParametricLIFNode, MultiStepLIFNode
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg


class TIMv2(nn.Module):
    '''
    input shape should be [TB C H W]
    '''
    def __init__(self, step = 4):
        super().__init__()
        self.step = step

        alpha_init = 0.75
        theta_init = torch.logit((torch.tensor(alpha_init) - 0.5) / 0.5)
        self.theta = nn.Parameter(theta_init)

    def forward(self, x):
        TB, C, H, W = x.shape
        x = rearrange(x, '(t b) c h w -> t (b c h w)', t=self.step)

        a = 0.5 + 0.5 * torch.sigmoid(self.theta) #alpha > 0.5
        # a = 0.75
        i_idx = torch.arange(self.step,  device=x.device).unsqueeze(1).expand(self.step, self.step)
        j_idx = torch.arange(self.step,  device=x.device).unsqueeze(0).expand(self.step, self.step)

        exponent = i_idx - j_idx
        mask = i_idx >= j_idx

        K = a * (1 - a) ** exponent * mask
        K[:, 0] = (1 - a) ** torch.arange(self.step, device=x.device)

        x = K @ x  # TIMv2

        return x.reshape(TB, C, H, W).contiguous()

class Token_QK_Attention(nn.Module):
    def __init__(self, embed_dim, step=4, num_heads=12):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.q_conv = nn.Conv1d(embed_dim, embed_dim, kernel_size=1, stride=1, bias=False)
        self.q_bn = nn.BatchNorm1d(embed_dim)
        self.q_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.k_conv = nn.Conv1d(embed_dim, embed_dim, kernel_size=1, stride=1, bias=False)
        self.k_bn = nn.BatchNorm1d(embed_dim)
        self.k_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.v_conv = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.v_bn = nn.BatchNorm1d(embed_dim)
        self.TIMv2 = TIMv2(step=step)
        self.v_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        #
        self.attn_lif = MultiStepLIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend='cupy') # special v_thres

        self.proj_conv = nn.Conv1d(embed_dim, embed_dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm1d(embed_dim)

        self.proj_lif = MultiStepLIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend='cupy')

    def forward(self, x):


        T, B, C, H, W = x.shape
        x = x.flatten(-2, -1) # T B C N
        T, B, C, N = x.shape
        x_for_qkv = x.flatten(0, 1) # TB C N

        q_conv_out = self.q_conv(x_for_qkv)
        q_conv_out = self.q_bn(q_conv_out).reshape(T, B, C, N)
        q_conv_out = self.q_lif(q_conv_out)
        q = q_conv_out.unsqueeze(2).reshape(T, B, self.num_heads, C // self.num_heads, N)

        k_conv_out = self.k_conv(x_for_qkv)
        k_conv_out = self.k_bn(k_conv_out).reshape(T, B, C, N)
        k_conv_out = self.k_lif(k_conv_out)
        k =  k_conv_out.unsqueeze(2).reshape(T, B, self.num_heads, C // self.num_heads, N)
        #

        v_conv_out = self.v_conv(x_for_qkv)
        v_conv_out = self.v_bn(v_conv_out).reshape(T*B, C, H, W).contiguous() # TB C H W
        v = self.TIMv2(v_conv_out).reshape(T, B, C, H, W)
        v = self.v_lif(v).flatten(0, 1).flatten(-2, -1) # TB C N

        # QK Attn
        q = torch.sum(q, dim=3, keepdim=True)
        attn = self.attn_lif(q)
        x = torch.mul(attn, k) # T B num_heads C//num_heads N

        x = x.flatten(2, 3).flatten(0, 1) # TB C N
        #
        x = torch.mul(x, v) # TB C N
        x = self.proj_bn(self.proj_conv(x)).reshape(T, B, C, H, W)
        x = self.proj_lif(x)

        return x # T, B, C, H, W

class SSA(nn.Module):
    def __init__(self, embed_dim, step=4, num_heads=12, scale=0.125):
        super().__init__()
        self.num_heads = num_heads
        self.scale = scale
        self.q_conv = nn.Conv1d(embed_dim, embed_dim, kernel_size=1, stride=1, bias=False)
        self.q_bn = nn.BatchNorm1d(embed_dim)
        self.q_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.k_conv = nn.Conv1d(embed_dim, embed_dim, kernel_size=1, stride=1, bias=False)
        self.k_bn = nn.BatchNorm1d(embed_dim)
        self.k_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.v_conv = nn.Conv1d(embed_dim, embed_dim, kernel_size=1, stride=1, bias=False)
        self.v_bn = nn.BatchNorm1d(embed_dim)
        self.v_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.attn_lif =MultiStepLIFNode(tau=2.0, v_threshold=0.5, detach_reset=True, backend='cupy') #special v_thres

        self.proj_conv = nn.Conv1d(embed_dim, embed_dim, kernel_size=1, stride=1)
        self.proj_bn = nn.BatchNorm1d(embed_dim)
        self.proj_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.qkv_mp = nn.MaxPool1d(4)

    def forward(self, x):


        T, B, C, H, W = x.shape

        x = x.flatten(3)
        T, B, C, N = x.shape
        x_for_qkv = x.flatten(0, 1)

        q_conv_out = self.q_conv(x_for_qkv)
        q_conv_out = self.q_bn(q_conv_out).reshape(T, B, C, N).contiguous()
        q_conv_out = self.q_lif(q_conv_out)
        q = q_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2,
                                                                                                       4).contiguous()

        k_conv_out = self.k_conv(x_for_qkv)
        k_conv_out = self.k_bn(k_conv_out).reshape(T, B, C, N).contiguous()
        k_conv_out = self.k_lif(k_conv_out)
        k = k_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2,
                                                                                                       4).contiguous()

        v_conv_out = self.v_conv(x_for_qkv)
        v_conv_out = self.v_bn(v_conv_out).reshape(T, B, C, N).contiguous()
        v_conv_out = self.v_lif(v_conv_out)
        v = v_conv_out.transpose(-1, -2).reshape(T, B, N, self.num_heads, C // self.num_heads).permute(0, 1, 3, 2,
                                                                                                       4).contiguous()

        attn = (q @ k.transpose(-2, -1))
        x = (attn @ v) * 0.125
        # attn = self.hsa(q,k)
        # x = attn @ v

        x = x.transpose(3, 4).reshape(T, B, C, N).contiguous()
        x = self.attn_lif(x)
        x = x.flatten(0, 1)
        x = self.proj_lif(self.proj_bn(self.proj_conv(x))).reshape(T, B, C, W, H)

        return x # T, B, C, H, W

class MLP(nn.Module):
    def __init__(self, in_features, step=4, mlp_ratio = 4.0, out_features=None):
        super().__init__()

        self.in_features = in_features
        self.mlp_ratio = mlp_ratio
        self.out_features = out_features or in_features
        self.hidden_features = int(self.in_features * self.mlp_ratio)

        self.fc_conv1 = nn.Conv2d(in_features, self.hidden_features, kernel_size=1, stride=1)
        self.fc_bn1 = nn.BatchNorm2d(self.hidden_features)
        self.fc_lif1 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.fc_conv2 = nn.Conv2d(self.hidden_features, self.out_features, kernel_size=1, stride=1)
        self.fc_bn2 = nn.BatchNorm2d(self.out_features)
        self.fc_lif2 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

    def forward(self, x):


        T, B, C, H, W = x.shape

        x = self.fc_conv1 (x.flatten(0, 1))
        x = self.fc_bn1(x).reshape(T, B, self.hidden_features, H, W)
        x = self.fc_lif1(x)

        x = self.fc_conv2(x.flatten(0, 1))
        x = self.fc_bn2(x).reshape(T, B, C, H, W)
        x = self.fc_lif2(x)

        return x #T B, C, H, W

class T_MLP(nn.Module):
    def __init__(self, in_features, step=4, mlp_ratio = 2.0, out_features=None):
        super().__init__()
        self.step = step

        self.in_features = in_features
        self.mlp_ratio = mlp_ratio
        self.out_features = out_features or in_features
        self.hidden_features = int(self.in_features * self.mlp_ratio)

        self.forget_x = nn.Conv1d(in_features, self.hidden_features, kernel_size=1, stride=1)
        self.forget_h = nn.Conv1d(self.hidden_features, self.hidden_features, kernel_size=1, stride=1)

        self.input_x = nn.Conv1d(in_features, self.hidden_features, kernel_size=1, stride=1)
        self.gate_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.fc_conv2 = nn.Conv1d(self.hidden_features, self.out_features, kernel_size=1, stride=1)
        self.fc_bn2 = nn.BatchNorm1d(self.out_features)
        self.fc_lif2 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

    def forward(self, x):

        T, B, C, H, W = x.shape
        x = x.flatten(0, 1) # TB C N
        x = x.flatten(-2, -1)  # TB C N
        TB, _, N = x.shape
        x = rearrange(x, '(t b) c n -> t b c n', t=self.step)  # T B C N

        ht = torch.zeros_like(x[0])
        hidden_state = []

        for t in range(self.step):
            xt = x[self.step - t - 1]
            if t == 0:
                ht = self.input_x(xt)
                ht = self.gate_lif(ht)
                hidden_state.insert(0, ht)
            else :
                f = torch.sigmoid(self.forget_x(xt) + self.forget_h(ht))
                c = self.input_x(xt)
                ht = f * ht + (1 - f) * c
                ht = self.gate_lif(ht)
                hidden_state.insert(0, ht)

        x = torch.stack(hidden_state, dim=0)  # T B C N
        # x = self.gate_lif(x.flatten(0, 1))

        x = self.fc_bn2(self.fc_conv2(x.flatten(0, 1))) # TB C N
        x = self.fc_lif2(x.reshape(T, B, C, H, W).contiguous()) # T B C H W
        return x

class TokenSpikingTransformer(nn.Module):
    def __init__(self, embed_dim, step=4, num_heads=12,mlp_ratio=4.0):
        super().__init__()

        self.tssa = Token_QK_Attention(embed_dim=embed_dim, num_heads=num_heads, step=step)
        self.mlp = T_MLP(in_features=embed_dim, step=step,mlp_ratio=mlp_ratio)
        # self.mlp = MLP(in_features=embed_dim, step=step, mlp_ratio=mlp_ratio)

    def forward(self, x):

        x = x + self.tssa(x)
        x = x + self.mlp(x)

        return x

class SpikingTransformer(nn.Module):
    def __init__(self, embed_dim, step=4,num_heads=12,mlp_ratio=4.0, scale=0.125):
        super().__init__()
        self.ssa = SSA(embed_dim=embed_dim, num_heads=num_heads, scale=scale,step=step)
        self.mlp = MLP(in_features=embed_dim, step=step, mlp_ratio=mlp_ratio)

    def forward(self, x):

        x = x + self.ssa(x)
        x = x + self.mlp(x)

        return x

class PatchInit(nn.Module):
    def __init__(self, img_size_h=128, img_size_w=128, patch_size=4, in_channels=2, embed_dims=256):
        super().__init__()
        self.image_size = [img_size_h, img_size_w]
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.C = in_channels
        self.H, self.W = self.image_size[0] // patch_size[0], self.image_size[1] // patch_size[1]
        self.num_patches = self.H * self.W

        self.proj_conv = nn.Conv2d(in_channels, embed_dims // 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn = nn.BatchNorm2d(embed_dims // 8)
        self.proj_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.proj1_conv = nn.Conv2d(embed_dims // 8, embed_dims // 4, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj1_bn = nn.BatchNorm2d(embed_dims // 4)
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.proj1_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.proj2_conv = nn.Conv2d(embed_dims//4, embed_dims // 2, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj2_bn = nn.BatchNorm2d(embed_dims // 2)
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.proj2_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.proj3_conv = nn.Conv2d(embed_dims // 2, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj3_bn = nn.BatchNorm2d(embed_dims)
        self.maxpool3 = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.proj3_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.proj_res_conv = nn.Conv2d(embed_dims // 4, embed_dims, kernel_size=1, stride=4, padding=0, bias=False)
        self.proj_res_bn = nn.BatchNorm2d(embed_dims)
        self.proj_res_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')


    def forward(self, x):
        T, B, C, H, W = x.shape
        # Downsampling + Res
        # x_feat = x.flatten(0, 1)
        x = self.proj_conv(x.flatten(0, 1))
        x = self.proj_bn(x).reshape(T, B, -1, H, W)
        x = self.proj_lif(x).flatten(0, 1).contiguous()

        x = self.proj1_conv(x)
        x = self.proj1_bn(x)
        x = self.maxpool1(x).reshape(T, B, -1, H//2, W//2).contiguous()
        x = self.proj1_lif(x).flatten(0, 1).contiguous()

        x_feat = x
        x = self.proj2_conv(x)
        x = self.proj2_bn(x)
        x = self.maxpool2(x).reshape(T, B, -1, H//4, W//4).contiguous()
        x = self.proj2_lif(x).flatten(0, 1).contiguous()

        x = self.proj3_conv(x)
        x = self.proj3_bn(x)
        x = self.maxpool3(x).reshape(T, B, -1, H // 8, W // 8).contiguous()
        x = self.proj3_lif(x)

        x_feat = self.proj_res_conv(x_feat)
        x_feat = self.proj_res_bn(x_feat).reshape(T, B, -1, H//8, W//8).contiguous()
        x_feat = self.proj_res_lif(x_feat)
        x = x + x_feat # shortcut

        return x


class PatchEmbedding(nn.Module):
    def __init__(self, step=4, encode_type='direct', img_h=32, img_w=32, patch_size=4, in_channels=3, embed_dim=384):

        super().__init__()

        self.img_h = img_h
        self.img_w = img_w
        self.patch_size = patch_size
        self.patch_nums = self.img_h // self.patch_size * self.img_w // self.patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim

        self.proj3_conv = nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj3_bn = nn.BatchNorm2d(embed_dim)
        self.proj3_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.proj4_conv = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj4_bn = nn.BatchNorm2d(embed_dim)
        self.proj4_maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.proj4_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

        self.proj_res_conv = nn.Conv2d(embed_dim // 2, embed_dim, kernel_size=1, stride=2, padding=0, bias=False)
        self.proj_res_bn = nn.BatchNorm2d(embed_dim)
        self.proj_res_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

    def forward(self, x):

        T, B, C, H, W = x.shape
            # Downsampling + Res
        x = x.flatten(0, 1).contiguous()
        x_feat = x

        x = self.proj3_conv(x)
        x = self.proj3_bn(x).reshape(T, B, -1, H, W).contiguous()
        x = self.proj3_lif(x).flatten(0, 1).contiguous()

        x = self.proj4_conv(x)
        x = self.proj4_bn(x)
        x = self.proj4_maxpool(x).reshape(T, B, -1, H // 2, W // 2).contiguous()
        x = self.proj4_lif(x)

        x_feat = self.proj_res_conv(x_feat)
        x_feat = self.proj_res_bn(x_feat).reshape(T, B, -1, H//2, W//2).contiguous()
        x_feat = self.proj_res_lif(x_feat)

        x = x + x_feat  # shortcut

        return x

class QKFormer(nn.Module):
    def __init__(self,
                 step=16, img_size=32, patch_size=4, in_channels=3, num_classes=101,
                 embed_dim=384, num_heads=12, mlp_ratio=4, scale=0.125, depths=4, ):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.T = step
        # dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]  # stochastic depth decay rule

        # embed_dim // 4
        patch_embed1 = PatchInit( patch_size=patch_size, in_channels=in_channels,embed_dims=embed_dim//2)

        stage1 = nn.ModuleList([TokenSpikingTransformer(
                    embed_dim=embed_dim//2, step=step,num_heads=num_heads, mlp_ratio=mlp_ratio)
                for j in range(1)])

        # embed_dim // 2
        patch_embed2 = PatchEmbedding(step=step, img_h=img_size, img_w=img_size, patch_size=patch_size,
                                      in_channels=in_channels,
                                      embed_dim=embed_dim)

        stage2 = nn.ModuleList([TokenSpikingTransformer(
                                 embed_dim=embed_dim, step=step, num_heads=num_heads, mlp_ratio=mlp_ratio,
        )
                            for j in range(1)])


        setattr(self, f"patch_embed1", patch_embed1)
        setattr(self, f"patch_embed2", patch_embed2)
        setattr(self, f"stage1", stage1)
        setattr(self, f"stage2", stage2)


        # classification head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        stage1 = getattr(self, f"stage1")
        patch_embed1 = getattr(self, f"patch_embed1")
        stage2 = getattr(self, f"stage2")
        patch_embed2 = getattr(self, f"patch_embed2")

        x = patch_embed1(x)
        for blk in stage1:
            x = blk(x)

        x = patch_embed2(x)
        for blk in stage2:
            x = blk(x)

        return x.flatten(3).mean(3)

    def forward(self, x):
        x = x.permute(1, 0, 2, 3, 4)  # [T, N, 2, *, *]
        x = self.forward_features(x)
        x = self.head(x.mean(0))
        return x

@register_model
def TEFormer(pretrained=False, **kwargs):
    model = QKFormer(
        step=kwargs.get('step', 4),
        img_size=kwargs.get('img_size', 128),
        patch_size=kwargs.get('patch_size', 16),
        in_channels=kwargs.get('in_channels', 2),
        num_classes=kwargs.get('num_classes', 101),
        embed_dim=kwargs.get('embed_dim', 256),
        num_heads=kwargs.get('num_heads', 16),
        mlp_ratio=kwargs.get('mlp_ratio', 4),
        scale=kwargs.get('attn_scale', 0.125),
        depths=kwargs.get('depths', 2),
    )
    model.default_cfg = _cfg()
    return model
