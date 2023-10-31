import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function
from torch.nn.modules.module import Module
from torch.nn.functional import fold, unfold
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torchvision.utils import make_grid
import math
from utils import resize_image
import custom_transforms as transforms
from custom_blocks import PatchEmbed, TransformerBlock, trunc_normal_


#################################################################################
# You will need to fill in the missing code in this file
#################################################################################


#################################################################################
# Part I: Understanding Convolutions
#################################################################################
class CustomConv2DFunction(Function):
    @staticmethod
    def forward(ctx, input_feats, weight, bias, stride=1, padding=0):
        """
        Forward propagation of convolution operation.
        We only consider square filters with equal stride/padding in width and height!

        Args:
          input_feats: input feature map of size N * C_i * H * W
          weight: filter weight of size C_o * C_i * K * K
          bias: (optional) filter bias of size C_o
          stride: (int, optional) stride for the convolution. Default: 1
          padding: (int, optional) Zero-padding added to both sides of the input. Default: 0

        Outputs:
          output: responses of the convolution  w*x+b

        """
        # sanity check
        assert weight.size(2) == weight.size(3)
        assert input_feats.size(1) == weight.size(1)
        assert isinstance(stride, int) and (stride > 0)
        assert isinstance(padding, int) and (padding >= 0)

        # save the conv params
        kernel_size = weight.size(2)
        ctx.stride = stride
        ctx.padding = padding
        ctx.input_height = input_feats.size(2)
        ctx.input_width = input_feats.size(3)

        # make sure this is a valid convolution
        assert kernel_size <= (input_feats.size(2) + 2 * padding)
        assert kernel_size <= (input_feats.size(3) + 2 * padding)

        #################################################################################
        if padding > 0:
            input_feats = nn.functional.pad(input_feats, (padding, padding, padding, padding))
        unfold_input = unfold(input_feats, kernel_size, stride=stride)
        new_weight = weight.view(weight.size(0), -1)
        output = new_weight @ unfold_input
        if bias is not None:
            output += bias.unsqueeze(1)
        H = (ctx.input_height - kernel_size + 2 * padding) // stride + 1
        W = (ctx.input_width - kernel_size + 2 * padding) // stride + 1
        output = output.view(input_feats.size(0), weight.size(0), H, W).clone()

        #################################################################################

        # save for backward (you need to save the unfolded tensor into ctx)
        ctx.save_for_backward(unfold_input, weight, bias)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        
        """
        Backward propagation of convolution operation

        Args:
          grad_output: gradients of the outputs

        Outputs:
          grad_input: gradients of the input features
          grad_weight: gradients of the convolution weight
          grad_bias: gradients of the bias term

        """
        # unpack tensors and initialize the grads
        unfold_input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        # recover the conv params
        kernel_size = weight.size(2)
        stride = ctx.stride
        padding = ctx.padding
        input_height = ctx.input_height
        input_width = ctx.input_width

        #################################################################################
        C_i = weight.size(1)
    
        if ctx.needs_input_grad[0]:
            C_o = weight.size(0)
            new_weight = weight.view(C_o, C_i * kernel_size * kernel_size).T
            new_grad_output = grad_output.view(grad_output.size(0),
                                                grad_output.size(1), -1).permute(0, 2, 1)
            new_grad_input = new_grad_output @ new_weight.T
            new_grad_input = new_grad_input.permute(0, 2, 1)
            grad_input = fold(new_grad_input, (input_height, input_width), (kernel_size, kernel_size), padding=padding, stride=stride)

        if ctx.needs_input_grad[1]:
            new_grad_output = grad_output.view(grad_output.size(0),
                                                grad_output.size(1), -1)
            grad_weight = new_grad_output @ unfold_input.permute(0, 2, 1)
            grad_weight = grad_weight.sum(dim=0).view_as(weight)
        #################################################################################
        # compute the gradients w.r.t. input and params

        if bias is not None and ctx.needs_input_grad[2]:
            # compute the gradients w.r.t. bias (if any)
            grad_bias = grad_output.sum((0, 2, 3))

        return grad_input, grad_weight, grad_bias, None, None


custom_conv2d = CustomConv2DFunction.apply


class CustomConv2d(Module):
    """
    The same interface as torch.nn.Conv2D
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        super(CustomConv2d, self).__init__()
        assert isinstance(kernel_size, int), "We only support squared filters"
        assert isinstance(stride, int), "We only support equal stride"
        assert isinstance(padding, int), "We only support equal padding"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # not used (for compatibility)
        self.dilation = dilation
        self.groups = groups

        # register weight and bias as parameters
        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels, kernel_size, kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        # initialization using Kaiming uniform
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        # call our custom conv2d op
        return custom_conv2d(input, self.weight, self.bias, self.stride, self.padding)

    def extra_repr(self):
        s = (
            "{in_channels}, {out_channels}, kernel_size={kernel_size}"
            ", stride={stride}, padding={padding}"
        )
        if self.bias is None:
            s += ", bias=False"
        return s.format(**self.__dict__)


#################################################################################
# Part II: Design and train a network
#################################################################################
class SimpleNet(nn.Module):
    # a simple CNN for image classifcation
    def __init__(self, conv_op=nn.Conv2d, num_classes=100):
        super(SimpleNet, self).__init__()
        # you can start from here and create a better model
        bn_op = nn.BatchNorm2d
        self.conv_op = conv_op

        self.features = nn.Sequential(
            conv_op(3, 64, kernel_size=7, stride=2, padding=3),
            bn_op(64),
            nn.ReLU(inplace=True),
            # max pooling 1/2
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # conv2 block: simple bottleneck
            conv_op(64, 128, kernel_size=3, stride=1, padding=1),
            bn_op(128),
            nn.ReLU(inplace=True),
            conv_op(128, 128, kernel_size=3, stride=1, padding=1),
            bn_op(128),
            nn.ReLU(inplace=True),
            # max pooling 1/2
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # conv3 block: simple bottleneck
            conv_op(128, 256, kernel_size=3, stride=1, padding=1),
            bn_op(256),
            nn.ReLU(inplace=True),
            conv_op(256, 256, kernel_size=3, stride=1, padding=1),
            bn_op(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # conv4
            conv_op(256, 512, kernel_size=3, stride=1, padding=1),
            bn_op(512),
            nn.ReLU(inplace=True),
            conv_op(512, 512, kernel_size=3, stride=1, padding=1),
            bn_op(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        # global avg pooling + FC
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),  # Adding dropout for regularization
            nn.Linear(512, num_classes)
        )
        self.attacker = default_attack

    def reset_parameters(self):
        # init all params
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, self.conv_op)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        # you can implement adversarial training here
        if self.training and not hasattr(self, 'is_adversarial'):
            self.is_adversarial = True  # Set flag to indicate we are generating an adversarial sample
            adv_x = self.attack.perturb(self, x)
            x = adv_x
            del self.is_adversarial
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class SimpleViT(nn.Module):
    """
    This module implements Vision Transformer (ViT) backbone in
    "Exploring Plain Vision Transformer Backbones for Object Detection",
    https://arxiv.org/abs/2203.16527
    """

    def __init__(
        self,
        img_size=128,
        num_classes=100,
        patch_size=16,
        in_chans=3,
        embed_dim=192,
        depth=4,
        num_heads=4,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        use_abs_pos=True,
        window_size=4,
        window_block_indexes=(0, 2),
    ):
        """
        Args:
            img_size (int): Input image size.
            num_classes (int): Number of object categories
            patch_size (int): Patch size.
            in_chans (int): Number of input image channels.
            embed_dim (int): Patch embedding dimension.
            depth (int): Depth of ViT.
            num_heads (int): Number of attention heads in each ViT block.
            mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
            qkv_bias (bool): If True, add a learnable bias to query, key, value.
            drop_path_rate (float): Stochastic depth rate.
            norm_layer (nn.Module): Normalization layer.
            act_layer (nn.Module): Activation layer.
            use_abs_pos (bool): If True, use absolute positional embeddings.
            window_size (int): Window size for local attention blocks.
            window_block_indexes (list): Indexes for blocks using local attention.
                Local window attention allows more efficient computation, and can be
                coupled with standard global attention.
                E.g., [0, 2] indicates the first and the third blocks will use
                local window attention, while other block use standard attention.

        Feel free to modify the default parameters here.
        """
        super(SimpleViT, self).__init__()

        if use_abs_pos:
            # Initialize absolute positional embedding with image size
            # The embedding is learned from data
            self.pos_embed = nn.Parameter(
                torch.zeros(
                    1, img_size // patch_size, img_size // patch_size, embed_dim
                )
            )
        else:
            self.pos_embed = None

        # stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        ########################################################################
        self.patch_embed = PatchEmbed(
            in_chans = in_chans,
            embed_dim = embed_dim,
            kernel_size = (patch_size, patch_size),
            stride = (patch_size, patch_size),
        )
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                window_size=window_size if i in window_block_indexes else 0
            )
            for i in range(depth)
        ])
        self.head = nn.Linear(embed_dim, num_classes)
        
        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)
        ########################################################################
        # the implementation shall start from embedding patches,
        # followed by some transformer blocks

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=0.02)

        self.apply(self._init_weights)
        # add any necessary weight initialization here

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        ########################################################################
        x = self.patch_embed(x)
        if self.pos_embed is not None:
            x = x + self.pos_embed

        for block in self.blocks:
            x = block(x)
            
        x = x.mean(dim=(1, 2))
        x = self.head(x)
        
        ########################################################################
        return x

# change this to your model!
default_cnn_model = SimpleNet
default_vit_model = SimpleViT

# define data augmentation used for training, you can tweak things if you want
def get_train_transforms(normalize):
    train_transforms = []
    train_transforms.append(transforms.Scale(144))
    train_transforms.append(transforms.RandomHorizontalFlip())
    train_transforms.append(transforms.RandomColor(0.15))
    train_transforms.append(transforms.RandomRotate(15))
    train_transforms.append(transforms.RandomSizedCrop(128))
    train_transforms.append(transforms.ToTensor())
    train_transforms.append(normalize)
    train_transforms = transforms.Compose(train_transforms)
    return train_transforms


# define data augmentation used for validation, you can tweak things if you want
def get_val_transforms(normalize):
    val_transforms = []
    val_transforms.append(transforms.Scale(144))
    val_transforms.append(transforms.CenterCrop(128))
    val_transforms.append(transforms.ToTensor())
    val_transforms.append(normalize)
    val_transforms = transforms.Compose(val_transforms)
    return val_transforms


#################################################################################
# Part III: Adversarial samples and Attention
#################################################################################
class PGDAttack(object):
    def __init__(self, loss_fn, num_steps=20, step_size=0.01, epsilon=0.1):
        """
        Attack a network by Project Gradient Descent. The attacker performs
        k steps of gradient descent of step size a, while always staying
        within the range of epsilon (under l infinity norm) from the input image.

        Args:
          loss_fn: loss function used for the attack
          num_steps: (int) number of steps for PGD
          step_size: (float) step size of PGD (i.e., alpha in our lecture)
          epsilon: (float) the range of acceptable samples
                   for our normalization, 0.1 ~ 6 pixel levels
        """
        self.loss_fn = loss_fn
        self.num_steps = num_steps
        self.step_size = step_size
        self.epsilon = epsilon

    def perturb(self, model, input):
        """
        Given input image X (torch tensor), return an adversarial sample
        (torch tensor) using PGD of the least confident label.

        See https://openreview.net/pdf?id=rJzIBfZAb

        Args:
          model: (nn.module) network to attack
          input: (torch tensor) input image of size N * C * H * W

        Outputs:
          output: (torch tensor) an adversarial sample of the given network
        """
        # clone the input tensor and disable the gradients
        output = input.clone()
        input.requires_grad = False

        # loop over the number of steps
        # for _ in range(self.num_steps):
        #################################################################################
        for _ in range(self.num_steps):
            output.requires_grad = True
            predictions = model(output)
            _, predicted_class = predictions.min(dim=1)
            loss = self.loss_fn(predictions, predicted_class)
            loss.backward()

            with torch.no_grad():
                perturbation = self.step_size * output.grad.sign()
                output = output + perturbation       # Applying perturbation
                output = torch.clamp(output, input - self.epsilon, input + self.epsilon)  # Ensuring perturbation doesn't exceed epsilon


            if output.grad is not None:
                output.grad.zero_()
        #################################################################################

        return output


default_attack = PGDAttack


class GradAttention(object):
    def __init__(self, loss_fn):
        """
        Visualize a network's decision using gradients

        Args:
          loss_fn: loss function used for the attack
        """
        self.loss_fn = loss_fn

    def explain(self, model, input):
        """
        Given input image X (torch tensor), return a saliency map
        (torch tensor) by computing the max of abs values of the gradients
        given by the predicted label

        See https://arxiv.org/pdf/1312.6034.pdf

        Args:
          model: (nn.module) network to attack
          input: (torch tensor) input image of size N * C * H * W

        Outputs:
          output: (torch tensor) a saliency map of size N * 1 * H * W
        """
        # Make sure input receives gradients
        input.requires_grad = True

        # Forward pass to obtain predictions
        output = model(input)
        
        # Get the predicted class (most confident prediction)
        _, predicted_class = output.max(dim=1)
        
        # Compute the loss with respect to the predicted class
        loss = self.loss_fn(output, predicted_class)

        # Zero out any existing gradients
        model.zero_grad()

        # Compute gradients of the loss with respect to the input
        loss.backward()

        # Calculate the absolute gradients
        gradients = torch.abs(input.grad)

        # Take the maximum gradient value across color channels
        saliency_map, _ = gradients.max(dim=1, keepdim=True)

        return saliency_map

default_attention = GradAttention


def vis_grad_attention(input, vis_alpha=2.0, n_rows=10, vis_output=None):
    """
    Given input image X (torch tensor) and a saliency map
    (torch tensor), compose the visualziations

    Args:
      input: (torch tensor) input image of size N * C * H * W
      output: (torch tensor) input map of size N * 1 * H * W

    Outputs:
      output: (torch tensor) visualizations of size 3 * HH * WW
    """
    # concat all images into a big picture
    input_imgs = make_grid(input.cpu(), nrow=n_rows, normalize=True)
    if vis_output is not None:
        output_maps = make_grid(vis_output.cpu(), nrow=n_rows, normalize=True)

        # somewhat awkward in PyTorch
        # add attention to R channel
        mask = torch.zeros_like(output_maps[0, :, :]) + 0.5
        mask = output_maps[0, :, :] > vis_alpha * output_maps[0, :, :].mean()
        mask = mask.float()
        input_imgs[0, :, :] = torch.max(input_imgs[0, :, :], mask)
    output = input_imgs
    return output


default_visfunction = vis_grad_attention
