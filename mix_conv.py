import torch
import torch.nn as nn


class MixConv:
    """MixConv with mixed depth-wise convolutional kernels.
      MDConv is an improved depth-wise convolution that mixes multiple kernels (e.g.
      3x3, 5x5, etc). Right now, we use an naive implementation that split channels
      into multiple groups and perform different kernels for each group.
      See MixNet paper https://arxiv.org/abs/1907.09595 for more details."""

    def __init__(self, ksize: list, stride: int, dilated=False, bias=False):
        """

        :param ksize: list of int, kernel_size of each group;
        :param stride: int, stride for each group;
        :param dilated: bool. indicate whether to use dilated conv to simulate large
                        kernel size, note that this only take effect when stride is 1;
        :param bias: convolution layer bias.
        """

        self.k_list = []
        self.d_list = []

        for k in ksize:
            d = 1
            # Only apply dilated conv for stride 1 if needed
            if stride == 1 and dilated:
                d = (k - 1) // 2
                k = 3

            self.k_list.append(k)
            self.d_list.append(d)

        self.stride = stride
        self.bias = bias

    def __call__(self, x):
        num_groups = len(self.k_list)
        # returns a tuple
        x_split = torch.split(x, num_groups, dim=1)

        # with dilation, the truth kernel size is: (k - 1) * d + 1
        conv_list = [nn.Conv2d(x.size(1), x.size(1), k, self.stride,
                               padding=(k - 1) * d // 2, dilation=d, groups=x.size(1), bias=self.bias)
                     for k, d, x in zip(self.k_list, self.d_list, x_split)]

        out_conv = [conv(inputs) for conv, inputs in zip(conv_list, x_split)]
        out = torch.cat(out_conv, dim=1)

        return out
