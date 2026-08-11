import numpy as np
import torch

# import the dependencies
from escnn import gspaces, nn


def is_odd(number: int):
    """check if integer number is odd"""
    assert isinstance(number, int)
    return bool(number & 1)


def is_even(number: int):
    """check if number is even"""
    return not is_odd(number)


def create_field(grp_space, n_filters=1, rep_type=True):
    """
    Function to create the escnn.nn.FieldType of trivial/regular representation
    of n_filters

    Parameters
    ----------------
    grp_space : escnn.gspace
            The group space in which the a given signal lives
    n_filters : int, [Optional] by default 1
            The no. of feature fields to be created
    rep_type : bool, [Optional], by default True
            The required represention of the gspace
            if True - regular representation
            if False - trivial representation

    Return
    ---------------
    FieldType of gspace with the n_filters of feature fields
    """
    if rep_type:
        return nn.FieldType(grp_space, n_filters * [grp_space.regular_repr])
    else:
        return nn.FieldType(grp_space, n_filters * [grp_space.trivial_repr])


def EqConv(
    in_field,
    out_field,
    k_size,
    pad,
    stride,
    grp_space,
    f_cutoff=0.8 * np.pi,
    b_norm=False,
    init=True,
    in_rep=True,
    out_rep=True,
    act=False,
    act_func="r",
    bias=True,
):
    """
    The equi. conv. operation
    - can be used for designing blocks for other model arch.

    Parameters
    ----------------
    in_field : int
            The no. of input feature fields
    out_field : int
            The no. of output feature fields
    k_size : int
            The kernel size for the conv. operation
    pad : int
            The padding size for the conv. operation
    stride : int
            The stride for the conv. operation
    grp_space : escnn.gspace
            The group space in which the a given signal lives
    f_cutoff : float [Optional, by default 0.8*np.pi
            The frequency cutoff for the conv. operation
            multiple of np.pi
    b_norm : bool [Optional], by default False
            The flag to perform batch normalization after conv. layers
    init : bool [Optional], by default True
            The defualt initialization in escnn is he' method, as suggested in
            escnn documentation vary this parameter depending on task
    act : bool [Optional], by default False
            The flag to add relu activation after batch norm layers
    act_func : str, [Optional], by default - 'r'
            The flag to chose betwwen relu('r) & elu('e')
    bias : bool [Optional], by default True
        Allow a bias term in addition to the conv kernel

    Return
    ----------------
    Sequential module of escnn with (1x) Conv. with batch normalization(if specified)

    """
    in_type = create_field(grp_space, in_field, in_rep)
    out_type = create_field(grp_space, out_field, out_rep)
    layer = nn.SequentialModule(
        nn.R2Conv(
            in_type,
            out_type,
            kernel_size=k_size,
            padding=pad,
            stride=stride,
            frequencies_cutoff=f_cutoff,
            initialize=init,
            bias=bias,
        )
    )
    if b_norm:
        layer.add_module("bn", nn.InnerBatchNorm(out_type))
    if act:
        if act_func == "r":
            layer.add_module("act", nn.ReLU(out_type))
        elif act_func == "e":
            layer.add_module("act", nn.ELU(out_type))

    return layer


def EqConvBlock(
    in_field,
    out_field,
    k_size,
    pad,
    stride,
    grp_space,
    f_cutoff=0.8 * np.pi,
    b_norm=False,
    res=False,
    act_func="r",
    bias=True,
):
    """
    The equi. conv. block for the U-Net.

    - the direct conversion of feature space to g-space is implemented within the function

    Parameters
    ----------------
    in_field : int
        The no. of input feature fields
    out_field : int
        The no. of output feature fields
    k_size : int
        The kernel size for the conv. operation
    pad : int
        The padding size for the conv. operation
    stride : int
        The stride for the conv. operation
    grp_space : escnn.gspace
        The group space in which the a given signal lives
    f_cutoff : float [Optional, by default 0.8*np.pi
        The frequency cutoff for the conv. operation
        multiple of np.pi
    b_norm : bool [Optional], by default False
        The flag to perform batch normalization after conv. layers
    res : bool [Optional], by default False
        The block for resnet models
    act_func : str, [Optional], by default - 'r'
            The flag to chose betwwen relu('r) & elu('e')
    bias : bool [Optional], by default True
        Allow a bias term in addition to the conv kernel

    Return
    ----------------
    Sequential module of escnn with (2x) Conv. each follwed by ReLU activation

    """
    in_type = (
        in_field
        if isinstance(in_field, nn.field_type.FieldType)
        else create_field(grp_space, in_field)
    )
    out_type = (
        out_field
        if isinstance(out_field, nn.field_type.FieldType)
        else create_field(grp_space, out_field)
    )
    layers = []

    # TODO: include option for other activation
    for i in range(2):
        layers.append(
            nn.R2Conv(
                in_type,
                out_type,
                kernel_size=k_size,
                padding=pad,
                stride=stride,
                frequencies_cutoff=f_cutoff,
                bias=bias,
            )
        )
        if b_norm:
            layers.append(nn.InnerBatchNorm(out_type))

        if act_func == "r":
            act_layer = nn.ReLU(out_type)
        elif act_func == "e":
            act_layer = nn.ELU(out_type)

        layers.append(act_layer)
        in_type = out_type
    return nn.SequentialModule(*layers)


def Pool(
    in_field, grp_space, k_size=2, pad=0, stride=2, pool_type="max", alias=True, sig=0.6
):
    """
    The equi. max pooling block

    Parameters
    -------------
    in_field : int
            The no. of input feature fields
    grp_space : escnn.gspace
            The group space in which the a given signal lives
    k_size : int, [Optional], by default 2
            The kernel size for the max operation
    pad : int, [Optional], by default 0
            The padding size for the max operation
    stride : int, [Optional], by default 2
            The stride for the max operation
    pool_type : str, [Optional] by default "max"
            "max" - MaxPool opration is applied
            "avg" - Avg. pool option is applied
            "strided" - use strided convs for pool to retain symmetries
    alias : bool, [Optiona], by default True
            if True - PointWise(Max/Avg)PoolAAntialisaed option based on shift-invariant
                        conv. operation
            if False - PointWise(Max/Avg)Pool option is selected
    sig : float, [Optional], by default 0.6
            The std. deviation of gaussian blur

    Return
    --------------
    Channel-wise max pooled feature maps of specified options
    """
    in_type = create_field(grp_space, in_field)
    pt = pool_type.lower()

    if pt == "max":
        if alias:
            return nn.PointwiseMaxPoolAntialiased(
                in_type, kernel_size=k_size, stride=stride, padding=pad, sigma=sig
            )
        else:
            return nn.PointwiseMaxPool(
                in_type, kernel_size=k_size, stride=stride, padding=pad
            )
    elif pt == "avg":
        if alias:
            return nn.PointwiseAvgPoolAntialiased(
                in_type, stride=stride, sigma=sig, padding=pad
            )
        else:
            return nn.PointwiseAvgPool(
                in_type, kernel_size=k_size, stride=stride, padding=pad
            )
    elif pt == "stride":
        # use EqConv as pooling operation, in_field == out_field
        # pooling is not along the feature field axis
        return EqConv(
            in_field=in_field,
            out_field=in_field,
            k_size=k_size,
            stride=stride,
            pad=pad,  # we assume zero padding when using a conv for pooling
            grp_space=grp_space,
            act=False,  # no activation
            b_norm=False,  # no batch norm
            bias=False,  # should learn pooling and not more
        )
    else:
        raise NotImplementedError


def UpSample(in_field, grp_space, scale=2, osize=None, align=True):
    """
    Function for upsampling on the given feature field

    NOTE: only "bilinear" mode preserves equivariance, so the default is used as in escnn
    Parameter
    ------------
    in_field : int
            The no. of input feature fields
    grp_space : escnn.gspace
            The group space in which the a given signal lives
    scale : int, [Optional], by default 2
        The scaling factor for the Upsampling
    osize : int, [Optional], by default None
        defines the output size to upsample to
        Ignore if set to None. If set to value larger than 0, takes precendence over scale.
    align : bool, [Optional], by default True
        For the alignment of corner pixels

    Return
    ----------------
    Sequential module of escnn with (1x) UPsampled feature fields
    """
    in_type = (
        in_field
        if isinstance(in_field, nn.field_type.FieldType)
        else create_field(grp_space, in_field)
    )

    value = nn.SequentialModule(
        nn.R2Upsampling(in_type, scale_factor=scale, align_corners=align)
    )
    if osize:
        value = nn.SequentialModule(
            nn.R2Upsampling(in_type, size=osize, align_corners=align)
        )
    return value


class EqEncode(torch.nn.Module):
    """
    an equivariant encoder module for groups belonging to the dihedral discrete group in 2D
    """

    def __init__(
        self,
        signal_shape: tuple[int, int, int],
        n_theta: int = 4,  # TODO: could be replaced by just the gspace object
        flip: bool = False,  # TODO: could be replaced by just the gspace object
        features: tuple[int] = (8, 8, 16, 32, 64),
        backbone: str = "unet",  # TODO: unclear if this is needed
        conv_kernel: int = 3,
        conv_pad: int = 1,
        conv_stride: int = 1,
        f_cutoff: float = 0.8 * np.pi,
        conv_bnorm: bool = False,
        pool_kernel: int = 3,
        pool_pad: int = 1,
        pool_stride: int = 2,
        pool_type: str = "stride",
        pool_alias: bool = True,
        act: str = "r",
    ):
        """

        Constructor of equivariant 2D Unet

        Parameters
        ----------
        signal_shape : tuple[int, int, int]
            the shape of the 2D input signal or image, expected to be encoded as (channels, height, width)
        n_theta : int
            The number of rotations for the cyclic group
        flip: bool
            The flag to include dihedral group
            False - only cyclic group is considered
        features: tuple[int]
            the output features of the encoder layers. the length of this tuple governs the depth of the encoder. For example, the default (8,8,16,32,64) would create a 3-layer encoder. The configuration for the default features would look like this:
            DoubleConv(8)
            DoubleConv(8)+Pool
            DoubleConv(16)+Pool
            DoubleConv(32)+Pool
            DoubleConv(64)

        backbone : str
            **uneffective at the moment!**
            The encoding part of the unet
            available encoders, "unet", "resnet", "vgg"
        conv_kernel: int
            The kernel size of the conv. layers
        conv_pad : int
            The pad size of the conv. layers
        conv_stride : int
            The stride size of the conv. layers
        f_cutoff : float
            The cut of frequncy for the filter bassi expansion
        conv_bnorm : bool
            to include batch normalization between conv. layers
        pool_kernel : int
            The kernel size of pooling layers
        pool_pad : int
            The pad size of pooling layers
        pool_stride : int
            The stride size of pooling layers
        pool_type : str
            The pool type for pooling layers
            available: "avg", "stride"
            The kernel size of pooling layers
        pool_alias : bool
            to include alias for avg or max pooling operations
        act : str
            the activation function to use after conv layers
            'r' - relu activation
            'e' - elu activation

        Examples
        --------
        # for a 16x16 single channel signal
        > enc = EqEncode((1,32,32))
        > y = enc(x)
        """

        super().__init__()

        assert (
            len(signal_shape) == 3
        ), f"EqEncode:: incompatible signal_shape {signal_shape} (should be (C,H,W))"

        self.in_shape = signal_shape
        self.n_theta = n_theta
        self.flip = flip
        self.features = features
        self.conv_kernel = conv_kernel
        self.conv_pad = conv_pad
        self.conv_stride = conv_stride
        self.f_cutoff = f_cutoff
        self.conv_bnorm = conv_bnorm
        self.pool_kernel = pool_kernel
        self.pool_pad = pool_pad
        self.pool_stride = pool_stride
        self.pool_type = pool_type
        self.pool_alias = pool_alias
        self.act = act
        self.pool_sigma = 0.4  # ksize = 3

        if "avg" in self.pool_type.lower() and self.pool_alias:
            # recompute sigma, see https://github.com/QUVA-Lab/escnn/blob/b101341b4c23fb68e7d5d9a093d86461a7b47c67/escnn/nn/modules/pooling/pointwise_avg.py#L171
            self.pool_sigma = (0.5 * (self.pool_kernel - 1)) / 3.0

        if self.flip:
            if n_theta == 1:
                self.gspace = gspaces.flip2dOnR2()
            else:
                self.gspace = gspaces.flipRot2dOnR2(self.n_theta)
        else:
            if n_theta == 1:
                self.gspace = gspaces.trivialOnR2()
            else:
                self.gspace = gspaces.rot2dOnR2(self.n_theta)

        self.in_type = create_field(
            self.gspace, n_filters=signal_shape[0], rep_type=False
        )
        # output needs to be in representation space (regular for now)
        self.out_type = create_field(self.gspace, n_filters=features[-1], rep_type=True)

        self.convs = torch.nn.ModuleList()  # conv. operation
        self.poolings = torch.nn.ModuleList()  # pooling operation

        self.out_shapes = []
        x = torch.zeros((4, *self.in_shape))
        x_ = self.in_type(x)

        out_type = create_field(self.gspace, self.features[0])
        self.convs.append(
            # this is always a double block
            EqConvBlock(
                in_field=self.in_type,
                out_field=out_type,
                k_size=self.conv_kernel,
                pad=self.conv_pad,
                stride=self.conv_stride,
                grp_space=self.gspace,
                f_cutoff=self.f_cutoff,
                b_norm=self.conv_bnorm,
                act_func=self.act,
            )
        )
        # x_ = x_.to(self.convs[-1].device)
        self.convs[-1] = self.convs[-1].to(x_.tensor.device)  # harmonize devices
        y = self.convs[-1].forward(x_)  # test run 'forward' method
        self.out_shapes.append(y.shape[-3:])
        x_ = y

        in_type = out_type
        for out_filter in self.features[1:-1]:
            out_type = create_field(self.gspace, out_filter)
            self.convs.append(
                # this is always a double block
                EqConvBlock(
                    in_field=in_type,
                    out_field=out_type,
                    k_size=self.conv_kernel,
                    pad=self.conv_pad,
                    stride=self.conv_stride,
                    grp_space=self.gspace,
                    f_cutoff=self.f_cutoff,
                    b_norm=self.conv_bnorm,
                    act_func=self.act,
                )
            )

            y = self.convs[-1].forward(x_)
            in_shape = y.shape
            pool_ksize = self.pool_kernel
            pool_pad = self.pool_pad
            pool_sigma = self.pool_sigma
            if self.pool_stride == 2:
                if "stride" in self.pool_type.lower():
                    if is_even(in_shape[-1]):
                        if is_odd(pool_ksize):
                            pool_ksize = self.pool_kernel + 1
                            pool_pad = pool_ksize // 2
                    elif is_odd(in_shape[-1]):
                        if is_even(pool_ksize):
                            pool_ksize = self.pool_kernel + 1
                            pool_pad = pool_ksize // 2
                elif "avg" in self.pool_type.lower():
                    if not self.pool_alias:
                        if is_even(in_shape[-1]):
                            if is_odd(pool_ksize):
                                pool_ksize = self.pool_kernel + 1
                                pool_pad = pool_ksize // 2
                        elif is_odd(in_shape[-1]):
                            if is_even(pool_ksize):
                                pool_ksize = self.pool_kernel + 1
                                pool_pad = pool_ksize // 2
                    else:
                        # LEARNING for stride 2 avgpool antialiased:
                        # - odd-sized input can be handled with odd kernels
                        # - even-sized input cannot be handled at all!
                        if is_even(in_shape[-1]):
                            # TODO: this could be replaced by an non antialias version or a strided conv
                            raise ValueError(
                                f"unable to treat even sized input {in_shape} with avg pooling and aliasing on"
                            )
                        else:
                            if is_even(pool_ksize):
                                pool_ksize = self.pool_kernel + 1
                                pool_pad = pool_ksize // 2
            self.poolings.append(
                Pool(
                    in_field=out_filter,
                    grp_space=self.gspace,
                    k_size=pool_ksize,
                    pad=pool_pad,
                    stride=self.pool_stride,
                    pool_type=self.pool_type,
                    alias=self.pool_alias,
                    sig=pool_sigma,
                )
            )
            self.poolings[-1] = self.poolings[-1].to(
                y.tensor.device
            )  # harmonize devices
            y = self.poolings[-1].forward(y)  # test-predict
            x_ = y
            self.out_shapes.append(y.shape[-3:])
            in_type = out_type

        out_type = create_field(self.gspace, self.features[-1])
        self.convs.append(
            EqConvBlock(
                in_field=in_type,
                out_field=out_type,
                k_size=self.conv_kernel,
                pad=self.conv_pad,
                stride=self.conv_stride,
                grp_space=self.gspace,
                f_cutoff=self.f_cutoff,
                b_norm=self.conv_bnorm,
                act_func=self.act,
            )
        )

    def forward(self, x: nn.GeometricTensor):
        """

        method to run the encoder onl (includes the bottle neck)

        Parameters
        ----------
        x : torch.Tensor
            input batch of data

        Returns
        -------
        y : torch.Tensor
            output predictions
        convs : List(torch.Tensor)
            outputs of all downward convolutions+activations (excluding pooling operations)


        Examples
        --------
        > y, convs = en.forward(x)


        """
        assert x.shape[-2:] == self.in_shape[-2:]
        x = self.in_type(x) if not isinstance(x, nn.GeometricTensor) else x
        x = self.convs[0](x)
        # downward path
        out_tensors = []
        for lay, pool in zip(self.convs[1:-1], self.poolings):
            c = lay(x)
            x = pool(c)
            out_tensors.append(c)  # store for skip connection

        # last downward conv (produces bottleneck)
        x = self.convs[-1](x)

        return x, out_tensors

    def nparams(self):
        """

        return the number of parameters of this model

        Examples
        --------
        > mdl = EqUnet((1,32,32))
        > mdl.nparams()

        """

        value = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return value


class EqDecode(torch.nn.Module):
    """
    an equivariant decoder module for groups belonging to the dihedral discrete group in 2D
    """

    def __init__(
        self, encoder_template: EqEncode, out_channels: int = 1, tail_channels: int = 32
    ):
        """

        Constructor of EqDecode

        Parameters
        ----------
        encoder_template : EqEncode
            encoder to bootstrap this decoder from
        out_channels : int
            number of channels to use as output
        tail_channels : int
            number of channels to use for the last conv double block before the
            output is generated. If smaller than 1. ignore this

        Examples
        --------
        > en = EqEncode()
        > dec = EqDecode(en)
        > y_prime = dec(z)

        """
        super().__init__()
        self.enc = encoder_template
        self.up_conv = torch.nn.ModuleList()
        self.up_sample = torch.nn.ModuleList()
        self.out_channels = out_channels
        self.tail_channels = tail_channels
        # repr_cardinality = self.enc.gspace.fibergroup.order()
        los = len(self.enc.out_shapes)

        for depth, (in_filter, out_filter, in_shape) in enumerate(
            zip(
                reversed(self.enc.features[1:]),
                reversed(self.enc.features[:-1]),
                reversed(self.enc.out_shapes),
            )
        ):
            exp_in_shape = in_shape[-2:]
            next_in_shape = self.enc.out_shapes[los - 2 - depth]
            exp_out_shape = [item * 2 for item in exp_in_shape]
            obs_out_shape = list(next_in_shape[-2:])
            if obs_out_shape != exp_out_shape:
                # override scale=2 in UpSample to match skip connection shape
                self.up_sample.append(
                    UpSample(
                        in_filter, osize=obs_out_shape[-1], grp_space=self.enc.gspace
                    )
                )
            else:
                # allow default scale=2 in UpSample
                self.up_sample.append(UpSample(in_filter, grp_space=self.enc.gspace))

            # divide-by-2 is based on an assumption
            # out_filter = in_filter // 2

            self.up_conv.append(
                EqConvBlock(
                    in_field=in_filter + out_filter,  # sum due to nn.tensor_directsum
                    out_field=out_filter,
                    k_size=self.enc.conv_kernel,
                    pad=self.enc.conv_pad,
                    stride=self.enc.conv_stride,
                    grp_space=self.enc.gspace,
                    f_cutoff=self.enc.f_cutoff,
                    b_norm=self.enc.conv_bnorm,
                    act_func=self.enc.act,
                )
            )

        self.tail_conv = None
        if tail_channels > 0 or not tail_channels:
            self.tail_conv = EqConvBlock(
                # NB. z = nn.tensor_directsum([z, c]) in forward call doubles
                # the number of feature fields (z and c are stacked
                # wrt to the regular repr axis, hence len(z) becomes 2),
                # conv operations in self.up_conv silently work anyhow,
                # BUT the output of EqConvBlock produces types with
                # len(z.type)==2, need to take this into accout here:
                in_field=in_filter,  # found empirically, tricky due to tensor_directsum
                out_field=tail_channels,
                k_size=self.enc.conv_kernel,
                pad=self.enc.conv_pad,
                stride=self.enc.conv_stride,
                grp_space=self.enc.gspace,
                f_cutoff=self.enc.f_cutoff,
                b_norm=self.enc.conv_bnorm,
                act_func=self.enc.act,
            )
            self.out_type = self.tail_conv.out_type
        else:
            self.out_type = self.up_conv[-1].out_type
            # self.out_type = create_field(self.enc.gspace, n_filters=in_filter)

    def forward(self, z, out_tensors):
        """

        decode the latents obtained from the encoding

        Parameters
        ----------
        z : torch.Tensor
            input data/predictions to the decoder (aka bottleneck that was produced by self.encode)
        out_tensors : List(torch.Tensor)
            outputs of all downward layers during encoding

        Examples
        --------
        > en = EQNet()
        > z, convs = en.encode(x)
        > y = en.decode(z,convs)

        """

        for upsample, lay, c in zip(
            self.up_sample, self.up_conv, reversed(out_tensors)
        ):
            z = upsample(z)
            z = nn.tensor_directsum([z, c])  # apply skip connect
            z = lay(z)

        if self.tail_conv:
            z = self.tail_conv(z)

        return z

    def nparams(self):
        """

        return the number of parameters of this model

        Examples
        --------
        > mdl = EqUnet((1,32,32))
        > mdl.nparams()

        """

        value = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return value


class EqUnet(torch.nn.Module):
    """
    an equivariant Unet for groups belonging to the dihedral discrete group in 2D
    """

    def __init__(
        self,
        signal_shape: tuple[int, int, int],
        n_theta: int = 4,  # TODO: could be replaced by just the gspace object
        flip: bool = False,  # TODO: could be replaced by just the gspace object
        features: tuple[int] = (8, 8, 16, 32, 64, 32),
        backbone: str = "unet",  # TODO: unclear if this is needed
        conv_kernel: int = 3,
        conv_pad: int = 1,
        conv_stride: int = 1,
        f_cutoff: float = 0.8 * np.pi,
        conv_bnorm: bool = False,
        pool_kernel: int = 3,
        pool_pad: int = 1,
        pool_stride: int = 2,
        pool_type: str = "stride",
        pool_alias: bool = True,
        act: str = "r",
        final: str = "gpool",
        out_channels: int = 1,
    ):
        """

        Constructor of equivariant 2D Unet

        Parameters
        ----------
        signal_shape : tuple[int, int, int]
            the shape of the 2D input signal or image, expected to be encoded as (channels, height, width)
        n_theta : int
            The number of rotations for the cyclic group
        flip: bool
            The flag to include dihedral group
            False - only cyclic group is considered
        features: tuple[int]
            the output features of the encoder layers. the length of this tuple governs the depth of the encoder and decoder. For example, the default (8,8,16,32,64) would create a 3-layer encoder. The configuration for the default features (8, 8, 16, 32, 64, 32) would look like this:
            DoubleConv->8
            DoubleConv+Pool->8
            DoubleConv+Pool->16
            DoubleConv+Pool->32
            DoubleConv->64
            #bottleneck
            UpSample+Cat+DoubleConv->32
            UpSample+Cat+DoubleConv->16
            UpSample+Cat+DoubleConv->8
            DoubleConv->32

        backbone : str
            **uneffective at the moment!**
            The encoding part of the unet
            available encoders, "unet", "resnet", "vgg"
        conv_kernel: int
            The kernel size of the conv. layers
        conv_pad : int
            The pad size of the conv. layers
        conv_stride : int
            The stride size of the conv. layers
        f_cutoff : float
            The cut of frequncy for the filter bassi expansion
        conv_bnorm : bool
            to include batch normalization between conv. layers
        pool_kernel : int
            The kernel size of pooling layers
        pool_pad : int
            The pad size of pooling layers
        pool_stride : int
            The stride size of pooling layers
        pool_type : str
            The pool type for pooling layers
            available: "avg", "stride"
            The kernel size of pooling layers
        pool_alias : bool
            to include alias for avg or max pooling operations
        act : str
            the activation function to use after conv layers
            'r' - relu activation
            'e' - elu activation
        final: str
            if equal to "gpool", append a GlobalPooling operation to the end of
        the Unet; else append regular R2Conv+Activation
        out_channels: int
            number of output channels to produce

        Examples
        --------
        # for a 16x16 single channel signal
        > mdl = EqUnet((1,32,32))
        > y = mdl(x)
        """

        super().__init__()

        self.enc_features = tuple(features[:-1])
        self.dec_tail_features = features[-1]

        self.enc = EqEncode(
            signal_shape,
            n_theta,
            flip,
            self.enc_features,
            backbone,
            conv_kernel,
            conv_pad,
            conv_stride,
            f_cutoff,
            conv_bnorm,
            pool_kernel,
            pool_pad,
            pool_stride,
            pool_type,
            pool_alias,
            act,
        )

        self.gspace = self.enc.gspace
        self.dec = EqDecode(self.enc, tail_channels=self.dec_tail_features)
        self.out_channels = out_channels
        self.final_type = final
        self.in_type = self.enc.in_type

        if self.final_type.lower() == "gpool":
            self.pool = nn.GroupPooling(self.dec.out_type)
            self.tfinal = torch.nn.Conv2d(
                len(self.pool.out_type), out_channels, kernel_size=3, padding=1
            )

        else:
            # this is a R2Conv plus an activation
            # but the outtype of EqConv has regular representations
            self.pool = EqConv(
                in_field=self.dec.out_type,
                out_field=out_channels,
                k_size=3,
                pad=1,
                stride=1,
                grp_space=self.gspace,
                f_cutoff=None,
                out_rep=False,  # returns trivial representation
            )
            self.tfinal = torch.nn.Conv2d(
                out_channels, out_channels, kernel_size=3, padding=1
            )
        self.out_type = torch.Tensor  # will return a torch.Tensor

    def forward(self, x: torch.Tensor):
        """

        method to run the model

        Parameters
        ----------
        x : torch.Tensor
            input batch of data

        Returns
        -------
        y : torch.Tensor
            output predictions
        convs : List(torch.Tensor)
            outputs of all downward convolutions+activations (excluding pooling operations)


        Examples
        --------
        > y = model(x)


        """
        x_ = x if isinstance(x, nn.GeometricTensor) else self.in_type(x)

        z, outt = self.enc(x_)
        y = self.dec(z, outt)

        y = self.pool(y)
        y = self.tfinal(y.tensor)

        return y

    def nparams(self):
        """

        return the number of parameters of this model

        Examples
        --------
        > mdl = EqUnet((1,32,32))
        > mdl.nparams()
        392201
        """
        value = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return value
