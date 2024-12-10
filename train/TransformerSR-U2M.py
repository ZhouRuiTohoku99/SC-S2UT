
from dataclasses import dataclass
from typing import Any, Optional

import torch  # noqa 42
from torch import nn

from speechbrain.dataio.dataio import length_to_mask
from speechbrain.lobes.models.transformer.Transformer import (
    TransformerDecoder,
    TransformerEncoder,
    NormalizedEmbedding,
    TransformerInterface,
    get_key_padding_mask,
)
from speechbrain.nnet.activations import Swish
from speechbrain.nnet.containers import ModuleList
from speechbrain.nnet.linear import Linear
import torch.nn.functional as F


class Adaptor(TransformerInterface):

    def __init__(
        self,
        mel_dim,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        d_ffn=2048,
        dropout=0.1,
        activation=nn.ReLU,
        positional_encoding="fixed_abs_sine",
        normalize_before=False,
        kernel_size: Optional[int] = 31,
        bias: Optional[bool] = True,
        encoder_module: Optional[str] = "transformer",
        conformer_activation: Optional[nn.Module] = Swish,
        branchformer_activation: Optional[nn.Module] = nn.GELU,
        attention_type: Optional[str] = "regularMHA",
        max_length: Optional[int] = 2500,
        causal: Optional[bool] = True,
        csgu_linear_units: Optional[int] = 3072,
        gate_activation: Optional[nn.Module] = nn.Identity,
        use_linear_after_conv: Optional[bool] = False,
    ):
        super().__init__(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            d_ffn=d_ffn,
            dropout=dropout,
            activation=activation,
            positional_encoding=positional_encoding,
            normalize_before=normalize_before,
            kernel_size=kernel_size,
            bias=bias,
            encoder_module=encoder_module,
            conformer_activation=conformer_activation,
            branchformer_activation=branchformer_activation,
            attention_type=attention_type,
            max_length=max_length,
            causal=causal,
            csgu_linear_units=csgu_linear_units,
            gate_activation=gate_activation,
            use_linear_after_conv=use_linear_after_conv,
        )
        
        self.custom_out_module = ModuleList(
            Linear(
                input_size=d_model,
                n_neurons=mel_dim,
                bias=True,
                combine_dims=False,
            ),
            torch.nn.Dropout(dropout),
        )
        
        self.adaptor = TransformerDecoder(
                    nhead=nhead,
                    num_layers=num_decoder_layers,
                    d_ffn=d_ffn,
                    d_model=d_model,
                    dropout=dropout,
                    activation=activation,
                    normalize_before=normalize_before,
                    causal=causal,
                    attention_type=attention_type,
                )

        # reset parameters using xavier_normal_

    def forward(self, src, tgt, target_length=None):
        """
        Arguments
        ----------
        src : torch.Tensor
            The sequence to the encoder.
        tgt : torch.Tensor
            The sequence to the decoder.
        wav_len: torch.Tensor, optional
            Torch Tensor of shape (batch, ) containing the relative length to padded length for each example.
        pad_idx : int, optional
            The index for <pad> token (default=0).
        """

        # reshape the src vector to [Batch, Time, Fea] is a 4d vector is given
        tgt_key_padding_mask = None
        if target_length is not None:
            abs_len = torch.round(target_length * tgt.shape[1])
            tgt_key_padding_mask = ~length_to_mask(abs_len).bool()
        src_key_padding_mask = None
        src_mask = None
        tgt_mask = None

    # mask out audio beyond the length of audio for each batch
        #src = self.custom_src_module(src)
        # add pos encoding to queries if are sinusoidal ones else
        tgt = tgt + self.positional_encoding(tgt)  # add the encodings here
        adaptor_out, _, _ = self.adaptor(
            tgt=tgt,
            memory=src,
            memory_mask=src_mask,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )
        adaptor_out = self.custom_out_module(adaptor_out)
        return adaptor_out
    
    def _init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_normal_(p)
    
class Adaptor_Pretrain(TransformerInterface):

    def __init__(
        self,
        mel_dim,
        embedding_size=256,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        d_ffn=2048,
        dropout=0.1,
        activation=nn.ReLU,
        positional_encoding="fixed_abs_sine",
        normalize_before=False,
        kernel_size: Optional[int] = 31,
        bias: Optional[bool] = True,
        encoder_module: Optional[str] = "transformer",
        conformer_activation: Optional[nn.Module] = Swish,
        branchformer_activation: Optional[nn.Module] = nn.GELU,
        attention_type: Optional[str] = "regularMHA",
        max_length: Optional[int] = 2500,
        causal: Optional[bool] = True,
        csgu_linear_units: Optional[int] = 3072,
        gate_activation: Optional[nn.Module] = nn.Identity,
        use_linear_after_conv: Optional[bool] = False,
    ):
        super().__init__(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            d_ffn=d_ffn,
            dropout=dropout,
            activation=activation,
            positional_encoding=positional_encoding,
            normalize_before=normalize_before,
            kernel_size=kernel_size,
            bias=bias,
            encoder_module=encoder_module,
            conformer_activation=conformer_activation,
            branchformer_activation=branchformer_activation,
            attention_type=attention_type,
            max_length=max_length,
            causal=causal,
            csgu_linear_units=csgu_linear_units,
            gate_activation=gate_activation,
            use_linear_after_conv=use_linear_after_conv,
        )
        
        self.custom_out_module = ModuleList(
            Linear(
                input_size=d_model,
                n_neurons=mel_dim,
                bias=True,
                combine_dims=False,
            ),
            torch.nn.Dropout(dropout),
        )

        self.custom_in_module = ModuleList(
            Linear(
                input_size=embedding_size,
                n_neurons=d_model,
                bias=True,
                combine_dims=False,
            ),
            torch.nn.Dropout(dropout),
        )
        
        self.adaptor = TransformerDecoder(
                    nhead=nhead,
                    num_layers=num_decoder_layers,
                    d_ffn=d_ffn,
                    d_model=d_model,
                    dropout=dropout,
                    activation=activation,
                    normalize_before=normalize_before,
                    causal=causal,
                    attention_type=attention_type,
                )

        # reset parameters using xavier_normal_

    def forward(self, src, tgt, target_length=None):
        """
        Arguments
        ----------
        src : torch.Tensor
            The sequence to the encoder.
        tgt : torch.Tensor
            The sequence to the decoder.
        wav_len: torch.Tensor, optional
            Torch Tensor of shape (batch, ) containing the relative length to padded length for each example.
        pad_idx : int, optional
            The index for <pad> token (default=0).
        """

        # reshape the src vector to [Batch, Time, Fea] is a 4d vector is given
        src = self.custom_in_module(src)
        tgt_key_padding_mask = None
        if target_length is not None:
            abs_len = torch.round(target_length * tgt.shape[1])
            tgt_key_padding_mask = ~length_to_mask(abs_len).bool()
        src_key_padding_mask = None
        src_mask = None
        tgt_mask = None
        
    # mask out audio beyond the length of audio for each batch
        #src = self.custom_src_module(src)
        # add pos encoding to queries if are sinusoidal ones else
        tgt = tgt + self.positional_encoding(tgt)  # add the encodings here
        adaptor_out, _, _ = self.adaptor(
            tgt=tgt,
            memory=src,
            memory_mask=src_mask,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )
        adaptor_out = self.custom_out_module(adaptor_out)
        return adaptor_out
    
    def _init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_normal_(p)

class Code_FFT(TransformerInterface):

    def __init__(
        self,
        vocab_size,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
        d_ffn=2048,
        dropout=0.1,
        activation=nn.ReLU,
        positional_encoding="fixed_abs_sine",
        normalize_before=False,
        kernel_size: Optional[int] = 31,
        bias: Optional[bool] = True,
        encoder_module: Optional[str] = "transformer",
        conformer_activation: Optional[nn.Module] = Swish,
        branchformer_activation: Optional[nn.Module] = nn.GELU,
        attention_type: Optional[str] = "regularMHA",
        ffn_type = "1dcnn",
        max_length: Optional[int] = 2500,
        causal: Optional[bool] = True,
        csgu_linear_units: Optional[int] = 3072,
        gate_activation: Optional[nn.Module] = nn.Identity,
        use_linear_after_conv: Optional[bool] = False,
    ):
        super().__init__(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            d_ffn=d_ffn,
            dropout=dropout,
            activation=activation,
            positional_encoding=positional_encoding,
            normalize_before=normalize_before,
            kernel_size=kernel_size,
            bias=bias,
            encoder_module=encoder_module,
            conformer_activation=conformer_activation,
            branchformer_activation=branchformer_activation,
            attention_type=attention_type,
            max_length=max_length,
            causal=causal,
            csgu_linear_units=csgu_linear_units,
            gate_activation=gate_activation,
            use_linear_after_conv=use_linear_after_conv,
        )
        self.encoder = TransformerEncoder(
                    nhead=nhead,
                    num_layers=num_encoder_layers,
                    d_ffn=d_ffn,
                    d_model=d_model,
                    dropout=dropout,
                    activation=activation,
                    normalize_before=normalize_before,
                    causal=causal,
                    attention_type=attention_type,
                    ffn_type=ffn_type,
                )
        self.custom_emb_module = ModuleList(
                NormalizedEmbedding(d_model, vocab_size)
            )
        self._init_params()

    def forward(self, src, pad_idx=0):
        """
        Arguments
        ----------
        src : torch.Tensor
            The sequence to the encoder.
        tgt : torch.Tensor
            The sequence to the decoder.
        wav_len: torch.Tensor, optional
            Torch Tensor of shape (batch, ) containing the relative length to padded length for each example.
        pad_idx : int, optional
            The index for <pad> token (default=0).
        """

        # reshape the src vector to [Batch, Time, Fea] is a 4d vector is given
        src_key_padding_mask = get_key_padding_mask(src, pad_idx=pad_idx)
        src_mask = None
        src = self.custom_emb_module(src)
    # mask out audio beyond the length of audio for each batch

        #src = self.custom_src_module(src)
        # add pos encoding to queries if are sinusoidal ones else
        src = src + self.positional_encoding(src)  # add the encodings here
        pos_embs_encoder = None

        encoder_out, _ = self.encoder(
            src=src,
            src_mask=src_mask,
            src_key_padding_mask=src_key_padding_mask,
            pos_embs=pos_embs_encoder,
        )
        return encoder_out
    
    def _init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_normal_(p)
