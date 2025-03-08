"""
16ビット精度（半精度浮動小数点数）でトレーニングするためのヘルパー関数とクラス。
"""

import numpy as np
import torch
import torch.nn as nn
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

from .models.unet import UNetModel

INITIAL_LOG_LOSS_SCALE = 20.0


def convert_module_to_f16(layer):
    """
    プリミティブなモジュールをfloat16に変換する。
    """
    if isinstance(layer, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        layer.weight.data = layer.weight.data.half()
        if layer.bias is not None:
            layer.bias.data = layer.bias.data.half()


class MixedPrecisionTrainer:
    def __init__(
        self,
        *,
        model: UNetModel,
        fp16_scale_growth=1e-3,
        initial_lg_loss_scale=INITIAL_LOG_LOSS_SCALE,
    ):
        self.model = model
        self.fp16_scale_growth = fp16_scale_growth  # 損失スケールの成長率

        self.model_params = list(self.model.parameters())  # モデルのパラメータリスト
        self.lg_loss_scale = initial_lg_loss_scale  # 現在の損失スケールの対数値

        named_params = self.model.named_parameters()
        # パラメータグループとその形状を取得
        self.param_groups_and_shapes = self._get_param_groups_and_shapes(named_params)
        # マスターパラメータ（float32）を作成
        self.master_params = self._make_master_params(self.param_groups_and_shapes)
        self.model.convert_to_fp16()  # モデルをfloat16に変換

    def zero_grad(self):
        for param in self.model_params:
            # Taken from https://pytorch.org/docs/stable/_modules/torch/optim/optimizer.html#Optimizer.add_param_group
            if param.grad is not None:
                param.grad.detach_()
                param.grad.zero_()

    def backward(self, loss: torch.Tensor):
        """
        損失をスケーリングして逆伝播を行う。
        """
        loss_scale = 2**self.lg_loss_scale
        (loss * loss_scale).backward()

    def optimize(self, opt: torch.optim.Optimizer):
        self._model_grads_to_master_grads()
        grad_norm, param_norm = self._compute_norms(grad_scale=2**self.lg_loss_scale)
        if self._check_overflow(grad_norm):
            self.lg_loss_scale -= 1
            print(f"Found NaN, decreased lg_loss_scale to {self.lg_loss_scale}")
            self._zero_master_grads()
            return False

        for p in self.master_params:
            p.grad.mul_(1.0 / (2**self.lg_loss_scale))
        # self.master_params[0].grad.mul_(1.0 / (2 ** self.lg_loss_scale))
        opt.step()
        self._zero_master_grads()
        self._master_params_to_model_params()
        self.lg_loss_scale += self.fp16_scale_growth
        return True

    def _zero_master_grads(self):
        for param in self.master_params:
            param.grad = None

    def _master_params_to_model_params(self):
        """
        マスターパラメータのデータをモデルパラメータにコピーする。
        """
        # Without copying to a list, if a generator is passed, this will
        # silently not copy any parameters.
        _iter = zip(self.master_params, self.param_groups_and_shapes)
        for master_param, (param_group, _) in _iter:
            _params = self._unflatten_master_params(param_group, master_param.view(-1))
            for (_, param), unflat_master_param in zip(param_group, _params):
                param.detach().copy_(unflat_master_param)

    def _model_grads_to_master_grads(self):
        """
        モデルパラメータの勾配をマスターパラメータにコピーする。
        """
        _iter = zip(self.master_params, self.param_groups_and_shapes)
        for master_param, (param_group, shape) in _iter:
            master_param.grad = _flatten_dense_tensors(
                [self._param_grad_or_zeros(param) for (_, param) in param_group]
            ).view(shape)

    def _compute_norms(self, grad_scale=1.0):
        """
        勾配とパラメータのL2ノルムを計算する。
        """
        grad_norm = 0.0
        param_norm = 0.0
        for p in self.master_params:
            with torch.no_grad():
                param_norm += torch.norm(p, p=2, dtype=torch.float32).item() ** 2
                if p.grad is not None:
                    grad_norm += (
                        torch.norm(p.grad, p=2, dtype=torch.float32).item() ** 2
                    )
        return np.sqrt(grad_norm) / grad_scale, np.sqrt(param_norm)

    def _param_grad_or_zeros(self, param):
        """
        パラメータの勾配が存在しない場合はゼロを返す。
        """
        if param.grad is not None:
            return param.grad.data.detach()
        else:
            return torch.zeros_like(param)

    def master_params_to_state_dict(self, master_params):
        """
        マスターパラメータをモデルのstate_dictに変換する。
        """
        state_dict = self.model.state_dict()
        _iter = zip(master_params, self.param_groups_and_shapes)
        for master_param, (param_group, _) in _iter:
            _params = self._unflatten_master_params(param_group, master_param.view(-1))
            for (name, _), unflat_master_param in zip(param_group, _params):
                assert name in state_dict
                state_dict[name] = unflat_master_param
        return state_dict

    def state_dict_to_master_params(self, state_dict):
        """
        state_dictからマスターパラメータを生成する。
        """
        named_model_params = [
            (name, state_dict[name]) for name, _ in self.model.named_parameters()
        ]
        param_groups_and_shapes = self._get_param_groups_and_shapes(named_model_params)
        master_params = self._make_master_params(param_groups_and_shapes)
        return master_params

    def _get_param_groups_and_shapes(self, named_model_params):
        """
        パラメータをグループ化し、その形状を取得する。
        """
        named_model_params = list(named_model_params)
        scalar_vector_named_params = (
            [(n, p) for (n, p) in named_model_params if p.ndim <= 1],
            (-1),
        )
        matrix_named_params = (
            [(n, p) for (n, p) in named_model_params if p.ndim > 1],
            (1, -1),
        )
        return [scalar_vector_named_params, matrix_named_params]

    def _make_master_params(self, param_groups_and_shapes):
        """
        モデルパラメータをコピーして、異なる形状のfloat32パラメータリストを作成する。
        """
        master_params = []
        for param_group, shape in param_groups_and_shapes:
            master_param = nn.Parameter(
                _flatten_dense_tensors(
                    [param.detach().float() for (_, param) in param_group]
                ).view(shape)
            )
            master_param.requires_grad = True
            master_params.append(master_param)
        return master_params

    def _unflatten_master_params(self, param_group, master_param):
        """
        マスターパラメータを元の形状に戻す。
        """
        return _unflatten_dense_tensors(
            master_param, [param for (_, param) in param_group]
        )

    def _check_overflow(self, value):
        return (value == float("inf")) or (value == -float("inf")) or (value != value)
