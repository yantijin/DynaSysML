import pytest
import DynaSysML as dsl
import torch


class TestGated(object):

    def test_Gated(self):
        gated = dsl.Layers.Gated(feature_axis=-2, num_features=3,
                                gate_bias=1.5)
        assert (
            'feature_axis=-2, num_features=3, gate_bias=1.5' in
            repr(gated)
        )
        # gated = T.jit_compile(gated)

        x = torch.randn([6, 5])
        assert (gated(x) - x[:3, ...] * torch.sigmoid(x[3:, ...] + 1.5) < 1e-6).all()

        x = torch.randn([3, 6, 5])
        assert (gated(x)- x[:, :3, ...] * torch.sigmoid(x[:, 3:, ...] + 1.5) < 1e-6).all()

        with pytest.raises(Exception,
                           match='shape of the pre-gated output is invalid'):
            _ = gated(torch.randn([7, 3]))

    def test_GatedWithActivation(self):
        gated = dsl.Layers.GatedWithActivation(
            feature_axis=-2, num_features=3, gate_bias=1.5,
            activation=dsl.Layers.LeakyReLU(),
        )
        assert (
            'feature_axis=-2, num_features=3, gate_bias=1.5' in
            repr(gated)
        )
        # gated = T.jit_compile(gated)

        x = torch.randn([6, 5])
        assert(
            gated(x) -torch.nn.functional.leaky_relu(x[:3, ...]) * torch.sigmoid(x[3:, ...] + 1.5)<1e-6
        ).all()

        x = torch.randn([3, 6, 5])
        assert(
            gated(x) - torch.nn.functional.leaky_relu(x[:, :3, ...]) * torch.sigmoid(x[:, 3:, ...] + 1.5)<1e-6
        ).all()

        with pytest.raises(Exception,
                           match='shape of the pre-gated output is invalid'):
            _ = gated(torch.randn([7, 3]))