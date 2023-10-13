import torch
import unittest

from ..training.modules import *

class TestModelComponents(unittest.TestCase):

    def setUp(self):
        # Some common tensors for testing
        self.x = torch.randn(16, 256, 100)
        self.x_mask = torch.ones(16, 256, 100)
        self.g = torch.randn(16, 10, 100)

    def test_layer_norm(self):
        model = LayerNorm(256)
        out = model(self.x)
        self.assertEqual(out.shape, self.x.shape)

    def test_conv_relu_norm(self):
        model = ConvReluNorm(256, 512, 256, 3, 3, 0.1)
        out = model(self.x, self.x_mask)
        self.assertEqual(out.shape, self.x.shape)

    def test_dds_conv(self):
        model = DDSConv(256, 3, 3)
        out = model(self.x, self.x_mask, self.g)
        self.assertEqual(out.shape, self.x.shape)

    def test_wn(self):
        model = WN(256, 3, 2, 3)
        out = model(self.x, self.x_mask)
        self.assertEqual(out.shape, self.x.shape)

    def test_log(self):
        layer = Log()
        out, logdet = layer(self.x, self.x_mask)
        self.assertEqual(out.shape, self.x.shape)

    def test_flip(self):
        layer = Flip()
        out, logdet = layer(self.x, self.x_mask)
        self.assertEqual(out.shape, self.x.shape)

    def test_elementwise_affine(self):
        model = ElementwiseAffine(256)
        out, logdet = model(self.x, self.x_mask)
        self.assertEqual(out.shape, self.x.shape)

    def test_residual_coupling_layer(self):
        model = ResidualCouplingLayer(256, 512, 3, 2, 3)
        out, logdet = model(self.x, self.x_mask)
        self.assertEqual(out.shape, self.x.shape)

    def test_conv_flow(self):
        model = ConvFlow(256, 512, 3, 3)
        out, logdet = model(self.x, self.x_mask)
        self.assertEqual(out.shape, self.x.shape)

    def test_res_block_1(self):
        model = ResBlock1(256)
        out = model(self.x, self.x_mask)
        self.assertEqual(out.shape, self.x.shape)

    def test_res_block_2(self):
        model = ResBlock2(256)
        out = model(self.x, self.x_mask)
        self.assertEqual(out.shape, self.x.shape)

class TestReverseComponents(unittest.TestCase):

    def test_layer_norm(self):
        model = LayerNorm(256)
        x = torch.randn(16, 256, 100)
        out = model(x)
        self.assertEqual(out.shape, x.shape)

    def test_conv_relu_norm(self):
        model = ConvReluNorm(256, 512, 256, 3, 3, 0.1)
        x = torch.randn(16, 256, 100)
        mask = torch.ones(16, 256, 100)
        out = model(x, mask)
        self.assertEqual(out.shape, x.shape)

    def test_dds_conv(self):
        model = DDSConv(256, 3, 3)
        x = torch.randn(16, 256, 100)
        mask = torch.ones(16, 256, 100)
        out = model(x, mask)
        self.assertEqual(out.shape, x.shape)

    def test_wn(self):
        model = WN(256, 3, 3, 3)
        x = torch.randn(16, 256, 100)
        mask = torch.ones(16, 256, 100)
        out, _ = model(x, mask)
        self.assertEqual(out.shape, x.shape)

    def test_res_block1(self):
        model = ResBlock1(256)
        x = torch.randn(16, 256, 100)
        mask = torch.ones(16, 256, 100)
        out = model(x, mask)
        self.assertEqual(out.shape, x.shape)

    def test_res_block2(self):
        model = ResBlock2(256)
        x = torch.randn(16, 256, 100)
        mask = torch.ones(16, 256, 100)
        out = model(x, mask)
        self.assertEqual(out.shape, x.shape)

    def test_log(self):
        model = Log()
        x = torch.abs(torch.randn(16, 256, 100))
        mask = torch.ones(16, 256, 100)
        out, _ = model(x, mask)
        rev_out = model(out, mask, reverse=True)
        self.assertTrue(torch.allclose(x, rev_out, atol=1e-4))

    def test_flip(self):
        model = Flip()
        x = torch.randn(16, 256, 100)
        out, _ = model(x)
        rev_out = model(out, reverse=True)
        self.assertTrue(torch.allclose(x, rev_out, atol=1e-4))

    def test_elementwise_affine(self):
        model = ElementwiseAffine(256)
        x = torch.randn(16, 256, 100)
        mask = torch.ones(16, 256, 100)
        out, _ = model(x, mask)
        rev_out = model(out, mask, reverse=True)
        self.assertTrue(torch.allclose(x, rev_out, atol=1e-4))

    def test_residual_coupling_layer(self):
        model = ResidualCouplingLayer(256, 256, 3, 3, 3)
        x = torch.randn(16, 256, 100)
        mask = torch.ones(16, 256, 100)
        out, _ = model(x, mask)
        rev_out = model(out, mask, reverse=True)
        self.assertTrue(torch.allclose(x, rev_out, atol=1e-4))

    def test_conv_flow(self):
        model = ConvFlow(256, 256, 3, 3)
        x = torch.randn(16, 256, 100)
        mask = torch.ones(16, 256, 100)
        out, _ = model(x, mask)
        rev_out = model(out, mask, reverse=True)
        self.assertTrue(torch.allclose(x, rev_out, atol=1e-4))

class TestModelComponentsEdgeCases(unittest.TestCase):
    
    def setUp(self):
        self.small_tensor = torch.randn(2, 32, 10)
        self.large_tensor = torch.randn(32, 512, 500)
        self.mask = torch.ones_like(self.small_tensor)

    def test_layer_norm_small_tensor(self):
        model = LayerNorm(32)
        out = model(self.small_tensor)
        self.assertEqual(out.shape, self.small_tensor.shape)

    def test_layer_norm_large_tensor(self):
        model = LayerNorm(512)
        out = model(self.large_tensor)
        self.assertEqual(out.shape, self.large_tensor.shape)

    def test_conv_relu_norm_small_tensor(self):
        model = ConvReluNorm(32, 64, 32, 3, 3, 0.1)
        out = model(self.small_tensor, self.mask)
        self.assertEqual(out.shape, self.small_tensor.shape)

    def test_conv_relu_norm_large_tensor(self):
        model = ConvReluNorm(512, 1024, 512, 3, 3, 0.1)
        out = model(self.large_tensor, torch.ones_like(self.large_tensor))
        self.assertEqual(out.shape, self.large_tensor.shape)

    def test_dds_conv_small_tensor(self):
        model = DDSConv(32, 3, 3)
        out = model(self.small_tensor, self.mask)
        self.assertEqual(out.shape, self.small_tensor.shape)

    def test_dds_conv_large_tensor(self):
        model = DDSConv(512, 3, 3)
        out = model(self.large_tensor, torch.ones_like(self.large_tensor))
        self.assertEqual(out.shape, self.large_tensor.shape)
    
    def test_wn_small_tensor(self):
        model = WN(32, 3, 2, 3)
        out, _ = model(self.small_tensor, self.mask)
        self.assertEqual(out.shape, self.small_tensor.shape)
        
    def test_wn_large_tensor(self):
        model = WN(512, 3, 2, 3)
        out, _ = model(self.large_tensor, torch.ones_like(self.large_tensor))
        self.assertEqual(out.shape, self.large_tensor.shape)

    def test_res_block1_small_tensor(self):
        model = ResBlock1(32)
        out = model(self.small_tensor, self.mask)
        self.assertEqual(out.shape, self.small_tensor.shape)
        
    def test_res_block1_large_tensor(self):
        model = ResBlock1(512)
        out = model(self.large_tensor, torch.ones_like(self.large_tensor))
        self.assertEqual(out.shape, self.large_tensor.shape)
    
    def test_res_block2_small_tensor(self):
        model = ResBlock2(32)
        out = model(self.small_tensor, self.mask)
        self.assertEqual(out.shape, self.small_tensor.shape)
        
    def test_res_block2_large_tensor(self):
        model = ResBlock2(512)
        out = model(self.large_tensor, torch.ones_like(self.large_tensor))
        self.assertEqual(out.shape, self.large_tensor.shape)

    # Tests for extreme values in tensors.
    def test_extreme_values(self):
        extreme_tensor = torch.tensor([1e6, -1e6, 1e-6, -1e-6])
        model = LayerNorm(4)
        out = model(extreme_tensor.unsqueeze(0).unsqueeze(0))
        self.assertEqual(out.shape, extreme_tensor.unsqueeze(0).unsqueeze(0).shape)

    def test_log_small_tensor(self):
        model = Log()
        out, _ = model(self.small_tensor, self.mask)
        self.assertEqual(out.shape, self.small_tensor.shape)

    def test_log_large_tensor(self):
        model = Log()
        out, _ = model(self.large_tensor, torch.ones_like(self.large_tensor))
        self.assertEqual(out.shape, self.large_tensor.shape)

    def test_flip_small_tensor(self):
        model = Flip()
        out, _ = model(self.small_tensor)
        self.assertEqual(out.shape, self.small_tensor.shape)
        self.assertTrue(torch.equal(out, torch.flip(self.small_tensor, [1])))

    def test_flip_large_tensor(self):
        model = Flip()
        out, _ = model(self.large_tensor)
        self.assertEqual(out.shape, self.large_tensor.shape)
        self.assertTrue(torch.equal(out, torch.flip(self.large_tensor, [1])))

    def test_elementwise_affine_small_tensor(self):
        model = ElementwiseAffine(32)
        out, _ = model(self.small_tensor, self.mask)
        self.assertEqual(out.shape, self.small_tensor.shape)

    def test_elementwise_affine_large_tensor(self):
        model = ElementwiseAffine(512)
        out, _ = model(self.large_tensor, torch.ones_like(self.large_tensor))
        self.assertEqual(out.shape, self.large_tensor.shape)

    def test_residual_coupling_small_tensor(self):
        model = ResidualCouplingLayer(32, 64, 3, 2, 3)
        out, _ = model(self.small_tensor, self.mask)
        self.assertEqual(out.shape, self.small_tensor.shape)

    def test_residual_coupling_large_tensor(self):
        model = ResidualCouplingLayer(512, 1024, 3, 2, 3)
        out, _ = model(self.large_tensor, torch.ones_like(self.large_tensor))
        self.assertEqual(out.shape, self.large_tensor.shape)

    def test_conv_flow_small_tensor(self):
        model = ConvFlow(32, 64, 3, 3)
        out, _ = model(self.small_tensor, self.mask)
        self.assertEqual(out.shape, self.small_tensor.shape)

    def test_conv_flow_large_tensor(self):
        model = ConvFlow(512, 1024, 3, 3)
        out, _ = model(self.large_tensor, torch.ones_like(self.large_tensor))
        self.assertEqual(out.shape, self.large_tensor.shape)

    # Additional test for extreme values
    def test_residual_coupling_extreme_values(self):
        extreme_tensor = torch.tensor([1e6, -1e6, 1e-6, -1e-6])
        model = ResidualCouplingLayer(4, 8, 3, 2, 3)
        out, _ = model(extreme_tensor.unsqueeze(0).unsqueeze(0), torch.ones(1, 1, 4))
        self.assertEqual(out.shape, extreme_tensor.unsqueeze(0).unsqueeze(0).shape)

if __name__ == '__main__':
    unittest.main()