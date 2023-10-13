import unittest
import torch

from ..training.losses import *

class TestLossFunctions(unittest.TestCase):
    
    def setUp(self):
        # Random inputs for testing
        self.batch_size = 3
        self.features = 5
        self.length = 4
        
        self.fmap_r = [torch.randn(self.batch_size, self.features, self.length) for _ in range(2)]
        self.fmap_g = [torch.randn(self.batch_size, self.features, self.length) for _ in range(2)]
        
        self.disc_real_outputs = [torch.randn(self.batch_size, 1) for _ in range(2)]
        self.disc_generated_outputs = [torch.randn(self.batch_size, 1) for _ in range(2)]
        
        self.disc_outputs = [torch.randn(self.batch_size, 1) for _ in range(2)]
        
        self.z_p = torch.randn(self.batch_size, self.features, self.length)
        self.logs_q = torch.randn(self.batch_size, self.features, self.length)
        self.m_p = torch.randn(self.batch_size, self.features, self.length)
        self.logs_p = torch.randn(self.batch_size, self.features, self.length)
        self.z_mask = torch.ones(self.batch_size, self.features, self.length)
    
    def test_feature_loss_basic(self):
        loss = feature_loss(self.fmap_r, self.fmap_g)
        self.assertIsInstance(loss, torch.Tensor)

    def test_feature_loss_identity(self):
        loss = feature_loss(self.fmap_r, self.fmap_r)
        self.assertAlmostEqual(loss.item(), 0, places=4)
    
    def test_discriminator_loss_basic(self):
        loss, r_losses, g_losses = discriminator_loss(self.disc_real_outputs, self.disc_generated_outputs)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(len(r_losses), 2)
        self.assertEqual(len(g_losses), 2)

    def test_discriminator_loss_ideal(self):
        ideal_real_outputs = [torch.ones_like(out) for out in self.disc_real_outputs]
        ideal_gen_outputs = [torch.zeros_like(out) for out in self.disc_generated_outputs]
        loss, _, _ = discriminator_loss(ideal_real_outputs, ideal_gen_outputs)
        self.assertAlmostEqual(loss.item(), 0, places=4)

    def test_generator_loss_basic(self):
        loss, gen_losses = generator_loss(self.disc_outputs)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(len(gen_losses), 2)
        
    def test_generator_loss_ideal(self):
        ideal_gen_outputs = [torch.ones_like(out) for out in self.disc_outputs]
        loss, _ = generator_loss(ideal_gen_outputs)
        self.assertAlmostEqual(loss.item(), 0, places=4)

    def test_kl_loss_basic(self):
        loss = kl_loss(self.z_p, self.logs_q, self.m_p, self.logs_p, self.z_mask)
        self.assertIsInstance(loss, torch.Tensor)

    def test_kl_loss_identity(self):
        loss = kl_loss(self.z_p, self.logs_q, self.z_p, self.logs_q, self.z_mask)
        self.assertAlmostEqual(loss.item(), 0, places=4)
    
if __name__ == '__main__':
    unittest.main()
