import torch
import unittest

from vits.training.commons import *

class TestUtils(unittest.TestCase):

    def test_init_weights(self):
        m = torch.nn.Conv1d(16, 32, 3)
        init_weights(m)
        self.assertIsNotNone(m.weight.data)

    def test_get_padding(self):
        self.assertEqual(get_padding(3), 1)

    def test_convert_pad_shape(self):
        self.assertEqual(convert_pad_shape([[1,2],[3,4]]), [2,1,4,3])

    def test_intersperse(self):
        self.assertEqual(intersperse([1,2], 0), [0,1,0,2,0])

    def test_kl_divergence(self):
        m_p = torch.tensor([0.5])
        logs_p = torch.tensor([0.2])
        m_q = torch.tensor([0.3])
        logs_q = torch.tensor([0.4])
        kl = kl_divergence(m_p, logs_p, m_q, logs_q)
        self.assertEqual(kl.shape, m_p.shape)

    def test_rand_gumbel(self):
        g = rand_gumbel((10,))
        self.assertEqual(g.shape, (10,))

    def test_rand_gumbel_like(self):
        x = torch.tensor([0.5, 0.3, 0.7])
        g = rand_gumbel_like(x)
        self.assertEqual(g.shape, x.shape)

    def test_add_timing_signal_1d(self):
        x = torch.zeros(1, 8, 10)
        y = add_timing_signal_1d(x)
        self.assertEqual(y.shape, x.shape)

    def test_cat_timing_signal_1d(self):
        x = torch.zeros(1, 8, 10)
        y = cat_timing_signal_1d(x)
        self.assertEqual(y.shape, (1, 16, 10))

    def test_subsequent_mask(self):
        m = subsequent_mask(5)
        expected = torch.tensor([[[[1,0,0,0,0], [1,1,0,0,0], [1,1,1,0,0], [1,1,1,1,0], [1,1,1,1,1]]]])
        self.assertTrue(torch.allclose(m, expected))

    def test_fused_add_tanh_sigmoid_multiply(self):
        input_a = torch.tensor([[[1.2, 2.1, 3.5], [0.4, 1.1, 0.2]]])
        input_b = torch.tensor([[[0.3, 1.7, 2.2], [0.1, 0.6, 0.8]]])
        n_channels = torch.tensor([2])
        fused_out = fused_add_tanh_sigmoid_multiply(input_a, input_b, n_channels)
        self.assertEqual(fused_out.shape, input_a.shape)

    def test_shift_1d(self):
        x = torch.tensor([[[1, 2, 3, 4]]])
        shifted = shift_1d(x)
        expected = torch.tensor([[[0, 1, 2, 3]]])
        self.assertTrue(torch.allclose(shifted, expected))

    def test_sequence_mask(self):
        lengths = torch.tensor([1, 3])
        mask = sequence_mask(lengths, 4)
        expected = torch.tensor([[True, False, False, False],
                                 [True, True, True, False]])
        self.assertTrue(torch.all(mask == expected))

    # Additional tests for other functions can be added here

    def setUp(self):
        self.input_tensor = torch.tensor([0.5, 0.3, 0.7])
        # Any other common initialization can go here.

    def test_rand_gumbel_output_range(self):
        SOME_MIN_VALUE = -5.5
        SOME_MAX_VALUE = 6.2

        ## WARNING: this only captures around 99% of the data, so there is a small chance the test could fail even if everything is 
        # functioning correctly. Adjust the range or the confidence level based on your tolerance for type I errors in the test.

        for _ in range(100):  # randomized testing
            g = rand_gumbel(self.input_tensor.shape)
            # assuming the gumbel distribution outputs should fall within some range
            self.assertTrue((g > SOME_MIN_VALUE) & (g < SOME_MAX_VALUE).all())

    def test_rand_gumbel_like_shape_match(self):
        g = rand_gumbel_like(self.input_tensor)
        self.assertEqual(g.shape, self.input_tensor.shape)

    def test_kl_divergence_value_range(self):
        m_p = torch.tensor([0.5])
        logs_p = torch.tensor([0.2])
        m_q = torch.tensor([0.3])
        logs_q = torch.tensor([0.4])
        kl = kl_divergence(m_p, logs_p, m_q, logs_q)
        # check value properties, not just shapes
        self.assertTrue((kl > 0).all())


if __name__ == '__main__':
    unittest.main()
