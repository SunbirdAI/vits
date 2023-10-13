import torch
import torch.nn.functional as F
import unittest

from ..training.transforms import *

class TestTransforms(unittest.TestCase):

    def test_piecewise_rational_quadratic_transform():
        inputs = torch.tensor([0.5])
        unnormalized_widths = torch.tensor([0.3])
        unnormalized_heights = torch.tensor([0.2])
        unnormalized_derivatives = torch.tensor([0.1])
        
        # Test when tails is None
        outputs, logabsdet = piecewise_rational_quadratic_transform(inputs, unnormalized_widths, unnormalized_heights, unnormalized_derivatives)
        assert torch.isfinite(outputs)
        assert torch.isfinite(logabsdet)
        
        # Test when tails is not None
        outputs, logabsdet = piecewise_rational_quadratic_transform(inputs, unnormalized_widths, unnormalized_heights, unnormalized_derivatives, tails='linear')
        assert torch.isfinite(outputs)
        assert torch.isfinite(logabsdet)

    def test_searchsorted():
        inputs = torch.tensor([0.5, 1.5, 2.5])
        bin_locations = torch.tensor([0., 1., 2., 3.])
        
        # Test with increasing bin_locations
        result = searchsorted(bin_locations, inputs)
        assert torch.all(result == torch.tensor([1, 2, 3]))
        
        # Test with duplicate bin_locations
        bin_locations_dup = torch.tensor([0., 1., 1., 2., 3.])
        result = searchsorted(bin_locations_dup, inputs)
        assert torch.all(result == torch.tensor([1, 3, 4]))

    def test_unconstrained_rational_quadratic_spline():
        inputs_outside = torch.tensor([-2., 2.])
        inputs_inside = torch.tensor([-0.5, 0.5])
        unnormalized_widths = torch.tensor([0.3])
        unnormalized_heights = torch.tensor([0.2])
        unnormalized_derivatives = torch.tensor([0.1])
        
        # Test with inputs outside tail_bound
        outputs, logabsdet = unconstrained_rational_quadratic_spline(inputs_outside, unnormalized_widths, unnormalized_heights, unnormalized_derivatives)
        assert torch.all(inputs_outside == outputs)
        assert torch.all(logabsdet == 0.)
        
        # Test with inputs inside tail_bound
        outputs, logabsdet = unconstrained_rational_quadratic_spline(inputs_inside, unnormalized_widths, unnormalized_heights, unnormalized_derivatives)
        assert torch.isfinite(outputs).all()
        assert torch.isfinite(logabsdet).all()

    def test_rational_quadratic_spline():
        inputs = torch.tensor([0.5])
        unnormalized_widths = torch.tensor([0.6])  # this will lead to a sum of widths > 1.0
        unnormalized_heights = torch.tensor([0.2])
        unnormalized_derivatives = torch.tensor([0.1])
        
        # Test when input is outside of domain
        try:
            outputs, logabsdet = rational_quadratic_spline(torch.tensor([-2.]), unnormalized_widths, unnormalized_heights, unnormalized_derivatives)
            assert False, "Expected ValueError"
        except ValueError:
            pass

        # Test when min_bin_width * num_bins > 1.0
        try:
            outputs, logabsdet = rational_quadratic_spline(inputs, unnormalized_widths, unnormalized_heights, unnormalized_derivatives)
            assert False, "Expected ValueError"
        except ValueError:
            pass


if __name__ == '__main__':
    unittest.main()



