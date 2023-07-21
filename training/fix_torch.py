
from torch import functional 

with open("functional_fixed.py", "r") as tfx:
    file_to_write = tfx.readlines()

with open(functional.__file__, "w") as tfx:
    tfx.writelines(file_to_write)
###
#if not return_complex:
#    return torch.view_as_real(_VF.stft(input, n_fft, hop_length, win_length, window,  # type: ignore[attr-defined]
#                                    normalized, onesided, return_complex=True))
###

