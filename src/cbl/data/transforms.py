import numpy as np
import numpy.typing as npt
from scipy.interpolate import interp1d

TARGET_WN = np.linspace(4000, 400, 1800) # MSS dataset has vector length 1800

_float_T = np.float32 | np.float64
_arr_T = npt.NDArray[_float_T]

class Interpolate():
    """Interpolate IR spectrum to target X-axis. (Originally written by Eva)"""

    kind: str
    target_x: _arr_T

    def __init__(
        self,
        target_range: tuple[int, int] = (4000, 400),
        target_len: int = 1800,
        kind: str = "linear"
    ):
        self.kind = kind
        self.target_x = np.linspace(target_range[0], target_range[1], target_len)

    def __call__(self, wavenumbers: _arr_T, transmittance: _arr_T) -> _arr_T:
        if len(transmittance) != len(self.target_x):
            order = np.argsort(wavenumbers)
            wavenumbers, transmittance = wavenumbers[order], transmittance[order]
            interp = interp1d(wavenumbers, transmittance, kind=self.kind, bounds_error=False, fill_value=(transmittance[0], transmittance[-1]))
            res = interp(self.target_x)
        
        # normalise every spectrum to [0, 1] range using min-max normalisation
        y_min, y_max = transmittance.min(), transmittance.max()
        if y_max == y_min: 
            return np.zeros_like(transmittance)
        return (res - y_min) / (y_max - y_min)
