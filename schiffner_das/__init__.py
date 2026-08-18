"""
schiffner_das — port a PyTorch de la conformacion con F-number dependiente de
frecuencia de M. F. Schiffner, para compresion coherente de ondas planas.

En este repositorio cumple dos funciones, ambas necesarias:

  das_pw                      genera los targets de entrenamiento
                              (compresion de 75 ondas planas)
  GratingAngleLB, Tukey,      los usa en tiempo de ejecucion el operador
  NormalizationOn             adjunto que alimenta a UNet-AG

Se han eliminado los modulos de carga de datos y de visualizacion del port
original: este repositorio usa sus propios cargadores (unet_ag/data/) y no
dibuja nada.
"""

from .f_numbers import ConstantFNumber, FNumber, GratingAngleLB
from .normalizations import Normalization, NormalizationOff, NormalizationOn
from .windows import Boxcar, Hann, Triangular, Tukey, Window
from .beamforming import DASResult, das_pw

__version__ = "0.1.0"
