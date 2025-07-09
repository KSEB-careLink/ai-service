# enums.py
from enum import Enum

class ToneEnum(str, Enum):
    kind = "다정하게"
    calm = "차분하게"
    bright = "밝게"
