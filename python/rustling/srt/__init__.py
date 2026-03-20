from rustling._lib_name import srt as _srt
from rustling.srt._read_srt import read_srt

SRT = _srt.SRT
Utterance = _srt.Utterance

__all__ = [
    "SRT",
    "Utterance",
    "read_srt",
]
