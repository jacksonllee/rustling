from rustling._lib_name import conllu as _conllu
from rustling.conllu._read_conllu import read_conllu

CoNLLU = _conllu.CoNLLU
Sentence = _conllu.Sentence
Token = _conllu.Token

__all__ = [
    "CoNLLU",
    "Sentence",
    "Token",
    "read_conllu",
]
