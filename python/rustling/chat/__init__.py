"""CHAT parsing.

This module provides a parser for CHAT transcription files
(CHILDES/TalkBank format) and data structures for accessing
utterances, tokens, and annotations.
"""

from rustling._lib_name import chat as _chat
from rustling.chat._read_chat import read_chat

Age = _chat.Age
CHAT = _chat.CHAT
ChangeableHeader = _chat.ChangeableHeader
Gra = _chat.Gra
Headers = _chat.Headers
Participant = _chat.Participant
Token = _chat.Token
Utterance = _chat.Utterance
Utterances = _chat.Utterances

__all__ = [
    "Age",
    "CHAT",
    "ChangeableHeader",
    "Gra",
    "Headers",
    "Participant",
    "Token",
    "Utterance",
    "Utterances",
    "read_chat",
]
