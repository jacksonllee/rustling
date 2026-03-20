"""Tests for rustling.chat.CHAT."""

import datetime
import warnings

import pytest

from rustling.chat import (
    CHAT,
    Age,
    ChangeableHeader,
    Gra,
    Headers,
    Participant,
    Token,
    Utterance,
)

BASIC_CHAT = (
    "@UTF8\n"
    "@Begin\n"
    "@Participants:\tCHI Child, MOT Mother\n"
    "*CHI:\tI want cookie .\n"
    "%mor:\tpro|I v|want n|cookie .\n"
    "%gra:\t1|2|SUBJ 2|0|ROOT 3|2|OBJ 4|2|PUNCT\n"
    "*MOT:\tno .\n"
    "%mor:\tco|no .\n"
    "%gra:\t1|0|ROOT 2|1|PUNCT\n"
    "@End\n"
)

TWO_FILES = [
    "@UTF8\n@Begin\n*CHI:\thi .\n@End\n",
    "@UTF8\n@Begin\n*MOT:\tbye .\n@End\n",
]


class TestCHATFromStrs:
    def test_basic_parsing(self):
        reader = CHAT.from_strs([BASIC_CHAT])
        assert reader.n_files == 1

    def test_empty_strs(self):
        reader = CHAT.from_strs([])
        assert reader.n_files == 0

    def test_ids_provided(self):
        reader = CHAT.from_strs([BASIC_CHAT], ids=["my_file"])
        assert reader.file_paths == ["my_file"]

    def test_ids_auto_generated(self):
        reader = CHAT.from_strs([BASIC_CHAT])
        paths = reader.file_paths
        assert len(paths) == 1
        # UUID format: 8-4-4-4-12 hex chars
        assert len(paths[0]) == 36

    def test_ids_length_mismatch(self):
        with pytest.raises(ValueError):
            CHAT.from_strs(["content1", "content2"], ids=["only_one"])

    def test_multiple_files(self):
        reader = CHAT.from_strs(TWO_FILES, ids=["a", "b"])
        assert reader.n_files == 2
        assert reader.file_paths == ["a", "b"]


class TestFromUtterances:
    def test_round_trip(self):
        reader = CHAT.from_strs([BASIC_CHAT])
        utts = reader.utterances()
        new_reader = CHAT.from_utterances(utts)
        assert new_reader.n_files == 1
        new_utts = new_reader.utterances()
        assert len(new_utts) == len(utts)
        for orig, rebuilt in zip(utts, new_utts):
            assert orig == rebuilt

    def test_subset(self):
        reader = CHAT.from_strs([BASIC_CHAT])
        utts = reader.utterances()
        new_reader = CHAT.from_utterances(utts[:1])
        new_utts = new_reader.utterances()
        assert len(new_utts) == 1
        assert new_utts[0].participant == "CHI"

    def test_empty(self):
        new_reader = CHAT.from_utterances([])
        assert new_reader.n_files == 1
        assert len(new_reader.utterances()) == 0

    def test_words(self):
        reader = CHAT.from_strs([BASIC_CHAT])
        utts = reader.utterances()
        new_reader = CHAT.from_utterances(utts)
        assert new_reader.words() == reader.words()

    def test_to_strs(self):
        reader = CHAT.from_strs([BASIC_CHAT])
        utts = reader.utterances()
        new_reader = CHAT.from_utterances(utts)
        strs = new_reader.to_strs()
        assert len(strs) == 1
        assert "*CHI:" in strs[0]
        assert "%mor:" in strs[0]
        assert "@End" in strs[0]

    def test_serialization_round_trip(self):
        reader = CHAT.from_strs([BASIC_CHAT])
        utts = reader.utterances()
        new_reader = CHAT.from_utterances(utts)
        strs = new_reader.to_strs()
        reparsed = CHAT.from_strs(strs)
        reparsed_utts = reparsed.utterances()
        assert len(reparsed_utts) == len(utts)
        for orig, reparsed_utt in zip(utts, reparsed_utts):
            assert orig.participant == reparsed_utt.participant
            assert orig.tokens == reparsed_utt.tokens


class TestUtterances:
    def test_utterances_flat(self):
        reader = CHAT.from_strs([BASIC_CHAT])
        utts = reader.utterances()
        assert len(utts) == 2
        assert utts[0].participant == "CHI"
        assert utts[1].participant == "MOT"

    def test_utterances_by_file(self):
        reader = CHAT.from_strs([BASIC_CHAT])
        utts = reader.utterances(by_file=True)
        assert len(utts) == 1  # one file
        assert len(utts[0]) == 2  # two utterances

    def test_utterances_multiple_files_by_file(self):
        reader = CHAT.from_strs(TWO_FILES, ids=["a", "b"])
        utts = reader.utterances(by_file=True)
        assert len(utts) == 2
        assert len(utts[0]) == 1
        assert len(utts[1]) == 1

    def test_utterance_type(self):
        reader = CHAT.from_strs([BASIC_CHAT])
        utts = reader.utterances()
        assert isinstance(utts[0], Utterance)

    def test_utterance_tokens(self):
        reader = CHAT.from_strs([BASIC_CHAT])
        utts = reader.utterances()
        tokens = utts[0].tokens
        assert tokens is not None
        assert len(tokens) == 4  # I, want, cookie, .
        assert tokens[0].word == "I"
        assert tokens[1].word == "want"
        assert tokens[2].word == "cookie"

    def test_utterance_tiers(self):
        reader = CHAT.from_strs([BASIC_CHAT])
        utts = reader.utterances()
        tiers = utts[0].tiers
        assert tiers is not None
        assert "CHI" in tiers
        assert "%mor" in tiers
        assert "%gra" in tiers

    def test_utterance_audible(self):
        reader = CHAT.from_strs([BASIC_CHAT])
        utts = reader.utterances()
        assert utts[0].audible == "I want cookie ."

    def test_utterance_audible_constructed(self):
        utt = Utterance(tokens=[Token("hello"), Token("world")])
        assert utt.audible == "hello world"

    def test_utterance_audible_none_tokens(self):
        utt = Utterance()
        assert utt.audible is None


class TestTokens:
    def test_token_pos(self):
        reader = CHAT.from_strs([BASIC_CHAT])
        utts = reader.utterances()
        tokens = utts[0].tokens
        assert tokens is not None
        assert tokens[0].pos == "pro"
        assert tokens[1].pos == "v"
        assert tokens[2].pos == "n"

    def test_token_mor(self):
        reader = CHAT.from_strs([BASIC_CHAT])
        utts = reader.utterances()
        tokens = utts[0].tokens
        assert tokens is not None
        assert tokens[0].mor == "I"
        assert tokens[1].mor == "want"
        assert tokens[2].mor == "cookie"

    def test_token_gra(self):
        reader = CHAT.from_strs([BASIC_CHAT])
        utts = reader.utterances()
        tokens = utts[0].tokens
        assert tokens is not None
        gra = tokens[0].gra
        assert isinstance(gra, Gra)
        assert gra.dep == 1
        assert gra.head == 2
        assert gra.rel == "SUBJ"

    def test_no_mor_tier(self):
        chat_str = "@UTF8\n@Begin\n*CHI:\thello world .\n@End\n"
        reader = CHAT.from_strs([chat_str])
        utts = reader.utterances()
        tokens = utts[0].tokens
        assert tokens is not None
        assert tokens[0].pos is None
        assert tokens[0].mor is None
        assert tokens[0].gra is None


class TestWords:
    def test_words_flat(self):
        reader = CHAT.from_strs([BASIC_CHAT])
        words = reader.words()
        assert "I" in words
        assert "want" in words
        assert "cookie" in words
        assert "no" in words

    def test_words_by_utterance(self):
        reader = CHAT.from_strs([BASIC_CHAT])
        words = reader.words(by_utterance=True)
        assert isinstance(words, list)
        assert isinstance(words[0], list)
        assert words[0] == ["I", "want", "cookie", "."]

    def test_words_by_file(self):
        reader = CHAT.from_strs(TWO_FILES, ids=["a", "b"])
        words = reader.words(by_file=True)
        assert len(words) == 2  # two files

    def test_words_by_utterance_and_files(self):
        reader = CHAT.from_strs(TWO_FILES, ids=["a", "b"])
        words = reader.words(by_utterance=True, by_file=True)
        assert len(words) == 2  # two files
        assert isinstance(words[0], list)
        assert isinstance(words[0][0], list)


class TestChatTokens:
    def test_tokens_flat(self):
        reader = CHAT.from_strs([BASIC_CHAT])
        tokens = reader.tokens()
        assert isinstance(tokens[0], Token)
        assert tokens[0].word == "I"
        assert tokens[0].pos == "pro"

    def test_tokens_by_utterance(self):
        reader = CHAT.from_strs([BASIC_CHAT])
        tokens = reader.tokens(by_utterance=True)
        assert isinstance(tokens[0], list)
        assert isinstance(tokens[0][0], Token)


class TestTimeMarks:
    def test_time_marks_present(self):
        chat_str = "@UTF8\n@Begin\n*CHI:\thello . \x15123_456\x15\n@End\n"
        reader = CHAT.from_strs([chat_str])
        utts = reader.utterances()
        assert utts[0].time_marks == (123, 456)

    def test_time_marks_absent(self):
        reader = CHAT.from_strs([BASIC_CHAT])
        utts = reader.utterances()
        assert utts[0].time_marks is None


class TestFilter:
    def test_filter_files(self):
        reader = CHAT.from_strs(TWO_FILES, ids=["action.cha", "codes.cha"])
        filtered = reader.filter(files="action")
        assert filtered.n_files == 1
        assert filtered.file_paths == ["action.cha"]

    def test_filter_single_participant(self):
        reader = CHAT.from_strs([BASIC_CHAT])
        filtered = reader.filter(participants="CHI")
        utts = filtered.utterances()
        assert len(utts) == 1
        assert utts[0].participant == "CHI"

    def test_filter_multiple_participants_list(self):
        reader = CHAT.from_strs([BASIC_CHAT])
        filtered = reader.filter(participants=["CHI", "MOT"])
        utts = filtered.utterances()
        assert len(utts) == 2

    def test_filter_single_participant_as_list(self):
        reader = CHAT.from_strs([BASIC_CHAT])
        filtered = reader.filter(participants=["MOT"])
        utts = filtered.utterances()
        assert len(utts) == 1
        assert utts[0].participant == "MOT"

    def test_filter_regex_alternation(self):
        reader = CHAT.from_strs([BASIC_CHAT])
        filtered = reader.filter(participants="CHI|MOT")
        utts = filtered.utterances()
        assert len(utts) == 2

    def test_filter_no_match(self):
        reader = CHAT.from_strs([BASIC_CHAT])
        filtered = reader.filter(participants="INV")
        utts = filtered.utterances()
        assert len(utts) == 0

    def test_filter_auto_anchored(self):
        reader = CHAT.from_strs([BASIC_CHAT])
        # "CH" should NOT match "CHI" due to auto-anchoring.
        filtered = reader.filter(participants="CH")
        utts = filtered.utterances()
        assert len(utts) == 0

    def test_filter_words(self):
        reader = CHAT.from_strs([BASIC_CHAT])
        filtered = reader.filter(participants="CHI")
        words = filtered.words()
        assert "I" in words
        assert "want" in words
        assert "no" not in words

    def test_filter_tokens(self):
        reader = CHAT.from_strs([BASIC_CHAT])
        filtered = reader.filter(participants="CHI")
        tokens = filtered.tokens()
        assert all(t.word != "no" for t in tokens)

    def test_filter_participants_header(self):
        reader = CHAT.from_strs([BASIC_CHAT])
        filtered = reader.filter(participants="CHI")
        parts = filtered.participants()
        codes = [p.code for p in parts]
        assert "CHI" in codes
        assert "MOT" not in codes

    def test_filter_files_and_participants(self):
        reader = CHAT.from_strs(TWO_FILES, ids=["action.cha", "codes.cha"])
        filtered = reader.filter(files="action", participants="CHI")
        assert filtered.n_files == 1
        utts = filtered.utterances()
        assert all(u.participant == "CHI" for u in utts)

    def test_filter_does_not_mutate_original(self):
        reader = CHAT.from_strs([BASIC_CHAT])
        _ = reader.filter(participants="CHI")
        assert len(reader.utterances()) == 2

    def test_filter_invalid_regex(self):
        reader = CHAT.from_strs([BASIC_CHAT])
        with pytest.raises(ValueError, match="Invalid participant regex"):
            reader.filter(participants="[invalid")

    def test_filter_wrong_type(self):
        reader = CHAT.from_strs([BASIC_CHAT])
        with pytest.raises(TypeError):
            reader.filter(participants=123)

    def test_filter_empty_list(self):
        reader = CHAT.from_strs([BASIC_CHAT])
        with pytest.raises(ValueError, match="must not be empty"):
            reader.filter(participants=[])

    def test_filter_participant_lookahead(self):
        reader = CHAT.from_strs([BASIC_CHAT])
        filtered = reader.filter(participants="(?=.*H).+")
        utts = filtered.utterances()
        assert len(utts) == 1
        assert utts[0].participant == "CHI"

    def test_filter_participant_negative_lookahead(self):
        reader = CHAT.from_strs([BASIC_CHAT])
        filtered = reader.filter(participants="(?!CHI).+")
        utts = filtered.utterances()
        assert len(utts) == 1
        assert utts[0].participant == "MOT"

    def test_filter_files_lookahead(self):
        reader = CHAT.from_strs(TWO_FILES, ids=["action.cha", "codes.cha"])
        filtered = reader.filter(files="(?=.*action).+")
        assert filtered.n_files == 1
        assert filtered.file_paths == ["action.cha"]

    def test_filter_files_negative_lookahead(self):
        reader = CHAT.from_strs(TWO_FILES, ids=["action.cha", "codes.cha"])
        filtered = reader.filter(files="^(?!.*action).+")
        assert filtered.n_files == 1
        assert filtered.file_paths == ["codes.cha"]


class TestLen:
    def test_len_raises(self):
        reader = CHAT.from_strs(TWO_FILES, ids=["a", "b"])
        with pytest.raises(TypeError):
            len(reader)  # type: ignore[arg-type]


class TestRepr:
    def test_repr(self):
        reader = CHAT.from_strs([BASIC_CHAT])
        assert "1 file(s)" in repr(reader)


class TestReprHtml:
    def test_basic_html(self):
        reader = CHAT.from_strs([BASIC_CHAT])
        utts = reader.utterances()
        html = utts[0]._repr_html_()
        assert isinstance(html, str)
        assert "<table" in html
        assert "*CHI:" in html
        assert "want" in html
        assert "%mor:" in html
        assert "%gra:" in html
        # Reconstructed %mor from token fields
        assert "pro|I" in html
        assert "v|want" in html
        assert "n|cookie" in html
        # Reconstructed %gra from token fields
        assert "1|2|SUBJ" in html
        assert "2|0|ROOT" in html

    def test_html_no_annotations(self):
        chat_str = "@UTF8\n@Begin\n*CHI:\thello world .\n@End\n"
        reader = CHAT.from_strs([chat_str])
        utts = reader.utterances()
        html = utts[0]._repr_html_()
        assert "*CHI:" in html
        assert "hello" in html
        assert "world" in html
        assert "%mor:" not in html
        assert "%gra:" not in html

    def test_html_time_marks(self):
        chat_str = (
            "@UTF8\n@Begin\n@Participants:\tCHI Child\n"
            "*CHI:\thi . \x150_1500\x15\n@End\n"
        )
        reader = CHAT.from_strs([chat_str])
        utts = reader.utterances()
        html = utts[0]._repr_html_()
        assert "0" in html
        assert "1500" in html
        assert "ms" in html
        assert "rustling-utterance-wrapper" in html

    def test_html_no_time_marks(self):
        reader = CHAT.from_strs([BASIC_CHAT])
        utts = reader.utterances()
        html = utts[0]._repr_html_()
        assert "rustling-utterance-wrapper" not in html

    def test_html_escaping(self):
        utt = Utterance(
            participant="CHI",
            tokens=[Token(word="a<b")],
            time_marks=None,
            tiers={"CHI": "a<b"},
        )
        html = utt._repr_html_()
        assert "a&lt;b" in html
        assert "a<b" not in html.replace("a&lt;b", "")

    def test_html_empty_tokens(self):
        utt = Utterance(
            participant="CHI",
            tokens=[],
            time_marks=None,
            tiers={"CHI": ""},
        )
        html = utt._repr_html_()
        assert "<table" in html
        assert "*CHI:" in html


class TestCleanUtterance:
    """Test that CHAT annotations are cleaned in parsed output."""

    def test_removes_overlap_markers(self):
        chat_str = "@UTF8\n@Begin\n*CHI:\tI [<] want cookie .\n@End\n"
        reader = CHAT.from_strs([chat_str])
        words = reader.words()
        assert "[<]" not in words

    def test_removes_error_markers(self):
        chat_str = "@UTF8\n@Begin\n*CHI:\tgoed [*] .\n@End\n"
        reader = CHAT.from_strs([chat_str])
        words = reader.words()
        assert "[*]" not in words
        assert "goed" in words

    def test_removes_explanations(self):
        chat_str = "@UTF8\n@Begin\n*CHI:\tI want [= desire] cookie .\n@End\n"
        reader = CHAT.from_strs([chat_str])
        words = reader.words()
        assert "[=" not in " ".join(words)
        assert "desire" not in " ".join(words)

    def test_removes_xxx(self):
        chat_str = "@UTF8\n@Begin\n*CHI:\txxx hello .\n@End\n"
        reader = CHAT.from_strs([chat_str])
        words = reader.words()
        assert "xxx" not in words
        assert "hello" in words


class TestMorAlignment:
    def test_postclitic(self):
        chat_str = (
            "@UTF8\n@Begin\n"
            "*CHI:\tthat's good .\n"
            "%mor:\tpro:dem|that~cop|be&3S adj|good .\n"
            "@End\n"
        )
        reader = CHAT.from_strs([chat_str])
        utts = reader.utterances()
        tokens = utts[0].tokens
        tokens = utts[0].tokens
        assert tokens is not None
        assert tokens[0].word == "that's"
        assert tokens[0].pos == "pro:dem"
        # Clitic token has empty word
        assert tokens[1].word == ""
        assert tokens[1].pos == "cop"
        assert tokens[2].word == "good"
        assert tokens[2].pos == "adj"


class TestContinuationLines:
    def test_continuation_joined(self):
        chat_str = "@UTF8\n@Begin\n" "*CHI:\tI want\n" "\tcookie .\n" "@End\n"
        reader = CHAT.from_strs([chat_str])
        words = reader.words()
        assert "I" in words
        assert "want" in words
        assert "cookie" in words


class TestLeadingWhitespace:
    def test_leading_whitespace_stripped(self):
        chat_str = (
            "  @UTF8\n"
            "  @Begin\n"
            "  @Participants:\tCHI Child, MOT Mother\n"
            "  *CHI:\tI want cookie .\n"
            "  %mor:\tpro|I v|want n|cookie .\n"
            "  *MOT:\tno .\n"
            "  %mor:\tco|no .\n"
            "  @End\n"
        )
        reader = CHAT.from_strs([chat_str])
        utts = reader.utterances()
        assert len(utts) == 2
        words = reader.words()
        assert words == ["I", "want", "cookie", ".", "no", "."]


class TestFromDir:
    def test_from_dir(self, testchat_good_dir):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reader = CHAT.from_dir(str(testchat_good_dir), strict=False)
        assert reader.n_files > 0

    def test_from_dir_with_path(self, testchat_good_dir):
        """from_dir accepts pathlib.Path directly."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reader = CHAT.from_dir(testchat_good_dir, strict=False)
        assert reader.n_files > 0

    def test_from_dir_with_match(self, testchat_good_dir):
        reader = CHAT.from_dir(str(testchat_good_dir), match="action")
        paths = reader.file_paths
        assert len(paths) > 0
        assert all("action" in p for p in paths)

    def test_utterances_from_real_files(self, testchat_good_dir):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reader = CHAT.from_dir(str(testchat_good_dir), strict=False)
        utts = reader.utterances()
        assert len(utts) > 0

    def test_words_from_real_files(self, testchat_good_dir):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reader = CHAT.from_dir(str(testchat_good_dir), strict=False)
        words = reader.words()
        assert len(words) > 0

    def test_testchat_strict_compliance(self, testchat_good_dir):
        """Report which testchat/good files would fail strict=True."""
        try:
            CHAT.from_dir(str(testchat_good_dir), strict=True)
        except ValueError as e:
            warnings.warn(
                f"testchat/good files failing strict=True:\n\n{e}",
                stacklevel=1,
            )

    def test_testchat_bad_files_catch_errors(self, testchat_bad_dir):
        """Every testchat/bad file must raise a parsing error with strict=True."""
        # Files requiring complex cross-tier or context-dependent checks
        # not yet implemented (Tier 3+4 validation rules).
        known_exceptions = {
            # Cross-tier checks (%mor, %gra, %pho):
            "mor-commas.cha",
            "mor-empty.cha",
            "latetalkers.cha",
            "mornumber-spanish.cha",
            "mornumber-spanish-2.cha",
            "pho-group-compound.cha",
            "pho-repetition.cha",
            "pho-repetition-bad.cha",
            # Media status vs bullet validation:
            "media-needs-bullets.cha",
            "media-notrans-bullets.cha",
            "media-unlinked-bullets.cha",
            # Retrace followed-by-content checks:
            "retrace-in-group-bad.cha",
            "retrace-multiple-no-following.cha",
            "retrace-no-following-content.cha",
            # Quotation nesting:
            "quotation-nested.cha",
            # Language-level checks:
            "language-different-speakers.cha",
            "zho-f.cha",
            # CA segment repetition:
            "ca-segment-repetition.cha",
            "ca-segment-repetition-bad-content.cha",
            # [x N] bracket context:
            "repetition.cha",
            "grouprepetition.cha",
            "x-repetition.cha",
            # @Options sign/heritage/bullet required:
            "sign.cha",
            "words-sign.cha",
            "heritage.cha",
            "heritage-lsfal14a.cha",
            "bs5.cha",
            # Participant code edge cases:
            "who.cha",
            # Other:
            "zero-others.cha",
            "space-bracket.cha",
        }
        no_error_files = []
        for path in sorted(testchat_bad_dir.glob("*.cha")):
            if path.name in known_exceptions:
                continue
            try:
                CHAT.from_files([str(path)], strict=True)
            except Exception:
                pass
            else:
                no_error_files.append(path.name)
        assert not no_error_files, (
            f"{len(no_error_files)} testchat/bad files that raised no parsing error:\n"
            + "\n".join(no_error_files)
        )

    def test_private_data_strict_compliance(self, private_data_dir):
        """Report which private test data files would fail strict=True."""
        try:
            CHAT.from_dir(str(private_data_dir), strict=True)
        except ValueError as e:
            warnings.warn(
                f"Private test data files failing strict=True:\n\n{e}",
                stacklevel=1,
            )


class TestFromFiles:
    def test_from_files(self, testchat_good_dir):
        import glob

        cha_files = sorted(glob.glob(str(testchat_good_dir / "*.cha")))[:3]
        if cha_files:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                reader = CHAT.from_files(cha_files, strict=False)
            assert reader.n_files == len(cha_files)

    def test_from_files_with_path(self, testchat_good_dir):
        """from_files accepts pathlib.Path objects in the list."""
        paths = sorted(testchat_good_dir.glob("*.cha"))[:3]
        if paths:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                reader = CHAT.from_files(paths, strict=False)
            assert reader.n_files == len(paths)

    def test_from_files_mixed_str_and_path(self, testchat_good_dir):
        """from_files accepts a mix of str and pathlib.Path."""
        all_paths = sorted(testchat_good_dir.glob("*.cha"))[:2]
        if len(all_paths) >= 2:
            mixed = [str(all_paths[0]), all_paths[1]]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                reader = CHAT.from_files(mixed, strict=False)
            assert reader.n_files == 2


class TestAppend:
    def test_append(self):
        a = CHAT.from_strs(["@UTF8\n@Begin\n*CHI:\thi .\n@End\n"], ids=["a"])
        b = CHAT.from_strs(["@UTF8\n@Begin\n*MOT:\tbye .\n@End\n"], ids=["b"])
        a.append(b)
        assert a.n_files == 2
        assert a.file_paths == ["a", "b"]

    def test_append_does_not_modify_other(self):
        a = CHAT.from_strs(TWO_FILES, ids=["a", "b"])
        b = CHAT.from_strs(["@UTF8\n@Begin\n*CHI:\tyes .\n@End\n"], ids=["c"])
        a.append(b)
        assert a.n_files == 3
        assert b.n_files == 1

    def test_append_left(self):
        a = CHAT.from_strs(["@UTF8\n@Begin\n*CHI:\thi .\n@End\n"], ids=["a"])
        b = CHAT.from_strs(["@UTF8\n@Begin\n*MOT:\tbye .\n@End\n"], ids=["b"])
        a.append_left(b)
        assert a.file_paths == ["b", "a"]

    def test_append_left_preserves_order(self):
        a = CHAT.from_strs(["@UTF8\n@Begin\n*CHI:\thi .\n@End\n"], ids=["a"])
        b = CHAT.from_strs(TWO_FILES, ids=["b", "c"])
        a.append_left(b)
        assert a.file_paths == ["b", "c", "a"]


class TestExtend:
    def test_extend(self):
        a = CHAT.from_strs(["@UTF8\n@Begin\n*CHI:\thi .\n@End\n"], ids=["a"])
        b = CHAT.from_strs(["@UTF8\n@Begin\n*MOT:\tbye .\n@End\n"], ids=["b"])
        c = CHAT.from_strs(["@UTF8\n@Begin\n*CHI:\tyes .\n@End\n"], ids=["c"])
        a.extend([b, c])
        assert a.n_files == 3
        assert a.file_paths == ["a", "b", "c"]

    def test_extend_left(self):
        a = CHAT.from_strs(["@UTF8\n@Begin\n*CHI:\thi .\n@End\n"], ids=["a"])
        b = CHAT.from_strs(["@UTF8\n@Begin\n*MOT:\tbye .\n@End\n"], ids=["b"])
        c = CHAT.from_strs(["@UTF8\n@Begin\n*CHI:\tyes .\n@End\n"], ids=["c"])
        a.extend_left([b, c])
        assert a.file_paths == ["b", "c", "a"]

    def test_extend_empty_list(self):
        a = CHAT.from_strs(TWO_FILES, ids=["a", "b"])
        a.extend([])
        assert a.n_files == 2


class TestPop:
    def test_pop(self):
        reader = CHAT.from_strs(TWO_FILES, ids=["a", "b"])
        popped = reader.pop()
        assert reader.n_files == 1
        assert reader.file_paths == ["a"]
        assert popped.n_files == 1
        assert popped.file_paths == ["b"]

    def test_pop_left(self):
        reader = CHAT.from_strs(TWO_FILES, ids=["a", "b"])
        popped = reader.pop_left()
        assert reader.n_files == 1
        assert reader.file_paths == ["b"]
        assert popped.n_files == 1
        assert popped.file_paths == ["a"]

    def test_pop_empty_raises(self):
        reader = CHAT.from_strs([])
        with pytest.raises(IndexError):
            reader.pop()

    def test_pop_left_empty_raises(self):
        reader = CHAT.from_strs([])
        with pytest.raises(IndexError):
            reader.pop_left()

    def test_pop_preserves_data(self):
        reader = CHAT.from_strs(TWO_FILES, ids=["a", "b"])
        popped = reader.pop()
        assert popped.words() == ["bye", "."]


class TestClear:
    def test_clear(self):
        reader = CHAT.from_strs(TWO_FILES, ids=["a", "b"])
        reader.clear()
        assert reader.n_files == 0

    def test_clear_empty(self):
        reader = CHAT.from_strs([])
        reader.clear()
        assert reader.n_files == 0


class TestAdd:
    def test_add_creates_new_chat(self):
        a = CHAT.from_strs(["@UTF8\n@Begin\n*CHI:\thi .\n@End\n"], ids=["a"])
        b = CHAT.from_strs(["@UTF8\n@Begin\n*MOT:\tbye .\n@End\n"], ids=["b"])
        c = a + b
        assert c.n_files == 2
        assert c.file_paths == ["a", "b"]

    def test_add_does_not_mutate_operands(self):
        a = CHAT.from_strs(["@UTF8\n@Begin\n*CHI:\thi .\n@End\n"], ids=["a"])
        b = CHAT.from_strs(["@UTF8\n@Begin\n*MOT:\tbye .\n@End\n"], ids=["b"])
        _ = a + b
        assert a.n_files == 1
        assert a.file_paths == ["a"]
        assert b.n_files == 1
        assert b.file_paths == ["b"]

    def test_add_multi_file(self):
        a = CHAT.from_strs(TWO_FILES, ids=["a", "b"])
        b = CHAT.from_strs(["@UTF8\n@Begin\n*CHI:\tyes .\n@End\n"], ids=["c"])
        c = a + b
        assert c.n_files == 3
        assert c.file_paths == ["a", "b", "c"]

    def test_iadd_mutates_in_place(self):
        a = CHAT.from_strs(["@UTF8\n@Begin\n*CHI:\thi .\n@End\n"], ids=["a"])
        b = CHAT.from_strs(["@UTF8\n@Begin\n*MOT:\tbye .\n@End\n"], ids=["b"])
        a += b
        assert a.n_files == 2
        assert a.file_paths == ["a", "b"]

    def test_iadd_does_not_mutate_other(self):
        a = CHAT.from_strs(["@UTF8\n@Begin\n*CHI:\thi .\n@End\n"], ids=["a"])
        b = CHAT.from_strs(["@UTF8\n@Begin\n*MOT:\tbye .\n@End\n"], ids=["b"])
        a += b
        assert b.n_files == 1
        assert b.file_paths == ["b"]


class TestBool:
    def test_empty_default_is_falsy(self):
        assert not CHAT()

    def test_empty_from_strs_is_falsy(self):
        assert not CHAT.from_strs([])

    def test_with_data_is_truthy(self):
        reader = CHAT.from_strs(["@UTF8\n@Begin\n*CHI:\thi .\n@End\n"])
        assert reader

    def test_after_clear_is_falsy(self):
        reader = CHAT.from_strs(["@UTF8\n@Begin\n*CHI:\thi .\n@End\n"])
        reader.clear()
        assert not reader


THREE_FILES = [
    "@UTF8\n@Begin\n*CHI:\thi .\n@End\n",
    "@UTF8\n@Begin\n*MOT:\tbye .\n@End\n",
    "@UTF8\n@Begin\n*CHI:\tyes .\n@End\n",
]

FOUR_FILES = [
    "@UTF8\n@Begin\n*CHI:\thi .\n@End\n",
    "@UTF8\n@Begin\n*MOT:\tbye .\n@End\n",
    "@UTF8\n@Begin\n*CHI:\tyes .\n@End\n",
    "@UTF8\n@Begin\n*MOT:\tok .\n@End\n",
]


class TestIterAndGetitem:
    # -- __iter__ --

    def test_iter_yields_chat_objects(self):
        reader = CHAT.from_strs(TWO_FILES, ids=["a", "b"])
        items = list(reader)
        assert len(items) == 2
        assert all(isinstance(item, CHAT) for item in items)

    def test_iter_each_has_one_file(self):
        reader = CHAT.from_strs(TWO_FILES, ids=["a", "b"])
        items = list(reader)
        assert items[0].n_files == 1
        assert items[1].n_files == 1

    def test_iter_preserves_order(self):
        reader = CHAT.from_strs(TWO_FILES, ids=["a", "b"])
        items = list(reader)
        assert items[0].file_paths == ["a"]
        assert items[1].file_paths == ["b"]

    def test_iter_preserves_data(self):
        reader = CHAT.from_strs(TWO_FILES, ids=["a", "b"])
        items = list(reader)
        assert items[0].words() == ["hi", "."]
        assert items[1].words() == ["bye", "."]

    def test_iter_empty_reader(self):
        reader = CHAT.from_strs([])
        assert list(reader) == []

    def test_iter_does_not_mutate(self):
        reader = CHAT.from_strs(TWO_FILES, ids=["a", "b"])
        _ = list(reader)
        assert reader.n_files == 2

    def test_iter_for_loop(self):
        reader = CHAT.from_strs(TWO_FILES, ids=["a", "b"])
        count = 0
        for item in reader:
            assert isinstance(item, CHAT)
            count += 1
        assert count == 2

    # -- __getitem__ with int --

    def test_getitem_positive_index(self):
        reader = CHAT.from_strs(TWO_FILES, ids=["a", "b"])
        item = reader[0]
        assert isinstance(item, CHAT)
        assert item.n_files == 1
        assert item.file_paths == ["a"]

    def test_getitem_second_index(self):
        reader = CHAT.from_strs(TWO_FILES, ids=["a", "b"])
        assert reader[1].file_paths == ["b"]

    def test_getitem_negative_index(self):
        reader = CHAT.from_strs(TWO_FILES, ids=["a", "b"])
        assert reader[-1].file_paths == ["b"]
        assert reader[-2].file_paths == ["a"]

    def test_getitem_out_of_range(self):
        reader = CHAT.from_strs(TWO_FILES, ids=["a", "b"])
        with pytest.raises(IndexError):
            reader[5]

    def test_getitem_negative_out_of_range(self):
        reader = CHAT.from_strs(TWO_FILES, ids=["a", "b"])
        with pytest.raises(IndexError):
            reader[-3]

    def test_getitem_preserves_data(self):
        reader = CHAT.from_strs(TWO_FILES, ids=["a", "b"])
        assert reader[0].words() == ["hi", "."]
        assert reader[1].words() == ["bye", "."]

    def test_getitem_does_not_mutate(self):
        reader = CHAT.from_strs(TWO_FILES, ids=["a", "b"])
        _ = reader[0]
        assert reader.n_files == 2

    def test_getitem_empty_reader(self):
        reader = CHAT.from_strs([])
        with pytest.raises(IndexError):
            reader[0]

    # -- __getitem__ with slice --

    def test_getitem_slice_basic(self):
        reader = CHAT.from_strs(THREE_FILES, ids=["a", "b", "c"])
        result = reader[:2]
        assert isinstance(result, CHAT)
        assert result.n_files == 2
        assert result.file_paths == ["a", "b"]

    def test_getitem_slice_with_start(self):
        reader = CHAT.from_strs(THREE_FILES, ids=["a", "b", "c"])
        result = reader[1:]
        assert result.n_files == 2
        assert result.file_paths == ["b", "c"]

    def test_getitem_slice_with_start_and_stop(self):
        reader = CHAT.from_strs(FOUR_FILES, ids=["a", "b", "c", "d"])
        result = reader[1:3]
        assert result.n_files == 2
        assert result.file_paths == ["b", "c"]

    def test_getitem_slice_with_step(self):
        reader = CHAT.from_strs(FOUR_FILES, ids=["a", "b", "c", "d"])
        result = reader[::2]
        assert result.n_files == 2
        assert result.file_paths == ["a", "c"]

    def test_getitem_slice_negative(self):
        reader = CHAT.from_strs(THREE_FILES, ids=["a", "b", "c"])
        result = reader[-2:]
        assert result.n_files == 2
        assert result.file_paths == ["b", "c"]

    def test_getitem_slice_negative_step(self):
        reader = CHAT.from_strs(THREE_FILES, ids=["a", "b", "c"])
        result = reader[::-1]
        assert result.n_files == 3
        assert result.file_paths == ["c", "b", "a"]

    def test_getitem_slice_empty_result(self):
        reader = CHAT.from_strs(TWO_FILES, ids=["a", "b"])
        result = reader[5:10]
        assert isinstance(result, CHAT)
        assert result.n_files == 0

    def test_getitem_slice_full(self):
        reader = CHAT.from_strs(TWO_FILES, ids=["a", "b"])
        result = reader[:]
        assert result.n_files == 2
        assert result.file_paths == ["a", "b"]

    def test_getitem_slice_does_not_mutate(self):
        reader = CHAT.from_strs(TWO_FILES, ids=["a", "b"])
        _ = reader[:1]
        assert reader.n_files == 2

    def test_getitem_slice_preserves_data(self):
        reader = CHAT.from_strs(TWO_FILES, ids=["a", "b"])
        result = reader[:1]
        assert result.words() == ["hi", "."]

    # -- __getitem__ type error --

    def test_getitem_invalid_type(self):
        reader = CHAT.from_strs(TWO_FILES, ids=["a", "b"])
        with pytest.raises(TypeError):
            reader["key"]  # type: ignore[index]


class TestToStrs:
    def test_to_strs_basic(self):
        reader = CHAT.from_strs([BASIC_CHAT])
        strs = reader.to_strs()
        assert len(strs) == 1
        assert "@UTF8" in strs[0]
        assert "@Begin" in strs[0]
        assert "@End" in strs[0]
        assert "*CHI:" in strs[0]
        assert "%mor:" in strs[0]

    def test_to_strs_multiple_files(self):
        reader = CHAT.from_strs(TWO_FILES, ids=["a", "b"])
        strs = reader.to_strs()
        assert len(strs) == 2

    def test_to_strs_round_trip(self):
        reader1 = CHAT.from_strs([BASIC_CHAT])
        strs = reader1.to_strs()
        reader2 = CHAT.from_strs(strs)
        assert reader1.words() == reader2.words()
        assert len(reader1.utterances()) == len(reader2.utterances())

    def test_to_strs_empty(self):
        reader = CHAT.from_strs([])
        assert reader.to_strs() == []

    def test_to_strs_round_trip_real_files(self, testchat_good_dir):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reader = CHAT.from_dir(str(testchat_good_dir), strict=False)
            strs = reader.to_strs()
            reader2 = CHAT.from_strs(strs, strict=False)
        assert reader.words() == reader2.words()


class TestToFiles:
    def test_to_files_single_file(self, tmp_path):
        reader = CHAT.from_strs([BASIC_CHAT])
        out_dir = str(tmp_path / "output")
        reader.to_files(out_dir)
        reader2 = CHAT.from_dir(out_dir)
        assert reader2.words() == reader.words()

    def test_to_files_with_path(self, tmp_path):
        """to_files accepts pathlib.Path directly."""
        reader = CHAT.from_strs([BASIC_CHAT])
        out_dir = tmp_path / "output"
        reader.to_files(out_dir)
        reader2 = CHAT.from_dir(out_dir)
        assert reader2.words() == reader.words()

    def test_to_files_directory(self, tmp_path):
        reader = CHAT.from_strs(TWO_FILES, ids=["a", "b"])
        out_dir = str(tmp_path / "output")
        reader.to_files(out_dir)
        reader2 = CHAT.from_dir(out_dir)
        assert reader2.n_files == 2

    def test_to_files_custom_filenames(self, tmp_path):
        import os

        reader = CHAT.from_strs(TWO_FILES, ids=["a", "b"])
        out_dir = str(tmp_path / "output")
        reader.to_files(out_dir, filenames=["file1.cha", "file2.cha"])
        assert os.path.exists(os.path.join(out_dir, "file1.cha"))
        assert os.path.exists(os.path.join(out_dir, "file2.cha"))

    def test_to_files_default_filenames_from_ids(self, tmp_path):
        import os

        reader = CHAT.from_strs(TWO_FILES, ids=["a", "b"])
        out_dir = str(tmp_path / "output")
        reader.to_files(out_dir)
        # IDs "a" and "b" are not UUIDs, so filenames are derived from them.
        assert os.path.exists(os.path.join(out_dir, "a.cha"))
        assert os.path.exists(os.path.join(out_dir, "b.cha"))

    def test_to_files_default_filenames_numbered(self, tmp_path):
        import os

        # No ids -> UUIDs are generated -> numbered fallback.
        reader = CHAT.from_strs(TWO_FILES)
        out_dir = str(tmp_path / "output")
        reader.to_files(out_dir)
        assert os.path.exists(os.path.join(out_dir, "0001.cha"))
        assert os.path.exists(os.path.join(out_dir, "0002.cha"))

    def test_to_files_filename_count_mismatch_raises(self, tmp_path):
        reader = CHAT.from_strs(TWO_FILES, ids=["a", "b"])
        with pytest.raises(ValueError):
            reader.to_files(
                str(tmp_path / "output"),
                filenames=["only_one.cha"],
            )

    def test_to_files_preserves_filenames(self, testchat_good_dir, tmp_path):
        import os

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reader = CHAT.from_dir(str(testchat_good_dir), strict=False)
        out_dir = str(tmp_path / "output")
        reader.to_files(out_dir)
        # Filenames should be derived from original file paths.
        for fp in reader.file_paths:
            expected = os.path.splitext(os.path.basename(fp))[0] + ".cha"
            assert os.path.exists(os.path.join(out_dir, expected)), expected

    def test_to_files_round_trip_real_files(self, testchat_good_dir, tmp_path):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reader = CHAT.from_dir(str(testchat_good_dir), strict=False)
            out_dir = str(tmp_path / "output")
            reader.to_files(out_dir)
            reader2 = CHAT.from_dir(out_dir, strict=False)
        assert reader.words() == reader2.words()


class TestToElan:
    def test_to_elan_returns_elan_object(self):
        from rustling.elan import ELAN

        reader = CHAT.from_strs([BASIC_CHAT])
        elan = reader.to_elan()
        assert isinstance(elan, ELAN)
        assert elan.n_files == 1

    def test_to_elan_tiers(self):
        reader = CHAT.from_strs([BASIC_CHAT])
        elan = reader.to_elan()
        tiers_dict = elan.tiers()[0]
        tier_ids = list(tiers_dict.keys())
        assert "CHI" in tier_ids
        assert "MOT" in tier_ids
        assert "mor@CHI" in tier_ids
        assert "gra@CHI" in tier_ids
        assert "mor@MOT" in tier_ids
        assert "gra@MOT" in tier_ids

    def test_to_elan_strs(self):
        reader = CHAT.from_strs([BASIC_CHAT])
        strs = reader.to_elan_strs()
        assert len(strs) == 1
        assert "ANNOTATION_DOCUMENT" in strs[0]
        assert "TIER_ID" in strs[0]

    def test_to_elan_dep_tier_structure(self):
        reader = CHAT.from_strs([BASIC_CHAT])
        elan = reader.to_elan()
        tiers_dict = elan.tiers()[0]
        chi_tier = tiers_dict["CHI"]
        mor_chi = tiers_dict["mor@CHI"]
        assert chi_tier.parent_id is None
        assert mor_chi.parent_id == "CHI"
        # CHI has 1 utterance in BASIC_CHAT.
        assert len(chi_tier.annotations) == 1
        assert chi_tier.annotations[0].value == "I want cookie ."
        assert len(mor_chi.annotations) == 1
        assert mor_chi.annotations[0].value == "pro|I v|want n|cookie ."

    def test_to_elan_multiple_files(self):
        reader = CHAT.from_strs(TWO_FILES, ids=["a", "b"])
        elan = reader.to_elan()
        assert elan.n_files == 2

    def test_to_elan_files_directory(self, tmp_path):
        from rustling.elan import ELAN

        reader = CHAT.from_strs(TWO_FILES, ids=["a", "b"])
        out_dir = str(tmp_path / "output")
        reader.to_elan_files(out_dir)
        elan = ELAN.from_dir(out_dir)
        assert elan.n_files == 2

    def test_to_elan_files_custom_filenames(self, tmp_path):
        import os

        reader = CHAT.from_strs(TWO_FILES, ids=["a", "b"])
        out_dir = str(tmp_path / "output")
        reader.to_elan_files(out_dir, filenames=["a.eaf", "b.eaf"])
        assert os.path.exists(os.path.join(out_dir, "a.eaf"))
        assert os.path.exists(os.path.join(out_dir, "b.eaf"))

    def test_to_elan_files_default_filenames_from_ids(self, tmp_path):
        import os

        reader = CHAT.from_strs(TWO_FILES, ids=["a", "b"])
        out_dir = str(tmp_path / "output")
        reader.to_elan_files(out_dir)
        # IDs "a" and "b" are not UUIDs, so filenames derive as a.eaf, b.eaf.
        assert os.path.exists(os.path.join(out_dir, "a.eaf"))
        assert os.path.exists(os.path.join(out_dir, "b.eaf"))

    def test_to_elan_files_default_filenames_numbered(self, tmp_path):
        import os

        reader = CHAT.from_strs(TWO_FILES)
        out_dir = str(tmp_path / "output")
        reader.to_elan_files(out_dir)
        assert os.path.exists(os.path.join(out_dir, "0001.eaf"))
        assert os.path.exists(os.path.join(out_dir, "0002.eaf"))

    def test_to_elan_files_preserves_filenames(self, testchat_good_dir, tmp_path):
        import os

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reader = CHAT.from_dir(str(testchat_good_dir), strict=False)
        out_dir = str(tmp_path / "output")
        reader.to_elan_files(out_dir)
        # Filenames should be derived from original .cha paths with .eaf extension.
        for fp in reader.file_paths:
            expected = os.path.splitext(os.path.basename(fp))[0] + ".eaf"
            assert os.path.exists(os.path.join(out_dir, expected)), expected

    def test_to_elan_files_round_trip_real_files(self, testchat_good_dir, tmp_path):
        from rustling.elan import ELAN

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reader = CHAT.from_dir(str(testchat_good_dir), strict=False)
        out_dir = str(tmp_path / "output")
        reader.to_elan_files(out_dir)
        elan = ELAN.from_dir(out_dir)
        assert elan.n_files == reader.n_files


class TestPopAndStitch:
    def test_pop_and_append(self):
        reader = CHAT.from_strs(TWO_FILES, ids=["a", "b"])
        n = reader.n_files
        popped = reader.pop()
        assert reader.n_files == n - 1
        reader.append(popped)
        assert reader.n_files == n

    def test_pop_and_append_real_files(self, testchat_good_dir):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reader = CHAT.from_dir(str(testchat_good_dir), strict=False)
        n = reader.n_files
        all_words = reader.words()
        popped = reader.pop()
        reader.append(popped)
        assert reader.n_files == n
        assert reader.words() == all_words


HEADER_CHAT = (
    "@UTF8\n"
    "@PID:\t11312/c-00044068-1\n"
    "@Begin\n"
    "@Languages:\teng, zho\n"
    "@Participants:\tCHI Child Target_Child, MOT Mary Mother\n"
    "@ID:\teng|brown|CHI|2;10.05|male|||Target_Child|||\n"
    "@ID:\teng|brown|MOT||female|||Mother|||\n"
    "@Date:\t25-JAN-1983\n"
    "@Location:\tBoston, MA, USA\n"
    "@Media:\tabe88, audio, missing\n"
    "@Situation:\tPlaying with toys\n"
    "@Comment:\tFirst recording\n"
    "@Comment:\tfor Yiddish/Hebrew glossary see file 4504\n"
    "*CHI:\thello .\n"
    "@New Episode\n"
    "*MOT:\thi .\n"
    "@Comment:\tChild laughs\n"
    "*CHI:\tcookie .\n"
    "@End\n"
)


class TestHeaders:
    def test_headers_returns_list(self):
        reader = CHAT.from_strs([HEADER_CHAT])
        headers = reader.headers()
        assert isinstance(headers, list)
        assert len(headers) == 1
        assert isinstance(headers[0], Headers)

    def test_pid(self):
        reader = CHAT.from_strs([HEADER_CHAT])
        h = reader.headers()[0]
        assert h.pid == "11312/c-00044068-1"

    def test_languages(self):
        reader = CHAT.from_strs([HEADER_CHAT])
        h = reader.headers()[0]
        assert h.languages == ["eng", "zho"]

    def test_languages_method(self):
        reader = CHAT.from_strs([HEADER_CHAT])
        assert reader.languages() == ["eng", "zho"]

    def test_languages_by_file(self):
        reader = CHAT.from_strs([HEADER_CHAT])
        result = reader.languages(by_file=True)
        assert result == [["eng", "zho"]]

    def test_date(self):
        reader = CHAT.from_strs([HEADER_CHAT])
        h = reader.headers()[0]
        assert h.date == datetime.date(1983, 1, 25)

    def test_date_iso(self):
        chat_str = HEADER_CHAT.replace("25-JAN-1983", "1983-01-25")
        reader = CHAT.from_strs([chat_str])
        h = reader.headers()[0]
        assert h.date == datetime.date(1983, 1, 25)

    def test_date_unparseable(self):
        chat_str = HEADER_CHAT.replace("25-JAN-1983", "not-a-date")
        reader = CHAT.from_strs([chat_str])
        h = reader.headers()[0]
        assert h.date is None

    def test_location(self):
        reader = CHAT.from_strs([HEADER_CHAT])
        h = reader.headers()[0]
        assert h.location == "Boston, MA, USA"

    def test_situation(self):
        reader = CHAT.from_strs([HEADER_CHAT])
        h = reader.headers()[0]
        assert h.situation == "Playing with toys"

    def test_media(self):
        reader = CHAT.from_strs([HEADER_CHAT])
        h = reader.headers()[0]
        assert isinstance(h.media, dict)
        assert h.media["filename"] == "abe88"
        assert h.media["format"] == "audio"
        assert h.media["status"] == "missing"

    def test_comments(self):
        reader = CHAT.from_strs([HEADER_CHAT])
        h = reader.headers()[0]
        assert isinstance(h.comments, list)
        assert len(h.comments) == 2
        assert h.comments[0] == "First recording"
        assert h.comments[1] == "for Yiddish/Hebrew glossary see file 4504"

    def test_comments_empty(self):
        simple = (
            "@UTF8\n@Begin\n@Participants:\tCHI Child Target_Child\n"
            "*CHI:\thello .\n@End\n"
        )
        reader = CHAT.from_strs([simple])
        h = reader.headers()[0]
        assert h.comments is None

    def test_other_empty_by_default(self):
        reader = CHAT.from_strs([HEADER_CHAT])
        h = reader.headers()[0]
        assert isinstance(h.other, dict)


class TestParticipants:
    def test_participants_count(self):
        reader = CHAT.from_strs([HEADER_CHAT])
        h = reader.headers()[0]
        assert len(h.participants) == 2

    def test_participant_fields(self):
        reader = CHAT.from_strs([HEADER_CHAT])
        chi = reader.headers()[0].participants[0]
        assert isinstance(chi, Participant)
        assert chi.code == "CHI"
        assert chi.name == "Child"
        assert chi.role == "Target_Child"

    def test_participant_id_fields(self):
        reader = CHAT.from_strs([HEADER_CHAT])
        chi = reader.headers()[0].participants[0]
        assert chi.language == "eng"
        assert chi.corpus == "brown"
        assert chi.sex == "male"

    def test_participant_age(self):
        reader = CHAT.from_strs([HEADER_CHAT])
        chi = reader.headers()[0].participants[0]
        assert isinstance(chi.age, Age)
        assert chi.age.years == 2
        assert chi.age.months == 10
        assert chi.age.days == 5

    def test_age_in_months(self):
        reader = CHAT.from_strs([HEADER_CHAT])
        chi = reader.headers()[0].participants[0]
        assert chi.age is not None
        months = chi.age.in_months()
        assert months == pytest.approx(34.1667, abs=0.01)

    def test_participant_no_age(self):
        reader = CHAT.from_strs([HEADER_CHAT])
        mot = reader.headers()[0].participants[1]
        assert mot.code == "MOT"
        assert mot.age is None
        assert mot.sex == "female"

    def test_participants_method(self):
        reader = CHAT.from_strs([HEADER_CHAT])
        parts = reader.participants()
        assert len(parts) == 2
        assert [p.code for p in parts] == ["CHI", "MOT"]

    def test_participants_by_file(self):
        reader = CHAT.from_strs([HEADER_CHAT])
        result = reader.participants(by_file=True)
        assert len(result) == 1
        assert isinstance(result[0], list)
        assert len(result[0]) == 2

    def test_participant_specific_headers(self):
        chat_str = (
            "@UTF8\n"
            "@Begin\n"
            "@Languages:\teng\n"
            "@Participants:\tCHI Ross Target_Child\n"
            "@ID:\teng|macwhinney|CHI|2;06.||||Target_Child|||\n"
            "@Birth of CHI:\t28-JUN-2001\n"
            "@Birthplace of CHI:\tPittsburgh, PA\n"
            "@L1 of CHI:\teng\n"
            "*CHI:\thello .\n"
            "@End\n"
        )
        reader = CHAT.from_strs([chat_str])
        chi = reader.headers()[0].participants[0]
        assert chi.birth == "28-JUN-2001"
        assert chi.birthplace == "Pittsburgh, PA"
        assert chi.l1 == "eng"


class TestHeadersMultipleFiles:
    def test_headers_multiple_files(self):
        strs = [
            "@UTF8\n@Begin\n@Languages:\teng\n"
            "@Participants:\tCHI Child Target_Child\n"
            "*CHI:\thi .\n@End\n",
            "@UTF8\n@Begin\n@Languages:\tzho\n"
            "@Participants:\tMOT Mother Mother\n"
            "*MOT:\tbye .\n@End\n",
        ]
        reader = CHAT.from_strs(strs)
        headers = reader.headers()
        assert len(headers) == 2
        assert headers[0].languages == ["eng"]
        assert headers[1].languages == ["zho"]

    def test_languages_flat_multiple_files(self):
        strs = [
            "@UTF8\n@Begin\n@Languages:\teng\n"
            "@Participants:\tCHI Child Target_Child\n"
            "*CHI:\thi .\n@End\n",
            "@UTF8\n@Begin\n@Languages:\tzho\n"
            "@Participants:\tMOT Mother Mother\n"
            "*MOT:\tbye .\n@End\n",
        ]
        reader = CHAT.from_strs(strs)
        assert reader.languages() == ["eng", "zho"]

    def test_participants_flat_multiple_files(self):
        strs = [
            "@UTF8\n@Begin\n@Languages:\teng\n"
            "@Participants:\tCHI Child Target_Child\n"
            "*CHI:\thi .\n@End\n",
            "@UTF8\n@Begin\n@Languages:\tzho\n"
            "@Participants:\tMOT Mother Mother\n"
            "*MOT:\tbye .\n@End\n",
        ]
        reader = CHAT.from_strs(strs)
        parts = reader.participants()
        assert len(parts) == 2
        assert [p.code for p in parts] == ["CHI", "MOT"]


class TestHeadersRealFiles:
    def test_headers_from_dir(self, testchat_good_dir):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reader = CHAT.from_dir(str(testchat_good_dir), strict=False)
        headers = reader.headers()
        assert len(headers) == reader.n_files
        for h in headers:
            assert isinstance(h, Headers)

    def test_languages_from_dir(self, testchat_good_dir):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reader = CHAT.from_dir(str(testchat_good_dir), strict=False)
        langs = reader.languages(by_file=True)
        assert len(langs) == reader.n_files
        for file_langs in langs:
            assert isinstance(file_langs, list)

    def test_participants_from_dir(self, testchat_good_dir):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reader = CHAT.from_dir(str(testchat_good_dir), strict=False)
        parts = reader.participants(by_file=True)
        assert len(parts) == reader.n_files
        for file_parts in parts:
            assert isinstance(file_parts, list)
            for p in file_parts:
                assert isinstance(p, Participant)
                assert p.code  # non-empty


MISALIGNED_CHAT = (
    "@UTF8\n"
    "@Begin\n"
    "@Participants:\tCHI Child\n"
    "*CHI:\tI want cookie .\n"
    "%mor:\tpro|I v|want .\n"  # 2 non-clitic mor items vs 3 words + punct
    "@End\n"
)

WELL_FORMED_CHAT = (
    "@UTF8\n"
    "@Begin\n"
    "@Participants:\tCHI Child\n"
    "*CHI:\tI want cookie .\n"
    "%mor:\tpro|I v|want n|cookie .\n"
    "@End\n"
)


class TestStrictMode:
    def test_strict_true_raises(self):
        with pytest.raises(ValueError, match="misalignment"):
            CHAT.from_strs([MISALIGNED_CHAT], strict=True)

    def test_strict_default_raises(self):
        """Default strict=True should raise on misalignment."""
        with pytest.raises(ValueError, match="misalignment"):
            CHAT.from_strs([MISALIGNED_CHAT])

    def test_strict_true_error_message_details(self):
        with pytest.raises(ValueError, match="CHI") as exc_info:
            CHAT.from_strs([MISALIGNED_CHAT], strict=True)
        msg = str(exc_info.value)
        assert "%mor" in msg or "mor" in msg
        assert "strict=False" in msg

    def test_strict_false_warns(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            CHAT.from_strs([MISALIGNED_CHAT], strict=False)
            assert len(w) == 1
            assert issubclass(w[0].category, UserWarning)
            assert "misalignment" in str(w[0].message)

    def test_strict_false_empty_tokens(self):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            reader = CHAT.from_strs([MISALIGNED_CHAT], strict=False)
        utts = reader.utterances()
        assert len(utts) == 1
        assert utts[0].tokens == []

    def test_strict_false_preserves_tiers(self):
        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            reader = CHAT.from_strs([MISALIGNED_CHAT], strict=False)
        utts = reader.utterances()
        tiers = utts[0].tiers
        assert tiers is not None
        assert "CHI" in tiers
        assert "%mor" in tiers
        assert "cookie" in tiers["CHI"]

    def test_no_misalignment_strict_true_ok(self):
        reader = CHAT.from_strs([WELL_FORMED_CHAT], strict=True)
        utts = reader.utterances()
        assert len(utts) == 1
        assert utts[0].tokens is not None
        assert len(utts[0].tokens) == 4  # I, want, cookie, .

    def test_strict_parallel_false(self):
        with pytest.raises(ValueError, match="misalignment"):
            CHAT.from_strs([MISALIGNED_CHAT], parallel=False, strict=True)

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            reader = CHAT.from_strs([MISALIGNED_CHAT], parallel=False, strict=False)
        utts = reader.utterances()
        assert utts[0].tokens == []
        assert utts[0].tiers is not None
        assert "%mor" in utts[0].tiers


class TestDevelopmentalMeasures:
    def test_mlum_basic(self):
        reader = CHAT.from_strs([BASIC_CHAT])
        result = reader.mlum()
        assert result == [3.0]

    def test_mlu_aliases_mlum(self):
        reader = CHAT.from_strs([BASIC_CHAT])
        assert reader.mlu() == reader.mlum()

    def test_mluw_basic(self):
        reader = CHAT.from_strs([BASIC_CHAT])
        result = reader.mluw()
        assert result == [3.0]

    def test_ttr_basic(self):
        reader = CHAT.from_strs([BASIC_CHAT])
        result = reader.ttr()
        assert result == [1.0]

    def test_mlum_multiple_files(self):
        reader = CHAT.from_strs([BASIC_CHAT, BASIC_CHAT], ids=["a", "b"])
        result = reader.mlum()
        assert len(result) == 2
        assert result == [3.0, 3.0]

    def test_mluw_multiple_files(self):
        reader = CHAT.from_strs([BASIC_CHAT, BASIC_CHAT], ids=["a", "b"])
        result = reader.mluw()
        assert len(result) == 2
        assert result == [3.0, 3.0]

    def test_ttr_multiple_files(self):
        reader = CHAT.from_strs([BASIC_CHAT, BASIC_CHAT], ids=["a", "b"])
        result = reader.ttr()
        assert len(result) == 2
        assert result == [1.0, 1.0]

    def test_mlum_empty(self):
        reader = CHAT.from_strs([])
        assert reader.mlum() == []

    def test_mluw_empty(self):
        reader = CHAT.from_strs([])
        assert reader.mluw() == []

    def test_ttr_empty(self):
        reader = CHAT.from_strs([])
        assert reader.ttr() == []

    def test_ttr_repeated_words(self):
        chat_str = (
            "@UTF8\n@Begin\n"
            "@Participants:\tCHI Child\n"
            "*CHI:\tno no no .\n"
            "%mor:\tco|no co|no co|no .\n"
            "@End\n"
        )
        reader = CHAT.from_strs([chat_str])
        result = reader.ttr()
        assert result == pytest.approx([1.0 / 3.0])

    def test_mlum_participant(self):
        reader = CHAT.from_strs([BASIC_CHAT])
        assert reader.mlum(participant="MOT") == [1.0]

    def test_mluw_participant(self):
        reader = CHAT.from_strs([BASIC_CHAT])
        assert reader.mluw(participant="MOT") == [1.0]

    def test_mlum_n_none(self):
        reader = CHAT.from_strs([BASIC_CHAT])
        assert reader.mlum(n=None) == [3.0]

    def test_mluw_n_none(self):
        reader = CHAT.from_strs([BASIC_CHAT])
        assert reader.mluw(n=None) == [3.0]

    def test_mlum_n_truncation(self):
        chat_str = (
            "@UTF8\n@Begin\n"
            "@Participants:\tCHI Child\n"
            "*CHI:\tI want cookie .\n"
            "%mor:\tpro|I v|want n|cookie .\n"
            "*CHI:\tno .\n"
            "%mor:\tco|no .\n"
            "@End\n"
        )
        reader = CHAT.from_strs([chat_str])
        # All CHI utterances: (3 + 1) / 2 = 2.0
        assert reader.mlum(n=None) == [2.0]
        # First 1 utterance only: 3 / 1 = 3.0
        assert reader.mlum(n=1) == [3.0]

    def test_mluw_n_truncation(self):
        chat_str = (
            "@UTF8\n@Begin\n"
            "@Participants:\tCHI Child\n"
            "*CHI:\tI want cookie .\n"
            "%mor:\tpro|I v|want n|cookie .\n"
            "*CHI:\tno .\n"
            "%mor:\tco|no .\n"
            "@End\n"
        )
        reader = CHAT.from_strs([chat_str])
        # All CHI utterances: (3 + 1) / 2 = 2.0
        assert reader.mluw(n=None) == [2.0]
        # First 1 utterance only: 3 / 1 = 3.0
        assert reader.mluw(n=1) == [3.0]

    def test_ttr_participant(self):
        reader = CHAT.from_strs([BASIC_CHAT])
        # MOT says "no" -> 1 unique / 1 total = 1.0
        assert reader.ttr(participant="MOT") == [1.0]

    def test_ttr_n_none(self):
        reader = CHAT.from_strs([BASIC_CHAT])
        assert reader.ttr(n=None) == [1.0]

    def test_ttr_n_truncation(self):
        chat_str = (
            "@UTF8\n@Begin\n"
            "@Participants:\tCHI Child\n"
            "*CHI:\tno no no yes yes .\n"
            "%mor:\tco|no co|no co|no co|yes co|yes .\n"
            "@End\n"
        )
        reader = CHAT.from_strs([chat_str])
        # All 5 tokens: 2 unique / 5 total = 0.4
        assert reader.ttr(n=None) == pytest.approx([2.0 / 5.0])
        # First 3 tokens ("no", "no", "no"): 1 unique / 3 total = 1/3
        assert reader.ttr(n=3) == pytest.approx([1.0 / 3.0])

    def test_measures_with_filter(self):
        reader = CHAT.from_strs([BASIC_CHAT])
        filtered = reader.filter(participants="CHI")
        assert filtered.mlum() == [3.0]
        assert filtered.mluw() == [3.0]
        assert filtered.ttr() == [1.0]

    def test_measures_return_type(self):
        reader = CHAT.from_strs([BASIC_CHAT])
        assert isinstance(reader.mlum(), list)
        assert isinstance(reader.mluw(), list)
        assert isinstance(reader.ttr(), list)
        assert all(isinstance(v, float) for v in reader.mlum())


class TestAges:
    def test_ages_basic(self):
        reader = CHAT.from_strs([HEADER_CHAT])
        result = reader.ages()
        assert len(result) == 1
        age = result[0]
        assert isinstance(age, Age)
        assert age.years == 2
        assert age.months == 10
        assert age.days == 5

    def test_ages_no_chi(self):
        chat_str = (
            "@UTF8\n@Begin\n" "@Participants:\tMOT Mother\n" "*MOT:\thello .\n" "@End\n"
        )
        reader = CHAT.from_strs([chat_str])
        assert reader.ages() == [None]

    def test_ages_empty(self):
        reader = CHAT.from_strs([])
        assert reader.ages() == []

    def test_ages_multiple_files(self):
        reader = CHAT.from_strs([HEADER_CHAT, HEADER_CHAT], ids=["a", "b"])
        result = reader.ages()
        assert len(result) == 2
        assert result[0] is not None and result[0].years == 2
        assert result[1] is not None and result[1].years == 2


class TestWordNgrams:
    def test_word_ngrams_unigrams(self):
        reader = CHAT.from_strs([BASIC_CHAT])
        counter = reader.word_ngrams(1)
        # BASIC_CHAT: "*CHI: I want cookie ." and "*MOT: no ."
        # Words: I, want, cookie, ., no, .
        assert counter.get(["I"]) == 1
        assert counter.get(["want"]) == 1
        assert counter.get(["cookie"]) == 1
        assert counter.get(["no"]) == 1

    def test_word_ngrams_bigrams(self):
        reader = CHAT.from_strs([BASIC_CHAT])
        counter = reader.word_ngrams(2)
        assert counter.get(["I", "want"]) == 1
        assert counter.get(["want", "cookie"]) == 1

    def test_word_ngrams_returns_ngrams(self):
        from rustling.ngram import Ngrams

        reader = CHAT.from_strs([BASIC_CHAT])
        counter = reader.word_ngrams(1)
        assert isinstance(counter, Ngrams)
        assert counter.n == 1

    def test_word_ngrams_no_cross_utterance(self):
        reader = CHAT.from_strs([BASIC_CHAT])
        counter = reader.word_ngrams(2)
        # "." ends first utterance, "no" starts second. Should not have (".", "no").
        assert counter.get([".", "no"]) == 0

    def test_word_ngrams_empty_reader(self):
        reader = CHAT.from_strs([])
        counter = reader.word_ngrams(1)
        assert len(counter) == 0
        assert counter.total() == 0

    def test_word_ngrams_multiple_files(self):
        reader = CHAT.from_strs(
            [
                "@UTF8\n@Begin\n*CHI:\thi .\n@End\n",
                "@UTF8\n@Begin\n*MOT:\thi .\n@End\n",
            ],
            ids=["a", "b"],
        )
        counter = reader.word_ngrams(1)
        assert counter.get(["hi"]) == 2


class TestHeadTail:
    def test_head_returns_utterances(self):
        reader = CHAT.from_strs([BASIC_CHAT])
        result = reader.head()
        assert type(result).__name__ == "Utterances"

    def test_head_repr_displays_formatted(self):
        reader = CHAT.from_strs([BASIC_CHAT])
        result = reader.head()
        text = repr(result)
        assert "*CHI:" in text
        assert "*MOT:" in text
        # repr should contain actual newlines, not escaped ones
        assert "\n" in text

    def test_head_str_matches_repr(self):
        reader = CHAT.from_strs([BASIC_CHAT])
        result = reader.head()
        assert str(result) == repr(result)

    def test_head_len(self):
        reader = CHAT.from_strs([BASIC_CHAT])
        assert len(reader.head()) == 2
        assert len(reader.head(1)) == 1

    def test_head_getitem(self):
        reader = CHAT.from_strs([BASIC_CHAT])
        result = reader.head()
        assert isinstance(result[0], Utterance)
        assert result[0].participant == "CHI"
        assert result[-1].participant == "MOT"

    def test_head_getitem_out_of_range(self):
        reader = CHAT.from_strs([BASIC_CHAT])
        result = reader.head(1)
        with pytest.raises(IndexError):
            result[5]

    def test_head_iter(self):
        reader = CHAT.from_strs([BASIC_CHAT])
        result = reader.head()
        participants = [u.participant for u in result]
        assert participants == ["CHI", "MOT"]

    def test_head_n_1(self):
        reader = CHAT.from_strs([BASIC_CHAT])
        result = reader.head(1)
        text = repr(result)
        assert "*CHI:" in text
        assert "*MOT:" not in text

    def test_tail_n_1(self):
        reader = CHAT.from_strs([BASIC_CHAT])
        result = reader.tail(1)
        text = repr(result)
        assert "*MOT:" in text
        assert "*CHI:" not in text

    def test_head_with_mor_and_gra(self):
        reader = CHAT.from_strs([BASIC_CHAT])
        text = repr(reader.head(1))
        assert "%mor:" in text
        assert "%gra:" in text
        assert "pro|I" in text
        assert "1|2|SUBJ" in text

    def test_head_no_mor(self):
        chat_str = "@UTF8\n@Begin\n*CHI:\thello world .\n@End\n"
        reader = CHAT.from_strs([chat_str])
        text = repr(reader.head())
        assert "*CHI:" in text
        assert "%mor:" not in text
        assert "%gra:" not in text

    def test_head_empty_reader(self):
        reader = CHAT.from_strs([])
        result = reader.head()
        assert len(result) == 0
        assert repr(result) == ""

    def test_tail_empty_reader(self):
        reader = CHAT.from_strs([])
        result = reader.tail()
        assert len(result) == 0
        assert repr(result) == ""

    def test_head_separation(self):
        reader = CHAT.from_strs([BASIC_CHAT])
        text = repr(reader.head())
        assert "\n\n" in text

    def test_utterance_to_str(self):
        reader = CHAT.from_strs([BASIC_CHAT])
        utts = reader.utterances()
        s = utts[0].to_str()
        assert isinstance(s, str)
        assert "*CHI:" in s

    def test_head_time_marks(self):
        chat_str = (
            "@UTF8\n@Begin\n@Participants:\tCHI Child\n"
            "*CHI:\thi . \x150_1500\x15\n@End\n"
        )
        reader = CHAT.from_strs([chat_str])
        text = repr(reader.head())
        assert "0" in text
        assert "1500" in text
        assert "ms" in text

    def test_head_column_alignment(self):
        reader = CHAT.from_strs([BASIC_CHAT])
        text = repr(reader.head(1))
        lines = text.strip().split("\n")
        # participant, %mor, %gra rows should all be present
        assert len(lines) >= 3
        # All lines should start with a label of the same padded width
        first_data_offsets = []
        for line in lines:
            # Find the position after the label where data starts
            idx = line.find("  ")
            if idx >= 0:
                first_data_offsets.append(idx)
        # All offsets should be the same
        assert len(set(first_data_offsets)) == 1

    def test_head_repr_html(self):
        reader = CHAT.from_strs([BASIC_CHAT])
        result = reader.head()
        html = result._repr_html_()
        assert "<table" in html
        assert "*CHI:" in html


class TestChangeableHeaders:
    """Test that changeable headers appear interleaved with utterances."""

    def test_changeable_headers_in_utterances(self):
        reader = CHAT.from_strs([HEADER_CHAT])
        utts = reader.utterances()
        # HEADER_CHAT has 3 real utterances + 2 mid-file changeable headers
        # (@New Episode and @Comment: Child laughs)
        # File-level headers (@Comment: First recording, etc.) are NOT included.
        assert len(utts) == 5

    def test_ordering_preserved(self):
        reader = CHAT.from_strs([HEADER_CHAT])
        utts = reader.utterances()
        # Expected mid-file order:
        # 0: *CHI: hello .
        # 1: @New Episode
        # 2: *MOT: hi .
        # 3: @Comment: Child laughs
        # 4: *CHI: cookie .
        assert utts[0].participant == "CHI"
        assert utts[0].changeable_header is None
        assert isinstance(utts[1].changeable_header, ChangeableHeader.NewEpisode)
        assert utts[2].participant == "MOT"
        assert isinstance(utts[3].changeable_header, ChangeableHeader.Comment)
        assert utts[3].changeable_header.value == "Child laughs"
        assert utts[4].participant == "CHI"

    def test_changeable_header_fields_none(self):
        reader = CHAT.from_strs([HEADER_CHAT])
        utts = reader.utterances()
        headers = [u for u in utts if u.changeable_header is not None]
        assert len(headers) == 2
        for h in headers:
            assert h.participant is None
            assert h.tokens is None
            assert h.tiers is None

    def test_real_utterance_has_no_changeable_header(self):
        reader = CHAT.from_strs([HEADER_CHAT])
        utts = reader.utterances()
        real = [u for u in utts if u.changeable_header is None]
        assert len(real) == 3
        for u in real:
            assert u.participant is not None
            assert u.tokens is not None
            assert u.tiers is not None

    def test_file_level_headers_excluded(self):
        reader = CHAT.from_strs([HEADER_CHAT])
        utts = reader.utterances()
        # The file-level @Comment headers should NOT appear in utterances.
        comments = [
            u
            for u in utts
            if u.changeable_header is not None
            and isinstance(u.changeable_header, ChangeableHeader.Comment)
        ]
        # Only the mid-file @Comment: Child laughs, not the file-level ones.
        assert len(comments) == 1
        assert comments[0].changeable_header.value == "Child laughs"

    def test_words_skips_headers(self):
        reader = CHAT.from_strs([HEADER_CHAT])
        words = reader.words()
        assert "hello" in words
        assert "hi" in words
        assert "cookie" in words
        # No changeable header content in words
        assert len(words) == 6  # hello, ., hi, ., cookie, .

    def test_filter_drops_changeable_headers(self):
        reader = CHAT.from_strs([HEADER_CHAT])
        filtered = reader.filter(participants="CHI")
        utts = filtered.utterances()
        # Only CHI utterances (hello, cookie); changeable headers dropped
        assert len(utts) == 2
        assert all(u.participant == "CHI" for u in utts)
        assert all(u.changeable_header is None for u in utts)


BASIC_CHAT_2 = (
    "@UTF8\n"
    "@Begin\n"
    "@Participants:\tMOT Mother\n"
    "*MOT:\tgood morning .\n"
    "%mor:\tadj|good n|morning .\n"
    "%gra:\t1|2|MOD 2|0|ROOT 3|2|PUNCT\n"
    "@End\n"
)


class TestInfo:
    def test_info_single_file(self, capsys):
        reader = CHAT.from_strs([BASIC_CHAT])
        reader.info()
        output = capsys.readouterr().out
        lines = output.strip().split("\n")
        assert lines[0] == "1 files"
        assert lines[1] == "2 utterances"
        assert lines[2] == "6 words"
        # Single file: no table
        assert len(lines) == 3

    def test_info_multiple_files(self, capsys):
        reader = CHAT.from_strs([BASIC_CHAT, BASIC_CHAT_2], ids=["a.cha", "b.cha"])
        reader.info()
        output = capsys.readouterr().out
        lines = output.strip().split("\n")
        assert lines[0] == "2 files"
        assert lines[1] == "3 utterances"
        assert lines[2] == "9 words"
        # Table header, separator, 2 data rows, truncation
        assert "Utterance Count" in lines[3]
        assert "Word Count" in lines[3]
        assert "File Path" in lines[3]
        # Separator line (all dashes and spaces)
        assert set(lines[4].strip()) <= {"-", " "}
        # Data rows
        assert "#1" in lines[5]
        assert "a.cha" in lines[5]
        assert "#2" in lines[6]
        assert "b.cha" in lines[6]

    def test_info_verbose(self, capsys):
        files = [BASIC_CHAT] * 7
        ids = [f"file{i}.cha" for i in range(7)]
        reader = CHAT.from_strs(files, ids=ids)
        reader.info(verbose=True)
        output = capsys.readouterr().out
        assert "7 files" in output
        assert "#7" in output
        assert "verbose" not in output

    def test_info_not_verbose_truncates(self, capsys):
        files = [BASIC_CHAT] * 7
        ids = [f"file{i}.cha" for i in range(7)]
        reader = CHAT.from_strs(files, ids=ids)
        reader.info()
        output = capsys.readouterr().out
        assert "#5" in output
        assert "#6" not in output
        assert "set `verbose` to True for all the files" in output

    def test_info_empty(self, capsys):
        reader = CHAT.from_strs([])
        reader.info()
        output = capsys.readouterr().out
        lines = output.strip().split("\n")
        assert lines[0] == "0 files"
        assert lines[1] == "0 utterances"
        assert lines[2] == "0 words"

    def test_no_changeable_headers_in_basic(self):
        reader = CHAT.from_strs([BASIC_CHAT])
        utts = reader.utterances()
        assert len(utts) == 2
        assert all(u.changeable_header is None for u in utts)

    def test_to_str_changeable_header(self):
        reader = CHAT.from_strs([HEADER_CHAT])
        utts = reader.utterances()
        # @New Episode
        assert utts[1].to_str() == "@New Episode"
        # @Comment: Child laughs
        assert utts[3].to_str() == "@Comment:\tChild laughs"

    def test_head_includes_headers(self):
        reader = CHAT.from_strs([HEADER_CHAT])
        result = reader.head(10)
        has_header = any(u.changeable_header is not None for u in result)
        assert has_header


class TestIPSyn:
    def test_ipsyn_basic(self):
        # BASIC_CHAT has "I want cookie ." with SUBJ/OBJ/ROOT GRA
        reader = CHAT.from_strs([BASIC_CHAT])
        result = reader.ipsyn()
        assert len(result) == 1
        # Should score: N1 (noun), N2 (pronoun), V1 (verb), S1 (2+ words),
        # S2 (subj-verb -> credits S1), S3 (verb-obj -> credits S1),
        # S4 (SVO -> credits S2, S3), plus more
        assert result[0] > 0

    def test_ipsyn_empty(self):
        reader = CHAT.from_strs([])
        assert reader.ipsyn() == []

    def test_ipsyn_no_matching_participant(self):
        reader = CHAT.from_strs([BASIC_CHAT])
        assert reader.ipsyn(participant="INV") == [0]

    def test_ipsyn_participant(self):
        reader = CHAT.from_strs([BASIC_CHAT])
        # MOT only says "no ." — not much to score
        result = reader.ipsyn(participant="MOT")
        assert len(result) == 1
        assert result[0] < reader.ipsyn()[0]

    def test_ipsyn_n_truncation(self):
        chat_str = (
            "@UTF8\n@Begin\n"
            "@Participants:\tCHI Child\n"
            "*CHI:\tI want cookie .\n"
            "%mor:\tpro|I v|want n|cookie .\n"
            "%gra:\t1|2|SUBJ 2|0|ROOT 3|2|OBJ 4|2|PUNCT\n"
            "*CHI:\tthe big dog is running .\n"
            "%mor:\tdet|the adj|big n|dog cop|be&3S v|run-PRESP .\n"
            "%gra:\t1|3|DET 2|3|MOD 3|4|SUBJ 4|0|ROOT 5|4|PRED 6|4|PUNCT\n"
            "@End\n"
        )
        reader = CHAT.from_strs([chat_str])
        score_all = reader.ipsyn(n=None)[0]
        score_one = reader.ipsyn(n=1)[0]
        # More utterances should give >= score
        assert score_all >= score_one

    def test_ipsyn_n_none(self):
        reader = CHAT.from_strs([BASIC_CHAT])
        result = reader.ipsyn(n=None)
        assert len(result) == 1

    def test_ipsyn_multiple_files(self):
        reader = CHAT.from_strs([BASIC_CHAT, BASIC_CHAT], ids=["a", "b"])
        result = reader.ipsyn()
        assert len(result) == 2
        assert result[0] == result[1]

    def test_ipsyn_return_type(self):
        reader = CHAT.from_strs([BASIC_CHAT])
        result = reader.ipsyn()
        assert isinstance(result, list)
        assert all(isinstance(v, int) for v in result)

    def test_ipsyn_max_score_bounded(self):
        reader = CHAT.from_strs([BASIC_CHAT])
        result = reader.ipsyn()
        assert 0 <= result[0] <= 112

    def test_ipsyn_noun_phrase_items(self):
        # Test N1 (noun), N4 (det/mod + noun), N5 (article + noun, credits N4)
        chat_str = (
            "@UTF8\n@Begin\n"
            "@Participants:\tCHI Child\n"
            "*CHI:\tthe cookie .\n"
            "%mor:\tdet|the n|cookie .\n"
            "%gra:\t1|2|DET 2|0|ROOT 3|2|PUNCT\n"
            "*CHI:\tthe cookie .\n"
            "%mor:\tdet|the n|cookie .\n"
            "%gra:\t1|2|DET 2|0|ROOT 3|2|PUNCT\n"
            "@End\n"
        )
        reader = CHAT.from_strs([chat_str])
        result = reader.ipsyn()[0]
        # N1 (noun) = 2, N5 (det+noun) = 2, N4 (credited by N5) = 2, S1 = 2
        assert result >= 8

    def test_ipsyn_verb_items(self):
        # Test V1 (verb), V7 (progressive -PRESP), V12 (past -PAST)
        chat_str = (
            "@UTF8\n@Begin\n"
            "@Participants:\tCHI Child\n"
            "*CHI:\the is running .\n"
            "%mor:\tpro|he cop|be&3S v|run-PRESP .\n"
            "%gra:\t1|2|SUBJ 2|0|ROOT 3|2|PRED 4|2|PUNCT\n"
            "*CHI:\the walked .\n"
            "%mor:\tpro|he v|walk-PAST .\n"
            "%gra:\t1|2|SUBJ 2|0|ROOT 3|2|PUNCT\n"
            "@End\n"
        )
        reader = CHAT.from_strs([chat_str])
        result = reader.ipsyn()[0]
        # V1 (verb) >= 1, V7 (progressive) >= 1, V12 (past) >= 1
        assert result >= 3

    def test_ipsyn_question_items(self):
        # Test Q1 (intonation question), Q9 (why/when/which/whose)
        chat_str = (
            "@UTF8\n@Begin\n"
            "@Participants:\tCHI Child\n"
            "*CHI:\tis it good ?\n"
            "%mor:\tcop|be&3S pro|it adj|good ?\n"
            "%gra:\t1|0|ROOT 2|1|SUBJ 3|1|PRED 4|1|PUNCT\n"
            "*CHI:\twhy ?\n"
            "%mor:\tadv:wh|why ?\n"
            "%gra:\t1|0|ROOT 2|1|PUNCT\n"
            "@End\n"
        )
        reader = CHAT.from_strs([chat_str])
        result = reader.ipsyn()[0]
        # Q1 (intonation question) >= 1, Q9 (why) >= 1
        assert result >= 2

    def test_ipsyn_no_mor_no_gra(self):
        chat_str = (
            "@UTF8\n@Begin\n"
            "@Participants:\tCHI Child\n"
            "*CHI:\thello world .\n"
            "@End\n"
        )
        reader = CHAT.from_strs([chat_str])
        # No %mor/%gra means tokens are empty, should get 0
        assert reader.ipsyn() == [0]

    def test_ipsyn_svo_scores_s4_and_credits(self):
        # "I want cookie" has S-V-O structure
        # S4 credits S2 and S3; S2 credits S1; S3 credits S1
        chat_str = (
            "@UTF8\n@Begin\n"
            "@Participants:\tCHI Child\n"
            "*CHI:\tI want cookie .\n"
            "%mor:\tpro|I v|want n|cookie .\n"
            "%gra:\t1|2|SUBJ 2|0|ROOT 3|2|OBJ 4|2|PUNCT\n"
            "*CHI:\tI eat food .\n"
            "%mor:\tpro|I v|eat n|food .\n"
            "%gra:\t1|2|SUBJ 2|0|ROOT 3|2|OBJ 4|2|PUNCT\n"
            "@End\n"
        )
        reader = CHAT.from_strs([chat_str])
        result = reader.ipsyn()[0]
        # S4(2) + S2(2) + S3(2) + S1(2) + N1(2) + N2(2) + V1(2) = 14 minimum
        assert result >= 14


# ---------------------------------------------------------------------------
# Tests for mor_tier / gra_tier kwargs
# ---------------------------------------------------------------------------

CHAT_WITH_XMOR = (
    "@UTF8\n"
    "@Begin\n"
    "@Participants:\tCHI Child\n"
    "*CHI:\tI want cookie .\n"
    "%xmor:\tpro|I v|want n|cookie .\n"
    "%xgra:\t1|2|SUBJ 2|0|ROOT 3|2|OBJ 4|2|PUNCT\n"
    "@End\n"
)


class TestMorGraTierKwargs:
    def test_custom_tier_names(self):
        reader = CHAT.from_strs([CHAT_WITH_XMOR], mor_tier="xmor", gra_tier="xgra")
        tokens = reader.tokens()
        assert tokens[0].pos == "pro"
        assert tokens[0].mor == "I"
        assert tokens[0].gra == Gra(dep=1, head=2, rel="SUBJ")

    def test_custom_tiers_default_ignores(self):
        """Default tiers should not pick up %xmor/%xgra data."""
        reader = CHAT.from_strs([CHAT_WITH_XMOR])
        tokens = reader.tokens()
        assert tokens[0].mor is None
        assert tokens[0].gra is None

    def test_none_mor_disables_both(self):
        reader = CHAT.from_strs([BASIC_CHAT], mor_tier=None)
        tokens = reader.tokens()
        assert tokens[0].mor is None
        assert tokens[0].gra is None

    def test_none_gra_disables_both(self):
        reader = CHAT.from_strs([BASIC_CHAT], gra_tier=None)
        tokens = reader.tokens()
        assert tokens[0].mor is None
        assert tokens[0].gra is None

    def test_both_none_disables(self):
        reader = CHAT.from_strs([BASIC_CHAT], mor_tier=None, gra_tier=None)
        tokens = reader.tokens()
        assert tokens[0].mor is None
        assert tokens[0].gra is None

    def test_utterance_tier_names_custom(self):
        reader = CHAT.from_strs([CHAT_WITH_XMOR], mor_tier="xmor", gra_tier="xgra")
        utts = reader.utterances()
        assert utts[0].mor_tier_name == "%xmor"
        assert utts[0].gra_tier_name == "%xgra"

    def test_utterance_tier_names_default(self):
        reader = CHAT.from_strs([BASIC_CHAT])
        utts = reader.utterances()
        assert utts[0].mor_tier_name == "%mor"
        assert utts[0].gra_tier_name == "%gra"

    def test_utterance_tier_names_disabled(self):
        reader = CHAT.from_strs([BASIC_CHAT], mor_tier=None, gra_tier=None)
        utts = reader.utterances()
        assert utts[0].mor_tier_name is None
        assert utts[0].gra_tier_name is None

    def test_to_strs_roundtrip_custom_tiers(self):
        reader = CHAT.from_strs([CHAT_WITH_XMOR], mor_tier="xmor", gra_tier="xgra")
        strs = reader.to_strs()
        assert "%xmor:" in strs[0]
        assert "%xgra:" in strs[0]
        assert "%mor:" not in strs[0]
        assert "%gra:" not in strs[0]
