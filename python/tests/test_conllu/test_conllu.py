import glob

import pytest

from rustling.conllu import CoNLLU, Sentence, Token, read_conllu

SAMPLE_CONLLU = (
    "# sent_id = 1\n"
    "# text = The cat sat on the mat.\n"
    "1\tThe\tthe\tDET\tDT\tDefinite=Def|PronType=Art\t2\tdet\t_\t_\n"
    "2\tcat\tcat\tNOUN\tNN\tNumber=Sing\t3\tnsubj\t_\t_\n"
    "3\tsat\tsit\tVERB\tVBD\tMood=Ind|Tense=Past\t0\troot\t_\t_\n"
    "4\ton\ton\tADP\tIN\t_\t6\tcase\t_\t_\n"
    "5\tthe\tthe\tDET\tDT\tDefinite=Def|PronType=Art\t6\tdet\t_\t_\n"
    "6\tmat\tmat\tNOUN\tNN\tNumber=Sing\t3\tnmod\t_\t_\n"
    "7\t.\t.\tPUNCT\t.\t_\t3\tpunct\t_\tSpaceAfter=No\n"
    "\n"
    "# sent_id = 2\n"
    "# text = I like it.\n"
    "1\tI\tI\tPRON\tPRP\tCase=Nom|Number=Sing|Person=1|PronType=Prs\t2\tnsubj\t_\t_\n"
    "2\tlike\tlike\tVERB\tVBP\tMood=Ind|Number=Sing|Person=1|Tense=Pres\t0\troot\t_\t_\n"  # noqa: E501
    "3\tit\tit\tPRON\tPRP\tCase=Acc|Gender=Neut|Number=Sing|Person=3|PronType=Prs\t2\tobj\t_\tSpaceAfter=No\n"  # noqa: E501
    "4\t.\t.\tPUNCT\t.\t_\t2\tpunct\t_\tSpaceAfter=No\n"
)


class TestCoNLLUFromStrs:
    def test_basic_parsing(self):
        reader = CoNLLU.from_strs([SAMPLE_CONLLU])
        assert reader.n_files == 1

    def test_sentences(self):
        reader = CoNLLU.from_strs([SAMPLE_CONLLU])
        sentences = reader.sentences()
        assert len(sentences) == 2

        s1 = sentences[0]
        assert isinstance(s1, Sentence)
        assert s1.comments is not None
        assert len(s1.comments) == 2
        assert s1.comments[0] == "sent_id = 1"
        assert s1.comments[1] == "text = The cat sat on the mat."

    def test_tokens(self):
        reader = CoNLLU.from_strs([SAMPLE_CONLLU])
        sentences = reader.sentences()
        tokens = sentences[0].tokens()
        assert len(tokens) == 7

        t0 = tokens[0]
        assert isinstance(t0, Token)
        assert t0.id == "1"
        assert t0.form == "The"
        assert t0.lemma == "the"
        assert t0.upos == "DET"
        assert t0.xpos == "DT"
        assert t0.feats == "Definite=Def|PronType=Art"
        assert t0.head == "2"
        assert t0.deprel == "det"
        assert t0.deps == "_"
        assert t0.misc == "_"

    def test_token_misc(self):
        reader = CoNLLU.from_strs([SAMPLE_CONLLU])
        tokens = reader.sentences()[0].tokens()
        assert tokens[6].misc == "SpaceAfter=No"

    def test_no_comments(self):
        conllu_str = "1\tHello\thello\tNOUN\t_\t_\t0\troot\t_\t_\n"
        reader = CoNLLU.from_strs([conllu_str])
        sentences = reader.sentences()
        assert len(sentences) == 1
        assert sentences[0].comments is None

    def test_multiple_files(self):
        reader = CoNLLU.from_strs(
            [SAMPLE_CONLLU, SAMPLE_CONLLU],
            ids=["file1.conllu", "file2.conllu"],
        )
        assert reader.n_files == 2
        assert reader.file_paths == ["file1.conllu", "file2.conllu"]
        assert len(reader.sentences()) == 4

    def test_empty_input(self):
        reader = CoNLLU.from_strs([""])
        assert reader.n_files == 1
        assert len(reader.sentences()) == 0


class TestCoNLLURoundTrip:
    def test_to_strs_round_trip(self):
        reader = CoNLLU.from_strs([SAMPLE_CONLLU])
        strs = reader.to_strs()
        reader2 = CoNLLU.from_strs(strs)
        assert len(reader.sentences()) == len(reader2.sentences())
        for s1, s2 in zip(reader.sentences(), reader2.sentences()):
            assert s1 == s2


class TestCoNLLUCollectionOps:
    def test_append(self):
        r1 = CoNLLU.from_strs([SAMPLE_CONLLU], ids=["a.conllu"])
        r2 = CoNLLU.from_strs([SAMPLE_CONLLU], ids=["b.conllu"])
        r1.append(r2)
        assert r1.n_files == 2
        assert r1.file_paths == ["a.conllu", "b.conllu"]

    def test_append_left(self):
        r1 = CoNLLU.from_strs([SAMPLE_CONLLU], ids=["a.conllu"])
        r2 = CoNLLU.from_strs([SAMPLE_CONLLU], ids=["b.conllu"])
        r1.append_left(r2)
        assert r1.file_paths == ["b.conllu", "a.conllu"]

    def test_pop(self):
        r = CoNLLU.from_strs(
            [SAMPLE_CONLLU, SAMPLE_CONLLU],
            ids=["a.conllu", "b.conllu"],
        )
        popped = r.pop()
        assert popped.file_paths == ["b.conllu"]
        assert r.n_files == 1

    def test_pop_left(self):
        r = CoNLLU.from_strs(
            [SAMPLE_CONLLU, SAMPLE_CONLLU],
            ids=["a.conllu", "b.conllu"],
        )
        popped = r.pop_left()
        assert popped.file_paths == ["a.conllu"]
        assert r.n_files == 1

    def test_clear(self):
        r = CoNLLU.from_strs([SAMPLE_CONLLU])
        r.clear()
        assert r.n_files == 0
        assert not r

    def test_add(self):
        r1 = CoNLLU.from_strs([SAMPLE_CONLLU], ids=["a.conllu"])
        r2 = CoNLLU.from_strs([SAMPLE_CONLLU], ids=["b.conllu"])
        r3 = r1 + r2
        assert r3.n_files == 2
        assert r1.n_files == 1  # original unchanged

    def test_getitem(self):
        r = CoNLLU.from_strs(
            [SAMPLE_CONLLU, SAMPLE_CONLLU],
            ids=["a.conllu", "b.conllu"],
        )
        assert r[0].file_paths == ["a.conllu"]
        assert r[-1].file_paths == ["b.conllu"]

    def test_iter(self):
        r = CoNLLU.from_strs(
            [SAMPLE_CONLLU, SAMPLE_CONLLU],
            ids=["a.conllu", "b.conllu"],
        )
        paths = [item.file_paths[0] for item in r]
        assert paths == ["a.conllu", "b.conllu"]

    def test_bool(self):
        assert CoNLLU.from_strs([SAMPLE_CONLLU])
        assert not CoNLLU()

    def test_repr(self):
        r = CoNLLU.from_strs([SAMPLE_CONLLU])
        assert "1 file" in repr(r)

    def test_eq(self):
        r1 = CoNLLU.from_strs([SAMPLE_CONLLU], ids=["a.conllu"])
        r2 = CoNLLU.from_strs([SAMPLE_CONLLU], ids=["a.conllu"])
        assert r1 == r2


class TestCoNLLUToCHAT:
    def test_to_chat(self):
        reader = CoNLLU.from_strs([SAMPLE_CONLLU])
        chat = reader.to_chat()
        assert chat.n_files == 1
        utts = chat.utterances()
        assert len(utts) == 2

    def test_to_chat_strs(self):
        reader = CoNLLU.from_strs([SAMPLE_CONLLU])
        strs = reader.to_chat_strs()
        assert len(strs) == 1
        assert "@UTF8" in strs[0]
        assert "@Begin" in strs[0]
        assert "@End" in strs[0]


class TestReadConllu:
    def test_read_conllu_with_strs(self, tmp_path):
        conllu_file = tmp_path / "test.conllu"
        conllu_file.write_text(SAMPLE_CONLLU)
        reader = read_conllu(conllu_file)
        assert reader.n_files == 1
        assert len(reader.sentences()) == 2

    def test_read_conllu_from_dir(self, tmp_path):
        conllu_file = tmp_path / "test.conllu"
        conllu_file.write_text(SAMPLE_CONLLU)
        reader = read_conllu(tmp_path)
        assert reader.n_files == 1

    def test_read_conllu_invalid_path(self):
        with pytest.raises(ValueError):
            read_conllu("not_a_valid_source.txt")

    def test_read_conllu_invalid_cls(self):
        with pytest.raises(TypeError):
            read_conllu("test.conllu", cls=str)


class TestUDEnglishEWT:
    def test_parse_test_file(self, ud_english_ewt_dir):
        test_files = glob.glob(str(ud_english_ewt_dir / "*test.conllu"))
        assert len(test_files) > 0, "No test.conllu files found"
        reader = CoNLLU.from_files(test_files)
        assert reader.n_files == len(test_files)
        sentences = reader.sentences()
        assert len(sentences) > 0
        # Verify tokens are properly parsed.
        for sentence in sentences[:5]:
            tokens = sentence.tokens()
            assert len(tokens) > 0
            for token in tokens:
                assert token.form
                assert token.id

    def test_from_dir(self, ud_english_ewt_dir):
        reader = CoNLLU.from_dir(
            ud_english_ewt_dir,
            match="test",
        )
        assert reader.n_files > 0
        assert len(reader.sentences()) > 0

    def test_round_trip(self, ud_english_ewt_dir):
        test_files = glob.glob(str(ud_english_ewt_dir / "*test.conllu"))
        reader = CoNLLU.from_files(test_files)
        strs = reader.to_strs()
        reader2 = CoNLLU.from_strs(strs)
        assert len(reader.sentences()) == len(reader2.sentences())

    def test_to_chat_conversion(self, ud_english_ewt_dir):
        test_files = glob.glob(str(ud_english_ewt_dir / "*test.conllu"))
        reader = CoNLLU.from_files(test_files)
        chat = reader.to_chat()
        assert chat.n_files == len(test_files)
        utts = chat.utterances()
        assert len(utts) > 0


class TestUDCantoneseHK:
    def test_parse_test_file(self, ud_cantonese_hk_dir):
        test_files = glob.glob(str(ud_cantonese_hk_dir / "*test.conllu"))
        assert len(test_files) > 0, "No test.conllu files found"
        reader = CoNLLU.from_files(test_files)
        assert reader.n_files == len(test_files)
        sentences = reader.sentences()
        assert len(sentences) > 0

    def test_from_dir(self, ud_cantonese_hk_dir):
        reader = CoNLLU.from_dir(
            ud_cantonese_hk_dir,
            match="test",
        )
        assert reader.n_files > 0
        assert len(reader.sentences()) > 0

    def test_to_chat_conversion(self, ud_cantonese_hk_dir):
        test_files = glob.glob(str(ud_cantonese_hk_dir / "*test.conllu"))
        reader = CoNLLU.from_files(test_files)
        chat = reader.to_chat()
        assert chat.n_files > 0
