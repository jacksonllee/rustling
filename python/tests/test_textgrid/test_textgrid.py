import zipfile

import pytest

from rustling.textgrid import TextGrid, IntervalTier, TextTier

SAMPLE_TEXT_FORMAT = """\
File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0
xmax = 2.3
tiers? <exists>
size = 2
item []:
    item [1]:
        class = "IntervalTier"
        name = "words"
        xmin = 0
        xmax = 2.3
        intervals: size = 2
            intervals [1]:
                xmin = 0
                xmax = 1.5
                text = "hello"
            intervals [2]:
                xmin = 1.5
                xmax = 2.3
                text = "world"
    item [2]:
        class = "TextTier"
        name = "events"
        xmin = 0
        xmax = 2.3
        points: size = 1
            points [1]:
                number = 0.75
                mark = "click"
"""

SAMPLE_SHORT_TEXT_FORMAT = """\
File type = "ooTextFile"
Object class = "TextGrid"

0
2.3
<exists>
2
"IntervalTier"
"words"
0
2.3
2
0
1.5
"hello"
1.5
2.3
"world"
"TextTier"
"events"
0
2.3
1
0.75
"click"
"""


class TestFromStrs:
    def test_basic(self):
        tg = TextGrid.from_strs([SAMPLE_TEXT_FORMAT])
        assert tg.n_files == 1
        tiers_list = tg.tiers()
        assert isinstance(tiers_list, list)
        assert len(tiers_list) == 1
        tiers = tiers_list[0]
        assert len(tiers) == 2

        # IntervalTier
        tier0 = tiers[0]
        assert isinstance(tier0, IntervalTier)
        assert tier0.name == "words"
        assert tier0.tier_class == "IntervalTier"
        assert tier0.xmin == 0.0
        assert tier0.xmax == 2.3
        intervals = tier0.intervals
        assert len(intervals) == 2
        assert intervals[0].xmin == 0.0
        assert intervals[0].xmax == 1.5
        assert intervals[0].text == "hello"
        assert intervals[1].xmin == 1.5
        assert intervals[1].xmax == 2.3
        assert intervals[1].text == "world"

        # TextTier
        tier1 = tiers[1]
        assert isinstance(tier1, TextTier)
        assert tier1.name == "events"
        assert tier1.tier_class == "TextTier"
        points = tier1.points
        assert len(points) == 1
        assert points[0].number == 0.75
        assert points[0].mark == "click"

    def test_short_text_format(self):
        tg = TextGrid.from_strs([SAMPLE_SHORT_TEXT_FORMAT])
        assert tg.n_files == 1
        tiers = tg.tiers()[0]
        assert len(tiers) == 2
        assert isinstance(tiers[0], IntervalTier)
        assert isinstance(tiers[1], TextTier)
        assert tiers[0].intervals[0].text == "hello"

    def test_with_ids(self):
        tg = TextGrid.from_strs([SAMPLE_TEXT_FORMAT], ids=["my_file.TextGrid"])
        assert tg.file_paths == ["my_file.TextGrid"]

    def test_ids_length_mismatch(self):
        with pytest.raises(ValueError, match="same length"):
            TextGrid.from_strs([SAMPLE_TEXT_FORMAT], ids=["a", "b"])

    def test_multiple_strs(self):
        tg = TextGrid.from_strs(
            [SAMPLE_TEXT_FORMAT, SAMPLE_SHORT_TEXT_FORMAT],
            ids=["file1.TextGrid", "file2.TextGrid"],
        )
        assert tg.n_files == 2
        assert tg.file_paths == ["file1.TextGrid", "file2.TextGrid"]


class TestFilePathsAndNFiles:
    def test_file_paths(self):
        tg = TextGrid.from_strs([SAMPLE_TEXT_FORMAT], ids=["test.TextGrid"])
        assert tg.file_paths == ["test.TextGrid"]

    def test_n_files(self):
        tg = TextGrid.from_strs([SAMPLE_TEXT_FORMAT, SAMPLE_TEXT_FORMAT])
        assert tg.n_files == 2

    def test_empty(self):
        tg = TextGrid()
        assert tg.n_files == 0
        assert tg.file_paths == []


class TestFromDir:
    def test_from_dir(self, tmp_path):
        (tmp_path / "a.TextGrid").write_text(SAMPLE_TEXT_FORMAT, encoding="utf-8")
        (tmp_path / "b.TextGrid").write_text(SAMPLE_SHORT_TEXT_FORMAT, encoding="utf-8")
        (tmp_path / "c.txt").write_text("not textgrid", encoding="utf-8")

        tg = TextGrid.from_dir(tmp_path)
        assert tg.n_files == 2
        assert all(fp.endswith(".TextGrid") for fp in tg.file_paths)

    def test_from_dir_with_match(self, tmp_path):
        (tmp_path / "a.TextGrid").write_text(SAMPLE_TEXT_FORMAT, encoding="utf-8")
        (tmp_path / "b.TextGrid").write_text(SAMPLE_SHORT_TEXT_FORMAT, encoding="utf-8")

        tg = TextGrid.from_dir(tmp_path, match=r"a\.TextGrid")
        assert tg.n_files == 1


class TestFromFiles:
    def test_from_files(self, tmp_path):
        f = tmp_path / "test.TextGrid"
        f.write_text(SAMPLE_TEXT_FORMAT, encoding="utf-8")
        tg = TextGrid.from_files([f])
        assert tg.n_files == 1
        assert len(tg.tiers()[0]) == 2


class TestFromZip:
    def test_from_zip(self, tmp_path):
        zip_path = tmp_path / "data.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("a.TextGrid", SAMPLE_TEXT_FORMAT)
            zf.writestr("b.TextGrid", SAMPLE_SHORT_TEXT_FORMAT)
            zf.writestr("c.txt", "not textgrid")

        tg = TextGrid.from_zip(zip_path)
        assert tg.n_files == 2


class TestCollectionOps:
    def test_iter(self):
        tg = TextGrid.from_strs(
            [SAMPLE_TEXT_FORMAT, SAMPLE_SHORT_TEXT_FORMAT],
            ids=["f1.TextGrid", "f2.TextGrid"],
        )
        items = list(tg)
        assert len(items) == 2
        assert items[0].n_files == 1
        assert items[1].n_files == 1

    def test_getitem_int(self):
        tg = TextGrid.from_strs(
            [SAMPLE_TEXT_FORMAT, SAMPLE_SHORT_TEXT_FORMAT],
            ids=["f1.TextGrid", "f2.TextGrid"],
        )
        first = tg[0]
        assert first.file_paths == ["f1.TextGrid"]
        last = tg[-1]
        assert last.file_paths == ["f2.TextGrid"]

    def test_getitem_slice(self):
        tg = TextGrid.from_strs(
            [SAMPLE_TEXT_FORMAT, SAMPLE_SHORT_TEXT_FORMAT],
            ids=["f1.TextGrid", "f2.TextGrid"],
        )
        sliced = tg[0:1]
        assert sliced.n_files == 1

    def test_getitem_out_of_range(self):
        tg = TextGrid.from_strs([SAMPLE_TEXT_FORMAT])
        with pytest.raises(IndexError):
            tg[5]

    def test_add(self):
        t1 = TextGrid.from_strs([SAMPLE_TEXT_FORMAT], ids=["f1.TextGrid"])
        t2 = TextGrid.from_strs([SAMPLE_SHORT_TEXT_FORMAT], ids=["f2.TextGrid"])
        combined = t1 + t2
        assert combined.n_files == 2

    def test_iadd(self):
        t1 = TextGrid.from_strs([SAMPLE_TEXT_FORMAT], ids=["f1.TextGrid"])
        t2 = TextGrid.from_strs([SAMPLE_SHORT_TEXT_FORMAT], ids=["f2.TextGrid"])
        t1 += t2
        assert t1.n_files == 2

    def test_append_pop(self):
        t1 = TextGrid.from_strs([SAMPLE_TEXT_FORMAT], ids=["f1.TextGrid"])
        t2 = TextGrid.from_strs([SAMPLE_SHORT_TEXT_FORMAT], ids=["f2.TextGrid"])
        t1.append(t2)
        assert t1.n_files == 2
        popped = t1.pop()
        assert popped.file_paths == ["f2.TextGrid"]
        assert t1.n_files == 1

    def test_append_left_pop_left(self):
        t1 = TextGrid.from_strs([SAMPLE_TEXT_FORMAT], ids=["f1.TextGrid"])
        t2 = TextGrid.from_strs([SAMPLE_SHORT_TEXT_FORMAT], ids=["f2.TextGrid"])
        t1.append_left(t2)
        assert t1.file_paths == ["f2.TextGrid", "f1.TextGrid"]
        popped = t1.pop_left()
        assert popped.file_paths == ["f2.TextGrid"]

    def test_clear(self):
        tg = TextGrid.from_strs([SAMPLE_TEXT_FORMAT])
        tg.clear()
        assert tg.n_files == 0

    def test_extend(self):
        t1 = TextGrid.from_strs([SAMPLE_TEXT_FORMAT], ids=["f1.TextGrid"])
        t2 = TextGrid.from_strs([SAMPLE_SHORT_TEXT_FORMAT], ids=["f2.TextGrid"])
        t3 = TextGrid.from_strs([SAMPLE_TEXT_FORMAT], ids=["f3.TextGrid"])
        t1.extend([t2, t3])
        assert t1.n_files == 3

    def test_pop_empty(self):
        tg = TextGrid()
        with pytest.raises(IndexError):
            tg.pop()
        with pytest.raises(IndexError):
            tg.pop_left()


class TestReprAndBool:
    def test_repr(self):
        tg = TextGrid.from_strs([SAMPLE_TEXT_FORMAT])
        assert "1 file(s)" in repr(tg)

    def test_bool_true(self):
        tg = TextGrid.from_strs([SAMPLE_TEXT_FORMAT])
        assert bool(tg) is True

    def test_bool_false(self):
        tg = TextGrid()
        assert bool(tg) is False


class TestEquality:
    def test_equal(self):
        t1 = TextGrid.from_strs([SAMPLE_TEXT_FORMAT], ids=["f.TextGrid"])
        t2 = TextGrid.from_strs([SAMPLE_TEXT_FORMAT], ids=["f.TextGrid"])
        assert t1 == t2

    def test_not_equal(self):
        t1 = TextGrid.from_strs([SAMPLE_TEXT_FORMAT], ids=["f1.TextGrid"])
        t2 = TextGrid.from_strs([SAMPLE_TEXT_FORMAT], ids=["f2.TextGrid"])
        assert t1 != t2


class TestToStrs:
    def test_to_strs_basic(self):
        tg = TextGrid.from_strs([SAMPLE_TEXT_FORMAT], ids=["test.TextGrid"])
        strs = tg.to_strs()
        assert len(strs) == 1
        assert strs[0] == SAMPLE_TEXT_FORMAT

    def test_to_strs_round_trip(self):
        tg = TextGrid.from_strs([SAMPLE_TEXT_FORMAT], ids=["test.TextGrid"])
        strs = tg.to_strs()
        tg2 = TextGrid.from_strs(strs, ids=["test.TextGrid"])
        assert tg == tg2


class TestToFiles:
    def test_single_file(self, tmp_path):
        tg = TextGrid.from_strs([SAMPLE_TEXT_FORMAT], ids=["test.TextGrid"])
        out_dir = tmp_path / "output"
        tg.to_files(out_dir)
        content = (out_dir / "test.TextGrid").read_text(encoding="utf-8")
        assert content == SAMPLE_TEXT_FORMAT
        # Re-read and verify
        tg2 = TextGrid.from_dir(out_dir)
        tiers = tg2.tiers()[0]
        assert len(tiers) == 2
        assert isinstance(tiers[0], IntervalTier)

    def test_custom_filenames(self, tmp_path):
        tg = TextGrid.from_strs(
            [SAMPLE_TEXT_FORMAT, SAMPLE_SHORT_TEXT_FORMAT],
            ids=["f1.TextGrid", "f2.TextGrid"],
        )
        out_dir = tmp_path / "output"
        tg.to_files(out_dir, filenames=["a.TextGrid", "b.TextGrid"])
        assert (out_dir / "a.TextGrid").exists()
        assert (out_dir / "b.TextGrid").exists()

    def test_filename_mismatch_raises(self, tmp_path):
        tg = TextGrid.from_strs(
            [SAMPLE_TEXT_FORMAT, SAMPLE_SHORT_TEXT_FORMAT],
            ids=["f1.TextGrid", "f2.TextGrid"],
        )
        with pytest.raises(ValueError, match="filenames"):
            tg.to_files(tmp_path, filenames=["only_one.TextGrid"])


class TestToElan:
    def test_to_elan_returns_elan_object(self):
        from rustling.elan import ELAN

        tg = TextGrid.from_strs([SAMPLE_TEXT_FORMAT], ids=["test.TextGrid"])
        elan = tg.to_elan()
        assert isinstance(elan, ELAN)
        assert elan.n_files == 1

    def test_to_elan_strs(self):
        tg = TextGrid.from_strs([SAMPLE_TEXT_FORMAT], ids=["test.TextGrid"])
        strs = tg.to_elan_strs()
        assert len(strs) == 1
        assert "ANNOTATION_DOCUMENT" in strs[0]

    def test_to_elan_round_trip(self):
        tg = TextGrid.from_strs([SAMPLE_TEXT_FORMAT], ids=["test.TextGrid"])
        elan = tg.to_elan()
        tg2 = elan.to_textgrid()
        assert tg2.n_files == 1
        tiers = tg2.tiers()[0]
        # Only IntervalTiers are converted; TextTier is skipped
        assert len(tiers) == 1
        assert isinstance(tiers[0], IntervalTier)
        assert tiers[0].name == "words"


class TestToChat:
    def test_to_chat_returns_chat_object(self):
        from rustling.chat import CHAT

        tg = TextGrid.from_strs([SAMPLE_TEXT_FORMAT], ids=["test.TextGrid"])
        chat = tg.to_chat(participants=["words"])
        assert isinstance(chat, CHAT)
        assert chat.n_files == 1

    def test_to_chat_strs(self):
        tg = TextGrid.from_strs([SAMPLE_TEXT_FORMAT], ids=["test.TextGrid"])
        strs = tg.to_chat_strs(participants=["words"])
        assert len(strs) == 1
        assert "@Begin" in strs[0]
        assert "@End" in strs[0]


class TestToSrt:
    def test_to_srt_returns_srt_object(self):
        from rustling.srt import SRT

        tg = TextGrid.from_strs([SAMPLE_TEXT_FORMAT], ids=["test.TextGrid"])
        srt = tg.to_srt(participants=["words"])
        assert isinstance(srt, SRT)
        assert srt.n_files == 1

    def test_to_srt_strs(self):
        tg = TextGrid.from_strs([SAMPLE_TEXT_FORMAT], ids=["test.TextGrid"])
        strs = tg.to_srt_strs(participants=["words"])
        assert len(strs) == 1
        assert "-->" in strs[0]
