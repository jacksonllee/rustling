import zipfile
from collections import OrderedDict

import pytest

from rustling.elan import ELAN

SAMPLE_EAF = """\
<?xml version="1.0" encoding="UTF-8"?>
<ANNOTATION_DOCUMENT AUTHOR=""
    DATE="2024-01-01T00:00:00+00:00"
    FORMAT="3.0" VERSION="3.0">
  <HEADER MEDIA_FILE="" TIME_UNITS="milliseconds"/>
  <TIME_ORDER>
    <TIME_SLOT TIME_SLOT_ID="ts1" TIME_VALUE="0"/>
    <TIME_SLOT TIME_SLOT_ID="ts2" TIME_VALUE="1500"/>
    <TIME_SLOT TIME_SLOT_ID="ts3" TIME_VALUE="2000"/>
    <TIME_SLOT TIME_SLOT_ID="ts4" TIME_VALUE="3500"/>
  </TIME_ORDER>
  <TIER TIER_ID="Speaker1" PARTICIPANT="Alice"
      ANNOTATOR="Ann" LINGUISTIC_TYPE_REF="default-lt">
    <ANNOTATION>
      <ALIGNABLE_ANNOTATION ANNOTATION_ID="a1"
          TIME_SLOT_REF1="ts1" TIME_SLOT_REF2="ts2">
        <ANNOTATION_VALUE>hello world</ANNOTATION_VALUE>
      </ALIGNABLE_ANNOTATION>
    </ANNOTATION>
    <ANNOTATION>
      <ALIGNABLE_ANNOTATION ANNOTATION_ID="a2"
          TIME_SLOT_REF1="ts3" TIME_SLOT_REF2="ts4">
        <ANNOTATION_VALUE>goodbye</ANNOTATION_VALUE>
      </ALIGNABLE_ANNOTATION>
    </ANNOTATION>
  </TIER>
</ANNOTATION_DOCUMENT>"""

SAMPLE_EAF_WITH_REF = """\
<?xml version="1.0" encoding="UTF-8"?>
<ANNOTATION_DOCUMENT>
  <HEADER MEDIA_FILE="" TIME_UNITS="milliseconds"/>
  <TIME_ORDER>
    <TIME_SLOT TIME_SLOT_ID="ts1" TIME_VALUE="0"/>
    <TIME_SLOT TIME_SLOT_ID="ts2" TIME_VALUE="1500"/>
  </TIME_ORDER>
  <TIER TIER_ID="Main" PARTICIPANT="Alice"
      ANNOTATOR="" LINGUISTIC_TYPE_REF="default-lt">
    <ANNOTATION>
      <ALIGNABLE_ANNOTATION ANNOTATION_ID="a1"
          TIME_SLOT_REF1="ts1" TIME_SLOT_REF2="ts2">
        <ANNOTATION_VALUE>hello</ANNOTATION_VALUE>
      </ALIGNABLE_ANNOTATION>
    </ANNOTATION>
  </TIER>
  <TIER TIER_ID="Gloss" PARTICIPANT=""
      ANNOTATOR="" LINGUISTIC_TYPE_REF="gloss-lt"
      PARENT_REF="Main">
    <ANNOTATION>
      <REF_ANNOTATION ANNOTATION_ID="a2"
          ANNOTATION_REF="a1">
        <ANNOTATION_VALUE>greeting</ANNOTATION_VALUE>
      </REF_ANNOTATION>
    </ANNOTATION>
  </TIER>
</ANNOTATION_DOCUMENT>"""


class TestFromStrs:
    def test_basic(self):
        elan = ELAN.from_strs([SAMPLE_EAF])
        assert elan.n_files == 1
        tiers_list = elan.tiers()
        assert isinstance(tiers_list, list)
        assert len(tiers_list) == 1
        tiers = tiers_list[0]
        assert isinstance(tiers, OrderedDict)
        assert list(tiers.keys()) == ["Speaker1"]
        tier = tiers["Speaker1"]
        assert tier.id == "Speaker1"
        assert tier.participant == "Alice"
        assert tier.annotator == "Ann"
        assert tier.linguistic_type_ref == "default-lt"
        assert tier.parent_id is None
        assert tier.child_ids is None
        assert len(tier.annotations) == 2

        a1 = tier.annotations[0]
        assert a1.id == "a1"
        assert a1.start_time == 0
        assert a1.end_time == 1500
        assert a1.value == "hello world"
        assert a1.parent_id is None

        a2 = tier.annotations[1]
        assert a2.id == "a2"
        assert a2.start_time == 2000
        assert a2.end_time == 3500
        assert a2.value == "goodbye"
        assert a2.parent_id is None

    def test_with_ids(self):
        elan = ELAN.from_strs([SAMPLE_EAF], ids=["my_file.eaf"])
        assert elan.file_paths == ["my_file.eaf"]

    def test_ids_length_mismatch(self):
        with pytest.raises(ValueError, match="same length"):
            ELAN.from_strs([SAMPLE_EAF], ids=["a", "b"])

    def test_multiple_strs(self):
        elan = ELAN.from_strs(
            [SAMPLE_EAF, SAMPLE_EAF_WITH_REF],
            ids=["file1.eaf", "file2.eaf"],
        )
        assert elan.n_files == 2
        assert elan.file_paths == ["file1.eaf", "file2.eaf"]


class TestRefAnnotation:
    def test_ref_annotation_resolves_time(self):
        elan = ELAN.from_strs([SAMPLE_EAF_WITH_REF])
        tiers = elan.tiers()[0]
        assert list(tiers.keys()) == ["Main", "Gloss"]

        main_tier = tiers["Main"]
        assert main_tier.child_ids == ["Gloss"]
        assert main_tier.annotations[0].start_time == 0
        assert main_tier.annotations[0].end_time == 1500
        assert main_tier.annotations[0].parent_id is None

        gloss_tier = tiers["Gloss"]
        assert gloss_tier.parent_id == "Main"
        assert gloss_tier.child_ids is None
        ref_ann = gloss_tier.annotations[0]
        assert ref_ann.id == "a2"
        assert ref_ann.value == "greeting"
        assert ref_ann.start_time == 0
        assert ref_ann.end_time == 1500
        assert ref_ann.parent_id == "a1"


class TestTiersMultipleFiles:
    def test_tiers_across_files(self):
        elan = ELAN.from_strs(
            [SAMPLE_EAF, SAMPLE_EAF_WITH_REF],
            ids=["f1.eaf", "f2.eaf"],
        )
        tiers_list = elan.tiers()
        assert isinstance(tiers_list, list)
        assert len(tiers_list) == 2
        # File 1: Speaker1
        assert isinstance(tiers_list[0], OrderedDict)
        assert list(tiers_list[0].keys()) == ["Speaker1"]
        # File 2: Main, Gloss
        assert isinstance(tiers_list[1], OrderedDict)
        assert list(tiers_list[1].keys()) == ["Main", "Gloss"]


class TestFilePathsAndNFiles:
    def test_file_paths(self):
        elan = ELAN.from_strs([SAMPLE_EAF], ids=["test.eaf"])
        assert elan.file_paths == ["test.eaf"]

    def test_n_files(self):
        elan = ELAN.from_strs([SAMPLE_EAF, SAMPLE_EAF])
        assert elan.n_files == 2

    def test_empty(self):
        elan = ELAN()
        assert elan.n_files == 0
        assert elan.file_paths == []


class TestFromDir:
    def test_from_dir(self, tmp_path):
        (tmp_path / "a.eaf").write_text(SAMPLE_EAF, encoding="utf-8")
        (tmp_path / "b.eaf").write_text(SAMPLE_EAF_WITH_REF, encoding="utf-8")
        (tmp_path / "c.txt").write_text("not elan", encoding="utf-8")

        elan = ELAN.from_dir(tmp_path)
        assert elan.n_files == 2
        # Sorted by path
        assert all(fp.endswith(".eaf") for fp in elan.file_paths)

    def test_from_dir_with_match(self, tmp_path):
        (tmp_path / "a.eaf").write_text(SAMPLE_EAF, encoding="utf-8")
        (tmp_path / "b.eaf").write_text(SAMPLE_EAF_WITH_REF, encoding="utf-8")

        elan = ELAN.from_dir(tmp_path, match="a\\.eaf")
        assert elan.n_files == 1


class TestFromFiles:
    def test_from_files(self, tmp_path):
        f = tmp_path / "test.eaf"
        f.write_text(SAMPLE_EAF, encoding="utf-8")
        elan = ELAN.from_files([f])
        assert elan.n_files == 1
        assert len(elan.tiers()[0]) == 1


class TestFromZip:
    def test_from_zip(self, tmp_path):
        zip_path = tmp_path / "data.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("a.eaf", SAMPLE_EAF)
            zf.writestr("b.eaf", SAMPLE_EAF_WITH_REF)
            zf.writestr("c.txt", "not elan")

        elan = ELAN.from_zip(zip_path)
        assert elan.n_files == 2

    def test_from_zip_with_match(self, tmp_path):
        zip_path = tmp_path / "data.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("a.eaf", SAMPLE_EAF)
            zf.writestr("b.eaf", SAMPLE_EAF_WITH_REF)

        elan = ELAN.from_zip(zip_path, match="a\\.eaf")
        assert elan.n_files == 1


class TestCollectionOps:
    def test_iter(self):
        elan = ELAN.from_strs(
            [SAMPLE_EAF, SAMPLE_EAF_WITH_REF],
            ids=["f1.eaf", "f2.eaf"],
        )
        items = list(elan)
        assert len(items) == 2
        assert items[0].n_files == 1
        assert items[1].n_files == 1

    def test_getitem_int(self):
        elan = ELAN.from_strs(
            [SAMPLE_EAF, SAMPLE_EAF_WITH_REF],
            ids=["f1.eaf", "f2.eaf"],
        )
        first = elan[0]
        assert first.file_paths == ["f1.eaf"]
        last = elan[-1]
        assert last.file_paths == ["f2.eaf"]

    def test_getitem_slice(self):
        elan = ELAN.from_strs(
            [SAMPLE_EAF, SAMPLE_EAF_WITH_REF],
            ids=["f1.eaf", "f2.eaf"],
        )
        sliced = elan[0:1]
        assert sliced.n_files == 1

    def test_getitem_out_of_range(self):
        elan = ELAN.from_strs([SAMPLE_EAF])
        with pytest.raises(IndexError):
            elan[5]

    def test_add(self):
        e1 = ELAN.from_strs([SAMPLE_EAF], ids=["f1.eaf"])
        e2 = ELAN.from_strs([SAMPLE_EAF_WITH_REF], ids=["f2.eaf"])
        combined = e1 + e2
        assert combined.n_files == 2

    def test_iadd(self):
        e1 = ELAN.from_strs([SAMPLE_EAF], ids=["f1.eaf"])
        e2 = ELAN.from_strs([SAMPLE_EAF_WITH_REF], ids=["f2.eaf"])
        e1 += e2
        assert e1.n_files == 2

    def test_append_pop(self):
        e1 = ELAN.from_strs([SAMPLE_EAF], ids=["f1.eaf"])
        e2 = ELAN.from_strs([SAMPLE_EAF_WITH_REF], ids=["f2.eaf"])
        e1.append(e2)
        assert e1.n_files == 2
        popped = e1.pop()
        assert popped.file_paths == ["f2.eaf"]
        assert e1.n_files == 1

    def test_append_left_pop_left(self):
        e1 = ELAN.from_strs([SAMPLE_EAF], ids=["f1.eaf"])
        e2 = ELAN.from_strs([SAMPLE_EAF_WITH_REF], ids=["f2.eaf"])
        e1.append_left(e2)
        assert e1.file_paths == ["f2.eaf", "f1.eaf"]
        popped = e1.pop_left()
        assert popped.file_paths == ["f2.eaf"]

    def test_clear(self):
        elan = ELAN.from_strs([SAMPLE_EAF])
        elan.clear()
        assert elan.n_files == 0

    def test_extend(self):
        e1 = ELAN.from_strs([SAMPLE_EAF], ids=["f1.eaf"])
        e2 = ELAN.from_strs([SAMPLE_EAF_WITH_REF], ids=["f2.eaf"])
        e3 = ELAN.from_strs([SAMPLE_EAF], ids=["f3.eaf"])
        e1.extend([e2, e3])
        assert e1.n_files == 3

    def test_pop_empty(self):
        elan = ELAN()
        with pytest.raises(IndexError):
            elan.pop()
        with pytest.raises(IndexError):
            elan.pop_left()


class TestReprAndBool:
    def test_repr(self):
        elan = ELAN.from_strs([SAMPLE_EAF])
        assert "1 file(s)" in repr(elan)

    def test_bool_true(self):
        elan = ELAN.from_strs([SAMPLE_EAF])
        assert bool(elan) is True

    def test_bool_false(self):
        elan = ELAN()
        assert bool(elan) is False


class TestEquality:
    def test_equal(self):
        e1 = ELAN.from_strs([SAMPLE_EAF], ids=["f.eaf"])
        e2 = ELAN.from_strs([SAMPLE_EAF], ids=["f.eaf"])
        assert e1 == e2

    def test_not_equal(self):
        e1 = ELAN.from_strs([SAMPLE_EAF], ids=["f1.eaf"])
        e2 = ELAN.from_strs([SAMPLE_EAF], ids=["f2.eaf"])
        assert e1 != e2


class TestToStrs:
    def test_to_strs_basic(self):
        elan = ELAN.from_strs([SAMPLE_EAF], ids=["test.eaf"])
        strs = elan.to_strs()
        assert len(strs) == 1
        assert strs[0] == SAMPLE_EAF

    def test_to_strs_multiple(self):
        elan = ELAN.from_strs(
            [SAMPLE_EAF, SAMPLE_EAF_WITH_REF],
            ids=["f1.eaf", "f2.eaf"],
        )
        strs = elan.to_strs()
        assert len(strs) == 2
        assert strs[0] == SAMPLE_EAF
        assert strs[1] == SAMPLE_EAF_WITH_REF

    def test_to_strs_round_trip(self):
        elan = ELAN.from_strs([SAMPLE_EAF], ids=["test.eaf"])
        strs = elan.to_strs()
        elan2 = ELAN.from_strs(strs, ids=["test.eaf"])
        assert elan == elan2


class TestToFiles:
    def test_single_file(self, tmp_path):
        elan = ELAN.from_strs([SAMPLE_EAF], ids=["test.eaf"])
        out_dir = tmp_path / "output"
        elan.to_files(out_dir)
        content = (out_dir / "test.eaf").read_text(encoding="utf-8")
        assert content == SAMPLE_EAF
        # Re-read and verify
        elan2 = ELAN.from_dir(out_dir)
        tiers = elan2.tiers()[0]
        assert list(tiers.keys()) == ["Speaker1"]
        assert len(tiers["Speaker1"].annotations) == 2

    def test_single_file_pathlib(self, tmp_path):
        import pathlib

        elan = ELAN.from_strs([SAMPLE_EAF], ids=["test.eaf"])
        out_dir = pathlib.Path(tmp_path) / "output"
        elan.to_files(out_dir)
        assert (out_dir / "test.eaf").exists()

    def test_directory(self, tmp_path):
        elan = ELAN.from_strs(
            [SAMPLE_EAF, SAMPLE_EAF_WITH_REF],
            ids=["f1.eaf", "f2.eaf"],
        )
        out_dir = tmp_path / "output"
        elan.to_files(out_dir)
        assert (out_dir / "f1.eaf").exists()
        assert (out_dir / "f2.eaf").exists()
        content1 = (out_dir / "f1.eaf").read_text(encoding="utf-8")
        content2 = (out_dir / "f2.eaf").read_text(encoding="utf-8")
        assert content1 == SAMPLE_EAF
        assert content2 == SAMPLE_EAF_WITH_REF

    def test_custom_filenames(self, tmp_path):
        elan = ELAN.from_strs(
            [SAMPLE_EAF, SAMPLE_EAF_WITH_REF],
            ids=["f1.eaf", "f2.eaf"],
        )
        out_dir = tmp_path / "output"
        elan.to_files(out_dir, filenames=["alice.eaf", "bob.eaf"])
        assert (out_dir / "alice.eaf").exists()
        assert (out_dir / "bob.eaf").exists()

    def test_filename_mismatch_raises(self, tmp_path):
        elan = ELAN.from_strs(
            [SAMPLE_EAF, SAMPLE_EAF_WITH_REF],
            ids=["f1.eaf", "f2.eaf"],
        )
        with pytest.raises(ValueError, match="filenames"):
            elan.to_files(tmp_path, filenames=["only_one.eaf"])

    def test_round_trip_with_ref(self, tmp_path):
        elan = ELAN.from_strs([SAMPLE_EAF_WITH_REF], ids=["test.eaf"])
        out_dir = tmp_path / "output"
        elan.to_files(out_dir)
        elan2 = ELAN.from_dir(out_dir)
        tiers = elan2.tiers()[0]
        assert list(tiers.keys()) == ["Main", "Gloss"]
        ref_ann = tiers["Gloss"].annotations[0]
        assert ref_ann.value == "greeting"
        assert ref_ann.parent_id == "a1"


class TestToChat:
    """Tests for ELAN -> CHAT conversion."""

    def test_to_chat_returns_chat_object(self):
        from rustling.chat import CHAT

        elan = ELAN.from_strs([SAMPLE_EAF], ids=["test.eaf"])
        chat = elan.to_chat(participants=["Speaker1"])
        assert isinstance(chat, CHAT)
        assert chat.n_files == 1

    def test_to_chat_strs(self):
        elan = ELAN.from_strs([SAMPLE_EAF], ids=["test.eaf"])
        strs = elan.to_chat_strs(participants=["Speaker1"])
        assert len(strs) == 1
        assert "@Begin" in strs[0]
        assert "*Speaker1:" in strs[0]
        assert "@End" in strs[0]

    def test_to_chat_with_ref_tiers(self):
        elan = ELAN.from_strs([SAMPLE_EAF_WITH_REF], ids=["test.eaf"])
        # "Main" is 4 chars so auto-detect won't pick it up; use explicit.
        chat = elan.to_chat(participants=["Main"])
        utts = chat.utterances()
        assert len(utts) == 1
        tiers = utts[0].tiers
        assert "Main" in tiers
        assert tiers["Main"].startswith("hello")

    def test_to_chat_files(self, tmp_path):
        from rustling.chat import CHAT

        elan = ELAN.from_strs([SAMPLE_EAF], ids=["test.eaf"])
        out_dir = str(tmp_path / "output")
        elan.to_chat_files(out_dir, participants=["Speaker1"])
        chat = CHAT.from_dir(out_dir, strict=False)
        assert chat.n_files == 1

    def test_to_chat_auto_detect_3char(self):
        """Auto-detect: only 3-char parent tiers are treated as main tiers."""
        # Build ELAN data with a 3-char tier "CHI" from CHAT round-trip.
        from rustling.chat import CHAT

        chat_in = CHAT.from_strs(
            [
                "@UTF8\n@Begin\n@Participants:\tCHI Target_Child\n"
                "*CHI:\thello .\n%mor:\tn|hello .\n@End\n"
            ]
        )
        elan = chat_in.to_elan()
        # Now convert back — auto-detect should find "CHI" (3 chars).
        chat_out = elan.to_chat()
        assert chat_out.n_files == 1
        utts = chat_out.utterances()
        assert len(utts) == 1
        assert utts[0].participant == "CHI"

    def test_to_chat_round_trip_preserves_words(self):
        from rustling.chat import CHAT

        chat_in = CHAT.from_strs(
            [
                "@UTF8\n@Begin\n@Participants:\tCHI Target_Child, MOT Mother\n"
                "*CHI:\tI want cookie .\n"
                "*MOT:\tno .\n"
                "@End\n"
            ]
        )
        elan = chat_in.to_elan()
        chat_out = elan.to_chat()
        assert chat_out.words() == chat_in.words()

    def test_to_chat_files_custom_filenames(self, tmp_path):
        import os

        elan = ELAN.from_strs([SAMPLE_EAF], ids=["test.eaf"])
        out_dir = str(tmp_path / "output")
        elan.to_chat_files(out_dir, participants=["Speaker1"], filenames=["custom.cha"])
        assert os.path.exists(os.path.join(out_dir, "custom.cha"))

    def test_to_chat_participant_info(self):
        """Tier.participant populates the @Participants line."""
        from rustling.chat import CHAT

        chat_in = CHAT.from_strs(
            [
                "@UTF8\n@Begin\n@Participants:\tCHI Eve Target_Child\n"
                "*CHI:\thi .\n@End\n"
            ]
        )
        elan = chat_in.to_elan()
        # ELAN Tier.participant should be "Eve" (from CHAT Participant.name).
        tiers = elan.tiers()[0]
        assert tiers["CHI"].participant == "Eve"
        # Round-trip: ELAN -> CHAT preserves the name in @Participants.
        chat_out = elan.to_chat()
        strs = chat_out.to_strs()
        assert "CHI Eve" in strs[0]


class TestCantoMap:
    def test_read_cantomap_eaf(self, cantomap_dir):
        eaf_path = (
            cantomap_dir / "ConversationData" / "Subjects-1_2" / "160725_009_1_2_A1.eaf"
        )
        if not eaf_path.exists():
            pytest.skip("CantoMap data not available")
        elan = ELAN.from_files([eaf_path])
        assert elan.n_files == 1
        tiers = elan.tiers()[0]
        assert isinstance(tiers, OrderedDict)
        assert "G-jyutping" in tiers
        assert "G-word" in tiers
        assert "E" in tiers
        assert "F" in tiers
        assert "G" in tiers

        # Check a specific tier has annotations
        g_tier = tiers["G"]
        assert len(g_tier.annotations) > 0
        assert g_tier.participant == "Yau Wau Shan"

        # Check annotations have resolved times
        for ann in g_tier.annotations:
            assert ann.start_time is not None
            assert ann.end_time is not None
            assert ann.start_time >= 0
            assert ann.end_time >= ann.start_time
