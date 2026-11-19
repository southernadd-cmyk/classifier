"""Small synthetic ZIP ensures the CommonsSpeech data preparation works offline."""
import csv
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from scripts.prepare_commons_speech import normalise_party, clean_text, process_archive

class PrepareCommonsSpeechTest(unittest.TestCase):
    def test_parties(self):
        self.assertEqual(normalise_party("Lab"), "Labour")
        self.assertEqual(normalise_party("Labour Co-op"), "Labour")
        self.assertEqual(normalise_party("Con"), "Conservative")
        self.assertEqual(normalise_party("Conservative"), "Conservative")
        self.assertIsNone(normalise_party("Lib Dem"))
    def test_clean(self):
        self.assertIn("hello & goodbye", clean_text("<p>hello &amp; goodbye</p>"))

    def test_generate(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.zip"
            out = Path(directory) / "sample.json"
            content = io.StringIO()
            writer = csv.DictWriter(content, fieldnames=["person_id","party_short","hdate","body"])
            writer.writeheader()
            for party in ["Labour","Conservative"]:
                for mp in range(12):
                    for speech in range(6):
                        writer.writerow({
                            "person_id": party + str(mp),
                            "party_short": party,
                            "hdate": "2008-05-11",
                            "body": ("Citizens deserve transparent public decision making, careful analysis of national priorities and effective services. " * 8) + str(speech)
                        })
            with zipfile.ZipFile(path,"w",zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("2008.csv",content.getvalue())
            result = process_archive(path,out,per_period=45,max_per_mp=8)
            self.assertTrue(out.exists())
            self.assertEqual(result["counts"]["Labour"],result["counts"]["Conservative"])
            self.assertGreaterEqual(result["counts"]["Labour"],30)
            self.assertGreaterEqual(result["speakers"]["Labour"],8)
            json.loads(out.read_text())
if __name__ == "__main__":
    unittest.main()
