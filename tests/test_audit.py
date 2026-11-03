import ast
from contextlib import redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from speech_emotion_recognition_claim_audit import __version__, phrases
from speech_emotion_recognition_claim_audit.audit import audit, claims, evidence, main, GENERIC

# Fixed independent oracle, transcribed from the requirements.
EXPECTED_POS = '''speech emotion recognition
recognizes speech emotion
recognises speech emotion
identifies speech emotion
detects speech emotion
classifies speech emotion
predicts speech emotion
speech emotion detection
speech emotion classification
voice emotion recognition
recognizes voice emotion
recognises voice emotion
identifies voice emotion
detects voice emotion
classifies voice emotion
predicts voice emotion
voice emotion detection
voice emotion classification
emotion recognition from speech
emotion recognition in speech
detects emotion in speech
identifies emotion in speech
recognizes emotion in speech
recognises emotion in speech
classifies emotion in speech
predicts emotion in speech
emotion detection from speech
spoken emotion recognition
recognizes spoken emotion
detects spoken emotion
classifies spoken emotion
emotion recognition from voice
detects emotion in voice
identifies emotion in voice
recognizes emotion in voice
speech emotion analysis
analyzes speech emotion
analyses speech emotion
speech emotion prediction
voice emotion analysis
analyzes voice emotion
detects the emotion in speech
identifies the emotion in speech
recognizes the emotion in speech
classifies speech by emotion
classifies voice by emotion
detects speech emotion in voice
identifies speech emotion in voice
speech emotion recognition system
automatic speech emotion recognition'''.splitlines()
EXPECTED_NEG = '''no speech emotion recognition
does not recognize speech emotion
does not recognise speech emotion
without speech emotion recognition
does not identify speech emotion
does not detect speech emotion
does not classify speech emotion
does not predict speech emotion
no speech emotion detection
does not detect emotion in speech
does not identify emotion in speech
does not recognize emotion in speech
does not recognise emotion in speech
no voice emotion recognition
does not recognize voice emotion
does not detect voice emotion
no spoken emotion recognition
does not recognize spoken emotion
speech emotion recognition not supported
does not classify emotion in speech
no speech emotion classification
does not detect spoken emotion
no emotion recognition from speech
no speech emotion prediction'''.splitlines()
EXPECTED_GENERIC = '''detect_speech_emotion detect_speech_emotions classify_speech_emotion classify_speech_emotions predict_speech_emotion predict_speech_emotions recognize_speech_emotion recognise_speech_emotion recognize_speech_emotions recognise_speech_emotions identify_speech_emotion identify_speech_emotions get_speech_emotion get_speech_emotions extract_speech_emotion extract_speech_emotions analyze_speech_emotion analyse_speech_emotion analyze_speech_emotions analyse_speech_emotions detect_voice_emotion detect_voice_emotions classify_voice_emotion classify_voice_emotions predict_voice_emotion predict_voice_emotions recognize_voice_emotion recognise_voice_emotion identify_voice_emotion get_voice_emotion detect_emotion_in_speech classify_emotion_in_speech predict_emotion_in_speech recognize_emotion_in_speech recognise_emotion_in_speech detect_spoken_emotion classify_spoken_emotion predict_spoken_emotion speech_emotion_detect speech_emotion_classify speech_emotion_predict'''.split()
POS_ID = 'documented-speech-emotion-recognition-claim-false'
NEG_ID = 'documented-no-speech-emotion-recognition-claim-false'


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def write(self, name, text):
        p = self.root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding='utf-8')
        return p

    def cli(self, *args):
        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            try:
                status = main(list(args))
            except SystemExit as exc:
                status = exc.code
        return status, out.getvalue(), err.getvalue()

    def test_phrase_oracle(self):
        tree = ast.parse(Path(phrases.__file__).read_text())
        values = {n.targets[0].id: ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)}
        self.assertEqual(tuple(EXPECTED_POS), values['SPEECHEMOTION'])
        self.assertEqual(tuple(EXPECTED_NEG), values['NOSPEECHEMOTION'])
        self.assertEqual(50, len(values['SPEECHEMOTION']))
        self.assertEqual(24, len(values['NOSPEECHEMOTION']))

    def test_every_phrase(self):
        for p in EXPECTED_POS:
            with self.subTest(p=p):
                self.assertEqual((True, False), claims(p))
        for p in EXPECTED_NEG:
            with self.subTest(p=p):
                self.assertEqual((False, True), claims(p))

    def test_normalization_boundaries_paragraphs(self):
        self.assertEqual((True, False), claims('SPEECH-emotion\n  recognition'))
        self.assertEqual((False, True), claims("doesn't recognize speech emotion"))
        self.assertEqual((False, False), claims('speech emotion\n\nrecognition'))
        self.assertEqual((False, False), claims('xspeech emotion recognitionz'))
        self.assertEqual((True, True), claims('no speech emotion recognition; detects voice emotion'))

    def test_fences_and_inline(self):
        for fence in ('```', '~~~', '   ````'):
            with self.subTest(fence=fence):
                self.assertEqual((False, False), claims(fence+'py\nspeech emotion recognition\n'+fence.strip()))
        self.assertEqual((True, False), claims('`speech emotion recognition`'))
        self.assertEqual((False, False), claims('```\nspeech emotion recognition'))

    def test_generic_all_forms(self):
        self.assertEqual(set(EXPECTED_GENERIC), GENERIC)
        self.assertEqual(41, len(GENERIC))
        for name in EXPECTED_GENERIC:
            for source in ('obj.'+name+'(audio)', 'from lib import '+name+'\n'+name+'(audio)',
                           'from lib import '+name+' as run\nrun(audio)',
                           'import lib as tool\ntool.'+name+'(audio)'):
                with self.subTest(source=source):
                    self.assertTrue(evidence(source))
            self.assertFalse(evidence(name+'(audio)'))
            self.assertFalse(evidence('obj.'+name))
            self.assertFalse(evidence('def '+name+'(x): pass\nobj.'+name+'(audio)'))
            self.assertFalse(evidence('from lib import '+name+' as run\ndef run(x): pass\nrun(audio)'))

    def test_classifier_pairs(self):
        for name in ('SpeechEmotionClassifier', 'VoiceSpeechEmotionClassifier', 'AudioSpeechEmotionClassifier',
                     'SpeechEmotionRecognizer', 'VoiceSpeechEmotionRecognizer'):
            for source in ('model = '+name+'()\nmodel.predict(audio)', name+'().predict(audio)',
                           'model: object = lib.'+name+'()\nmodel.predict(audio)',
                           'from lib import '+name+' as C\nmodel = C()\nmodel.predict(audio)',
                           'a = model = '+name+'()\na.predict(audio)'):
                with self.subTest(source=source):
                    self.assertTrue(evidence(source))
            for source in (name+'()', 'model = '+name+'()\nother.predict(audio)',
                           'class '+name+': pass\nm = '+name+'()\nm.predict(audio)',
                           'def '+name+'(): pass\nlib.'+name+'().predict(audio)'):
                with self.subTest(source=source):
                    self.assertFalse(evidence(source))

    def test_speechbrain_forms(self):
        sources = [
            'import speechbrain\nspeechbrain.pretrained.EmotionRecognition.from_hparams()',
            'import speechbrain.pretrained as sb\nsb.EmotionRecognition.from_hparams()',
            'import speechbrain as sb\nsb.inference.interfaces.EmotionRecognition.from_hparams()',
            'from speechbrain.pretrained import EmotionRecognition\nEmotionRecognition.from_hparams()',
            'from speechbrain.inference.interfaces import EmotionRecognition as ER\nER.from_hparams()',
            'from speechbrain import pretrained as sb\nsb.EmotionRecognition.from_hparams()',
            'import speechbrain.pretrained\nspeechbrain.pretrained.EmotionRecognition.from_hparams()',
        ]
        for source in sources:
            with self.subTest(source=source):
                self.assertTrue(evidence(source))
        for source in ('import speechbrain\nspeechbrain.from_hparams()',
                       'from unrelated import EmotionRecognition\nEmotionRecognition.from_hparams()',
                       'import speechbrain as sb\nsb.EncoderClassifier.from_hparams()',
                       'EmotionRecognition.from_hparams()',
                       'from speechbrain.pretrained import EmotionRecognition\nEmotionRecognition()',
                       'from speechbrain.pretrained import EmotionRecognition'):
            with self.subTest(source=source):
                self.assertFalse(evidence(source))

    def test_not_evidence(self):
        # Separate adjacent operations, each parsed and checked independently.
        sources = [
            'import speech_emotion', 'librosa.load(path)', 'soundfile.read(path)', 'wave.open(path).readframes(1)', 'torchaudio.load(path)',
            'mic.listen()', 'mic.record()', 'sounddevice.rec(100)', 'stream.read(100)', 'audio.play()',
            'whisper.transcribe(audio)', 'recognizer.recognize_google(audio)', 'engine.say(text)',
            'vad.is_speech(audio)', 'get_speech_timestamps(audio)', 'pipeline.segmentation(audio)',
            'pvporcupine.create()', 'model = Model()\nmodel.predict(audio)', 'obj.detect_keyword(audio)',
            'sb.verify_files(a,b)', 'encoder.embed_utterance(audio)', 'obj.verify_speaker(a,b)',
            'obj.diarize(audio)', 'Pipeline.from_pretrained("diarization")',
            'noisereduce.reduce_noise(y=audio)', 'df.enhance(audio)', 'obj.suppress_noise(audio)', 'obj.remove_noise(audio)',
            'acoustid.fingerprint_file(path)', 'd = Dejavu()\nd.recognize(audio)', 'obj.fingerprint_audio(audio)',
            'm = AudioTagging()\nm.inference(audio)', 'YAMNet()(audio)', 'obj.detect_sound_event(audio)',
            'demucs.separate(audio)', 's = Separator()\ns.separate(audio)', 'obj.separate_stems(audio)',
            'librosa.beat.beat_track(y=audio)', 'librosa.feature.tempo(y=audio)', 'aubio.tempo()(audio)', 'RhythmExtractor()(audio)',
            'librosa.piptrack(y=audio)', 'librosa.yin(audio)', 'crepe.predict(audio)', 'aubio.pitch()(audio)', 'snd.to_pitch()',
            'librosa.onset.onset_detect(y=audio)', 'librosa.onset.onset_strength(y=audio)', 'obj.detect_onsets(audio)',
            'librosa.feature.chroma_cqt(y=audio)', 'librosa.feature.chroma_cens(y=audio)', 'librosa.feature.chroma_stft(y=audio)',
            'librosa.feature.mfcc(y=audio)', 'librosa.feature.spectral_centroid(y=audio)', 'Chromagram()(audio)',
            'librosa.feature.zero_crossing_rate(audio)', 'obj.energy(audio)', 'obj.pitch_contour(audio)',
            'ChordsDetection()(audio)', 'ChordsDetectionBeats()(audio)', 'obj.detect_chords(audio)', 'KeyExtractor()(audio)', 'obj.detect_key(audio)',
            'basic_pitch.predict(audio)', 'm = PianoTranscription()\nm.transcribe(audio)', 'obj.transcribe_music(audio)', 'obj.audio_to_midi(audio)', 'obj.detect_notes(audio)',
            'obj.generate_audio(prompt)', 'obj.synthesize(text)', 'obj.classify_genre(audio)', 'm = GenreClassifier()\nm.predict(audio)',
            'obj.recognize_instruments(audio)', 'm = InstrumentClassifier()\nm.predict(audio)',
            'obj.detect_mood(audio)', 'obj.classify_emotion(audio)', 'm = MoodClassifier()\nm.predict(audio)', 'm = EmotionClassifier()\nm.predict(audio)',
            'obj.sentiment(lyrics)', 'obj.sentiment(transcript)', 'obj.valence(audio)', 'obj.arousal(audio)', 'obj.tag_audio(audio)',
            'classifier.predict(audio)', 'model.forward(audio)', 'obj.classify(audio)',
            'from transformers import pipeline\npipeline("audio-classification")(audio)',
            'm = pretty_midi.PrettyMIDI()\ni = pretty_midi.Instrument(0)\nn = pretty_midi.Note(1,2,3,4)',
            'music21.note.Note("C")', 'music21.stream.Stream().show()', 'verovio.renderToSVG(notes)',
            'hashlib.md5(data)', 'hashlib.sha256(data)', 'obj.filter(audio)', 'obj.equalize(audio)', 'obj.mix(a,b)', 'obj.normalize(audio)',
            'obj.detect_language(text)', 'obj.translate(text)', 'SpeechEmotionClassifier()', 'SpeechEmotionRecognizer()',
            'obj.predict(audio)', 'obj.from_hparams()', 'from speechbrain.pretrained import EmotionRecognition',
            'subprocess.run(["speech-emotion", "audio.wav"])', 'x = "detect_speech_emotion"', 'obj.detect_speech_emotion',
            'obj.detect_speech_emotion(',
        ]
        for source in sources:
            with self.subTest(source=source):
                self.assertFalse(evidence(source))

    def test_good_and_bad(self):
        self.write('README.md', 'speech emotion recognition')
        self.write('app.py', 'obj.detect_speech_emotion(audio)')
        self.assertEqual('PASS', audit(self.root)['verdict'])
        self.write('app.py', 'librosa.load(path)')
        result = audit(self.root)
        self.assertEqual([{'id': POS_ID, 'severity': 'HIGH', 'message': 'Affirmative documentation lacks production call evidence.'}], result['findings'])
        self.assertEqual(2, self.cli(str(self.root))[0])

    def test_test_only(self):
        self.write('README.md', 'speech emotion recognition')
        self.write('tests/test_app.py', 'obj.detect_speech_emotion(audio)')
        result = audit(self.root)
        self.assertFalse(result['production_evidence'])
        self.assertTrue(result['test_evidence'])
        self.assertEqual(POS_ID, result['findings'][0]['id'])
        self.assertEqual('MEDIUM', result['findings'][0]['severity'])
        self.assertEqual(0, self.cli(str(self.root))[0])
        self.assertEqual(2, self.cli(str(self.root), '--fail-on', 'MEDIUM')[0])

    def test_denial(self):
        self.write('README.md', 'does not recognize speech emotion')
        self.write('app.py', 'obj.detect_speech_emotion(audio)')
        self.assertEqual([{'id': NEG_ID, 'severity': 'HIGH', 'message': 'Negative documentation conflicts with production call evidence.'}], audit(self.root)['findings'])
        self.assertEqual(2, self.cli(str(self.root))[0])

    def test_no_claim_and_no_source(self):
        self.write('README.md', 'An ordinary tool.')
        self.assertEqual('PASS', audit(self.root)['verdict'])
        self.write('README.md', 'speech emotion recognition')
        self.assertEqual('HIGH', audit(self.root)['findings'][0]['severity'])
        self.write('README.md', 'no speech emotion recognition')
        self.assertEqual('PASS', audit(self.root)['verdict'])

    def test_source_and_doc_paths(self):
        self.write('nested/docs/guide.md', 'speech emotion recognition')
        for path in ('fixtures/a.py', 'docs/a.py', 'examples/a.py'):
            self.write(path, 'obj.detect_speech_emotion(audio)')
        self.assertEqual('HIGH', audit(self.root)['findings'][0]['severity'])
        for path in ('test_a.py', 'a_test.py', 'nested/tests/a.py'):
            p = self.write(path, 'obj.detect_speech_emotion(audio)')
            self.assertEqual('MEDIUM', audit(self.root)['findings'][0]['severity'])
            p.unlink()
        self.write('nested/README.md', 'no speech emotion recognition')
        self.write('app.py', 'obj.detect_speech_emotion(audio)')
        self.assertEqual('PASS', audit(self.root)['verdict'])

    def test_split_module_pairs(self):
        self.write('README.md', 'speech emotion recognition')
        self.write('a.py', 'm = SpeechEmotionClassifier()')
        self.write('b.py', 'm.predict(audio)')
        self.assertEqual('HIGH', audit(self.root)['findings'][0]['severity'])
        self.write('a.py', 'from lib import detect_speech_emotion as run')
        self.write('b.py', 'run(audio)')
        self.assertFalse(audit(self.root)['production_evidence'])
        self.write('a.py', 'from speechbrain.pretrained import EmotionRecognition')
        self.write('b.py', 'EmotionRecognition.from_hparams()')
        self.assertFalse(audit(self.root)['production_evidence'])

    def test_symlinks(self):
        self.write('README.md', 'speech emotion recognition')
        target = self.write('fixtures/real.py', 'obj.detect_speech_emotion(audio)')
        (self.root/'linked.py').symlink_to(target)
        (self.root/'linked_dir').symlink_to(target.parent, target_is_directory=True)
        self.assertFalse(audit(self.root)['production_evidence'])
        self.write('README.md', 'ordinary')
        doc = self.write('fixtures/claim.md', 'speech emotion recognition')
        (self.root/'docs').mkdir()
        (self.root/'docs'/'claim.md').symlink_to(doc)
        self.assertEqual('PASS', audit(self.root)['verdict'])

    def test_ignore_prefix_and_verbose(self):
        self.write('README.md', 'speech emotion recognition')
        self.write('src/app.py', 'obj.detect_speech_emotion(audio)')
        self.assertEqual(2, self.cli(str(self.root), '--ignore', 'src')[0])
        self.assertEqual(0, self.cli(str(self.root), '--ignore', 'sr')[0])
        self.assertEqual(0, self.cli(str(self.root), '--ignore', 'src', '--ignore', 'README.md')[0])
        status, out, _ = self.cli(str(self.root), '--verbose')
        self.assertEqual(0, status)
        self.assertIn('src/app.py', out)

    def test_json(self):
        self.write('README.md', 'speech emotion recognition')
        status, out, _ = self.cli(str(self.root), '--json')
        result = json.loads(out)
        self.assertEqual(2, status)
        self.assertEqual('FAIL', result['verdict'])
        self.assertEqual(POS_ID, result['findings'][0]['id'])
        self.assertTrue(result['findings'][0]['message'])
        self.assertEqual({'verdict', 'findings', 'production_evidence', 'test_evidence', 'inspected'}, set(result))

    def test_usage_io_and_version(self):
        for args in ((), ('/does/not/exist',), (str(self.root), '--unknown'), (str(self.root), '--fail-on', 'BOGUS')):
            self.assertEqual(1, self.cli(*args)[0])
        (self.root/'README.md').write_bytes(b'\xff')
        self.assertEqual(1, self.cli(str(self.root))[0])
        self.assertEqual((0, '0.1.0\n', ''), self.cli('--version'))
        self.assertEqual('0.1.0', __version__)

    def test_module_entry(self):
        result = subprocess.run([sys.executable, '-m', 'speech_emotion_recognition_claim_audit', '--version'], capture_output=True, text=True)
        self.assertEqual(0, result.returncode)
        self.assertEqual('0.1.0\n', result.stdout)

    def test_console_declaration(self):
        # Works on 3.9 without an extra TOML parser dependency.
        project = Path(__file__).resolve().parents[1]
        if (project/'pyproject.toml').exists():
            self.assertIn('speech-emotion-recognition-claim-audit = "speech_emotion_recognition_claim_audit.audit:main"', (project/'pyproject.toml').read_text())
        executable = shutil.which('speech-emotion-recognition-claim-audit')
        if executable:
            p = subprocess.run([executable, '--version'], capture_output=True, text=True)
            self.assertEqual(0, p.returncode)
            self.assertEqual('0.1.0\n', p.stdout)

    def test_readme_self_audit(self):
        project = Path(__file__).resolve().parents[1]
        self.assertEqual((False, False), claims((project/'README.md').read_text()))
        self.assertEqual('PASS', audit(project)['verdict'])


if __name__ == '__main__':
    unittest.main()
