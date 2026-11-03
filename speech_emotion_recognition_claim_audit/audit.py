"""Offline prose matching and conservative module-level AST evidence."""
import argparse
import ast
import json
import os
from pathlib import Path
import re
import sys

from . import __version__
from .phrases import SPEECHEMOTION, NOSPEECHEMOTION

GENERIC = frozenset(('detect_speech_emotion', 'detect_speech_emotions', 'classify_speech_emotion', 'classify_speech_emotions', 'predict_speech_emotion', 'predict_speech_emotions', 'recognize_speech_emotion', 'recognise_speech_emotion', 'recognize_speech_emotions', 'recognise_speech_emotions', 'identify_speech_emotion', 'identify_speech_emotions', 'get_speech_emotion', 'get_speech_emotions', 'extract_speech_emotion', 'extract_speech_emotions', 'analyze_speech_emotion', 'analyse_speech_emotion', 'analyze_speech_emotions', 'analyse_speech_emotions', 'detect_voice_emotion', 'detect_voice_emotions', 'classify_voice_emotion', 'classify_voice_emotions', 'predict_voice_emotion', 'predict_voice_emotions', 'recognize_voice_emotion', 'recognise_voice_emotion', 'identify_voice_emotion', 'get_voice_emotion', 'detect_emotion_in_speech', 'classify_emotion_in_speech', 'predict_emotion_in_speech', 'recognize_emotion_in_speech', 'recognise_emotion_in_speech', 'detect_spoken_emotion', 'classify_spoken_emotion', 'predict_spoken_emotion', 'speech_emotion_detect', 'speech_emotion_classify', 'speech_emotion_predict'))

def prose(text):
    lines = []
    fence = None
    for line in text.splitlines():
        match = re.match(r'^ {0,3}(`{3,}|~{3,})(.*)$', line)
        if fence:
            if match and match[1][0] == fence[0] and len(match[1]) >= len(fence) and not match[2].strip():
                fence = None
            lines.append('')
        elif match:
            fence = match[1]
            lines.append('')
        else:
            lines.append(line)
    return '\n'.join(lines)


def claims(text):
    positive = negative = False
    for paragraph in re.split(r'\n\s*\n', prose(text)):
        normalized = ' '.join(paragraph.lower().replace("n't", ' not').replace('-', ' ').split())
        for phrase in sorted(NOSPEECHEMOTION, key=lambda p: (-len(p), p)):
            normalized, count = re.subn(r'\b' + re.escape(phrase) + r'\b', ' ', normalized)
            negative |= bool(count)
        for phrase in sorted(SPEECHEMOTION, key=lambda p: (-len(p), p)):
            normalized, count = re.subn(r'\b' + re.escape(phrase) + r'\b', ' ', normalized)
            positive |= bool(count)
    return positive, negative


def evidence(source):
    """Recognize only the three specified, module-local AST routes."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return False
    nodes = list(ast.walk(tree))
    local = {n.name for n in nodes if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
    functions = {n.name for n in nodes if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    aliases = {}
    speechbrain = {}
    for node in nodes:
        if isinstance(node, ast.ImportFrom):
            for item in node.names:
                binding = item.asname or item.name
                aliases[binding] = item.name
                if node.module and (node.module == 'speechbrain' or node.module.startswith('speechbrain.')):
                    speechbrain[binding] = node.module + '.' + item.name
        elif isinstance(node, ast.Import):
            for item in node.names:
                binding = item.asname or item.name.split('.')[0]
                aliases[binding] = item.name
                if item.name == 'speechbrain' or item.name.startswith('speechbrain.'):
                    speechbrain[binding] = item.name if item.asname else 'speechbrain'

    def name(func):
        if isinstance(func, ast.Attribute):
            return func.attr
        if isinstance(func, ast.Name):
            return aliases.get(func.id, func.id).split('.')[-1]
        return ''

    def constructor(call):
        if not isinstance(call, ast.Call):
            return False
        target = name(call.func)
        return (target.endswith(('SpeechEmotionClassifier', 'SpeechEmotionRecognizer'))
                and target not in local
                and not (isinstance(call.func, ast.Name) and call.func.id in local))

    def brain_base(expr):
        tail = []
        while isinstance(expr, ast.Attribute):
            tail.append(expr.attr)
            expr = expr.value
        if not isinstance(expr, ast.Name) or expr.id not in speechbrain:
            return False
        resolved = speechbrain[expr.id].split('.') + list(reversed(tail))
        return resolved[-1] == 'EmotionRecognition'

    bindings = set()
    for node in nodes:
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and constructor(node.value):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            bindings.update(t.id for t in targets if isinstance(t, ast.Name))
    for node in nodes:
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        target = name(func)
        if target in GENERIC and target not in functions:
            if isinstance(func, ast.Attribute) or (isinstance(func, ast.Name) and func.id in aliases and func.id not in local):
                return True
        if isinstance(func, ast.Attribute):
            if func.attr == 'predict' and (constructor(func.value) or (isinstance(func.value, ast.Name) and func.value.id in bindings)):
                return True
            if func.attr == 'from_hparams' and brain_base(func.value):
                return True
    return False


def audit(root, ignore=()):
    root = Path(root)
    if not root.is_dir() or root.is_symlink():
        raise OSError('Root must be a readable directory')
    ignored = [Path(p).as_posix().strip('/') for p in ignore]
    positive = negative = production = testing = False
    inspected = []
    def onerror(error):
        raise error
    for directory, dirs, files in os.walk(root, followlinks=False, onerror=onerror):
        base = Path(directory)
        def excluded(path):
            rel = path.relative_to(root).as_posix()
            return path.is_symlink() or any(rel == p or rel.startswith(p + '/') for p in ignored)
        dirs[:] = sorted(d for d in dirs if not excluded(base / d))
        for filename in sorted(files):
            path = base / filename
            if excluded(path):
                continue
            rel = path.relative_to(root)
            parts = rel.parts
            doc = rel.as_posix() == 'README.md' or (path.suffix == '.md' and 'docs' in parts[:-1])
            if doc:
                p, n = claims(path.read_text(encoding='utf-8'))
                positive |= p
                negative |= n
                inspected.append(rel.as_posix())
            if path.suffix == '.py' and not {'fixtures', 'docs', 'examples'}.intersection(parts[:-1]):
                found = evidence(path.read_text(encoding='utf-8'))
                test = 'tests' in parts[:-1] or filename.startswith('test_') or filename.endswith('_test.py')
                testing |= found and test
                production |= found and not test
                inspected.append(rel.as_posix())
    findings = []
    if positive and not production:
        findings.append({'id': 'documented-speech-emotion-recognition-claim-false', 'severity': 'MEDIUM' if testing else 'HIGH', 'message': 'Affirmative documentation lacks production call evidence.'})
    if negative and production:
        findings.append({'id': 'documented-no-speech-emotion-recognition-claim-false', 'severity': 'HIGH', 'message': 'Negative documentation conflicts with production call evidence.'})
    return {'verdict': 'FAIL' if findings else 'PASS', 'findings': findings, 'production_evidence': production, 'test_evidence': testing, 'inspected': inspected}


class Parser(argparse.ArgumentParser):
    def error(self, message):
        self.print_usage(sys.stderr)
        self.exit(1, self.prog + ': error: ' + message + '\n')


def main(argv=None):
    parser = Parser(description=__doc__)
    parser.add_argument('root')
    parser.add_argument('--version', action='version', version=__version__)
    parser.add_argument('--json', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--ignore', action='append', default=[])
    parser.add_argument('--fail-on', choices=['LOW', 'MEDIUM', 'HIGH'], default='HIGH')
    args = parser.parse_args(argv)
    try:
        result = audit(args.root, args.ignore)
    except (OSError, UnicodeError) as error:
        print(str(error), file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(result))
    else:
        print(result['verdict'])
        for finding in result['findings']:
            print('{id} {severity}: {message}'.format(**finding))
        if args.verbose:
            print(json.dumps({k: v for k, v in result.items() if k not in ('verdict', 'findings')}))
    levels = {'LOW': 0, 'MEDIUM': 1, 'HIGH': 2}
    return 2 if any(levels[f['severity']] >= levels[args.fail_on] for f in result['findings']) else 0
