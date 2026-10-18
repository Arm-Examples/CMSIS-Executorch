#!/usr/bin/env python3
"""Prepare the isolated v4 firmware and bounded FVP fixtures. Never builds or touches a board.

Initial preparation uses the published 20 A03 vectors. Once data preparation is
complete, --data --start N --count 240 prepares a part of the 1080-trial suite.
--indices accepts a deliberate order (including repeats) for determinism checks.
The FVP fixture bound preserves the template's 2 MiB code region; production has
no trial-count/run-duration limit. Existing Model 2 sources/projects are read-only.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import struct
import zlib
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MODEL_SHA = '6caa8a9876d3cd0c36b39b3d4474a0fe7dbdbfb32a4d6f80ed4776623a19e02e'
PARAMS_SHA = '3d34f6326c67441e9e6b636454436c63951e81f6e221da977723700f1c8152f7'
SAMPLES = 5632
SESSION = 2026101804

def sha(data):
    return hashlib.sha256(data).hexdigest()

def frame(typ, request, trial, payload=b''):
    raw = struct.pack('<4sBBHQIIIHH', b'EEG2', 1, typ, 0, SESSION, request, trial, 0, len(payload), 0) + payload
    raw += struct.pack('<I', zlib.crc32(raw))
    out = bytearray([0]); at = 0; code = 1
    for value in raw:
        if value == 0:
            out[at] = code; at = len(out); out.append(0); code = 1
        else:
            out.append(value); code += 1
            if code == 255:
                out[at] = code; at = len(out); out.append(0); code = 1
    out[at] = code; out.append(0)
    return bytes(out)

def initialize(template, project):
    project.mkdir(parents=True, exist_ok=False)
    for folder in ('src', 'board', 'ai_layer'):
        shutil.copytree(template / folder, project / folder,
                        ignore=shutil.ignore_patterns('test_frames.h', '__pycache__', '.DS_Store', '*.o', '*.d'))
    for name in ('cmsis-executorch.csolution.yml', 'cmsis-executorch.cproject.yml',
                 'cmsis-executorch.cbuild-pack.yml', 'cmsis-executorch.cdefault.yml', 'vcpkg-configuration.json'):
        src = template / name
        if src.is_file():
            shutil.copy2(src, project / name)
    (project / '.vscode').mkdir()
    for name in ('fvp-base.sh', 'fvp.sh', 'fvp.Dockerfile', 'settings.json'):
        src = template / '.vscode' / name
        if src.is_file():
            shutil.copy2(src, project / '.vscode' / name)
    solution = project / 'cmsis-executorch.csolution.yml'
    text = solution.read_text().replace(str(template), str(project))
    text = text.replace('input-shape: 1x1x22x500', 'input-shape: 1x1x22x256')
    text = text.replace('calibration-samples: 2', 'calibration-samples: 200')
    text = text.replace('name: TinyCNN', 'name: TinyEEGNet-v4-prebuilt')
    text = text.replace('description: TinyCNN int8 image classifier for Ethos-U85',
                        'description: Unmodified prebuilt Moamen v4 model; do not run an exporter')
    # This is documentation of the precompiled model's profile, not a re-export.
    text = text.replace('system: Ethos_U85_SRAM_MRAM', 'system: Ethos_U85_SYS_DRAM_Mid')
    text = text.replace('memory: Shared_Sram', 'memory: Sram_Only')
    # Drop inherited exporter/provisioning instructions: this project only embeds
    # the validated bundle and has no authorized physical deployment step.
    text = '\n'.join(line for line in text.splitlines() if not line.lstrip().startswith('#')) + '\n'
    solution.write_text('# Isolated v4 prebuilt program; do not re-export or provision.\n' + text)
    (project / 'cmsis-executorch.cbuild-set.yml').write_text(
        'cbuild-set:\n  contexts:\n    - context: cmsis-executorch.Debug+SSE-320-U85\n')
    marker = {'schema': 1, 'template': str(template), 'board_actions': 'none', 'default_target': 'SSE-320-U85'}
    (project / '.multisubject-project.json').write_text(json.dumps(marker, indent=2) + '\n')

def source_cases(bundle):
    with np.load(bundle / 'test_vectors.npz', allow_pickle=False) as vectors:
        inputs, labels, logits = vectors['x'], vectors['y'], vectors['logits']
        assert inputs.dtype == np.int8 and inputs.shape == (20, 1, 22, 256)
        return [{'trial': 240+i, 'trial_id': f'BNCI2014_001/S03/session2/lr{25+i:03d}',
                 'ground_truth': int(labels[i]), 'payload': inputs[i].tobytes(),
                 'qdq_logits': logits[i].tolist()} for i in range(20)]

def data_cases(data, indexes):
    metadata = json.loads((data / 'metadata.json').read_text())
    rows = metadata['trials']; binary = (data / 'inputs.bin').read_bytes()
    assert len(rows) == 1080 and len(binary) == 1080*SAMPLES
    assert metadata['model_sha256'] == MODEL_SHA
    cases = []
    for index in indexes:
        assert 0 <= index < 1080
        row = rows[index]; assert row['index'] == index
        payload = binary[index*SAMPLES:(index+1)*SAMPLES]
        assert sha(payload) == row['input_sha256']
        cases.append({'trial': index, 'trial_id': row['trial_id'],
                      'ground_truth': row['ground_truth'], 'payload': payload})
    return cases

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--template', type=Path, default=ROOT/'artifacts/streaming-firmware/project')
    parser.add_argument('--project', type=Path, default=ROOT/'artifacts/multisubject-v4/firmware/project')
    parser.add_argument('--bundle', type=Path, default=ROOT/'artifacts/multisubject-v4/bundle')
    parser.add_argument('--data', type=Path, help='prepared metadata.json + inputs.bin; otherwise use the 20 published vectors')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--count', type=int, default=120)
    parser.add_argument('--indices', help='comma-separated order for repeated/reordered inputs')
    args = parser.parse_args()
    template, project, bundle = args.template.resolve(), args.project.resolve(), args.bundle.resolve()
    if project == template or template in project.parents or project in template.parents:
        raise ValueError('The multisubject project must be separate from the working Model 2 template')
    pte = (bundle/'tinyeegnet_int8.pte').read_bytes()
    params_bytes = (bundle/'preprocess_params.json').read_bytes()
    assert len(pte) == 11456 and sha(pte) == MODEL_SHA
    assert sha(params_bytes) == PARAMS_SHA
    params = json.loads(params_bytes)
    assert params['input_shape'] == [1, 1, 22, 256] and params['input_zero_point'] == -12
    assert params['input_quant_scale'] == 0.06955837458372116
    if not project.exists():
        initialize(template, project)
    elif not (project/'.multisubject-project.json').is_file():
        raise ValueError('Refusing to modify an existing project without the isolated-project marker')
    # Arm publishes no Darwin FVP binary. The existing macOS Docker shim pins
    # its own image version; do not export nonexistent native paths into CMSIS.
    if sys.platform == 'darwin':
        manifest = project/'vcpkg-configuration.json'
        config = json.loads(manifest.read_text())
        if config.get('requires', {}).pop('arm:models/arm/avh-fvp', None) is not None:
            manifest.write_text(json.dumps(config, indent=2)+'\n')
    for name in ('app_main.cpp', 'protocol.hpp'):
        shutil.copy2(ROOT/'firmware/multisubject'/name, project/'src'/name)
    # Reuse the exact proven transport, not a second modified UART implementation.
    uart = ROOT/'firmware/streaming/uart_stream.c'
    shutil.copy2(uart, project/'board/DevKit-E8/retarget_stdio.c')
    model_c = '#include "model_pte.h"\n__attribute__((aligned(16))) const unsigned char model_pte[] = {\n'
    model_c += '\n'.join(','.join(f'0x{x:02x}' for x in pte[i:i+24])+',' for i in range(0, len(pte), 24))
    model_c += '\n};\nconst unsigned long model_pte_size = sizeof(model_pte);\n'
    (project/'ai_layer/model_pte.c').write_text(model_c)
    layer = project/'ai_layer/ai_layer.clayer.yml'
    text = layer.read_text()
    component = '    - component: Machine Learning:ExecuTorch Operators:Quantized quantize\n'
    if component not in text:
        text = text.replace('    - component: Machine Learning:ExecuTorch Operators:Quantized dequantize\n',
                            '    - component: Machine Learning:ExecuTorch Operators:Quantized dequantize\n'+component)
    layer.write_text(text)
    if args.data:
        indexes = [int(x) for x in args.indices.split(',')] if args.indices else list(range(args.start, args.start+args.count))
        cases = data_cases(args.data.resolve(), indexes)
    else:
        if args.indices:
            raise ValueError('--indices requires --data')
        cases = source_cases(bundle)
    if not 1 <= len(cases) <= 240:
        raise ValueError('Use 1..240 FVP fixtures per build to preserve the template memory map; production has no fixed run limit')
    wire = bytearray(frame(1, 0, 0xffffffff))
    bad = bytearray(frame(3, 1, cases[0]['trial'], cases[0]['payload'])); bad[100] ^= 1
    wire.extend(bad); wire.extend(b'\x05bad\0')
    expected = []
    for request, case in enumerate(cases, 1):
        payload = case['payload']; assert len(payload) == SAMPLES
        wire.extend(frame(3, request, case['trial'], payload))
        expected.append({k: v for k, v in case.items() if k != 'payload'} | {
            'session': SESSION, 'request': request, 'input_sha256': sha(payload), 'input_crc32': zlib.crc32(payload)})
    header = '#pragma once\nstatic const unsigned expected_results='+str(len(cases))+';\nstatic const unsigned expected_rejected=2;\nstatic const unsigned char test_frames[]={\n'
    header += '\n'.join(','.join(map(str, wire[i:i+32]))+',' for i in range(0, len(wire), 32))+'\n};\n'
    (project/'src/test_frames.h').write_text(header)
    manifest = {'schema': 1, 'model_sha256': MODEL_SHA, 'params_sha256': PARAMS_SHA,
                'uart_source_sha256': sha(uart.read_bytes()), 'wire_dtype': 'int8', 'pte_input_dtype': 'float32',
                'input_shape': [1,1,22,256], 'input_quant_scale': params['input_quant_scale'],
                'input_zero_point': -12, 'test_frames_sha256': sha(bytes(wire)), 'test_frames_bytes': len(wire),
                'default_target': 'SSE-320-U85', 'physical_board_actions': 'none', 'cases': expected}
    (project.parent/'expected.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({'project': str(project), 'model_sha256': MODEL_SHA,
                      'fvp_cases': len(cases), 'wire_bytes': len(wire), 'default_target': 'SSE-320-U85'}))

if __name__ == '__main__':
    main()
