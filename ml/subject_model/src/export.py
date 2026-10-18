"""Investigate TinyEEGNet export using Arm CMSIS-ExecuTorch's pinned PT2E flow.

Run after src.data.export_data, in the official template's isolated environment.
No architecture/training changes. Host Q/DQ metrics are not FVP/NPU metrics.
"""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import traceback

import numpy as np
import torch

from src.models.tinyeegnet import TinyEEGNet


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def metrics(logits, labels):
    y, pred = labels.numpy(), logits.argmax(1).numpy()
    cm = np.zeros((2, 2), dtype=np.int64)
    np.add.at(cm, (y, pred), 1)
    return {'accuracy': float(np.mean(y == pred)),
            'balanced_accuracy': float(np.mean(cm.diagonal() / cm.sum(1))),
            'confusion_matrix': cm.tolist(), 'n': len(y)}


def run(directory, official, config_ini=None, system_config="Ethos_U85_SYS_DRAM_Mid", checkpoint_dir=None):
    from executorch.backends.arm.ethosu import EthosUCompileSpec, EthosUPartitioner
    from executorch.backends.arm.quantizer import EthosUQuantizer, get_symmetric_quantization_config
    from executorch.exir import EdgeCompileConfig, ExecutorchBackendConfig, to_edge_transform_and_lower
    from executorch.exir.passes.quantize_io_pass import QuantizeInputs
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

    report = {'official_repository': 'https://github.com/Arm-Examples/CMSIS-Executorch',
              'official_commit': subprocess.check_output(['git', '-C', str(official), 'rev-parse', 'HEAD'], text=True).strip(),
              'target': 'ethos-u85-256', 'system_config': system_config,
              'memory_mode': 'Shared_Sram', 'profile': ('Alif official E8 SRAM/MRAM profile; physical execution unvalidated' if config_ini else 'official Corstone-320 template; E8 memory mapping unvalidated'),
              'config_ini': str(config_ini) if config_ini else None,
              'fvp': {'ran': False, 'blocker': 'No native macOS FVP, Docker, CMSIS-Toolbox, or Arm firmware compiler found'},
              'versions': {name: importlib.metadata.version(name) for name in
                           ('torch', 'torchao', 'executorch', 'ethos-u-vela', 'tosa-tools', 'flatbuffers')},
              'stages': {}}
    try:
        torch.set_num_threads(4)
        torch.manual_seed(42)
        baseline = Path(checkpoint_dir) if checkpoint_dir else Path('artifacts/combined-seed42-20261016')
        data = torch.load(directory / 'export_data.pt', weights_only=True)
        checkpoint = torch.load(baseline / 'best_checkpoint.pt', weights_only=True)
        model = TinyEEGNet(dropout=0.5).eval()
        model.load_state_dict(checkpoint['model_state'])
        report['parameters'] = sum(p.numel() for p in model.parameters())
        report['checkpoint_sha256'] = hashlib.sha256((baseline / 'best_checkpoint.pt').read_bytes()).hexdigest()
        calibration = data['calibration']
        development = data['development_x']
        example = (calibration[:1],)
        with torch.inference_mode():
            fl = torch.cat([model(x[None]) for x in development])
        torch.testing.assert_close(fl, data['reference_logits'], atol=1e-5, rtol=1e-5)
        report['float_development'] = metrics(fl, data['development_y'])
        report['stages']['checkpoint_reload'] = 'passed; matches original environment logits'
        exported = torch.export.export(deepcopy(model), example)
        (directory / 'float_graph.txt').write_text(str(exported.graph_module.graph))
        report['float_operators'] = dict(Counter(str(n.target) for n in exported.graph.nodes if n.op == 'call_function'))
        spec_kwargs = {'config_ini': str(config_ini)} if config_ini else {}
        spec = EthosUCompileSpec(target=report['target'], system_config=report['system_config'], memory_mode=report['memory_mode'], **spec_kwargs)
        spec.dump_intermediate_artifacts_to(str(directory / 'compiler_debug'))
        quantizer = EthosUQuantizer(spec)
        quantizer.set_global(get_symmetric_quantization_config(is_per_channel=True))
        prepared = prepare_pt2e(exported.module(), quantizer)
        with torch.no_grad():
            for sample in calibration:
                prepared(sample[None])
        quantized = convert_pt2e(prepared)
        qexport = torch.export.export(quantized, example)
        torch.export.save(qexport, directory / 'quantized_qdq.pt2')
        (directory / 'quantized_graph.txt').write_text(str(qexport.graph_module.graph))
        report['stages']['quantization'] = f'passed; {len(calibration)} training-only calibration windows'
        report['calibration_class_counts'] = torch.bincount(data['calibration_y'], minlength=2).tolist()
        with torch.no_grad():
            ql = torch.cat([quantized(x[None]) for x in development])
        report['quantized_qdq_development'] = metrics(ql, data['development_y'])
        report['accuracy_change_points'] = 100 * (report['quantized_qdq_development']['accuracy'] - report['float_development']['accuracy'])
        report['prediction_agreement'] = float((ql.argmax(1) == fl.argmax(1)).float().mean())
        report['maximum_logit_difference'] = float(torch.max(torch.abs(ql - fl)))
        report['quantized_metrics_scope'] = 'host PT2E Q/DQ reference; not compiled Vela/FVP numerical execution'
        np.savez(directory / 'development_predictions.npz', float_logits=fl.numpy(), qdq_logits=ql.numpy(), labels=data['development_y'].numpy())
        print('QUANTIZED_REFERENCE', json.dumps({k:v for k,v in report.items() if 'development' in k or 'agreement' in k}), flush=True)
        edge = to_edge_transform_and_lower(qexport, partitioner=[EthosUPartitioner(spec)],
                                            compile_config=EdgeCompileConfig(_check_ir_validity=False))
        (directory / 'edge_graph.txt').write_text(str(edge.exported_program().graph_module.graph))
        delegate_reports = []
        for name, module in edge.exported_program().graph_module.named_modules():
            if hasattr(module, 'backend_id') and hasattr(module, 'processed_bytes'):
                delegate_reports.append({'name': name, 'backend': module.backend_id,
                    'compiled_bytes': len(module.processed_bytes),
                    'operators': dict(Counter(str(n.target) for n in module.original_module.graph.nodes if n.op == 'call_function'))})
        report['delegates'] = delegate_reports
        report['residual_cpu_operators'] = [str(n.target) for n in edge.exported_program().graph.nodes
                                            if n.op == 'call_function' and 'executorch_call_delegate' not in str(n.target)]
        if not delegate_reports:
            raise RuntimeError('No Ethos-U delegate produced')
        # Runtime lowering mutates IR; preserve functional Edge IR for the int8 boundary pass.
        program = deepcopy(edge).to_executorch(ExecutorchBackendConfig(extract_delegate_segments=False))
        (directory / 'tinyeegnet_quantized_float_io.pte').write_bytes(bytes(program.buffer))
        report['float_io_pte_bytes'] = len(program.buffer)
        report['stages']['vela_compilation_and_serialization'] = 'passed'
        # Supported ExecuTorch boundary pass removes only the CPU input quantizer.
        # The compiled delegate already takes int8; no model architecture change.
        quantize_input = QuantizeInputs(edge_program_manager=edge, quantized_inputs_idx=[0], method_name='forward')
        int8_edge = edge.transform([quantize_input])
        qargs = quantize_input.quant_args[0]
        scale, zero_point, qmin, qmax, dtype = qargs
        report['input_quantization'] = {'scale': float(scale), 'zero_point': int(zero_point),
                                       'quant_min': int(qmin), 'quant_max': int(qmax), 'dtype': str(dtype),
                                       'shape': [1, 1, 22, 500],
                                       'formula': 'clip(round(normalized_float / scale) + zero_point, quant_min, quant_max)'}
        int8_program = int8_edge.to_executorch(ExecutorchBackendConfig(extract_delegate_segments=False))
        (directory / 'tinyeegnet_int8.pte').write_bytes(bytes(int8_program.buffer))
        report['int8_io_pte_bytes'] = len(int8_program.buffer)
        report['stages']['int8_input_boundary'] = 'passed'
        (directory / 'int8_edge_graph.txt').write_text(str(int8_edge.exported_program().graph_module.graph))
        params = json.loads((directory / 'float_preprocess_params.json').read_text())
        params['quantization'] = {'input': report['input_quantization'], 'output': 'dequantized float32 logits'}
        write_json(directory / 'preprocess_params.json', params)
        # Training-derived calibration examples only, for later FVP/board checks.
        examples = calibration[:20]
        qinputs = torch.clamp(torch.round(examples / scale) + zero_point, qmin, qmax).to(dtype)
        with torch.no_grad():
            expected = torch.cat([quantized(v[None]) for v in examples])
        np.savez(directory / 'test_vectors.npz', inputs=qinputs.numpy(), expected_logits=expected.numpy(),
                 reference_kind=np.array('host_qdq_not_npu'))
        report['status'] = 'compiled; FVP and physical E8 execution not validated'
    except Exception as error:
        report['status'] = 'blocked at next stage; see completed stages'
        report['error'] = f'{type(error).__name__}: {error}'
        (directory / 'export_error.txt').write_text(traceback.format_exc())
        traceback.print_exc()
    finally:
        old = json.loads((directory / 'preserved_hashes.json').read_text())
        report['existing_files_unchanged'] = all(Path(p).is_file() and hashlib.sha256(Path(p).read_bytes()).hexdigest() == digest for p,digest in old.items())
        write_json(directory / 'export_report.json', report)
        print(json.dumps(report, indent=2), flush=True)
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workdir', type=Path, default=Path('artifacts/ethosu85-export-seed42-20261016'))
    parser.add_argument('--official-repo', type=Path, default=Path('/private/tmp/tinyeegnet-cmsis-executorch'))
    parser.add_argument('--config-ini', type=Path)
    parser.add_argument('--system-config', default='Ethos_U85_SYS_DRAM_Mid')
    parser.add_argument('--checkpoint-dir', type=Path, help='Separate trained run; defaults to original multi-subject baseline')
    args = parser.parse_args()
    if (args.workdir / 'export_report.json').exists():
        raise FileExistsError('Refusing to overwrite an export investigation')
    report = run(args.workdir, args.official_repo, args.config_ini, args.system_config, args.checkpoint_dir)
    results = Path('results') / args.workdir.name
    results.mkdir(exist_ok=False)
    shutil.copyfile(args.workdir / 'export_report.json', results / 'export_report.json')
    if 'error' in report:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
