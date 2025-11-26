#!/usr/bin/env python3
"""
Generate a comprehensive AI layer build report including:
- Library assets and sizes
- Compile options and configuration
- Model conversion logs and metadata
- Build environment information
- Operator selection details
"""

import os
import re
import sys
import json
import hashlib
import shutil
import subprocess
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

ROOT = Path(__file__).resolve().parent.parent
AI_LAYER = ROOT / "ai_layer"
ENGINE = AI_LAYER / "engine"
LIB_DIR = ENGINE / "lib"
INCLUDE_DIR = ENGINE / "include" / "executorch"
META_DIR = ENGINE / "meta"
MODEL_DIR = ENGINE / "model"
OUT_DIR = ROOT / "out"
STAGE1_DIR = OUT_DIR / "stage1"
STAGE2_DIR = OUT_DIR / "stage2"
REPORT_PATH = AI_LAYER / "REPORT.md"

def human_size(size_bytes: int) -> str:
    """Convert bytes to human readable format."""
    if size_bytes == 0:
        return "0 B"
    elif size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KiB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MiB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.1f} GiB"

def hash_file(path: Path) -> str:
    """Generate SHA256 hash of file (first 16 chars)."""
    h = hashlib.sha256()
    try:
        with path.open('rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                h.update(chunk)
        return h.hexdigest()[:16]
    except Exception:
        return "error"

def run_command(cmd: List[str], cwd: Optional[Path] = None) -> Optional[str]:
    """Run command and return stdout, or None on failure."""
    try:
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            cwd=cwd,
            timeout=30
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return None
    except Exception:
        return None

def get_git_info() -> Dict[str, str]:
    """Get git repository information."""
    info = {}
    
    # Get current commit hash
    sha = run_command(['git', 'rev-parse', 'HEAD'], ROOT)
    info['commit_hash'] = sha[:12] if sha else "unknown"
    
    # Get current branch
    branch = run_command(['git', 'rev-parse', '--abbrev-ref', 'HEAD'], ROOT)
    info['branch'] = branch if branch else "unknown"
    
    # Get last commit date
    date = run_command(['git', 'log', '-1', '--format=%ci'], ROOT)
    info['last_commit'] = date if date else "unknown"
    
    # Get repository status
    status = run_command(['git', 'status', '--porcelain'], ROOT)
    info['dirty'] = bool(status and status.strip())
    
    return info

def collect_library_assets() -> List[Dict[str, Any]]:
    """Collect information about all library files."""
    libraries = []
    
    if not LIB_DIR.exists():
        return libraries
    
    total_size = 0
    for lib_file in sorted(LIB_DIR.glob('*.a')):
        stat = lib_file.stat()
        size = stat.st_size
        total_size += size
        
        libraries.append({
            'name': lib_file.name,
            'size': size,
            'size_human': human_size(size),
            'modified': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
            'hash': hash_file(lib_file),
            'path': str(lib_file.relative_to(ROOT))
        })
    
    # Calculate percentages
    for lib in libraries:
        lib['percentage'] = (lib['size'] / total_size * 100) if total_size > 0 else 0
    
    return libraries

def collect_model_assets() -> List[Dict[str, Any]]:
    """Collect information about model files."""
    models = []
    
    # Check model directory
    if MODEL_DIR.exists():
        for model_file in sorted(MODEL_DIR.glob('*.pte')):
            stat = model_file.stat()
            models.append({
                'name': model_file.name,
                'size': stat.st_size,
                'size_human': human_size(stat.st_size),
                'modified': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
                'hash': hash_file(model_file),
                'path': str(model_file.relative_to(ROOT))
            })
    
    # Also check for header files
    for header_file in sorted(MODEL_DIR.glob('*.h')):
        stat = header_file.stat()
        models.append({
            'name': header_file.name,
            'size': stat.st_size,
            'size_human': human_size(stat.st_size),
            'modified': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
            'hash': hash_file(header_file),
            'path': str(header_file.relative_to(ROOT))
        })
    
    return models

def get_build_configuration() -> Dict[str, Any]:
    """Extract build configuration from CMake cache and other sources."""
    config = {}
    
    # Check Stage1 CMake cache
    stage1_cache = STAGE1_DIR / "CMakeCache.txt"
    if stage1_cache.exists():
        cmake_vars = {}
        try:
            content = stage1_cache.read_text()
            for line in content.splitlines():
                if '=' in line and not line.startswith('//') and not line.startswith('#'):
                    key, value = line.split('=', 1)
                    if ':' in key:
                        key = key.split(':')[0]
                    cmake_vars[key] = value
        except Exception:
            pass
        
        config['cmake_build_type'] = cmake_vars.get('CMAKE_BUILD_TYPE', 'unknown')
        config['cmake_toolchain_file'] = cmake_vars.get('CMAKE_TOOLCHAIN_FILE', 'none')
        config['executorch_build_arm_baremetal'] = cmake_vars.get('EXECUTORCH_BUILD_ARM_BAREMETAL', 'unknown')
        config['executorch_build_cortex_m'] = cmake_vars.get('EXECUTORCH_BUILD_CORTEX_M', 'unknown')
        config['executorch_build_portable_ops'] = cmake_vars.get('EXECUTORCH_BUILD_PORTABLE_OPS', 'unknown')
        config['executorch_build_kernels_quantized'] = cmake_vars.get('EXECUTORCH_BUILD_KERNELS_QUANTIZED', 'unknown')
    
    # Check for operators file
    operators_file = ROOT / "model" / "operators_minimal.txt"
    if operators_file.exists():
        try:
            operators = []
            for line in operators_file.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith('#'):
                    operators.append(line)
            config['selected_operators'] = operators
            config['operator_count'] = len(operators)
        except Exception:
            config['selected_operators'] = []
            config['operator_count'] = 0
    
    return config

def get_model_conversion_log() -> str:
    """Try to extract model conversion information."""
    log_info = []
    
    # Check if model conversion script exists and get info
    model_script = ROOT / "model" / "aot_model.py"
    if model_script.exists():
        try:
            content = model_script.read_text()
            
            # Extract key configuration from the script
            compile_spec_match = re.search(r'EthosUCompileSpec\((.*?)\)', content, re.DOTALL)
            if compile_spec_match:
                log_info.append("**Ethos-U Compile Specification:**")
                spec_content = compile_spec_match.group(1)
                for line in spec_content.split(','):
                    line = line.strip()
                    if '=' in line:
                        log_info.append(f"  - {line}")
            
            # Extract quantization config
            if 'quantizer' in content:
                log_info.append("\n**Quantization Configuration:**")
                log_info.append("  - Using EthosUQuantizer with symmetric quantization")
                log_info.append("  - Post-training quantization enabled")
            
            # Extract model details
            class_match = re.search(r'class (\w+)\(torch\.nn\.Module\):', content)
            if class_match:
                model_name = class_match.group(1)
                log_info.append(f"\n**Model Architecture:**")
                log_info.append(f"  - Model class: {model_name}")
                
                # Try to extract forward method
                forward_match = re.search(r'def forward\(self.*?\):(.*?)(?=def|\Z)', content, re.DOTALL)
                if forward_match:
                    forward_content = forward_match.group(1).strip()
                    log_info.append(f"  - Forward method: {forward_content.splitlines()[0].strip()}")
        
        except Exception:
            log_info.append("Error reading model conversion script")
    
    return "\n".join(log_info) if log_info else "No model conversion information available"

def get_build_environment() -> Dict[str, str]:
    """Get build environment information."""
    env = {}
    
    # Python version
    python_version = run_command(['python3', '--version'])
    env['python'] = python_version if python_version else "unknown"
    
    # CMake version
    cmake_version = run_command(['cmake', '--version'])
    if cmake_version:
        env['cmake'] = cmake_version.splitlines()[0]
    else:
        env['cmake'] = "unknown"
    
    # GCC version (if available)
    gcc_version = run_command(['arm-none-eabi-gcc', '--version'])
    if gcc_version:
        env['gcc_arm'] = gcc_version.splitlines()[0]
    else:
        env['gcc_arm'] = "not available"
    
    # System info
    env['platform'] = run_command(['uname', '-a']) or "unknown"
    
    return env

def main():
    """Generate comprehensive AI layer build report."""
    
    print("Generating comprehensive AI layer build report...")
    
    # Collect all information
    git_info = get_git_info()
    libraries = collect_library_assets()
    models = collect_model_assets()
    build_config = get_build_configuration()
    model_log = get_model_conversion_log()
    build_env = get_build_environment()
    
    timestamp = datetime.utcnow().isoformat(timespec='seconds') + 'Z'
    
    # Generate report
    lines = [
        "# ExecuTorch AI Layer Build Report",
        "",
        f"**Generated:** {timestamp}",
        f"**Git Commit:** {git_info.get('commit_hash', 'unknown')} on {git_info.get('branch', 'unknown')}",
        f"**Repository Status:** {'🔄 Modified' if git_info.get('dirty', False) else '✅ Clean'}",
        f"**Last Commit:** {git_info.get('last_commit', 'unknown')}",
        "",
        "## 📊 Build Summary",
        "",
        f"- **Libraries:** {len(libraries)} static libraries",
        f"- **Models:** {len(models)} model assets",
        f"- **Operators:** {build_config.get('operator_count', 0)} selected operators",
        f"- **Build Type:** {build_config.get('cmake_build_type', 'unknown')}",
        "",
        "## 📚 Library Assets",
        ""
    ]
    
    if libraries:
        total_size = sum(lib['size'] for lib in libraries)
        lines.extend([
            f"**Total Size:** {human_size(total_size)}",
            "",
            "| Library | Size | Percentage | Modified | Hash |",
            "|---------|------|------------|----------|------|"
        ])
        
        for lib in libraries:
            lines.append(
                f"| {lib['name']} | {lib['size_human']} | {lib['percentage']:.1f}% | {lib['modified']} | `{lib['hash']}` |"
            )
    else:
        lines.append("*No library assets found.*")
    
    lines.extend([
        "",
        "## 🤖 Model Assets",
        ""
    ])
    
    if models:
        lines.extend([
            "| Asset | Type | Size | Modified | Hash |",
            "|-------|------|------|----------|------|"
        ])
        
        for model in models:
            asset_type = "Model" if model['name'].endswith('.pte') else "Header"
            lines.append(
                f"| {model['name']} | {asset_type} | {model['size_human']} | {model['modified']} | `{model['hash']}` |"
            )
    else:
        lines.append("*No model assets found.*")
    
    lines.extend([
        "",
        "## ⚙️ Build Configuration",
        "",
        "### CMake Configuration",
        f"- **Build Type:** `{build_config.get('cmake_build_type', 'unknown')}`",
        f"- **Toolchain File:** `{build_config.get('cmake_toolchain_file', 'none')}`",
        f"- **ARM Baremetal:** `{build_config.get('executorch_build_arm_baremetal', 'unknown')}`",
        f"- **Cortex-M Support:** `{build_config.get('executorch_build_cortex_m', 'unknown')}`",
        f"- **Portable Ops:** `{build_config.get('executorch_build_portable_ops', 'unknown')}`",
        f"- **Quantized Kernels:** `{build_config.get('executorch_build_kernels_quantized', 'unknown')}`",
        "",
        "### Selected Operators",
        ""
    ])
    
    selected_ops = build_config.get('selected_operators', [])
    if selected_ops:
        lines.append(f"**Count:** {len(selected_ops)} operators")
        lines.append("")
        lines.append("```")
        for op in selected_ops:  # Show all operators
            lines.append(op)
        lines.append("```")
    else:
        lines.append("*No operator selection file found.*")
    
    lines.extend([
        "",
        "## 🔄 Model Conversion Details",
        "",
        model_log,
        "",
        "## 🛠️ Build Environment",
        "",
        f"- **Platform:** `{build_env.get('platform', 'unknown')}`",
        f"- **Python:** `{build_env.get('python', 'unknown')}`",
        f"- **CMake:** `{build_env.get('cmake', 'unknown')}`",
        f"- **ARM GCC:** `{build_env.get('gcc_arm', 'not available')}`",
        "",
        "## 📁 Asset Locations",
        "",
        "```",
        "ai_layer/",
        "├── engine/",
        "│   ├── lib/           # Static libraries",
        "│   ├── include/       # Header files", 
        "│   └── model/         # Model assets",
        "└── REPORT.md          # This report",
        "```",
        "",
        "---",
        f"*Report generated by ExecuTorch AI Layer build system at {timestamp}*"
    ])
    
    # Write report
    REPORT_PATH.write_text("\n".join(lines))
    print(f"✅ Comprehensive report written to {REPORT_PATH}")
    print(f"📊 Found {len(libraries)} libraries and {len(models)} model assets")

if __name__ == '__main__':
    main()
