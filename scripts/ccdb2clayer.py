#!/usr/bin/env python3
"""
Convert compile_commands.json to CMSIS *.clayer.yml format.

This tool uses the ccdb module to parse compilation databases and
generate CMSIS layer files with proper grouping and organization.
"""

import argparse
import shutil
import yaml
from pathlib import Path
from ccdb import CompilationDatabase


def format_defines(defines: list[str]) -> list:
    """
    Convert define list from ['NAME=VALUE', 'NAME'] format
    to CMSIS format: [{'NAME': 'VALUE'}, 'NAME']
    """
    result = []
    for define in defines:
        if '=' in define:
            # Split into name and value
            name, value = define.split('=', 1)
            result.append({name: value})
        else:
            # No value, just the name
            result.append(define)
    return result


def convert_include_paths(includes: list[str], strip_paths: list[str], layer_root: Path) -> list[str]:
    """
    Convert include paths from absolute container paths to relative paths
    pointing to the engine/include directory.
    
    Container paths like:
      /workspace/executorch/..
      /workspace/executorch/runtime/core/portable_type/c10
      /workspace2/out/stage1/schema/include
    
    Should become:
      engine/include
      engine/include/c10
      engine/include
    """
    result = []
    seen = set()
    
    for inc in includes:
        # Remove strip_path prefix if any match
        for strip_path in (strip_paths or []):
            if strip_path and inc.startswith(strip_path):
                inc = inc[len(strip_path):].lstrip('/')
                break
        
        # Handle common patterns
        if inc == '..' or inc.endswith('/..'):
            # Parent directory reference -> engine/include
            converted = 'engine/include'
        elif 'out/stage1/schema/include' in inc or 'schema/include' in inc:
            # Generated schema headers -> engine/include
            converted = 'engine/include'
        elif 'third-party/flatbuffers/include' in inc:
            # Flatbuffers includes -> engine/include
            converted = 'engine/include'
        elif inc.startswith('/workspace'):
            # Any other workspace path - check if it's a subpath
            # Extract the meaningful part after /workspace/executorch/
            if '/workspace/executorch/' in inc:
                subpath = inc.split('/workspace/executorch/', 1)[1]
                # Map known subdirectories
                if subpath.startswith('runtime/core/portable_type/'):
                    # Extract subfolder like c10
                    subfolder = subpath.split('/')[-1]
                    converted = f'engine/include/{subfolder}'
                else:
                    converted = 'engine/include'
            else:
                converted = 'engine/include'
        else:
            # Already relative or unknown - convert to engine/include
            converted = 'engine/include'
        
        # Deduplicate
        if converted not in seen:
            result.append(converted)
            seen.add(converted)
    
    return result


def parse_args():
    p = argparse.ArgumentParser(
        description="Convert compile_commands.json to a CMSIS *.clayer.yml"
    )
    p.add_argument(
        "--compile-commands",
        "-c",
        default="compile_commands.json",
        help="Path to compile_commands.json",
    )
    p.add_argument(
        "--layer-root",
        "-L",
        required=True,
        help="Path to the layer root directory (used as filter and relative base)",
    )
    p.add_argument(
        "--output",
        "-o",
        default="layer.clayer.yml",
        help="Output clayer YAML file",
    )
    p.add_argument(
        "--name",
        "-n",
        default="MyLayer",
        help="Layer name in clayer.yml",
    )
    p.add_argument(
        "--strip-path",
        "-A",
        action="append",
        default=[],
        help="Path prefix to strip from all paths in compile_commands.json (e.g., '/workspace/executorch'). Can be specified multiple times.",
    )
    p.add_argument(
        "--group",
        "-g",
        action="append",
        default=[],
        help="Top-level folder to group by (e.g., 'runtime', 'kernels'). Can be specified multiple times.",
    )
    p.add_argument(
        "--filter-flags",
        "-f",
        action="append",
        default=[],
        help="Flag patterns to filter out (matches substring, e.g., '-MF', '-MT')",
    )
    p.add_argument(
        "--remove-flags",
        "-r",
        action="append",
        default=[],
        help="Exact flags to remove completely. Use = syntax for flags starting with dash: --remove-flags=-fPIC. Use for AC6 compatibility.",
    )
    p.add_argument(
        "--exclude",
        "-e",
        action="append",
        default=[],
        help="File patterns to exclude (supports wildcards, e.g., '*/posix.cpp', 'runtime/platform/default/*'). Can be specified multiple times.",
    )
    p.add_argument(
        "--copy-sources",
        action="store_true",
        help="Copy source files to a subdirectory relative to the output clayer file",
    )
    p.add_argument(
        "--source-dir",
        default=None,
        help="Directory name for copied sources (default: uses --layer-root basename)",
    )
    p.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output with detailed information about parameters and parsing",
    )
    return p.parse_args()


def main():
    args = parse_args()
    
    if args.verbose:
        print("\n" + "="*60)
        print("ccdb2clayer - Compilation Database to CMSIS Layer Converter")
        print("="*60)
        print("\n📋 Parameters:")
        print(f"  Input:           {args.compile_commands}")
        print(f"  Output:          {args.output}")
        print(f"  Layer name:      {args.name}")
        print(f"  Layer root:      {args.layer_root}")
        print(f"  Strip paths:     {', '.join(args.strip_path) if args.strip_path else 'None'}")
        print(f"  Groups:          {', '.join(args.group) if args.group else 'None (single group)'}")
        print(f"  Filter flags:    {', '.join(args.filter_flags) if args.filter_flags else 'None'}")
        print(f"  Remove flags:    {', '.join(args.remove_flags) if args.remove_flags else 'None'}")
        print(f"  Exclude files:   {', '.join(args.exclude) if args.exclude else 'None'}")
        print(f"  Copy sources:    {args.copy_sources}")
        if args.copy_sources:
            print(f"  Source dir:      {args.source_dir or '(auto)'}")
        print()
    
    layer_root = Path(args.layer_root)
    if layer_root != Path('.'):
        layer_root = layer_root.resolve()
    
    # Determine source directory name (default to layer_root basename if not specified)
    if args.source_dir is None:
        args.source_dir = layer_root.name if layer_root.name != '.' else 'src'
    
    if args.verbose:
        print("📂 Loading compilation database...")
    
    # Load and filter compilation database
    db = CompilationDatabase.load(args.compile_commands, strip_paths=args.strip_path)
    db = db.filter_by_path(layer_root, recursive=True)
    
    if args.verbose:
        print(f"  Total entries loaded: {len(db)}")
        print(f"  Entries after filtering by layer root: {len(db)}")
    
    # Apply exclusion patterns if specified
    if args.exclude:
        original_count = len(db)
        db = db.exclude_by_patterns(args.exclude)
        excluded_count = original_count - len(db)
        if args.verbose:
            print(f"  Entries after exclusion: {len(db)} (excluded {excluded_count})")
    
    if args.verbose:
        print()
    
    if len(db) == 0:
        print(f"Warning: No compilation entries found under {layer_root}")
        return
    
    # Build glob patterns from top-level folders
    if args.group:
        group_patterns = {folder.capitalize(): f"{folder}/*" for folder in args.group}
        if args.verbose:
            print("📊 Grouping files by patterns:")
            for name, pattern in group_patterns.items():
                print(f"  {name}: {pattern}")
            print()
        # When specific groups are requested, don't include "Other" group
        groups = db.group_by_glob(group_patterns, filter_flags=args.filter_flags, remove_flags=args.remove_flags, include_other=False)
    else:
        if args.verbose:
            print("📊 No grouping specified - using single 'Source' group")
            print()
        # No grouping - single group with all files
        from ccdb import CompileGroup
        groups = [CompileGroup(name="Source", entries=db.entries)]
        db._extract_group_commons(groups[0], filter_flags=args.filter_flags, remove_flags=args.remove_flags)
    
    if args.verbose:
        print("📦 Groups parsed:")
        for group in groups:
            print(f"  {group.name}: {len(group.entries)} files")
            if group.includes:
                print(f"    Includes: {len(group.includes)} paths")
            if group.defines:
                print(f"    Defines: {len(group.defines)} items")
            if group.compiler_flags:
                print(f"    Flags: {len(group.compiler_flags)} items")
        print()
    
    # Build clayer structure
    clayer = {
        "layer": {
            "type": args.name,
            "description": f"Generated from compile_commands.json ({len(db)} files)",
        }
    }
    
    # Add global includes, defines, and flags
    if db.global_includes:
        clayer["layer"]["add-path"] = convert_include_paths(db.global_includes, args.strip_path, layer_root)
    if db.global_defines:
        clayer["layer"]["define"] = format_defines(db.global_defines)
    if db.global_flags:
        # Format misc as required by schema: object with C-CPP property
        clayer["layer"]["misc"] = [{"C-CPP": db.global_flags}]
    
    # Determine output directory for source copying
    output_path = Path(args.output).resolve()
    source_dir = output_path.parent / args.source_dir if args.copy_sources else None
    
    # Add groups
    file_groups = []
    copied_files = []
    
    for group in groups:
        if not group.entries:
            continue
        
        # Build file list with optional copying
        file_list = []
        for entry in sorted(group.entries, key=lambda x: x.relative_to(layer_root)):
            rel_path = entry.relative_to(layer_root)
            
            # Clean the relative path (remove leading slashes)
            clean_path = str(rel_path).lstrip('/')
            
            if args.copy_sources:
                # Copy source file to source directory
                # If strip_paths were used, try prepending each to reconstruct original path
                src_file = None
                if args.strip_path and not entry.file.is_absolute():
                    # Try each strip path to find the source file
                    for strip_path in args.strip_path:
                        candidate = Path(strip_path) / entry.file
                        if candidate.exists():
                            src_file = candidate
                            break
                elif entry.file.is_absolute():
                    src_file = entry.file
                else:
                    src_file = entry.directory / entry.file
                
                if src_file and src_file.exists():
                    dst_file = source_dir / clean_path
                    dst_file.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src_file, dst_file)
                    copied_files.append(clean_path)
            
            # Always use the same path in YAML (regardless of copying)
            file_list.append({"file": clean_path})
        
        group_dict = {
            "group": group.name,
            "files": file_list
        }
        
        # Add group-level includes, defines, and flags
        if group.includes:
            group_dict["add-path"] = convert_include_paths(group.includes, args.strip_path, layer_root)
        if group.defines:
            group_dict["define"] = format_defines(group.defines)
        if group.compiler_flags:
            # Format misc as required by schema: object with C-CPP property
            group_dict["misc"] = [{"C-CPP": group.compiler_flags}]
        
        file_groups.append(group_dict)
    
    clayer["layer"]["groups"] = file_groups
    
    # Write output
    if args.verbose:
        print("💾 Writing YAML output...")
    
    with open(args.output, "w", encoding="utf-8") as f:
        yaml.safe_dump(clayer, f, sort_keys=False, default_flow_style=False)
    
    print(f"\n✓ Wrote {args.output}")
    print(f"  Layer root: {layer_root}")
    print(f"  Files: {len(db)}")
    print(f"  Groups: {len(file_groups)}")
    print(f"  Global includes: {len(db.global_includes)}")
    print(f"  Global defines: {len(db.global_defines)}")
    print(f"  Global flags: {len(db.global_flags)}")
    
    if args.copy_sources:
        print(f"  📁 Copied {len(copied_files)} source files to {source_dir}")
    
    if args.verbose:
        print("\n" + "="*60)
        print("Conversion completed successfully")
        print("="*60 + "\n")


if __name__ == "__main__":
    main()
