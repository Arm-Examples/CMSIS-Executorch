#!/usr/bin/env python3
"""
Compilation Database (compile_commands.json) parser and analyzer.

This module provides a structured interface to parse, filter, and group
compilation commands from CMake/Ninja/Make generated compile_commands.json files.
"""

import json
import os
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Set, Dict, Optional, Callable


@dataclass
class CompileCommand:
    """Represents a single compilation command entry."""
    file: Path
    directory: Path
    command: List[str]
    output: Optional[Path] = None
    
    # Parsed flags
    includes: List[str] = field(default_factory=list)
    defines: List[str] = field(default_factory=list)
    compiler_flags: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Parse command into structured flags."""
        self.includes, self.defines, self.compiler_flags = self._parse_command()
    
    def _parse_command(self) -> tuple[List[str], List[str], List[str]]:
        """Extract includes, defines, and other flags from command."""
        includes = []
        defines = []
        flags = []
        
        it = iter(self.command)
        for token in it:
            if token.startswith("-I"):
                # -Ipath or -I path
                inc = token[2:] or next(it, "")
                if inc:
                    includes.append(inc)
            elif token.startswith("-D"):
                # -DNAME or -D NAME or -DNAME=VALUE
                define = token[2:] or next(it, "")
                if define:
                    defines.append(define)
            elif token == "-isystem":
                # -isystem path
                inc = next(it, "")
                if inc:
                    includes.append(inc)
            elif token.startswith("-"):
                # Skip command structure flags
                if token in ("-c", "-o", "-MF", "-MT", "-MQ"):
                    # These are command structure, not compilation flags
                    if token in ("-o", "-MF", "-MT", "-MQ"):
                        next(it, None)  # Skip their argument too
                    continue
                    
                # Other compiler flags (optimization, warnings, etc.)
                flags.append(token)
                # Handle flags that take arguments
                if token in ("-include",):
                    arg = next(it, None)
                    if arg:
                        flags.append(arg)
        
        return includes, defines, flags
    
    def relative_to(self, base: Path) -> str:
        """Get file path relative to base directory."""
        # If file is already relative, check if we need to strip base prefix
        if not self.file.is_absolute():
            file_str = str(self.file)
            base_str = str(base).rstrip('/\\')
            if base_str == '.':
                return file_str
            elif file_str.startswith(base_str + '/') or file_str.startswith(base_str + '\\'):
                return file_str[len(base_str)+1:]
            return file_str
        
        # Standard absolute path handling
        try:
            return str(self.file.relative_to(base))
        except ValueError:
            return str(self.file)


@dataclass
class CompileGroup:
    """Represents a group of compilation commands with common properties."""
    name: str
    entries: List[CompileCommand]
    includes: List[str] = field(default_factory=list)
    defines: List[str] = field(default_factory=list)
    compiler_flags: List[str] = field(default_factory=list)


class CompilationDatabase:
    """
    Compilation database loaded from compile_commands.json.
    
    Provides filtering, grouping, and analysis capabilities with automatic
    extraction of global and group-level compilation flags.
    """
    
    def __init__(self, entries: List[CompileCommand], extract_globals: bool = True):
        self.entries = entries
        self.global_includes: List[str] = []
        self.global_defines: List[str] = []
        self.global_flags: List[str] = []
        if extract_globals:
            self._extract_globals()
    
    @classmethod
    def load(cls, path: str, strip_path: Optional[str] = None, strip_paths: Optional[List[str]] = None) -> 'CompilationDatabase':
        """
        Load compilation database from JSON file.
        
        Args:
            path: Path to compile_commands.json
            strip_path: Optional path prefix to strip from all paths (makes paths relative)
            strip_paths: Optional list of path prefixes to strip (if any matches, path becomes relative)
        """
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Build list of strip prefixes
        strip_prefixes = []
        if strip_path:
            strip_prefixes.append(strip_path)
        if strip_paths:
            strip_prefixes.extend(strip_paths)
        
        entries = []
        for entry in data:
            file_path = Path(entry["file"])
            keep_relative = False
            
            # Strip prefix if any match
            if strip_prefixes:
                file_str = str(file_path)
                for strip_str in strip_prefixes:
                    # Handle both absolute and string prefix matching
                    if file_str.startswith(strip_str):
                        # Remove the prefix and any leading separator
                        file_path = Path(file_str[len(strip_str):].lstrip('/\\'))
                        keep_relative = True  # Keep as relative path after stripping
                        break
            
            # Only resolve to absolute if not keeping relative
            if not keep_relative:
                # Make relative paths absolute based on directory
                if not file_path.is_absolute():
                    file_path = Path(entry["directory"]) / file_path
                
                # Resolve to absolute path
                try:
                    file_path = file_path.resolve()
                except (OSError, RuntimeError):
                    # Path might not exist, keep as-is
                    pass
            
            # Parse command (can be string or list)
            cmd = entry.get("arguments") or entry.get("command", "")
            if isinstance(cmd, str):
                cmd = shlex.split(cmd)
            
            entries.append(CompileCommand(
                file=file_path,
                directory=Path(entry["directory"]).resolve(),
                command=cmd,
                output=Path(entry["output"]).resolve() if "output" in entry else None
            ))
        
        # Don't extract globals yet - wait until after filtering
        db = cls(entries, extract_globals=False)
        return db
    
    def _extract_globals(self):
        """Extract includes, defines, and flags common to ALL entries."""
        if not self.entries:
            return
        
        # Find common elements across all entries
        common_includes = set(self.entries[0].includes)
        common_defines = set(self.entries[0].defines)
        common_flags = set(self.entries[0].compiler_flags)
        
        for entry in self.entries[1:]:
            common_includes &= set(entry.includes)
            common_defines &= set(entry.defines)
            common_flags &= set(entry.compiler_flags)
        
        # Store as sorted lists
        self.global_includes = sorted(common_includes)
        self.global_defines = sorted(common_defines)
        self.global_flags = sorted(common_flags)
        
        # Remove globals from individual entries
        for entry in self.entries:
            entry.includes = [i for i in entry.includes if i not in common_includes]
            entry.defines = [d for d in entry.defines if d not in common_defines]
            entry.compiler_flags = [f for f in entry.compiler_flags if f not in common_flags]
    
    def filter_by_path(self, root: Path, recursive: bool = True) -> 'CompilationDatabase':
        """
        Filter entries by source file location.
        
        Handles both absolute paths and relative paths (after strip_path).
        For relative paths, root should be '.' or a relative path.
        """
        filtered = []
        
        # Check if we're dealing with relative paths
        has_relative = any(not e.file.is_absolute() for e in self.entries)
        
        if has_relative:
            # When paths are relative, filter by string prefix matching
            root_str = str(root).rstrip('/\\')
            if root_str == '.':
                # Accept all relative paths when root is current directory
                filtered = [e for e in self.entries if not e.file.is_absolute()]
            else:
                for entry in self.entries:
                    if entry.file.is_absolute():
                        continue
                    file_str = str(entry.file)
                    if recursive:
                        if file_str.startswith(root_str):
                            filtered.append(entry)
                    else:
                        if entry.file.parent == Path(root_str):
                            filtered.append(entry)
        else:
            # Standard absolute path filtering
            root = root.resolve()
            for entry in self.entries:
                if recursive:
                    try:
                        entry.file.relative_to(root)
                        filtered.append(entry)
                    except ValueError:
                        continue
                else:
                    if entry.file.parent == root:
                        filtered.append(entry)
        
        # Create new database and re-extract globals from filtered set
        new_db = CompilationDatabase.__new__(CompilationDatabase)
        new_db.entries = filtered
        new_db.global_includes = []
        new_db.global_defines = []
        new_db.global_flags = []
        new_db._extract_globals()
        return new_db
    
    def filter_by_extension(self, extensions: List[str]) -> 'CompilationDatabase':
        """Filter entries by file extension."""
        exts = set(ext if ext.startswith('.') else f'.{ext}' for ext in extensions)
        filtered = [e for e in self.entries if e.file.suffix in exts]
        return CompilationDatabase(filtered)
    
    def exclude_by_patterns(self, patterns: List[str]) -> 'CompilationDatabase':
        """
        Exclude entries matching any of the given patterns.
        
        Args:
            patterns: List of glob patterns to exclude
                     e.g., ['*/posix.cpp', 'runtime/platform/default/*', '*.test.cpp']
        
        Returns:
            New CompilationDatabase with matching files excluded
        """
        import fnmatch
        
        filtered = []
        for entry in self.entries:
            path_str = str(entry.file)
            exclude = False
            
            for pattern in patterns:
                if fnmatch.fnmatch(path_str, pattern) or fnmatch.fnmatch(path_str, f'*/{pattern}'):
                    exclude = True
                    break
            
            if not exclude:
                filtered.append(entry)
        
        # Create new database and re-extract globals from filtered set
        new_db = CompilationDatabase.__new__(CompilationDatabase)
        new_db.entries = filtered
        new_db.global_includes = []
        new_db.global_defines = []
        new_db.global_flags = []
        new_db._extract_globals()
        return new_db
    
    def filter_by_predicate(self, predicate: Callable[[CompileCommand], bool]) -> 'CompilationDatabase':
        """Filter entries using custom predicate function."""
        filtered = [e for e in self.entries if predicate(e)]
        return CompilationDatabase(filtered)
    
    def group_by_glob(self, patterns: Dict[str, str], filter_flags: Optional[List[str]] = None, remove_flags: Optional[List[str]] = None, include_other: bool = True) -> List[CompileGroup]:
        """
        Group entries by glob pattern matching with automatic extraction of group-common flags.
        
        Args:
            patterns: Dict of {group_name: glob_pattern}
                     e.g., {"Runtime": "runtime/*", "Kernels": "kernels/*"}
            filter_flags: Optional list of flag patterns to exclude (substring match)
            remove_flags: Optional list of exact flags to remove (e.g., '-fPIC', '-s')
            include_other: If True, include "Other" group for unmatched files (default: True)
        
        Returns:
            List of CompileGroup objects with group-level includes/defines/flags extracted
        """
        import fnmatch
        
        # First, group entries by pattern
        grouped_entries = {name: [] for name in patterns}
        if include_other:
            grouped_entries["Other"] = []
        
        # Track seen files to avoid duplicates
        seen_files = {name: set() for name in patterns}
        if include_other:
            seen_files["Other"] = set()
        
        for entry in self.entries:
            path_str = str(entry.file)
            matched = False
            
            for group_name, pattern in patterns.items():
                if fnmatch.fnmatch(path_str, pattern):
                    # Only add if we haven't seen this file in this group
                    if path_str not in seen_files[group_name]:
                        grouped_entries[group_name].append(entry)
                        seen_files[group_name].add(path_str)
                    matched = True
                    break
            
            if not matched and include_other:
                if path_str not in seen_files["Other"]:
                    grouped_entries["Other"].append(entry)
                    seen_files["Other"].add(path_str)
        
        # Create CompileGroup objects and extract group-level common flags
        groups = []
        for group_name, entries in grouped_entries.items():
            if not entries:
                continue
            
            group = CompileGroup(name=group_name, entries=entries)
            self._extract_group_commons(group, filter_flags, remove_flags)
            groups.append(group)
        
        return groups
    
    def group_by_regex(self, patterns: Dict[str, str], filter_flags: Optional[List[str]] = None, remove_flags: Optional[List[str]] = None) -> List[CompileGroup]:
        """
        Group entries by regex pattern matching with automatic extraction of group-common flags.
        
        Args:
            patterns: Dict of {group_name: regex_pattern}
                     e.g., {"Runtime": r".*/runtime/.*", "Kernels": r".*/kernels/.*"}
            filter_flags: Optional list of flag patterns to exclude (substring match)
            remove_flags: Optional list of exact flags to remove (e.g., '-fPIC', '-s')
        
        Returns:
            List of CompileGroup objects with group-level includes/defines/flags extracted
        """
        import re
        
        # First, group entries by pattern
        grouped_entries = {name: [] for name in patterns}
        grouped_entries["Other"] = []
        
        # Compile regex patterns
        compiled_patterns = {name: re.compile(pattern) for name, pattern in patterns.items()}
        
        for entry in self.entries:
            path_str = str(entry.file)
            matched = False
            
            for group_name, regex in compiled_patterns.items():
                if regex.search(path_str):
                    grouped_entries[group_name].append(entry)
                    matched = True
                    break
            
            if not matched:
                grouped_entries["Other"].append(entry)
        
        # Create CompileGroup objects and extract group-level common flags
        groups = []
        for group_name, entries in grouped_entries.items():
            if not entries:
                continue
            
            group = CompileGroup(name=group_name, entries=entries)
            self._extract_group_commons(group, filter_flags, remove_flags)
            groups.append(group)
        
        return groups
    
    def _extract_group_commons(self, group: CompileGroup, filter_flags: Optional[List[str]] = None, remove_flags: Optional[List[str]] = None):
        """
        Extract includes, defines, and flags common to all entries in a group.
        
        Args:
            group: The CompileGroup to process
            filter_flags: Optional list of flag patterns to exclude (substring match)
            remove_flags: Optional list of exact flags to remove (e.g., '-fPIC', '-s')
        """
        if not group.entries:
            return
        
        # Find common elements across all entries in the group
        common_includes = set(group.entries[0].includes)
        common_defines = set(group.entries[0].defines)
        common_flags = set(group.entries[0].compiler_flags)
        
        for entry in group.entries[1:]:
            common_includes &= set(entry.includes)
            common_defines &= set(entry.defines)
            common_flags &= set(entry.compiler_flags)
        
        # Apply pattern filter if provided (substring match)
        if filter_flags:
            common_flags = {f for f in common_flags if not any(filt in f for filt in filter_flags)}
        
        # Apply exact flag removal if provided
        if remove_flags:
            remove_set = set(remove_flags)
            common_flags = {f for f in common_flags if f not in remove_set}
        
        # Store as sorted lists
        group.includes = sorted(common_includes)
        group.defines = sorted(common_defines)
        group.compiler_flags = sorted(common_flags)
        
        # Remove group commons from individual entries
        for entry in group.entries:
            entry.includes = [i for i in entry.includes if i not in common_includes]
            entry.defines = [d for d in entry.defines if d not in common_defines]
            entry.compiler_flags = [f for f in entry.compiler_flags if f not in common_flags]
    
    def get_all_includes(self, relative_to: Optional[Path] = None) -> List[str]:
        """Get all unique include paths."""
        includes = set()
        for entry in self.entries:
            for inc in entry.includes:
                if relative_to:
                    inc_path = Path(inc)
                    if inc_path.is_absolute():
                        try:
                            inc = str(inc_path.relative_to(relative_to))
                        except ValueError:
                            pass
                includes.add(inc)
        return sorted(includes)
    
    def get_all_defines(self) -> List[str]:
        """Get all unique preprocessor defines."""
        defines = set()
        for entry in self.entries:
            defines.update(entry.defines)
        return sorted(defines)
    
    def get_all_files(self, relative_to: Optional[Path] = None) -> List[str]:
        """Get all source files."""
        files = []
        for entry in self.entries:
            if relative_to:
                files.append(entry.relative_to(relative_to))
            else:
                files.append(str(entry.file))
        return sorted(set(files))
    
    def __len__(self) -> int:
        return len(self.entries)
    
    def __iter__(self):
        return iter(self.entries)


class DatabaseAnalyzer:
    """Analyze compilation database for patterns and statistics."""
    
    def __init__(self, db: CompilationDatabase):
        self.db = db
    
    def get_file_types(self) -> Dict[str, int]:
        """Count files by extension."""
        counts = {}
        for entry in self.db:
            ext = entry.file.suffix or 'no_extension'
            counts[ext] = counts.get(ext, 0) + 1
        return counts
    
    def get_compiler_flags_histogram(self) -> Dict[str, int]:
        """Count usage of compiler flags."""
        flag_counts = {}
        for entry in self.db:
            for flag in entry.compiler_flags:
                # Group similar flags (e.g., -O2, -O3 -> -O*)
                base_flag = flag.split('=')[0] if '=' in flag else flag
                flag_counts[base_flag] = flag_counts.get(base_flag, 0) + 1
        return dict(sorted(flag_counts.items(), key=lambda x: x[1], reverse=True))
    
    def find_common_includes(self, min_usage: int = 2) -> List[str]:
        """Find include paths used by multiple files."""
        include_counts = {}
        for entry in self.db:
            for inc in entry.includes:
                include_counts[inc] = include_counts.get(inc, 0) + 1
        
        return [inc for inc, count in include_counts.items() if count >= min_usage]
    
    def find_common_defines(self, min_usage: int = 2) -> List[str]:
        """Find defines used by multiple files."""
        define_counts = {}
        for entry in self.db:
            for define in entry.defines:
                define_counts[define] = define_counts.get(define, 0) + 1
        
        return [d for d, count in define_counts.items() if count >= min_usage]
