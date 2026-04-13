#!/usr/bin/env python3
"""
Test code examples in markdown documentation files.

This script parses markdown files to find Python code blocks and executes them
to verify that the documentation examples are correct and up-to-date.
"""

import sys
import os
import re
import argparse
import subprocess
import tempfile
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import ast


class MarkdownCodeTester:
    """Test Python code examples in markdown files."""
    
    def __init__(self, merge_blocks: bool = False):
        self.merge_blocks = merge_blocks
        self.results: Dict[str, List[Tuple[int, str, Optional[Exception]]]] = {}
        self.skipped_tests = []
        
    def extract_code_blocks(self, content: str) -> List[Tuple[int, str]]:
        """Extract Python code blocks from markdown content."""
        code_blocks = []
        
        # Match fenced code blocks with python language
        pattern = r'```python\n(.*?)```'
        matches = re.finditer(pattern, content, re.DOTALL)
        
        for match in matches:
            code = match.group(1)
            # Find line number
            line_num = content[:match.start()].count('\n') + 1
            code_blocks.append((line_num, code))
        
        return code_blocks
    
    def extract_imports(self, code: str) -> List[str]:
        """Extract import statements from code."""
        imports = []
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(f"import {alias.name}")
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ''
                    for alias in node.names:
                        if module:
                            imports.append(f"from {module} import {alias.name}")
                        else:
                            imports.append(f"import {alias.name}")
        except SyntaxError:
            pass
        return imports
    
    def create_test_script(self, code: str) -> str:
        """Create a test script with necessary imports and setup."""
        # Extract imports
        imports = self.extract_imports(code)
        
        # Add pyyrtpet import if not present
        if not any('import pyyrtpet' in imp for imp in imports):
            imports.insert(0, 'import pyyrtpet as yrt')
        
        # Add numpy if used
        if 'numpy' in code or 'np.' in code:
            if not any('import numpy' in imp for imp in imports):
                imports.insert(0, 'import numpy as np')
        
        # Add torch if used
        if 'torch' in code:
            if not any('import torch' in imp for imp in imports):
                imports.insert(0, 'import torch')
        
        # Build script
        script_lines = [
            "# Auto-generated test script",
            "# noqa: E402",
            "",
        ]
        script_lines.extend(imports)
        script_lines.extend(['', '# Test code', code])
        
        return '\n'.join(script_lines)
    
    def test_code_block(self, code: str, block_name: str) -> Optional[Exception]:
        """Test a single code block."""
        script = self.create_test_script(code)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.py', delete=False
        ) as f:
            f.write(script)
            temp_file = f.name
        
        try:
            # Try to compile first
            compile(script, '<string>', 'exec')
            
            # Try to run
            result = subprocess.run(
                [sys.executable, '-c', script],
                capture_output=True,
                text=True,
                timeout=60,
                cwd=os.getcwd()
            )
            
            if result.returncode != 0:
                error_msg = result.stderr or result.stdout
                return Exception(error_msg)
            
            return None
            
        except subprocess.TimeoutExpired:
            return Exception("Test timed out after 60 seconds")
        except Exception as e:
            return e
        finally:
            os.unlink(temp_file)
    
    def test_markdown_file(self, file_path: str) -> Dict[str, List]:
        """Test all code blocks in a markdown file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        code_blocks = self.extract_code_blocks(content)
        
        results = {
            'passed': [],
            'failed': [],
            'skipped': []
        }

        if self.merge_blocks and code_blocks:
            passed_blocks = []
            skipped_blocks = []
            for line_num, code in code_blocks:
                if '...' in code and 'def ' not in code:
                    skipped_blocks.append((line_num, code[:50] + '...'))
                elif '<' in code and '>' in code:
                    skipped_blocks.append((line_num, code[:50] + '...'))
                else:
                    passed_blocks.append((line_num, code))
            
            if passed_blocks:
                merged_code = '\n\n'.join(code for _, code in passed_blocks)
                error = self.test_code_block(merged_code, Path(file_path).name)
                if error is None:
                    results['passed'].append((passed_blocks[0][0], merged_code[:50] + '...'))
                else:
                    results['failed'].append((passed_blocks[0][0], merged_code[:50] + '...', str(error)))
            
            for line_num, code_snippet in skipped_blocks:
                results['skipped'].append((line_num, code_snippet))
            
            return results
        
        for line_num, code in code_blocks:
            # Skip if it looks like pseudocode or incomplete
            if '...' in code and 'def ' not in code:
                results['skipped'].append((line_num, code[:50] + '...'))
                continue
            
            # Skip if it has placeholders
            if '<' in code and '>' in code:
                results['skipped'].append((line_num, code[:50] + '...'))
                continue
            
            block_name = f"{Path(file_path).name}:{line_num}"
            
            error = self.test_code_block(code, block_name)
            
            if error is None:
                results['passed'].append((line_num, code[:50] + '...'))
            else:
                results['failed'].append((line_num, code[:50] + '...', str(error)))
        
        return results
    
    def test_directory(self, directory: str) -> None:
        """Test all markdown files in a directory."""
        md_files = list(Path(directory).rglob('*.md'))
        
        print(f"Found {len(md_files)} markdown files")
        print("=" * 60)
        
        total_passed = 0
        total_failed = 0
        total_skipped = 0
        
        for md_file in md_files:
            print(f"\nTesting: {md_file}")
            print("-" * 40)
            
            results = self.test_markdown_file(str(md_file))
            
            total_passed += len(results['passed'])
            total_failed += len(results['failed'])
            total_skipped += len(results['skipped'])
            
            if results['passed']:
                print(f"  [SUCCESS] Passed: {len(results['passed'])}")
            if results['failed']:
                print(f"  [FAILURE] Failed: {len(results['failed'])}")
                for line, code_snippet, error in results['failed']:
                    print(f"      Line {line}: {code_snippet}")
                    print(f"      Error: {error[:200]}")
            if results['skipped']:
                print(f" [SKIP] Skipped: {len(results['skipped'])}")
        
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"Total passed:    {total_passed}")
        print(f"Total failed:    {total_failed}")
        print(f"Total skipped:  {total_skipped}")
        
        if total_failed > 0:
            print(f"\n[FAILED] {total_failed} test(s) failed!")
            return 1
        else:
            print(f"\n[SUCCESS] All tests passed!")
            return 0


def main():
    parser = argparse.ArgumentParser(
        description='Test Python code examples in markdown documentation'
    )
    parser.add_argument(
        'path',
        help='Path to markdown file or directory containing markdown files'
    )
    parser.add_argument(
        '--no_merge',
        action='store_true',
        help='Disable merging of code blocks (test each block separately)'
    )
    
    args = parser.parse_args()
    
    tester = MarkdownCodeTester(merge_blocks=not args.no_merge)
    
    if os.path.isfile(args.path):
        results = tester.test_markdown_file(args.path)
        print(f"Testing: {args.path}")
        print(f"Passed: {len(results['passed'])}")
        print(f"Failed: {len(results['failed'])}")
        print(f"Skipped: {len(results['skipped'])}")
        
        for line, code, error in results['failed']:
            print(f"\nFailed at line {line}:")
            print(code)
            print(f"Error: {error}")
        
        return 1 if results['failed'] else 0
    elif os.path.isdir(args.path):
        return tester.test_directory(args.path)
    else:
        print(f"Error: {args.path} is not a valid file or directory")
        return 1


if __name__ == '__main__':
    sys.exit(main())
