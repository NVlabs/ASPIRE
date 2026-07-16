import os
import argparse
import re
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Combine code.py files from subdirectories into a single text file.")
    parser.add_argument("input_dir", type=str, help="Path to the input directory containing trial subdirectories.")
    parser.add_argument("output_file", type=str, nargs="?", help="Path to the output text file. Defaults to input_dir/combined_code.txt")
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    if args.output_file:
        output_file = Path(args.output_file)
    else:
        output_file = input_dir / "combined_code.txt"
    
    # Ensure input directory exists
    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Error: Input directory '{input_dir}' does not exist or is not a directory.")
        return

    # Create parent directories for output file if they don't exist
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as outfile:
        # Sort directories for consistent order (e.g., by trial number if present, or just alphabetically)
        # We look for directories inside input_dir
        subdirs = sorted([d for d in input_dir.iterdir() if d.is_dir()])
        
        count = 0
        for subdir in subdirs:
            code_file = subdir / "code.py"
            
            if code_file.exists():
                count += 1
                try:
                    content = code_file.read_text(encoding='utf-8')
                    
                    # Parse metadata from directory name
                    # Expected format like: trial_02_sandboxrc_0_reward_1.000_taskcompleted_1
                    reward_match = re.search(r"reward_([0-9.]+)", subdir.name)
                    task_match = re.search(r"taskcompleted_([0-9]+)", subdir.name)
                    
                    reward = reward_match.group(1) if reward_match else "N/A"
                    task_completed = task_match.group(1) if task_match else "N/A"
                    
                    # Create a header/separator
                    header_lines = [
                        f"File: {subdir.name}/code.py",
                        f"Reward: {reward}",
                        f"Task Completed: {task_completed}"
                    ]
                    
                    max_len = max(len(line) for line in header_lines)
                    separator = "=" * max_len
                    
                    outfile.write(f"{separator}\n")
                    for line in header_lines:
                        outfile.write(f"{line}\n")
                    outfile.write(f"{separator}\n\n")
                    outfile.write("```python\n")
                    outfile.write(content)
                    if not content.endswith('\n'):
                        outfile.write('\n')
                    outfile.write("```\n\n")
                    
                except Exception as e:
                    print(f"Error reading {code_file}: {e}")
        
        print(f"Combined {count} files into '{output_file}'.")

if __name__ == "__main__":
    main()
