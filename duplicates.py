import json
from collections import defaultdict
from typing import Optional

def find_duplicates(filename: str, check_by: str = "title") -> dict:
    """
    Check for duplicate articles in a JSONL file.
    
    Args:
        filename: Path to the JSONL file
        check_by: Field to check for duplicates ("title", "id", or "content")
                  or "full" to compare entire JSON content
    
    Returns:
        Dictionary with duplicate keys and their line numbers
    """
    duplicates = defaultdict(list)
    total_articles = 0

    with open(filename, 'r', encoding='utf-8') as f:
        for line_number, line in enumerate(f, 1):
            total_articles += 1
            try:
                data = json.loads(line)
                
                if check_by == "full":
                    key = json.dumps(data, sort_keys=True)
                else:
                    key = data.get(check_by, None)
                    if key is None:
                        print(f"Warning: Missing '{check_by}' field in line {line_number}")
                        continue
                
                duplicates[key].append(line_number)
                
            except json.JSONDecodeError:
                print(f"! Invalid JSON on line {line_number}")
                continue

    # Filter out non-duplicates
    duplicates = {k: v for k, v in duplicates.items() if len(v) > 1}
    
    print(f"\nAnalyzed {total_articles} articles")
    print(f"Found {len(duplicates)} sets of duplicates\n")
    
    for i, (key, lines) in enumerate(duplicates.items(), 1):
        print(f"Duplicate set #{i}:")
        print(f"Key: {key[:100]}{'...' if len(str(key)) > 100 else ''}")
        print(f"Appears on lines: {', '.join(map(str, lines))}\n")
    
    return duplicates

def remove_duplicates(
    input_file: str,
    output_file: Optional[str] = None,
    check_by: str = "title",
    keep: str = "first"
) -> None:
    """
    Remove duplicate entries from a JSONL file.
    
    Args:
        input_file: Path to the input JSONL file
        output_file: Path to save the deduplicated file (None to modify in-place)
        check_by: Field to check for duplicates ("title", "id", "content", or "full")
        keep: Which duplicate to keep ("first" or "last")
    """
    if output_file is None:
        output_file = input_file + ".tmp"
    
    seen_keys = set()
    removed_count = 0
    total_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        lines = list(infile)  # Read all lines to allow keeping last occurrence
        if keep == "last":
            lines = reversed(lines)
        
        for line in lines:
            total_count += 1
            try:
                data = json.loads(line)
                
                if check_by == "full":
                    key = json.dumps(data, sort_keys=True)
                else:
                    key = data.get(check_by, None)
                    if key is None:
                        # Keep entries with missing key field
                        outfile.write(line)
                        continue
                
                if key not in seen_keys:
                    seen_keys.add(key)
                    outfile.write(line)
                else:
                    removed_count += 1
                    
            except json.JSONDecodeError:
                print(f"! Invalid JSON on line {total_count}, keeping as-is")
                outfile.write(line)
    
    # If we were working with reversed lines, reverse back
    if keep == "last":
        with open(output_file, 'r+', encoding='utf-8') as f:
            lines = f.readlines()
            f.seek(0)
            f.writelines(reversed(lines))
            f.truncate()
    
    # Replace original file if no output file was specified
    if output_file.endswith('.tmp'):
        import os
        os.replace(output_file, input_file)
    
    print(f"\nProcessed {total_count} entries")
    print(f"Removed {removed_count} duplicates")
    print(f"Kept {total_count - removed_count} unique entries")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Find and remove duplicates in JSONL files')
    parser.add_argument('file', help='JSONL file to process')
    parser.add_argument('--check-by', default='title',
                       help='Field to check for duplicates (title/id/content) or "full"')
    parser.add_argument('--remove', action='store_true',
                       help='Remove duplicates instead of just finding them')
    parser.add_argument('--output', default=None,
                       help='Output file (default: overwrite input)')
    parser.add_argument('--keep', choices=['first', 'last'], default='first',
                       help='Which duplicate to keep (first or last occurrence)')
    
    args = parser.parse_args()
    
    if args.remove:
        remove_duplicates(
            args.file,
            output_file=args.output,
            check_by=args.check_by,
            keep=args.keep
        )
    else:
        find_duplicates(args.file, check_by=args.check_by)
