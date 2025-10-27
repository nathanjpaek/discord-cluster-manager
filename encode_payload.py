import base64
import json
import sys
import zlib
from pathlib import Path


def encode_payload(config_dict: dict) -> str:
    """Convert a config dict to compressed base64 string"""
    json_str = json.dumps(config_dict)
    compressed = zlib.compress(json_str.encode('utf-8'))
    encoded = base64.b64encode(compressed).decode('utf-8')
    return encoded


def main():
    if len(sys.argv) > 1 and sys.argv[1] != '-':
        input_file = sys.argv[1]
        print(f"Reading from: {input_file}", file=sys.stderr)
        config = json.loads(Path(input_file).read_text())
    else:
        print("Reading from stdin...", file=sys.stderr)
        config = json.load(sys.stdin)
    
    encoded = encode_payload(config)
    
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
        Path(output_file).write_text(encoded)
        print(f"Written to: {output_file}", file=sys.stderr)
    elif len(sys.argv) > 1 and sys.argv[1] != '-':
        Path("payload.json").write_text(encoded)
        print(f"Written to: payload.json", file=sys.stderr)
    else:
        print(encoded)
    
    json_str = json.dumps(config)
    compressed = zlib.compress(json_str.encode('utf-8'))



if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help']:
        print(__doc__)
        sys.exit(0)
    main()