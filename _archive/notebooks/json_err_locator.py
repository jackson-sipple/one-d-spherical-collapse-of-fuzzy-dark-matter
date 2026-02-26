import json

filename = "collapse_neighbor_pressure.ipynb"
try:
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
        json.loads(content)
except json.JSONDecodeError as e:
    print(f"Error at position {e.pos}: {e.msg}")
    # Extract context around the error
    start = max(0, e.pos - 20)
    end = min(len(content), e.pos + 20)
    print(f"Context: {repr(content[start:end])}")
    # Print the problematic character in hex
    if e.pos < len(content):
        print(f"Character at position {e.pos}: {repr(content[e.pos])}")
        print(f"Hex value: {hex(ord(content[e.pos]))}")