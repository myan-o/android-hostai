import sys

with open(sys.argv[1], 'r', encoding='utf-8') as f:
    content = f.read()

stack = []
for i, char in enumerate(content):
    if char == '{':
        stack.append(i)
    elif char == '}':
        if not stack:
            print(f"Extra closing brace at index {i}")
            sys.exit(1)
        stack.pop()

if stack:
    print(f"Unclosed opening brace at index {stack[-1]}")
    sys.exit(1)
else:
    print("Braces are balanced")
