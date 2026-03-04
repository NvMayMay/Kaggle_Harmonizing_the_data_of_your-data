#!/usr/bin/env python3
"""Inspect first training example."""
import json, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

with open('training_data/sdrf_training_data.jsonl', encoding='utf-8') as f:
    ex = json.loads(f.readline())

pxd = ex['pxd']
msgs = ex['messages']
print(f'PXD: {pxd}')
print(f'Messages: {len(msgs)}')
print()

for i, m in enumerate(msgs):
    role = m['role']
    if role == 'system':
        print(f'[{i}] SYSTEM: ({len(m["content"])} chars)')
    elif role == 'user':
        print(f'[{i}] USER: ({len(m["content"])} chars)')
    elif role == 'assistant':
        if m.get('tool_calls'):
            for tc in m['tool_calls']:
                fn = tc['function']['name']
                args = tc['function']['arguments']
                print(f'[{i}] ASSISTANT tool_call: {fn}({args})')
        else:
            content = m.get('content', '')
            if '```json' in content:
                json_start = content.index('```json') + 7
                json_end = content.index('```', json_start)
                output = json.loads(content[json_start:json_end])
                print(f'[{i}] ASSISTANT final output:')
                for k, v in output.items():
                    if v and v != 'Not Applicable':
                        val = str(v)[:80]
                        print(f'     {k}: {val}')
            else:
                print(f'[{i}] ASSISTANT: ({len(content)} chars)')
    elif role == 'tool':
        content = m.get('content', '')
        tid = m.get("tool_call_id", "?")
        print(f'[{i}] TOOL ({tid}): ({len(content)} chars)')
