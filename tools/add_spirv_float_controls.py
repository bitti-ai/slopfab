#!/usr/bin/env python3
"""Add Vulkan 1.2 fp32 control modes to a glslang SPIR-V module.

Usage: add_spirv_float_controls.py INPUT NORMAL_OUTPUT DENORM_OUTPUT
       add_spirv_float_controls.py --preserve-only INPUT OUTPUT
NORMAL_OUTPUT requires signed-zero/Inf/NaN preservation and RTE. DENORM_OUTPUT
adds denormal preservation. The input may equal NORMAL_OUTPUT: it is read fully
before either output is written.
"""

import struct
import sys


def transform(raw: bytes, denorm: bool, rounding_rte: bool = True) -> bytes:
    if len(raw) % 4:
        raise ValueError("SPIR-V size is not word aligned")
    words = list(struct.unpack("<%dI" % (len(raw) // 4), raw))
    if len(words) < 5 or words[0] != 0x07230203:
        raise ValueError("invalid SPIR-V header")
    capability_end = 5
    execution_end = 0
    entry = 0
    existing = set()
    cleaned = words[:5]
    at = 5
    while at < len(words):
        count, opcode = words[at] >> 16, words[at] & 0xFFFF
        if count == 0 or at + count > len(words):
            raise ValueError("malformed SPIR-V instruction")
        instruction = tuple(words[at:at + count])
        if opcode in (16, 17) and instruction in existing:
            at += count
            continue
        existing.add(instruction)
        position = len(cleaned)
        cleaned.extend(instruction)
        if opcode == 17 and capability_end == position:
            capability_end = position + count
        if opcode == 15:
            entry = words[at + 2]
        if opcode in (15, 16):
            execution_end = position + count
        at += count
    words = cleaned
    if not entry or not execution_end:
        raise ValueError("SPIR-V has no entry point")
    capabilities = [(2 << 16) | 17, 4467 if rounding_rte else 4466]
    modes = [(4 << 16) | 16, entry, 4461, 32]
    if rounding_rte:
        capabilities[0:0] = [(2 << 16) | 17, 4466]
        modes.extend([(4 << 16) | 16, entry, 4462, 32])
    if denorm:
        capabilities[0:0] = [(2 << 16) | 17, 4464]
        modes[0:0] = [(4 << 16) | 16, entry, 4459, 32]
    capabilities = [word for i in range(0, len(capabilities), 2)
                    if tuple(capabilities[i:i + 2]) not in existing
                    for word in capabilities[i:i + 2]]
    modes = [word for i in range(0, len(modes), 4)
             if tuple(modes[i:i + 4]) not in existing
             for word in modes[i:i + 4]]
    words[capability_end:capability_end] = capabilities
    execution_end += len(capabilities)
    words[execution_end:execution_end] = modes
    return struct.pack("<%dI" % len(words), *words)


if len(sys.argv) == 4 and sys.argv[1] == "--preserve-only":
    with open(sys.argv[2], "rb") as source:
        input_bytes = source.read()
    with open(sys.argv[3], "wb") as output:
        output.write(transform(input_bytes, False, False))
    raise SystemExit(0)
if len(sys.argv) != 4:
    raise SystemExit(__doc__)
with open(sys.argv[1], "rb") as source:
    input_bytes = source.read()
normal = transform(input_bytes, False)
denorm = transform(input_bytes, True)
with open(sys.argv[2], "wb") as output:
    output.write(normal)
with open(sys.argv[3], "wb") as output:
    output.write(denorm)
