#ifndef MIPS_H
#define MIPS_H

#include "types.h"

/* see disasm.c */
void decode_instruction(uint32_t instruction_bits);

/* see emulator.c */
void execute_instruction(uint32_t instruction_bits, Processor* processor, Byte *memory);
void store(Byte *memory, Address address, Alignment alignment, Word value);
Word load(Byte *memory, Address address, Alignment alignment);

#endif
