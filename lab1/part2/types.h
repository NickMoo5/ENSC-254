#ifndef TYPES_H
#define TYPES_H

#include <stdint.h>

typedef union {

    /* access rtype with: instruction.rtype.(opcode|rd|funct3|rs1|rs2|funct7) */
    struct {
	unsigned int opcode : 7;
	unsigned int rd : 5;
	unsigned int funct3 : 3;
	unsigned int rs1 : 5;
	unsigned int rs2 : 5;
	unsigned int funct7 : 7;
    } rtype;
    
    /* access itype with: instruction.itype.(oppcode|rs|rt|imm) */
    struct {
	unsigned int opcode : 7;
	unsigned int rd : 5;	
	unsigned int funct3 : 3;
	unsigned int rs1 : 5;
	unsigned int imm : 12;
    } itype;
  
} Instruction;

Instruction parse_rtype(unsigned int number);
Instruction parse_itype(unsigned int number);
void print_rtype(Instruction instr);
void print_itype(Instruction instr);

#endif
