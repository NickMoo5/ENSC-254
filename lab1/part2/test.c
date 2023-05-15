#include <stdio.h>
#include "types.h"

const int LEFT_ONE = 1 << 31;

unsigned int getInstr(unsigned int num, int width, int pos) {
  /* extracts a set of bits for Instruction */
  int retVal = 0;
  int number = num >> pos;
  int mask = ~(LEFT_ONE >> (31 - width));
  //mask = mask << pos;
  retVal = number & mask;
  return retVal;
}

/* Your task: parse the unsigned int number into an Instruction
 * extract corresponding bits to fill the rtype struct fields
 */
Instruction parse_rtype(unsigned int number) {
  Instruction instr;  
  //int leftOne = 1 << 31;
  //int opCodeMask = ~(leftOne >> 24);
  //int* numptr = (int*)&instr.rtype;
  //printf("testing: %d\n", instr.rtype.opcode);
  instr.rtype.opcode = getInstr(number, 7, 0);
  instr.rtype.rd = getInstr(number, 5, 7);
  instr.rtype.funct3 = getInstr(number, 3, 12);
  instr.rtype.rs1 = getInstr(number, 5, 15);
  instr.rtype.rs2 = getInstr(number, 5, 20);
  instr.rtype.funct7 = getInstr(number, 7, 25);
  return instr;
}

/* Your task: parse the unsigned int number into an Instruction
 * extract corresponding bits to fill the itype struct fields
 */
Instruction parse_itype(unsigned int number) {
  Instruction instr;
  instr.itype.opcode = getInstr(number, 7, 0);
  instr.itype.rd = getInstr(number, 5, 7);
  instr.itype.funct3 = getInstr(number, 3, 12);
  instr.itype.rs1 = getInstr(number, 5, 15);
  instr.itype.imm = getInstr(number, 12, 20);
  return instr;
}

void print_rtype(Instruction instr) {
  printf("%d, ", instr.rtype.opcode);
  printf("%d, ", instr.rtype.rd);
  printf("%d, ", instr.rtype.funct3);
  printf("%d, ", instr.rtype.rs1);
  printf("%d, ", instr.rtype.rs2);
  printf("%d\n", instr.rtype.funct7);
}

void print_itype(Instruction instr) {
  printf("%d, ", instr.itype.opcode);
  printf("%d, ", instr.itype.rd);
  printf("%d, ", instr.itype.funct3);
  printf("%d, ", instr.itype.rs1);
  printf("%d\n", instr.itype.imm);
}

int main() {
  unsigned int numbers[4] = {0x015a04b3,0x009a84b3,0x00148493,0x00440413};
  Instruction instrs[4];

  //parse the above four numbers into corresponding instructions
  instrs[0] = parse_rtype(numbers[0]);
  print_rtype(instrs[0]);  
    
  instrs[1] = parse_rtype(numbers[1]);
  print_rtype(instrs[1]);  
    
  instrs[2] = parse_itype(numbers[2]);
  print_itype(instrs[2]);  
    
  instrs[3] = parse_itype(numbers[3]);
  print_itype(instrs[3]);  
    
  return 0;
}
