#include <stdio.h> // for stderr
#include <stdlib.h> // for exit()
#include "types.h"
#include "utils.h"

void print_rtype(char *, Instruction);
void print_itype_except_load(char *, Instruction, int);
void print_load(char *, Instruction);
void print_store(char *, Instruction);
void print_branch(char *, Instruction);
void print_lui(Instruction);
void print_jal(Instruction);
void print_ecall(Instruction);
void write_rtype(Instruction);
void write_itype_except_load(Instruction); 
void write_load(Instruction);
void write_store(Instruction);
void write_branch(Instruction);


void decode_instruction(uint32_t instruction_bits) {
    Instruction instruction = parse_instruction(instruction_bits);
    switch(instruction.opcode) {
        case 0x33:
            write_rtype(instruction);
            break;
        case 0x13:
            write_itype_except_load(instruction);
            break;
        case 0x3:
            write_load(instruction);
            break;
        case 0x23:
            write_store(instruction);
            break;
        case 0x63:
            write_branch(instruction);
            break;
        case 0x37:
            print_lui(instruction);
            break;
        case 0x6F:
            print_jal(instruction);
            break;
        case 0x73:
            print_ecall(instruction);
            break;
        default: // undefined opcode
            handle_invalid_instruction(instruction);
            break;
    }
}

void write_rtype(Instruction instruction) {
    /* YOUR CODE HERE */
    switch (instruction.rtype.funct3) {
        case 0x0:
            switch (instruction.rtype.funct7) {
                case 0x0:
		  print_rtype("add", instruction);
                    break;
		    case 0x1:
                    print_rtype("mul", instruction);
                    break;
                case 0x20:
                    print_rtype("sub", instruction);
                    break;
                default:
                    handle_invalid_instruction(instruction);
                break;      
            }
            break;
        /* YOUR CODE HERE */
        /* call print_rtype */
        default:
            handle_invalid_instruction(instruction);
        break;
    }
}

void write_itype_except_load(Instruction instruction) {
    switch (instruction.itype.funct3) {
      /* YOUR CODE HERE */
      /* call print_itype_except_load */
        //case 0x13:
            //switch (instruction.itype.funct3) {
                case 0x0:
                    print_itype_except_load("addi", instruction, instruction.itype.imm);
                    break;
                case 0x1:
                    print_itype_except_load("slli", instruction, instruction.itype.imm);
                    break;
                case 0x2:
                    print_itype_except_load("slti", instruction, instruction.itype.imm);
                    break;
                case 0x4:
                    print_itype_except_load("xori", instruction, instruction.itype.imm);
                    break;
                case 0x5:
                    switch (instruction.itype.imm) {
                        case 0x00:
                            print_itype_except_load("srli", instruction, 0x00);
                            break;
                        case 0x20:
                            print_itype_except_load("srai", instruction, 0x20);
                            break;
                    }
                    break;
                case 0x6:
                    print_itype_except_load("ori", instruction, instruction.itype.imm);
                    break;
                case 0x7:
                    print_itype_except_load("andi", instruction, instruction.itype.imm);
                    break;
            //}
            
                default:
                    handle_invalid_instruction(instruction);
                break;  
    }
}

void write_load(Instruction instruction) {
    switch (instruction.itype.funct3) {
      /* YOUR CODE HERE */
      /* call print_load */
        default:
            handle_invalid_instruction(instruction);
            break;
    }
}

void write_store(Instruction instruction) {
    switch (instruction.stype.funct3) {
      /* YOUR CODE HERE */
      /* call print_store */
        default:
            handle_invalid_instruction(instruction);
            break;
    }
}

void write_branch(Instruction instruction) {
    switch (instruction.sbtype.funct3) {
      /* YOUR CODE HERE */
      /* call print_branch */
        default:
            handle_invalid_instruction(instruction);
            break;
    }
}

void print_rtype(char *name, Instruction instruction) {
  printf(RTYPE_FORMAT, name, instruction.rtype.rd, instruction.rtype.rs1,
         instruction.rtype.rs2);
}

void print_itype_except_load(char *name, Instruction instruction, int imm) {
    printf(ITYPE_FORMAT, name, instruction.itype.rd, 
    instruction.itype.rs1, sign_extend_number(imm, 12));
}

void print_load(char *name, Instruction instruction) {
    printf(MEM_FORMAT, name, instruction.itype.rd, instruction.itype.imm,
    instruction.itype.rs1);
}

void print_store(char *name, Instruction instruction) {
    /* YOUR CODE HERE */
}

void print_branch(char *name, Instruction instruction) {
    /* YOUR CODE HERE */
}

void print_lui(Instruction instruction) {
    printf(LUI_FORMAT, instruction.utype.rd, instruction.utype.imm);
}

void print_jal(Instruction instruction) {
    printf(JAL_FORMAT, instruction.ujtype.rd, get_jump_offset(instruction));
}

void print_ecall(Instruction instruction) {
    /* YOUR CODE HERE */
}
