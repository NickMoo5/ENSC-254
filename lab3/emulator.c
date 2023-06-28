#include <stdio.h> // for stderr
#include <stdlib.h> // for exit()
#include "types.h"
#include "utils.h"
#include "riscv.h"

void execute_rtype(Instruction, Processor *);
void execute_itype_except_load(Instruction, Processor *);
void execute_branch(Instruction, Processor *);
void execute_jal(Instruction, Processor *);
void execute_load(Instruction, Processor *, Byte *);
void execute_store(Instruction, Processor *, Byte *);
void execute_ecall(Processor *, Byte *);
void execute_lui(Instruction, Processor *);

void execute_instruction(uint32_t instruction_bits, Processor *processor,Byte *memory) {    
    Instruction instruction = parse_instruction(instruction_bits);
    switch(instruction.opcode) {
        case 0x33:
            execute_rtype(instruction, processor);
            break;
        case 0x13:
            execute_itype_except_load(instruction, processor);
            break;
        case 0x73:
            execute_ecall(processor, memory);
            break;
        case 0x63:
            execute_branch(instruction, processor);
            break;
        case 0x6F:
            execute_jal(instruction, processor);
            break;
        case 0x23:
            execute_store(instruction, processor, memory);
            break;
        case 0x03:
            execute_load(instruction, processor, memory);
            break;
        case 0x37:
            execute_lui(instruction, processor);
            break;
        default: // undefined opcode
            handle_invalid_instruction(instruction);
            exit(-1);
            break;
    }
}

void execute_rtype(Instruction instruction, Processor *processor) {
    switch (instruction.rtype.funct3){
        case 0x0:
            switch (instruction.rtype.funct7) {
                case 0x0:
                    // Add
                    processor->R[instruction.rtype.rd] =
                        ((sWord)processor->R[instruction.rtype.rs1]) +
                        ((sWord)processor->R[instruction.rtype.rs2]);
                    break;
                case 0x1:
                    // Mul
                    processor->R[instruction.rtype.rd] =
                        ((sWord)processor->R[instruction.rtype.rs1]) *
                        ((sWord)processor->R[instruction.rtype.rs2]);
                    break;
                case 0x20:
                    // Sub
                    processor->R[instruction.rtype.rd] =
                        ((sWord)processor->R[instruction.rtype.rs1]) -
                        ((sWord)processor->R[instruction.rtype.rs2]);
                    break;
                default:
                    handle_invalid_instruction(instruction);
                    exit(-1);
                    break;
            }
            break;
        case 0x4:
            // xor
            processor->R[instruction.rtype.rd] = 
                ((sWord)processor->R[instruction.rtype.rs1]) ^ 
                ((sWord)processor->R[instruction.rtype.rs2]);
            break;
        case 0x6:
            // or
            processor->R[instruction.rtype.rd] = 
                ((sWord)processor->R[instruction.rtype.rs1]) | 
                ((sWord)processor->R[instruction.rtype.rs2]);
            break;
        case 0x7:
            // and
            processor->R[instruction.rtype.rd] = 
                ((sWord)processor->R[instruction.rtype.rs1]) & 
                ((sWord)processor->R[instruction.rtype.rs2]);
            break;
        case 0x1:
            switch (instruction.rtype.funct7) {
                case 0x0:
                    // shift left logical
                    processor->R[instruction.rtype.rd] = 
                        ((sWord)processor->R[instruction.rtype.rs1]) << 
                        ((sWord)processor->R[instruction.rtype.rs2]);
                    break;
                case 0x1:
                    // multiply high
                    processor->R[instruction.rtype.rd] = 
                        ((sDouble)sign_extend_number(processor->R[instruction.rtype.rs1], 5) * 
                        ((sDouble)sign_extend_number(processor->R[instruction.rtype.rs2], 5)) >> 32);
                    break;
                default:
                    handle_invalid_instruction(instruction);
                    exit(-1);
                    break;
            }
            break;
        case 0x5:
            switch (instruction.rtype.funct7) {
                case 0x0:
                    // shift right logical
                    processor->R[instruction.rtype.rd] = 
                        ((sWord)processor->R[instruction.rtype.rs1]) >> 
                        ((sWord)processor->R[instruction.rtype.rs2]);
                    break;
                case 0x2:
                    // shift right arithmetic
                    processor->R[instruction.rtype.rd] = 
                        sign_extend_number(((sWord)processor->R[instruction.rtype.rs1]) >>  
                        (sWord)processor->R[instruction.rtype.rs2], 32);
                    break;
                default:
                    handle_invalid_instruction(instruction);
                    exit(-1);
                    break;
                }
                break;
        case 0x2:
            // set less than
            processor->R[instruction.rtype.rd] = 
                (((sWord)processor->R[instruction.rtype.rs1]) < 
                ((sWord)processor->R[instruction.rtype.rs2])) ? 1 : 0;
            break;
        case 0x3:
            // set less than ( unsigned)
            processor->R[instruction.rtype.rd] = 
                sign_extend_number((((Word)processor->R[instruction.rtype.rs1]) < 
                ((Word)processor->R[instruction.rtype.rs2])) ? 1 : 0, 32);
            break;
        default:
            handle_invalid_instruction(instruction);
            exit(-1);
            break;
    }
    // update PC
    processor->PC += 4;
}

void execute_itype_except_load(Instruction instruction, Processor *processor) {
    switch (instruction.itype.funct3) {
        case 0x0:
            //addi
            processor->R[instruction.itype.rd] = 
                ((sWord)processor->R[instruction.itype.rs1]) + 
                ((sWord)sign_extend_number(instruction.itype.imm, 12));
            break;
        case 0x4:
            //xori
            processor->R[instruction.itype.rd] = 
                ((sWord)processor->R[instruction.itype.rs1]) ^ 
                ((sWord)sign_extend_number(instruction.itype.imm, 12));
            break;
        case 0x6:
            //ori
            processor->R[instruction.itype.rd] = 
                ((sWord)processor->R[instruction.itype.rs1]) | 
                ((sWord)sign_extend_number(instruction.itype.imm, 12));
            break;
        case 0x7:
            //andi
            processor->R[instruction.itype.rd] = 
                ((sWord)processor->R[instruction.itype.rs1]) & 
                ((sWord)sign_extend_number(instruction.itype.imm, 12));
            break;
        case 0x1:
            //slli
            processor->R[instruction.itype.rd] = 
                ((sWord)processor->R[instruction.itype.rs1]) << 
                ((sWord)sign_extend_number(instruction.itype.imm & ((1U << 5) - 1), 12));
            break;
        case 0x5:
            switch (instruction.itype.imm >> 5) {
                case 0x0:
                    //srli
                    // right 5 bits are imm
                    processor->R[instruction.itype.rd] = 
                        ((Word)processor->R[instruction.itype.rs1]) >> 
                        ((Word)sign_extend_number(instruction.itype.imm & ((1U << 5) - 1), 12));
                    break;
                case 0x20:
                    //srai
                    // right 5 bits are imm
                    processor->R[instruction.itype.rd] = 
                        ((sWord)processor->R[instruction.itype.rs1]) >> 
                        ((sWord)sign_extend_number(instruction.itype.imm & ((1U << 5) - 1), 12));
                    break;
                default:
                    handle_invalid_instruction(instruction);
                    exit(-1);
                    break;
                break; 
            }
            break;
        case 0x2:
            //slti
                processor->R[instruction.itype.rd] = 
                (((sWord)processor->R[instruction.itype.rs1]) < 
                ((sWord)sign_extend_number(instruction.itype.imm, 12))) ? 1 : 0;
            break;
        default:
            handle_invalid_instruction(instruction);
            break;
    }
    // update PC
    processor->PC += 4;    
}

void execute_ecall(Processor *p, Byte *memory) {
    Register i;
    
    // syscall number is given by a0 (x10)
    // argument is given by a1
    switch(p->R[10]) {
        case 1: // print an integer
            printf("%d",p->R[11]);
            p->PC += 4;
            break;
        case 4: // print a string
            for(i=p->R[11];i<MEMORY_SPACE && load(memory,i,LENGTH_BYTE);i++) {
                printf("%c",load(memory,i,LENGTH_BYTE));
            }
            p->PC += 4;
            break;
        case 10: // exit
            printf("exiting the simulator\n");
            exit(0);
            break;
        case 11: // print a character
            printf("%c",p->R[11]);
            p->PC += 4;
            break;
        default: // undefined ecall
            printf("Illegal ecall number %d\n", p->R[10]);
            exit(-1);
            break;
    }
}

void execute_branch(Instruction instruction, Processor *processor) {
    switch (instruction.sbtype.funct3) {
        case 0x0:
            //sWord offset = sign_extend_number(get_branch_offset(instruction), 12);
            if ((sWord)processor->R[instruction.sbtype.rs1] == (sWord)processor->R[instruction.sbtype.rs2]) {
                processor->PC += get_branch_offset(instruction);
            } else {
                processor->PC += 4;
            }
            break;
        case 0x1:
            //sWord offset = sign_extend_number(get_branch_offset(instruction), 12);
            if (processor->R[instruction.sbtype.rs1] != processor->R[instruction.sbtype.rs2]) {
                processor->PC += get_branch_offset(instruction);
            } else {
                processor->PC += 4;
            }            
            break;
        default:
            handle_invalid_instruction(instruction);
            exit(-1);
            break;
    }
}

void execute_load(Instruction instruction, Processor *processor, Byte *memory) {
    switch (instruction.itype.funct3) {
        case 0x0:
            processor->R[instruction.itype.rd] = 
                sign_extend_number(memory[((sWord)processor->R[instruction.itype.rs1]) + 
                    ((sWord)(sign_extend_number(instruction.itype.imm, 12)))], 8);
            break;
        case 0x1:
            processor->R[instruction.itype.rd] = 
                sign_extend_number((memory[((sWord)processor->R[instruction.itype.rs1]) + 
                    ((sWord)(sign_extend_number(instruction.itype.imm, 12))) + 1] << 8) + 
                    memory[((sWord)processor->R[instruction.itype.rs1]) + 
                    ((sWord)(sign_extend_number(instruction.itype.imm, 12)))], 16);
            break;
        case 0x2:
            processor->R[instruction.itype.rd] = 
                (memory[((sWord)processor->R[instruction.itype.rs1]) + 
                    ((sWord)(sign_extend_number(instruction.itype.imm, 12))) + 3] << 24) + 
                        (memory[((sWord)processor->R[instruction.itype.rs1]) + 
                    ((sWord)(sign_extend_number(instruction.itype.imm, 12))) + 2] << 16) + 
                        (memory[((sWord)processor->R[instruction.itype.rs1]) + 
                    ((sWord)(sign_extend_number(instruction.itype.imm, 12))) + 1] << 8) + 
                        memory[((sWord)processor->R[instruction.itype.rs1]) + 
                    ((sWord)(sign_extend_number(instruction.itype.imm, 12)))];
            break;
        default:
            handle_invalid_instruction(instruction);
            break;
    }
    // update PC
    processor->PC += 4;
}

void execute_store(Instruction instruction, Processor *processor, Byte *memory) {
    switch (instruction.stype.funct3) {
        case 0x0:
            // sb
            memory[(sWord)processor->R[instruction.stype.rs1] + (sWord)get_store_offset(instruction)] = 
                (Byte)(processor->R[instruction.stype.rs2] & ((1U << 8) - 1));
            break;
        case 0x1:
            // sh
            memory[(sWord)processor->R[instruction.stype.rs1] + (sWord)get_store_offset(instruction)] = 
                (Byte)(processor->R[instruction.stype.rs2] & ((1U << 8) - 1));
            memory[(sWord)processor->R[instruction.stype.rs1] + (sWord)get_store_offset(instruction) + 1] = 
                (Byte)((processor->R[instruction.stype.rs2] >> 8) & ((1U << 8) - 1));
            break;
        case 0x2:
            // sw
            memory[(sWord)processor->R[instruction.stype.rs1] + (sWord)get_store_offset(instruction)] = 
                (Byte)(processor->R[instruction.stype.rs2] & ((1U << 8) - 1));
            memory[(sWord)processor->R[instruction.stype.rs1] + (sWord)get_store_offset(instruction) + 1] = 
                (Byte)((processor->R[instruction.stype.rs2] >> 8) & ((1U << 8) - 1));
            memory[(sWord)processor->R[instruction.stype.rs1] + (sWord)get_store_offset(instruction) + 2] = 
                (Byte)((processor->R[instruction.stype.rs2] >> 16) & ((1U << 8) - 1));
            memory[(sWord)processor->R[instruction.stype.rs1] + (sWord)get_store_offset(instruction) + 3] = 
                (Byte)(processor->R[instruction.stype.rs2] >> 24);
            break;
        default:
            handle_invalid_instruction(instruction);
            exit(-1);
            break;
    }
    // update PC
    processor->PC += 4;
}

void execute_jal(Instruction instruction, Processor *processor) {
    sWord offset = sign_extend_number(get_jump_offset(instruction), 20);
    processor->R[instruction.ujtype.rd] = (Word)(processor->PC + 4);
    // update PC
    processor->PC += offset;
}

void execute_lui(Instruction instruction, Processor *processor) {
    processor->R[instruction.utype.rd] = instruction.utype.imm << 12;
    // update PC
    processor->PC += 4;
}

void store(Byte *memory, Address address, Alignment alignment, Word value) {
    if(alignment == LENGTH_BYTE) {
        return memory[address];
    } else if(alignment == LENGTH_HALF_WORD) {
        return (memory[address+1] << 8) + memory[address];
    } else if(alignment == LENGTH_WORD) {
        return (memory[address+3] << 24) + (memory[address+2] << 16)
               + (memory[address+1] << 8) + memory[address];
    } else {
        printf("Error: Unrecognized alignment %d\n", alignment);
        exit(-1);
    }
}

Word load(Byte *memory, Address address, Alignment alignment) {
    if(alignment == LENGTH_BYTE) {
        return memory[address];
    } else if(alignment == LENGTH_HALF_WORD) {
        return (memory[address+1] << 8) + memory[address];
    } else if(alignment == LENGTH_WORD) {
        return (memory[address+3] << 24) + (memory[address+2] << 16)
               + (memory[address+1] << 8) + memory[address];
    } else {
        printf("Error: Unrecognized alignment %d\n", alignment);
        exit(-1);
    }
}
