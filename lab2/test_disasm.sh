#R-type: 10 points
./riscv -d ./code/input/R/R.input > ./code/out/R/R.solution
echo "diff ./code/out/R/R.solution ./code/ref/R/R.solution"
diff ./code/out/R/R.solution ./code/ref/R/R.solution
#I-type: 19 points
./riscv -d ./code/input/I/I.input > ./code/out/I/I.solution
echo "diff ./code/out/I/I.solution ./code/ref/I/I.solution"
diff ./code/out/I/I.solution ./code/ref/I/I.solution
./riscv -d ./code/input/I/L.input > ./code/out/I/L.solution
echo "diff ./code/out/I/L.solution ./code/ref/I/L.solution"
diff ./code/out/I/L.solution ./code/ref/I/L.solution
#S-type: 10 points
./riscv -d ./code/input/S/S.input > ./code/out/S/S.solution
echo "diff ./code/out/I/L.solution ./code/ref/I/L.solution"
diff ./code/out/S/S.solution ./code/ref/S/S.solution
#SB-type: 10 points
./riscv -d ./code/input/SB/SB.input > ./code/out/SB/SB.solution
echo "diff ./code/out/SB/SB.solution ./code/ref/SB/SB.solution"
diff ./code/out/SB/SB.solution ./code/ref/SB/SB.solution
#U-type: 10 points
./riscv -d ./code/input/U/U.input > ./code/out/U/U.solution
echo "diff ./code/out/U/U.solution ./code/ref/U/U.solution"
diff ./code/out/U/U.solution ./code/ref/U/U.solution
#UJ-type: 9 points
./riscv -d ./code/input/UJ/UJ.input > ./code/out/UJ/UJ.solution
echo "diff ./code/out/UJ/UJ.solution ./code/ref/UJ/UJ.solution"
diff ./code/out/UJ/UJ.solution ./code/ref/UJ/UJ.solution
#multiply testcase: 24 points
./riscv -d ./code/input/multiply.input > ./code/out/multiply.solution
echo "diff ./code/out/multiply.solution ./code/ref/multiply.solution"
diff ./code/out/multiply.solution ./code/ref/multiply.solution
#random testcase: 21 points
./riscv -d ./code/input/random.input > ./code/out/random.solution
echo "diff ./code/out/random.solution ./code/ref/random.solution"
diff ./code/out/random.solution ./code/ref/random.solution
