#R-type: 7 points
./riscv -v -r ./code/input/Ri/Ri.input > ./code/out/Ri/Ri.trace
echo "\n python3 emulator_tester.py Ri/Ri"
python3 emulator_tester.py Ri/Ri
#I-type except load: 13 points
./riscv -r ./code/input/I/I.input > ./code/out/I/I.trace
echo "\n python3 emulator_tester.py I/I"
python3 emulator_tester.py I/I
#I-type load: 6 points
./riscv -r ./code/input/I/L.input > ./code/out/I/L.trace
echo "\n python3 emulator_tester.py I/L"
python3 emulator_tester.py I/L
#S-type: 10 points
./riscv -r ./code/input/S/S.input > ./code/out/S/S.trace
echo "\n python3 emulator_tester.py S/S"
python3 emulator_tester.py S/S
#SB-type: 10 points
./riscv -r ./code/input/SB/SB.input > ./code/out/SB/SB.trace
echo "\n python3 emulator_tester.py SB/SB"
python3 emulator_tester.py SB/SB
#U-type: 10 points
./riscv -r ./code/input/U/U.input > ./code/out/U/U.trace
echo "\n python3 emulator_tester.py U/U"
python3 emulator_tester.py U/U
#UJ-type: 9 points
./riscv -r ./code/input/UJ/UJ.input > ./code/out/UJ/UJ.trace
echo "\n python3 emulator_tester.py UJ/UJ"
python3 emulator_tester.py UJ/UJ
#multiply testcase: 15 points
./riscv -r -e ./code/input/multiply.input > ./code/out/multiply.trace
echo "\n python3 emulator_tester.py multiply"
python3 emulator_tester.py multiply
#random testcase: 15 points
./riscv -r -e ./code/input/random.input > ./code/out/random.trace
echo "\n python3 emulator_tester.py random"
python3 emulator_tester.py random
