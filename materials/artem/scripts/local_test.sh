problem_name="acc_disk_test_01"

rm -f *.tab

cp ~/athena/inputs/hydro/athinput.${problem_name} .

~/athena/bin/athena -i athinput.${problem_name}

ls

/bin/python3 /home/etrevol/work/all_scripts/bondi_disk.py