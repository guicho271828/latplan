nohup  ./plan.py blind 'run_hanoi4     ( "samples/hanoi4_fc2"               ,"fc2", import_module("puzzles.hanoi")             )'   &> blind-hanoi4     &
nohup  ./plan.py blind 'run_hanoi10     ( "samples/hanoi10_fc2"               ,"fc2", import_module("puzzles.hanoi")             )'   &> blind-hanoi10     &
nohup  ./plan.py blind 'run_puzzle    ( "samples/mnist_puzzle33p_fc2"     ,"fc2", import_module("puzzles.mnist_puzzle")      )'   &> blind-mnist     &
nohup  ./plan.py blind 'run_puzzle    ( "samples/lenna_puzzle33p_fc2"     ,"fc2", import_module("puzzles.lenna_puzzle")      )'   &> blind-lenna     &
nohup  ./plan.py blind 'run_puzzle    ( "samples/mandrill_puzzle33p_fc2"  ,"fc2", import_module("puzzles.mandrill_puzzle")   )'   &> blind-mandrill  &
nohup  ./plan.py blind 'run_lightsout ( "samples/digital_lightsout_fc2"   ,"fc2", import_module("puzzles.digital_lightsout") )'   &> blind-lightsout &
nohup  ./plan.py blind 'run_lightsout ( "samples/digital_lightsout_skewed_fc"   ,"fc", import_module("puzzles.digital_lightsout_skewed") )'   &> blind-lightsout-skewed &

nohup  ./plan.py pdb   'run_hanoi4     ( "samples/hanoi4_fc2"               ,"fc2", import_module("puzzles.hanoi")             )'   &> pdb-hanoi4     &
nohup  ./plan.py pdb   'run_hanoi10     ( "samples/hanoi10_fc2"               ,"fc2", import_module("puzzles.hanoi")             )'   &> pdb-hanoi10     &
nohup  ./plan.py pdb   'run_puzzle    ( "samples/mnist_puzzle33p_fc2"     ,"fc2", import_module("puzzles.mnist_puzzle")      )'   &> pdb-mnist     &
nohup  ./plan.py pdb   'run_puzzle    ( "samples/lenna_puzzle33p_fc2"     ,"fc2", import_module("puzzles.lenna_puzzle")      )'   &> pdb-lenna     &
nohup  ./plan.py pdb   'run_puzzle    ( "samples/mandrill_puzzle33p_fc2"  ,"fc2", import_module("puzzles.mandrill_puzzle")   )'   &> pdb-mandrill  &
nohup  ./plan.py pdb   'run_lightsout ( "samples/digital_lightsout_fc2"   ,"fc2", import_module("puzzles.digital_lightsout") )'   &> pdb-lightsout &
nohup  ./plan.py pdb 'run_lightsout ( "samples/digital_lightsout_skewed_fc"   ,"fc", import_module("puzzles.digital_lightsout_skewed") )'   &> pdb-lightsout-skewed &

nohup  ./plan.py mands   'run_hanoi4     ( "samples/hanoi4_fc2"               ,"fc2", import_module("puzzles.hanoi")             )'   &> mands-hanoi4     &
nohup  ./plan.py mands   'run_hanoi10     ( "samples/hanoi10_fc2"               ,"fc2", import_module("puzzles.hanoi")             )'   &> mands-hanoi10     &
nohup  ./plan.py mands   'run_puzzle    ( "samples/mnist_puzzle33p_fc2"     ,"fc2", import_module("puzzles.mnist_puzzle")      )'   &> mands-mnist     &
nohup  ./plan.py mands   'run_puzzle    ( "samples/lenna_puzzle33p_fc2"     ,"fc2", import_module("puzzles.lenna_puzzle")      )'   &> mands-lenna     &
nohup  ./plan.py mands   'run_puzzle    ( "samples/mandrill_puzzle33p_fc2"  ,"fc2", import_module("puzzles.mandrill_puzzle")   )'   &> mands-mandrill  &
nohup  ./plan.py mands   'run_lightsout ( "samples/digital_lightsout_fc2"   ,"fc2", import_module("puzzles.digital_lightsout") )'   &> mands-lightsout &
nohup  ./plan.py mands 'run_lightsout ( "samples/digital_lightsout_skewed_fc"   ,"fc", import_module("puzzles.digital_lightsout_skewed") )'   &> mands-lightsout-skewed &

