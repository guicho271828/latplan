# -*- truncate-lines : t -*-

./plan.py blind 'run_hanoi4    ( "samples/hanoi4_fc2"                  ,"fc2", import_module("puzzles.hanoi"                    ) )' |& tee blind-hanoi4
./plan.py blind 'run_hanoi10   ( "samples/hanoi10_fc2"                 ,"fc2", import_module("puzzles.hanoi"                    ) )' |& tee blind-hanoi10
./plan.py blind 'run_puzzle    ( "samples/mnist_puzzle33p_fc2"         ,"fc2", import_module("puzzles.mnist_puzzle"             ) )' |& tee blind-mnist     
./plan.py blind 'run_puzzle    ( "samples/mandrill_puzzle33p_fc2"      ,"fc2", import_module("puzzles.mandrill_puzzle"          ) )' |& tee blind-mandrill 
./plan.py blind 'run_lightsout ( "samples/digital_lightsout_fc2"       ,"fc2", import_module("puzzles.digital_lightsout"        ) )' |& tee blind-lightsout
./plan.py blind 'run_lightsout ( "samples/digital_lightsout_skewed_fc" ,"fc",  import_module("puzzles.digital_lightsout_skewed" ) )' |& tee blind-lightsout-skewed

################################################################

for h in blind pdb mands ; do
    ./plan.py $h 'run_puzzle    ( "samples/mandrill_puzzle33p_fc2"      ,"fc2", import_module("puzzles.mandrill_puzzle"          ) )' |& tee $h-mandrill
done

for h in blind pdb mands ; do
    ./plan_noise.py $h 'run_puzzle    ( "samples/mandrill_puzzle33p_fc2"      ,"fc2", import_module("puzzles.mandrill_puzzle"          ) )' |& tee $h-mandrill-noise
done


for h in blind pdb mands ; do
    ./plan.py $h 'run_puzzle    ( "samples/mnist_puzzle33p_fc2"         ,"fc2", import_module("puzzles.mnist_puzzle"             ) )' |& tee $h-mnist  
done


for h in blind pdb mands ; do
    ./plan_noise.py $h 'run_puzzle    ( "samples/mnist_puzzle33p_fc2"         ,"fc2", import_module("puzzles.mnist_puzzle"             ) )' |& tee $h-mnist-noise
done




./plan_iter.py blind 'run_puzzle    ( "samples/mnist_puzzle33p_fc2"         ,"fc2", import_module("puzzles.mnist_puzzle"             ) )' |& tee blind-mnist-iter
