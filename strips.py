
import numpy as np
from model import GumbelAE

def dump_actions(transitions,path):

    ae = GumbelAE(path)
    ae.train(np.concatenate(transitions,axis=0))

    orig, dest = transitions
    orig_b, dest_b = ae.encode_binary(orig), ae.encode_binary(dest)
    
    actions = np.concatenate((orig_b, dest_b), axis=1)
    import os
    np.savetxt(os.path.join(path,"actions.csv"),actions,"%d")
    return actions


if __name__ == '__main__':
    from counter import counter_transitions
    actions = dump_actions(counter_transitions(n=10000), "counter_model/")
    print actions[:3]

