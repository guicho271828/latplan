
import numpy as np
from model import GumbelAE

def reshape_batch1D(images):
    return np.reshape(images,(images.shape[0], -1))

def dump_actions(transitions,path):
    # assert 2 == transitions.shape[0]
    ae = GumbelAE(path)
    ae.train(np.concatenate(transitions,axis=0))

    orig, dest = transitions[0], transitions[1]
    orig_b, dest_b = ae.encode_binary(orig), ae.encode_binary(dest)
    
    actions = np.concatenate((orig_b, dest_b), axis=1)
    np.savetxt(ae.local("actions.csv"),actions,"%d")
    return actions


if __name__ == '__main__':
    def plot_grid(images,name="plan.png"):
        import matplotlib.pyplot as plt
        l = len(images)
        w = 6
        h = max(l//6,1)
        plt.figure(figsize=(20, h*2))
        for i,image in enumerate(images):
            # display original
            ax = plt.subplot(h,w,i+1)
            plt.imshow(image,interpolation='nearest',cmap='gray',)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.savefig(name)

    # from counter import counter_transitions
    # transitions = counter_transitions(n=1000)
    # counter_actions = dump_actions((reshape_batch1D(transitions[0]),
    #                                 reshape_batch1D(transitions[1])),
    #                                "samples/counter_model/")
    # print counter_actions[:3]
    # ae = GumbelAE("samples/counter_model/")
    # xs = transitions[0][:18]
    # ys = ae.autoencode(xs)
    # images = np.reshape(np.einsum('ab...->ba...',(xs,ys)),(36,28,28))
    # plot_grid(images, ae.local("autoencoding.png"))
    
    
    from puzzle import puzzle_transitions
    transitions = puzzle_transitions(2,2)
    puzzle_actions = dump_actions((reshape_batch1D(transitions[0]),
                                   reshape_batch1D(transitions[1])),
                                  "samples/puzzle_model/")
    print puzzle_actions[:3]
    ae = GumbelAE("samples/puzzle_model/")
    xs = transitions[0][:18]
    ys = ae.autoencode(xs)
    print xs.shape, ys.shape
    images = np.reshape(np.einsum('ab...->ba...',(xs,ys)),(36,6*2,5*2))
    plot_grid(images, ae.local("autoencoding.png"))
