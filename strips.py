
import numpy as np
from model import GumbelAE

def reshape_batch1D(images,repeat=None):
    result = np.reshape(images,(images.shape[0], -1))
    if repeat is not None:
        return np.repeat(result,repeat,axis=0)
    else:
        return result

def dump_actions(transitions,path):
    # assert 2 == transitions.shape[0]
    ae = GumbelAE(path)
    ae.train(np.concatenate(transitions,axis=0),
             # anneal_rate=0.0001,
             # epoch=400
    )

    orig, dest = transitions[0], transitions[1]
    orig_b, dest_b = ae.encode_binary(orig), ae.encode_binary(dest)
    
    actions = np.concatenate((orig_b, dest_b), axis=1)
    np.savetxt(ae.local("actions.csv"),actions,"%d")
    return actions


if __name__ == '__main__':
    import numpy.random as random
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

    def run(path, shape, transitions):
        actions = dump_actions(transitions, path)
        print actions[:3]
        ae = GumbelAE(path)
        xs = transitions[0][random.randint(0,transitions[0].shape[0],12)]
        zs = ae.encode_binary(xs).reshape((-1,4,4))
        ys = ae.autoencode(xs).reshape(shape)
        xs = xs.reshape(shape)
        images = []
        for x,z,y in zip(xs,zs,ys):
            images.append(x)
            images.append(z)
            images.append(y)
        plot_grid(images, ae.local("autoencoding.png"))

    from counter import counter_transitions
    transitions = counter_transitions(n=1000)
    run("samples/counter_model/", (-1,28,28), transitions)
    
    from puzzle import puzzle_transitions
    transitions = puzzle_transitions(2,2)
    transitions = np.repeat(transitions,100,axis=1)
    run("samples/puzzle_model/", (-1,6*2,5*2), transitions)

    from mnist_puzzle import mnist_puzzle_transitions
    transitions = mnist_puzzle_transitions(2,2)
    transitions = np.repeat(transitions,100,axis=1)
    run("samples/mnist_puzzle_model/", (-1,28*2,28*2), transitions)

    
