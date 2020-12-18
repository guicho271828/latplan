import time
import imageio
import os.path
import numpy as np
import latplan

from latplan.puzzles.objutil import tiled_bboxes, image_to_tiled_objects, bboxes_to_coord, random_object_masking

d = "random_object_masking"
if os.path.exists(d):
    import glob
    for f in glob.glob(os.path.join(d,"*")):
        print(f"removing {f}")
        os.remove(f)
else:
    os.makedirs(d)
print(d)


class dummy(latplan.model.BaseFirstOrderMixin):
    def __init__(self,parameters):
        self.parameters=parameters
    def unnormalize(self,x):
        return x



def sokoban_coord_tr(track="sokoban-27-True-False-True"):

    with np.load(os.path.join(latplan.__path__[0],"puzzles",track+".npz")) as data:
        pres   = data['pres'] / 255 # [B,25,sH*sW*C]
        sucs   = data['sucs'] / 255 # [B,25,sH*sW*C]
        bboxes = data['bboxes'] # [B,25,4]
        picsize = data['picsize']
        print("loaded")

    coord = bboxes_to_coord(bboxes)

    pres = np.concatenate([pres,coord], axis=-1)
    sucs = np.concatenate([sucs,coord], axis=-1)

    B, O, F = pres.shape
    transitions = np.stack([pres,sucs], axis=1) # [B, 2, O, F]

    print(transitions.shape)
    t1 = time.time()
    pruned = random_object_masking(transitions,int(O*0.9))
    t2 = time.time()
    print(pruned.shape)
    print(f"{track} -- {len(transitions)} took {t2-t1} sec")
    transitions = transitions[:10]
    pruned = pruned[:10]

    net = dummy(parameters={"picsize":picsize})
    render, render_each = net.blocks_coord_renderer()

    transitions_pres_rendered = render(transitions[:,0])
    transitions_sucs_rendered = render(transitions[:,1])
    pruned_pres_rendered = render(pruned[:,0])
    pruned_sucs_rendered = render(pruned[:,1])

    for i,render in enumerate(transitions_pres_rendered):
        imageio.imsave(os.path.join(d,f"{track}-transitions-{i}-pres.png"),render)
    for i,render in enumerate(transitions_sucs_rendered):
        imageio.imsave(os.path.join(d,f"{track}-transitions-{i}-sucs.png"),render)
    for i,render in enumerate(pruned_pres_rendered):
        imageio.imsave(os.path.join(d,f"{track}-pruned-{i}-pres.png"),render)
    for i,render in enumerate(pruned_sucs_rendered):
        imageio.imsave(os.path.join(d,f"{track}-pruned-{i}-sucs.png"),render)

    pass



sokoban_coord_tr("sokoban_image-5-False-True")
sokoban_coord_tr("sokoban_image-10-False-True")
sokoban_coord_tr("sokoban_image-20-False-True")
# sokoban_coord_tr("sokoban_image-28-False-True")
