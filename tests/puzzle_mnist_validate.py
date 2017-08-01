import importlib

import latplan
import latplan.puzzles.puzzle_mnist as p
import latplan.puzzles.model.puzzle as m

importlib.reload(m)
importlib.reload(p)

p.setup()


import itertools
c = [ c for c in itertools.islice(p.generate_configs(9), 100) ]
s = p.states(3,3,c)
print(m.validate_states(s,3,3))


