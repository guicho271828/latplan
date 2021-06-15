from . import common
from . import hanoi
from . import puzzle
from . import lightsout
from . import sokoban
from . import blocks


def main(parameters):
    import latplan.util.tuning
    latplan.util.tuning.parameters.update(parameters)

    import sys
    if len(sys.argv) == 1:
        print(f"""
Usage: {sys.argv[0]} MODE TASK parameters...

MODE is a single string which main contain "learn" "plot" "dump" "summary" "debug" "reproduce".
You can specify multiple modes by simply concatenating them (e.g. learnplotdump),
or separating it by an arbitrary separator (e.g. learn_plot_dump or learn-plot-dump).

the available tasks are as follows:
""")
        print({ k for k,v in common.tasks.items() if hasattr(v, '__call__')})
    else:
        print('args:',sys.argv)
        sys.argv.pop(0)
        common.mode     = sys.argv.pop(0)
        common.sae_path = "_".join(sys.argv)
        task = sys.argv.pop(0)

        def myeval(str):
            try:
                return eval(str)
            except:
                return str

        try:
            common.tasks[task](*map(myeval,sys.argv))
        except:
            latplan.util.stacktrace.format()
