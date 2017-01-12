
import subprocess

def echodo(cmd,file=None):
    subprocess.call(["echo"]+cmd)
    if file is None:
        subprocess.call(cmd)
    else:
        with open(file,"w") as f:
            subprocess.call(cmd,stdout=f)

init = ["0", "0"]
goal = ["1", "1"]
            
echodo(["./domain.bin","actions.csv"], "domain.pddl")
echodo(["./problem.bin", *init, *goal,], "problem.pddl")
echodo(["../planner-scripts/limit.sh","-v","--","fd-clean",
        "problem.pddl",
        "domain.pddl"])
echodo(["./parse-plan.bin", "problem.plan", *init])
