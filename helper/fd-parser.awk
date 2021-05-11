#!/usr/bin/awk -f


BEGIN {
    translator_variables = "null"
    translator_derived_variables = "null"
    translator_facts = "null"
    translator_goal_facts = "null"
    translator_mutex_groups = "null"
    translator_total_mutex_group_size = "null"
    translator_operators = "null"
    translator_axioms = "null"
    translator_task = "null"
    translator_peak_memory = "null"
    translate = "null"
    plan_length = "null"
    plan_cost = "null"
    expanded = "null"
    evaluated = "null"
    generated = "null"
    search = "null"
    initialization = "null"
}

# Translator variables: 32


/Translator variables:/{
    translator_variables = $3
}


# Translator derived variables: 0


/Translator derived /{
    translator_derived_variables = $4
}


# Translator facts: 64


/Translator facts:/{
    translator_facts = $3
}


# Translator goal facts: 32


/Translator goal facts:/{
    translator_goal_facts = $4
}


# Translator mutex groups: 13


/Translator mutex groups/{
    translator_mutex_groups = $4

}


# Translator total mutex groups size: 26


/Translator total mutex group size/{
    translator_total_mutex_group_size = $6

}


# Translator operators: 273


/Translator operators:/{
    translator_operators = $3
}


# Translator axioms: 0


/Translator axioms:/{
    translator_axioms = $3
}


# Translator task size: 5362


/Translator task /{
    translator_task = $4
}


# Translator peak memory: 306760 KB


/Translator peak memory/{
    translator_peak_memory = $4
}

# Done! [41.640s CPU, 42.502s wall-clock]

/^Done!/{
    sub(/s/,"",$4)
    translate=$4
}

# [t=1.22939s, 23808 KB] Plan length: 0 step(s).

/Plan length:/{
    plan_length = $6
}

# [t=1.22939s, 23808 KB] Plan cost: 0

/Plan cost:/{
    plan_cost = $6
}

# [t=1.22939s, 23808 KB] Expanded 1 state(s).

/Expanded [0-9]+ state\(s\)./{
    expanded=$5
}

# [t=1.22939s, 23808 KB] Evaluated 1 state(s).

/Evaluated [0-9]+ state\(s\)./{
    evaluated=$5
}

# [t=1.22939s, 23808 KB] Generated 0 state(s).

/Generated [0-9]+ state\(s\)./{
    generated=$5
}

/Search time: [.0-9]+s/{
    sub(/s/,"",$6)
    search = $6
}
/Total time: [.0-9]+s/{
    sub(/s/,"",$6)
    initialization = $6 - search
    # note : this is not total time!!!
}




END {
    total = translate+search+initialization
    print "{"
    print "\"translator_variables\":"translator_variables","
    print "\"translator_derived_variables\":"translator_derived_variables","
    print "\"translator_facts\":"translator_facts","
    print "\"translator_goal_facts\":"translator_goal_facts","
    print "\"translator_mutex_groups\":"translator_mutex_groups","
    print "\"translator_total_mutex_group_size\":"translator_total_mutex_group_size","
    print "\"translator_operators\":"translator_operators","
    print "\"translator_axioms\":"translator_axioms","
    print "\"translator_task\":"translator_task","
    print "\"translator_peak_memory\":"translator_peak_memory","
    print "\"translate\":"translate","
    print "\"plan_length\":"plan_length","
    print "\"plan_cost\":"plan_cost","
    print "\"expanded\":"expanded","
    print "\"evaluated\":"evaluated","
    print "\"generated\":"generated","
    print "\"search\":"search","
    print "\"initialization\":"initialization","
    print "\"total\":"translate+initialization+search # beware of unnecessary comma!
    print "}"
}
