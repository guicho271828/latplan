#!/usr/bin/awk -f


BEGIN {
    translate      = 10000000
    initialization = 100000000
    search         = 1000000000
    print "{"
}

# Translator variables: 32


/Translator variables:/{
    print "\"variables\":"$3","
}


# Translator derived variables: 0


/Translator derived /{
    print "\"derived_variables\":"$4","
}


# Translator facts: 64


/Translator facts:/{
    print "\"facts\":"$3","
}


# Translator goal facts: 32


/Translator goal facts:/{
    print "\"goal_facts\":"$4","
}


# Translator mutex groups: 13


/Translator mutex groups/{
    print "\"mutex_groups\":"$4","
}


# Translator total mutex groups size: 26


/Translator total mutex group size/{
    print "\"total\":"$6","
}


# Translator operators: 273


/Translator operators:/{
    print "\"operators\":"$3","
}


# Translator axioms: 0


/Translator axioms:/{
    print "\"axioms\":"$3","
}


# Translator task size: 5362


/Translator task /{
    print "\"task\":"$4","
}


# Translator peak memory: 306760 KB


/Translator peak memory/{
    print "\"peak\":"$4","
}

# Done! [41.640s CPU, 42.502s wall-clock]

/^Done!/{
    sub(/s/,"",$4)
    translate=$4
    print "\"translate\":"$4","
}

/Plan length:/{
    print "\"length\":"$3","
}
/Plan cost:/{
    print "\"cost\":"$3","
}



/Expanded [0-9]+ state\(s\)./{
    print "\"expanded\":"$2","
}
/Evaluated [0-9]+ state\(s\)./{
    print "\"evaluated\":"$2","
}
/Generated [0-9]+ state\(s\)./{
    print "\"generated\":"$2","
}

/Search time: [.0-9]+s/{
    sub(/s/,"",$3)
    search = $3
    print "\"search\":"$3","
}
/Total time: [.0-9]+s/{
    sub(/s/,"",$3)
    initialization = $3 - search
    print "\"initialization\":"initialization","
}




END {
    print "\"total\":"translate+search+initialization
    print "}"
}
