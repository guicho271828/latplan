
# usage:
# parallel awk -f summary.awk ::: samples/*/summary | column -t

function trim(str,begin,end){
    return substr(str,begin,length(str)-begin-end+1)
}

/Reconstruction MSE: [0-9.e\-]* \(validation\)/ { reconstruction = $3 }

# Reconstruction MSE (gaussian 0.3): 9.877090238143725e-15
# Reconstruction MSE (gaussian 0.3): 9.962790514176766e-15 \(validation\)
# Reconstruction MSE (salt 0.06): 9.877390134864439e-15
# Reconstruction MSE (salt 0.06): 9.962375891549085e-15 \(validation\)
# Reconstruction MSE (pepper 0.06): 4.568866535183683e-06
# Reconstruction MSE (pepper 0.06): 6.277518795880124e-06 \(validation\)
# Latent activation: 0.5041667222976685
/Latent activation: [0-9.e\-]* \(validation\)/{ activation = $3 }
# Inactive bits: 0.0
/Inactive bits \(always true or false\): [0-9.e\-]* \(validation\)/{ inactive = $7 }
# Latent variance (max,min,mean,median): [0.21174002, 0.0, 0.00012883308, 0.0]
/Latent variance \(max,min,mean,median\): \[[0-9.e\-]*, [0-9.e\-]*, [0-9.e\-]*, [0-9.e\-]*\] \(validation\)/{
    variance = trim($4,2,1)
}

BEGIN {
    variance = 0
    activation = 0
    inactive = 0
    reconstruction = 0
}
END {
    print variance, activation, inactive, reconstruction, FILENAME
}
