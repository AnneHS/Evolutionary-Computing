setwd("/Users/ricardoknauer/PycharmProjects/EC/")

# library(devtools)
library(SPOT)
# library(spotGUI)


#' wrapSystem_parseMatrixToString
#' 
#' Create a String that can be passed via the command line from an R-Matrix
#' 
#' @param m a matrix
#' 
#' @return parsed string
#' 
#' @keywords internal
#' 
wrapSystem_parseMatrixToString <- function(m){
    parseVecToString <- function(x){
        return(paste(x, sep = ",", collapse = ","))
    }
    return(apply(m, 1, parseVecToString))
}

#' wrapSystemCommand
#' 
#' Optimize parameters for a script that is accessible via Command Line
#' 
#' @param systemCall String that calls the command line script. 
#' 
#' @return callable function for SPOT
#' 
#' @export
#' 
wrapSystemCommand <- function(systemCall){
    return(
        function(x){
            paramStrings <- wrapSystem_parseMatrixToString(x)
            doSysCall <- function(p){
                system(paste(systemCall, p), intern = T)
            }
            res <- lapply(paramStrings, doSysCall)
            res <- matrix(as.numeric(unlist(res)), ncol = 1)
            return(res)
        }
    )
}


##############################
# SPOT FOR SPECIALIST CMA-ES #
##############################

# SPOT initially evaluates the utility of 10 hyperparameter settings (each on a separate run of optimization_
# specialist_v2.py), then builds a surrogate model to learn the relation between hyperparameters and utility,
# optimizes the surrogate model (= finds an optimal hyperparameter setting, based on the surrogate model),
# evaluates the model at the found optimum (on another run of the python script), refines the model, ...
# for a total of 20 evaluations

f <- wrapSystemCommand("python3 /Users/ricardoknauer/PycharmProjects/EC/optimization_specialist.py")

# runSpotGUI()

lambdaLo = 50
lambdaHi = 150
muLo = 10
muHi = 15
lower = c(lambdaLo, muLo)
upper = c(lambdaHi, muHi)

spotConfig <- list(
    types = c("integer", "integer"),        # data type of hyperparameters
    noise = TRUE,                           # for stochstic algorithms
    seedFun = 1,
    design = designUniformRandom,           # uniform random initial design (hyperparameter setting)
    model = buildEnsembleStack,             # Bartz-Beielstein, 2016; Bartz-Beielstein & Zaefferer, 2017
    optimizer = optimLBFGSB,                # find optimal hyperparameter setting, based on the model
    optimizerControl = list(funEvals=1000), # 1000 model evaluations -> sampling efficieny!
    plots = TRUE
)

res <- spot(, fun = f, lower = lower, upper = upper, control = spotConfig)
str(res)


##########################
# SPOT FOR SPECIALIST EA #
##########################

# SPOT initially evaluates the utility of 10 hyperparameter settings (each on a separate run of optimization_
# specialist_v2.py), then builds a surrogate model to learn the relation between hyperparameters and utility,
# optimizes the surrogate model (= finds an optimal hyperparameter setting, based on the surrogate model),
# evaluates the model at the found optimum (on another run of the python script), refines the model, ...
# for a total of 20 evaluations

f <- wrapSystemCommand("python3 /Users/ricardoknauer/PycharmProjects/EC/optimization_specialist_v2.py")

# runSpotGUI()

muLo = 50
muHi = 200
mutLo = 0
mutHi = 0.1
crossLo = 0
crossHi = 1
kLo = 2
kHi = 200
doomLo = 0
doomHi = 1
lower = c(muLo, mutLo, crossLo, kLo, doomLo)
upper = c(muHi, mutHi, crossHi, kHi, doomHi)

spotConfig <- list(
    types = c("integer", "numeric", "numeric", "integer", "numeric"), # data type of hyperparameters
    noise = TRUE,                           # for stochstic algorithms
    seedFun = 1,
    design = designUniformRandom,           # uniform random initial design (hyperparameter setting)
    model = buildEnsembleStack,             # Bartz-Beielstein, 2016; Bartz-Beielstein & Zaefferer, 2017
    optimizer = optimLBFGSB,                # find optimal hyperparameter setting, based on the model
    optimizerControl = list(funEvals=1000), # 1000 model evaluations -> sampling efficiency!
    plots = TRUE
)

res3 <- spot(, fun = f, lower = lower, upper = upper, control = spotConfig)
str(res3)


##############################
# SPOT FOR GENERALIST CMA-ES #
##############################

# SPOT initially evaluates the utility of 10 hyperparameter settings (each on a separate run of optimization_
# generalist_v2.py), then builds a surrogate model to learn the relation between hyperparameters and utility,
# optimizes the surrogate model (= finds an optimal hyperparameter setting, based on the surrogate model),
# evaluates the model at the found optimum (on another run of the python script), refines the model, ...
# for a total of 30 evaluations

g <- wrapSystemCommand("python3 /Users/ricardoknauer/PycharmProjects/EC/optimization_generalist_v2.py")

# runSpotGUI()

lambdaLo = 50
lambdaHi = 150
muLo = 10
muHi = 15
lower = c(lambdaLo, muLo)
upper = c(lambdaHi, muHi)

spotConfig <- list(
    types = c("integer", "integer"),        # data type of hyperparameters
    funEvals = 30,                          # 30 utility tests
    noise = TRUE,                           # for stochstic algorithms
    seedFun = 1,
    design = designUniformRandom,           # uniform random initial design (hyperparameter setting)
    model = buildEnsembleStack,             # Bartz-Beielstein, 2016; Bartz-Beielstein & Zaefferer, 2017
    optimizer = optimLBFGSB,                # find optimal hyperparameter setting, based on the model
    optimizerControl = list(funEvals=1000), # 1000 model evaluations -> sampling efficieny!
    plots = TRUE
)

res <- spot(, fun = g, lower = lower, upper = upper, control = spotConfig)
str(res)


##########################
# SPOT FOR GENERALIST EA #
##########################

# SPOT initially evaluates the utility of 10 hyperparameter settings (each on a separate run of optimization_
# generalist.py), then builds a surrogate model to learn the relation between hyperparameters and utility,
# optimizes the surrogate model (= finds an optimal hyperparameter setting, based on the surrogate model),
# evaluates the model at the found optimum (on another run of the python script), refines the model, ...
# for a total of 30 evaluations

h <- wrapSystemCommand("python3 /Users/ricardoknauer/PycharmProjects/EC/optimization_generalist.py")

# runSpotGUI()

muLo = 50
muHi = 200
mutLo = 0
mutHi = 0.1
crossLo = 0
crossHi = 1
kLo = 2
kHi = 200
doomLo = 0
doomHi = 1
lower = c(muLo, mutLo, crossLo, kLo, doomLo)
upper = c(muHi, mutHi, crossHi, kHi, doomHi)

spotConfig <- list(
    types = c("integer", "numeric", "numeric", "integer", "numeric"), # data type of hyperparameters
    funEvals = 30,                          # 30 utility tests
    noise = TRUE,                           # for stochstic algorithms
    seedFun = 1,
    design = designUniformRandom,           # uniform random initial design (hyperparameter setting)
    model = buildEnsembleStack,             # Bartz-Beielstein, 2016; Bartz-Beielstein & Zaefferer, 2017
    optimizer = optimLBFGSB,                # find optimal hyperparameter setting, based on the model
    optimizerControl = list(funEvals=1000), # 1000 model evaluations -> sampling efficiency!
    plots = TRUE
)

res12345678 <- spot(, fun = h, lower = lower, upper = upper, control = spotConfig)
str(res12345678)
