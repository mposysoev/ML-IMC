using Dates
using Statistics
using LinearAlgebra
using Distributed
using Chemfiles
using BSON: @save, @load

BLAS.set_num_threads(1)

@everywhere begin
    include("src/distances.jl")
    include("src/network.jl")
    include("src/base.jl")
    include("src/io.jl")
end

function main()
    # Start the timer
    startTime = Dates.now()
    println("Starting at: ", startTime)

    # Initialize the parameters
    globalParms, MCParms, NNParms, systemParmsList = parametersInit()

    # Check if the number of workers is divisble by the number of ref systems
    @assert nworkers() % length(systemParmsList) == 0

    # Initialize the input data
    inputs = inputInit(globalParms, NNParms, systemParmsList)
    if globalParms.mode == "training"
        model, opt, refRDFs = inputs
    else
        model = inputs
    end

    println("Running MC simulation on $(nworkers()) rank(s)...\n")
    println("Total number of steps: $(MCParms.steps * nworkers() / 1E6)M")
    println("Number of equilibration steps per rank: $(MCParms.Eqsteps / 1E6)M")

    if globalParms.mode == "training"
        nsystems = length(systemParmsList)
        println("Training a model using $(nsystems) reference system(s)")
        println("Using $(NNParms.activation) activation")
        println("Number of iterations: $(NNParms.iters)")
        println("Optimizer type: $(NNParms.optimizer)")
        println("Learning rate: $(NNParms.rate)")
        if NNParms.optimizer == "Momentum" #TODO: not only momentum, but many other optimizers
            println("Momentum coefficient: $(NNParms.momentum)")
        end
        
        # Run the training
        train!(globalParms, MCParms, NNParms, systemParmsList, model, opt, refRDFs)
    else
        @assert length(systemParmsList) == 1
        println("Running simulation with a trained model")
        # Run the simulation
        simulate!(model, globalParms, MCParms, NNParms, systemParmsList[1])
    end

    # Stop the timer
    stopTime = Dates.now()
    wallTime = Dates.canonicalize(stopTime - startTime)
    println("Stopping at: ", stopTime, "\n")
    println("Walltime: ", wallTime)
end

"""
Run the main() function
"""

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end