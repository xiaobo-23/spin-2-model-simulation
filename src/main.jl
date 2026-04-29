# 4/28/2026s
# Running DMRG simulation to obtain the ground-state wave function of spin-2 model on a 1D chain   

using ITensors, ITensorMPS
using LinearAlgebra
using HDF5
using Random
using Printf

include("spintwo.jl")
include("hamiltonian.jl")


# ---------------------------------------------------------------------------
# Threading
# ---------------------------------------------------------------------------
BLAS.set_num_threads(8)
@info "BLAS configuration" vendor=BLAS.vendor() threads=BLAS.get_num_threads() julia_threads=Threads.nthreads()



# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------
"""
    SimulationParameters

    Parameters for the spin-2 J₁–J₂ chain DMRG simulation.

    # Fields
    - `N::Int`           : number of sites.
    - `J₁::Float64`      : nearest-neighbor coupling.
    - `J₂::Float64`      : next-nearest-neighbor coupling.
    - `cutoff::Float64`  : truncation cutoff for DMRG.
    - `nsweeps::Int`     : number of DMRG sweeps.
    - `maxdim::Vector{Int}` : max bond dimension schedule across sweeps.
    - `seed::Int`        : RNG seed for the initial state.
"""

Base.@kwdef struct SimulationParameters
    N::Int                = 10
    J₁::Float64           = 1.0
    J₂::Float64           = 0.2
    cutoff::Float64       = 1e-8
    nsweeps::Int          = 10
    maxdim::Vector{Int}   = [20, 100, 500, 1000]
    seed::Int             = 1234

    function SimulationParameters(N, J₁, J₂, cutoff, nsweeps, maxdim, seed)
        N ≥ 2          || throw(ArgumentError("N must be ≥ 2, got $N"))
        cutoff > 0     || throw(ArgumentError("cutoff must be > 0, got $cutoff"))
        nsweeps ≥ 1    || throw(ArgumentError("nsweeps must be ≥ 1, got $nsweeps"))
        all(>(0), maxdim) || throw(ArgumentError("maxdim entries must all be > 0"))
        return new(N, J₁, J₂, cutoff, nsweeps, maxdim, seed)
    end
end



let
    header = repeat("#", 200)
    println(header)
    println(header)
    println("\nStarting DMRG simulation to obtain the ground-state wave function of the spin-2 model on a 1D chain \n")
    println("Parameters used in the simulation...")
    
    params = SimulationParameters(
        N       = 10,
        J₁      = 1.0,
        J₂      = 0.0,
        cutoff  = 1e-8,
        nsweeps = 20,
        maxdim  = [20, 50, 200, 1000],
        seed    = 1234
    )

    
    # Initialize a random MPS as the starting point of DMRG simulation
    Random.seed!(params.seed)
    sites = siteinds("S=2", params.N)
    ψ₀ = random_mps(sites, "+2"; linkdims=10)
    sz₀ = expect(ψ₀, "Sz"; sites=1:params.N)
    
    println("\nInitial magnetization profile <Sz> at each site:")
    @show sz₀


    # Set up the Hamiltonian as an MPO
    Hamiltonian = build_hamiltonian(params, sites)  
    

    # Perform DMRG simulation to obtain the ground state
    println("\nStarting DMRG simulation...")
    E, ψ = dmrg(Hamiltonian, ψ₀; 
                maxdim  = params.maxdim,
                cutoff  = params.cutoff,
                nsweeps = params.nsweeps
    )
    println("")


    sz  = expect(ψ, "Sz"; sites=1:params.N)
    czz  = correlation_matrix(ψ, "Sz", "Sz"; sites=1:params.N)
    println("\nFinal magnetization profile <Sz> at each site after DMRG:")
    @show sz
    

    # Check the variance of the energy
    H2 = inner(Hamiltonian, ψ, Hamiltonian, ψ)
    E₀ = inner(ψ', Hamiltonian, ψ)
    variance = H2 - E₀^2
    println("\nVariance of the energy is $variance")
    println(header)
    println(header)
    

    # output_filename = "data/heisenberg_input_n$(N).h5"
    # h5open(output_filename, "w") do file
    #     write(file, "Psi", ψ)
    # end

    return
end