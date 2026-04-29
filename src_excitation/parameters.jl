# 4/28/2026
# Parameters for spin-2 chain simulations.

"""
    SimulationParameters

    Parameters for the spin-2 J₁-J₂ chain DMRG simulation.

    # Fields
    - `N::Int`: number of sites.
    - `J₁::Float64`: nearest-neighbor coupling.
    - `J₂::Float64`: next-nearest-neighbor coupling.
    - `cutoff::Float64`: truncation cutoff for DMRG.
    - `nsweeps::Int`: number of DMRG sweeps.
    - `maxdim::Vector{Int}`: max bond dimension schedule across sweeps.
    - `seed::Int`: RNG seed for the initial state.
"""

Base.@kwdef struct SimulationParameters
    N::Int = 20
    J₁::Float64 = 1.0
    J₂::Float64 = 0.0
    cutoff::Float64 = 1e-8
    nsweeps::Int = 10
    maxdim::Vector{Int} = [20, 100, 500, 1000]
    seed::Int = 123456

    function SimulationParameters(N, J₁, J₂, cutoff, nsweeps, maxdim, seed)
        N ≥ 2 || throw(ArgumentError("N must be ≥ 2, got $N"))
        cutoff > 0 || throw(ArgumentError("cutoff must be > 0, got $cutoff"))
        nsweeps ≥ 1 || throw(ArgumentError("nsweeps must be ≥ 1, got $nsweeps"))
        all(>(0), maxdim) || throw(ArgumentError("maxdim entries must all be > 0"))
        return new(N, J₁, J₂, cutoff, nsweeps, maxdim, seed)
    end
end