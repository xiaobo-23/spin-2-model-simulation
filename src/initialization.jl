# 4/28/2025
# Set up the Hamiltonian for the spin-2 model on a 1D chain


using ITensors, ITensorMPS


include("parameters.jl")


# ---------------------------------------------------------------------------
# Initialize the MPS
# ---------------------------------------------------------------------------

"""
    initial_state(params; linkdims = 10) -> (sites, ψ₀)

    Build the site indices and a random product-of-`|+2⟩`-like MPS as DMRG's
    starting point. Logs the initial ⟨Sᶻ⟩ profile.
"""

function initial_state(params::SimulationParameters; linkdims::Int = 10)
    Random.seed!(params.seed)
    sites = siteinds("S=2", params.N)
    ψ₀    = random_mps(sites, "+2"; linkdims = linkdims)

    sz₀ = expect(ψ₀, "Sz"; sites = 1:params.N)
    @info "Initial ⟨Sᶻ⟩ profile" sz₀
    
    return sites, ψ₀
end