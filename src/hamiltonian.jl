# 4/28/2025
# Set up the Hamiltonian for the spin-2 model on a 1D chain


using ITensors, ITensorMPS


# ---------------------------------------------------------------------------
# Hamiltonian construction
# ---------------------------------------------------------------------------
"""
    build_hamiltonian(p, sites) -> MPO

    Construct the spin-2 Hamiltonian::
    
        H = Σⱼ J₁ Sⱼ · Sⱼ₊₁  +  Σⱼ J₂ Sⱼ · Sⱼ₊₂

    where `S · S = SᶻSᶻ + ½(S⁺S⁻ + S⁻S⁺)`.
"""

function build_hamiltonian(parameters::SimulationParameters, sites)
    os = OpSum()

    # Nearest-neighbor with dimerization
    if !iszero(parameters.J₁)
        for j in 1:(parameters.N - 1)
            os += 0.5 * parameters.J₁, "S+", j, "S-", j + 1
            os += 0.5 * parameters.J₁, "S-", j, "S+", j + 1
            os +=       parameters.J₁, "Sz", j, "Sz", j + 1
        end
    end

    # # Next-nearest-neighbor
    # if !iszero(parameters.J₂)
    #     for j in 1:(parameters.N - 2)
    #         os += 0.5 * parameters.J₂, "S+", j, "S-", j + 2
    #         os += 0.5 * parameters.J₂, "S-", j, "S+", j + 2
    #         os +=       parameters.J₂, "Sz", j, "Sz", j + 2
    #     end
    # end

    return MPO(os, sites)
end