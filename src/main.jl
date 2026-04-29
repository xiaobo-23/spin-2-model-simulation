# 4/28/2026
# Running DMRG simulation to obtain the ground-state wave function of spin-2 model on a 1D chain   

using ITensors, ITensorMPS
using LinearAlgebra
using HDF5
using Random, Printf

include("spintwo.jl")
include("parameters.jl")
include("initialization.jl")
include("hamiltonian.jl")


# ---------------------------------------------------------------------------
# Threading
# ---------------------------------------------------------------------------
BLAS.set_num_threads(8)
@info "BLAS configuration" vendor=BLAS.vendor() threads=BLAS.get_num_threads() julia_threads=Threads.nthreads()



let
    header = repeat("#", 70)
    println(header)
    println("\nRunning DMRG simulation to obtain the ground-state wave function of the spin-2 model on a 1D chain")
    println(header)

    
    # ---------------------------------------------------------------------------
    # Parameters
    # ---------------------------------------------------------------------------
    params = SimulationParameters(
        N       = 100,
        J₁      = 1.0,
        J₂      = 0.1799,
        Dxy     = 0.0161,
        Dz      = -0.0789,
        cutoff  = 1e-8,
        nsweeps = 20,
        maxdim  = [20, 50, 200, 1000],
        seed    = 1234
    )
    @info "Simulation parameters" params.N params.J₁ params.J₂ params.Dxy params.Dz params.cutoff params.nsweeps params.maxdim params.seed

    

    # ---------------------------------------------------------------------------
    # Set up the initial state as an MPS
    # ---------------------------------------------------------------------------
    sites, ψ₀ = initial_state(params; linkdims=10)

    
    # ---------------------------------------------------------------------------
    # Set up the Hamiltonian as an MPO
    # ---------------------------------------------------------------------------
    Hamiltonian = build_hamiltonian(params, sites)  
    

    # ---------------------------------------------------------------------------
    # Running DMRG to obtain the ground-state wave function
    # ---------------------------------------------------------------------------
    println("\nStarting DMRG simulation...\n")
    eigsolve_krylovdim = 50
    E, ψ = dmrg(Hamiltonian, ψ₀; 
                maxdim  = params.maxdim,
                cutoff  = params.cutoff,
                nsweeps = params.nsweeps,
                eigsolve_krylovdim = eigsolve_krylovdim = 50
    )
    
    
    # ---------------------------------------------------------------------------
    # Measure observables and check energy variance
    # ---------------------------------------------------------------------------
    sz  = expect(ψ, "Sz"; sites=1:params.N)
    czz  = correlation_matrix(ψ, "Sz", "Sz"; sites=1:params.N)
    
    # Check the variance of the energy
    H2 = inner(Hamiltonian, ψ, Hamiltonian, ψ)
    E₀ = inner(ψ', Hamiltonian, ψ)
    variance = H2 - E₀^2
    
    @printf("\nGround-state energy:        E = %.10f\n", E)
    @printf("Energy variance:    ⟨H²⟩ − ⟨H⟩² = %.3e\n", variance)
    @info "Final ⟨Sᶻ⟩ profile" sz       
    
   
    
    # ---------------------------------------------------------------------------
    # Save results
    # ---------------------------------------------------------------------------
    output_filename = "data/heisenberg_input_n$(params.N).h5"
    h5open(output_filename, "w") do file
        write(file, "Psi", ψ)
        write(file, "Sz", sz)
        write(file, "Czz", czz)
        write(file, "Energy", E)
        write(file, "EnergyVariance", variance)
    end

    return
end