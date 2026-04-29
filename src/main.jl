# 4/28/2026s
# Running DMRG simulation to obtain the ground-state wave function of spin-2 model on a 1D chain   

using ITensors, ITensorMPS
using LinearAlgebra
using HDF5
using Random
using Printf

include("spintwo.jl")
include("parameters.jl")
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
        N       = 10,
        J₁      = 1.0,
        J₂      = 0.0,
        cutoff  = 1e-8,
        nsweeps = 20,
        maxdim  = [20, 50, 200, 1000],
        seed    = 1234
    )
    @info "Simulation parameters" params.N params.J₁ params.J₂ params.cutoff params.nsweeps params.maxdim params.seed

    

    # ---------------------------------------------------------------------------
    # Initial state
    # ---------------------------------------------------------------------------
    Random.seed!(params.seed)
    sites = siteinds("S=2", params.N)
    ψ₀ = random_mps(sites, "+2"; linkdims=10)
    sz₀ = expect(ψ₀, "Sz"; sites=1:params.N)
    @info "Initial ⟨Sᶻ⟩ profile" sz₀


    
    # ---------------------------------------------------------------------------
    # Hamiltonian and DMRG
    # ---------------------------------------------------------------------------
    Hamiltonian = build_hamiltonian(params, sites)  
    

    # Perform DMRG simulation to obtain the ground state
    println("\nStarting DMRG simulation...\n")
    E, ψ = dmrg(Hamiltonian, ψ₀; 
                maxdim  = params.maxdim,
                cutoff  = params.cutoff,
                nsweeps = params.nsweeps
    )
    

    
    # ---------------------------------------------------------------------------
    # Observables
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
    # output_filename = "data/heisenberg_input_n$(N).h5"
    # h5open(output_filename, "w") do file
    #     write(file, "Psi", ψ)
    # end

    return
end