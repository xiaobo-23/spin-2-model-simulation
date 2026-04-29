# 4/28/2026
# Running DMRG simulation to obtain the gap of spin-1 Heisenberg chain. 

using ITensors, ITensorMPS
using LinearAlgebra
using HDF5
using Random, Printf


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
    println("\nRunning DMRG simulation to obtain the ground-state wave function of the spin-1 Heisenberg chain")
    println(header)

    
    # ---------------------------------------------------------------------------
    # Parameters
    # ---------------------------------------------------------------------------
    params = SimulationParameters(
        N       = 50,
        J₁      = 1.0,
        J₂      = 0.0,
        cutoff  = 1e-10,
        nsweeps = 15,
        maxdim  = [20, 50, 200, 1000],
        seed    = 123456
    )
    @info "Simulation parameters" params.N params.J₁ params.J₂ params.cutoff params.nsweeps params.maxdim params.seed

    
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
                eigsolve_krylovdim = eigsolve_krylovdim
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
    # Obtain the wave function of the first excited state using DMRG targeting
    # ---------------------------------------------------------------------------
    weight = 50.0
    ψ1_init = random_mps(sites, n -> isodd(n) ? "Up" : "Dn"; linkdims = 10)

    E1_penalty, ψ1 = dmrg(
        Hamiltonian,
        [ψ],
        ψ1_init;
        weight = weight,
        maxdim = params.maxdim,
        cutoff = params.cutoff,
        nsweeps = params.nsweeps, 
        eigsolve_krylovdim = eigsolve_krylovdim
    )


    # ---------------------------------------------------------------------------
    # Compute the energy gap and check the overlap between the ground and first excited states  
    # ---------------------------------------------------------------------------
    E₁ = inner(ψ1', Hamiltonian, ψ1)
    gap = E₁ - E
    overlap = inner(ψ, ψ1)
    @printf("First excited-state energy: E₁ = %.10f\n", E₁)
    @printf("Energy gap:                 ΔE = %.10f\n", gap)
    @printf("Overlap between ground and first excited states: %.3e\n", overlap)

    



    # # ---------------------------------------------------------------------------
    # # Save results
    # # ---------------------------------------------------------------------------
    # output_filename = "data/spin-1_heisenberg_n$(params.N).h5"
    # h5open(output_filename, "w") do file
    #     write(file, "E0", E)
    #     write(file, "Psi", ψ)
    # end

    return
end