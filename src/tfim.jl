# 4/28/2026
# DMRG for the 1D transverse-field Ising model (TFIM): H = J Σ ZⱼZⱼ₊₁ + h Σ Xⱼ.
# Computes the ground state, the first excited state, and the energy gap.

using ITensors, ITensorMPS
using LinearAlgebra
using HDF5
using Random, Printf

include("spintwo.jl")
include("parameters.jl")
include("initialization.jl")
include("hamiltonian.jl")



# -------- Setting up BLAS threading configuration for optimal performance ----------------------------------------
BLAS.set_num_threads(8)
@info "BLAS configuration" vendor=BLAS.vendor() threads=BLAS.get_num_threads() julia_threads=Threads.nthreads()



# -------- Setting up parameters used in the simulation ----------------------------------------------------------- 
const N = 20
const J = 1.0
const h = 1.5
const random_seed = 1234



# -------- Compute the first excited state and the energy gap (penalty method) -----------------------------------
"""
    compute_gap(Hamiltonian, ψ_ground, E_ground, sites; weight, nsweeps, maxdim,
                cutoff, eigsolve_krylovdim, linkdims) -> (gap, E₁, ψ₁, overlap)

Compute the first excited state of `Hamiltonian` and the energy gap above the
ground state `ψ_ground` (whose energy is `E_ground`).

DMRG is run on the modified problem `H + weight · |ψ_ground⟩⟨ψ_ground|`: the
penalty term raises the energy of any component along the ground state, so DMRG
relaxes to the lowest state orthogonal to `ψ_ground`. `weight` must exceed the
gap; a few times the relevant energy scale is a safe choice.

Returns the gap `E₁ − E_ground`, the excited-state energy `E₁ = ⟨ψ₁|H|ψ₁⟩`
(recomputed without the penalty term), the excited-state MPS `ψ₁`, and the
residual overlap `⟨ψ_ground|ψ₁⟩`, which should be ≈ 0 for a trustworthy result.
"""
function compute_gap(Hamiltonian, ψ_ground, E_ground, sites;
                     weight::Real             = 10.0,
                     nsweeps::Int             = 10,
                     maxdim::Vector{Int}      = [20, 50, 200, 1000],
                     cutoff::Real             = 1e-8,
                     eigsolve_krylovdim::Int  = 50,
                     linkdims::Int            = 10)
    ψ₁_init = random_mps(sites; linkdims = linkdims)

    _, ψ₁ = dmrg(Hamiltonian, [ψ_ground], ψ₁_init;
                 weight             = weight,
                 nsweeps            = nsweeps,
                 maxdim             = maxdim,
                 cutoff             = cutoff,
                 eigsolve_krylovdim = eigsolve_krylovdim)

    E₁      = real(inner(ψ₁', Hamiltonian, ψ₁))   # true ⟨H⟩, excludes the penalty term
    gap     = E₁ - E_ground
    overlap = inner(ψ_ground, ψ₁)
    return gap, E₁, ψ₁, overlap
end



let
    header = repeat("-", 150)
    println(header)
    println("Running DMRG for the transverse-field Ising model: ground state, first excited state, and energy gap")
    println(header)


    # ------- Set up the initial state as an MPS -------------------------------------------------------------------------
    Random.seed!(random_seed)
    sites = siteinds("S=1/2", N)
    ψ₀    = random_mps(sites; linkdims = 2)

    
    # ------- Set up the Hamiltonian as an MPO ---------------------------------------------------------------------------
    os = OpSum()
    for j in 1 : N - 1
        os .+= J, "Z", j, "Z", j + 1
    end

    for j in 1 : N
        os .+= h, "X", j
    end

    Hamiltonian = MPO(os, sites)
    

    
    # -------  Running DMRG to obtain the ground-state wave function -----------------------------------------------------
    println("\nRunning DMRG simulation to obtain the ground-state wave function...")
    eigsolve_krylovdim = 50
    cutoff  = 1e-8
    nsweeps = 10
    maxdim  = [20, 50, 200, 1000]
    seed    = 1234
    E, ψ = dmrg(Hamiltonian, ψ₀; 
                maxdim  = maxdim,
                cutoff  = cutoff,
                nsweeps = nsweeps,
                eigsolve_krylovdim = 50,
    )
    
    

    # ------- Measure observables and check the energy variance ----------------------------------------------------------
    sz  = expect(ψ, "Sz"; sites=1:N)
    czz  = correlation_matrix(ψ, "Sz", "Sz"; sites=1:N)
    

    # Check the variance of the energy
    H2 = inner(Hamiltonian, ψ, Hamiltonian, ψ)
    E₀ = inner(ψ', Hamiltonian, ψ)
    variance = H2 - E₀^2
    
    @printf("\nGround-state energy: E = %.10f\n", E)
    @printf("Energy variance: ⟨H²⟩ − ⟨H⟩² = %.3e\n", variance)



    # ------- First excited state and energy gap -------------------------------------------------------------------------
    println("\nRunning DMRG simulation to obtain the first excited state...")
    weight = 20
    gap, E₁, ψ₁, overlap = compute_gap(Hamiltonian, ψ, E, sites;
                                       weight             = weight,
                                       nsweeps            = nsweeps,
                                       maxdim             = maxdim,
                                       cutoff             = cutoff,
                                       eigsolve_krylovdim = eigsolve_krylovdim)

    exact_gap = 2 * abs(J - h)     # TFIM thermodynamic-limit gap (Pauli convention); exact for h ≥ J

    @printf("\nFirst excited-state energy: E₁ = %.10f\n", E₁)
    @printf("Energy gap:                 ΔE = %.10f\n", gap)
    @printf("Exact gap (N → ∞):     2|J − h| = %.10f\n", exact_gap)
    @printf("Overlap |⟨ψ₀|ψ₁⟩| (≈ 0 if clean): %.3e\n", abs(overlap))



    # ------- Save results -----------------------------------------------------------------------------------------------
    # output_filename = "data/heisenberg_input_n$(params.N)_J2$(params.J₂).h5"
    # h5open(output_filename, "w") do file
    #     write(file, "Psi", ψ)
    #     write(file, "Sz", sz)
    #     write(file, "Czz", czz)
    #     write(file, "Energy", E)
    #     write(file, "EnergyVariance", variance)
    # end

    return
end