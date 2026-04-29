# 4/28/2025
# Set an exact diagonalization code for the spin-2 model on a 1D chain, to benchmark against DMRG results


using LinearAlgebra
using SparseArrays
using KrylovKit


include("parameters.jl")



"""
    exact_diagonalization_sparse(params; sz_sector = 0) -> (E0, ψ0_vec, basis)

    Diagonalize H restricted to the total Sᶻ = `sz_sector` sector using sparse
    matrices and Lanczos. Returns the ground-state energy, the eigenvector in
    the restricted basis, and the basis itself (as integer-encoded states) so
    you can interpret the vector if needed.
"""

function exact_diagonalization_sparse(params::SimulationParameters;
                                       sz_sector::Int = 0)
    N = params.N
    d = 5                               # local Hilbert space dimension
    # Local Sᶻ values for S=2: state index 1..5 → m = +2,+1,0,-1,-2
    sz_vals = [+2, +1, 0, -1, -2]

    # 1. Enumerate all product states with total Sᶻ = sz_sector
    basis = Int[]                       # base-5 integer encoding of each state
    state_to_idx = Dict{Int, Int}()
    for s in 0:(d^N - 1)
        digits_vec = digits(s, base = d, pad = N)
        total_sz = sum(sz_vals[k + 1] for k in digits_vec)
        if total_sz == sz_sector
            push!(basis, s)
            state_to_idx[s] = length(basis)
        end
    end
    dim = length(basis)
    @info "Sector dim" sz_sector dim total = d^N

    # 2. Build sparse H by acting with each Hamiltonian term on each basis state
    rows = Int[]; cols = Int[]; vals = ComplexF64[]

    # Helper: get local state at site `j` (1-indexed), and set it
    get_local(s, j)         = (s ÷ d^(j-1)) % d           # 0-based result
    set_local(s, j, new_v)  = s + (new_v - get_local(s, j)) * d^(j-1)

    # S=2 matrix elements (in basis 1..5 = +2,+1,0,-1,-2, here using 0-based 0..4)
    # S+|m⟩ → coefficient and new state index
    sp_action = Dict(1 => (2.0,    0),     # m=+1 (idx 1 in 0-based) → m=+2 (idx 0)
                     2 => (sqrt(6),1),     # m=0  → m=+1
                     3 => (sqrt(6),2),     # m=-1 → m=0
                     4 => (2.0,    3))     # m=-2 → m=-1
    sm_action = Dict(0 => (2.0,    1),
                     1 => (sqrt(6),2),
                     2 => (sqrt(6),3),
                     3 => (2.0,    4))
    # Sᶻ acts diagonally; values from sz_vals indexed 0..4

    function add_term!(coeff::Real, ops::Vector{Tuple{Symbol,Int}})
        for (col_idx, s) in enumerate(basis)
            # Apply ops in sequence: returns (amplitude, new_state) or nothing
            amp = complex(coeff)
            cur = s
            ok  = true
            for (op_name, site) in ops
                local_v = get_local(cur, site)
                if op_name === :Sz
                    amp *= sz_vals[local_v + 1]
                elseif op_name === :Sp
                    haskey(sp_action, local_v) || (ok = false; break)
                    c, new_v = sp_action[local_v]
                    amp *= c
                    cur = set_local(cur, site, new_v)
                elseif op_name === :Sm
                    haskey(sm_action, local_v) || (ok = false; break)
                    c, new_v = sm_action[local_v]
                    amp *= c
                    cur = set_local(cur, site, new_v)
                else
                    error("unknown op $op_name")
                end
            end
            ok || continue
            row_idx = get(state_to_idx, cur, 0)
            row_idx == 0 && continue        # outside sector — shouldn't happen
            push!(rows, row_idx)
            push!(cols, col_idx)
            push!(vals, amp)
        end
    end

    # 3. Add all Hamiltonian terms (matching build_hamiltonian)
    for j in 1:(N-1)
        add_term!(0.5 * params.J₁, [(:Sp, j), (:Sm, j+1)])
        add_term!(0.5 * params.J₁, [(:Sm, j), (:Sp, j+1)])
        add_term!(      params.J₁, [(:Sz, j), (:Sz, j+1)])
    end
    if !iszero(params.J₂)
        for j in 1:(N-2)
            add_term!(0.5 * params.J₂, [(:Sp, j), (:Sm, j+2)])
            add_term!(0.5 * params.J₂, [(:Sm, j), (:Sp, j+2)])
            add_term!(      params.J₂, [(:Sz, j), (:Sz, j+2)])
        end
    end

    H = sparse(rows, cols, vals, dim, dim)
    @info "Sparse H built" nnz=nnz(H) hermitian=ishermitian(H)

    # 4. Lanczos for the ground state
    v0 = randn(ComplexF64, dim); v0 ./= norm(v0)
    vals_lanczos, vecs_lanczos, info = eigsolve(H, v0, 1, :SR;
                                                 ishermitian = true,
                                                 tol = 1e-12,
                                                 maxiter = 500)
    info.converged ≥ 1 || @warn "Lanczos did not fully converge" info

    return real(vals_lanczos[1]), vecs_lanczos[1], basis
end


let 
    params = SimulationParameters(
        N       = 10,
        J₁      = 1.0,
        J₂      = 0.0,
        cutoff  = 1e-12,
        nsweeps = 30,
        maxdim  = [50, 100, 200, 400, 600],
        seed    = 1234,
    )

    # ED
    println("\nRunning ED in Sᶻ_total = 0 sector...\n")
    E_ed, _, _ = exact_diagonalization_sparse(params; sz_sector = 0)


    @printf("\n%s\n", repeat("=", 70))
    @printf("ED   ground-state energy:  %.10f\n", E_ed)
    @printf("%s\n", repeat("=", 70))
end