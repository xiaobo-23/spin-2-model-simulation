# 4/28/2025
# Define the site types and operatos for the spin-2 model on a 1D chain


using ITensors, ITensorMPS


# Hilbert space dimension
ITensors.space(::SiteType"S=2") = 5


# Sz: diagonal with eigenvalues +2, +1, 0, -1, -2
function ITensors.op!(Op::ITensor, ::OpName"Sz", ::SiteType"S=2", s::Index)
    Op[s'=>1, s=>1] = +2.0
    Op[s'=>2, s=>2] = +1.0
    Op[s'=>3, s=>3] = 0.0
    Op[s'=>4, s=>4] = -1.0
    Op[s'=>5, s=>5] = -2.0
end



# S+: raises m -> m+1, with matrix elements sqrt(S(S+1) - m(m+1))
function ITensors.op!(Op::ITensor, ::OpName"S+", ::SiteType"S=2", s::Index)
    Op[s'=>1, s=>2] = 2.0               # |+2⟩⟨+1|
    Op[s'=>2, s=>3] = sqrt(6.0)         # |+1⟩⟨0|
    Op[s'=>3, s=>4] = sqrt(6.0)         # |0⟩⟨-1|
    Op[s'=>4, s=>5] = 2.0               # |-1⟩⟨-2|
end



# S-: lowers m -> m-1, with matrix elements sqrt(S(S+1) - m(m-1))
function ITensors.op!(Op::ITensor, ::OpName"S-", ::SiteType"S=2", s::Index)
    Op[s'=>2, s=>1] = 2.0               # |+1⟩⟨+2|
    Op[s'=>3, s=>2] = sqrt(6.0)         # |0⟩⟨+1|
    Op[s'=>4, s=>3] = sqrt(6.0)         # |-1⟩⟨0|
    Op[s'=>5, s=>4] = 2.0               # |-2⟩⟨-1|
end



# Sx = (S+ + S-) / 2, defined via composition
function ITensors.op!(Op::ITensor, ::OpName"Sx", ::SiteType"S=2", s::Index)
    sp = op("S+", s)
    sm = op("S-", s)
    Op .= 0.5 .* (sp .+ sm)
end



# Sy = (S+ - S-) / (2im), defined via composition
function ITensors.op!(Op::ITensor, ::OpName"Sy", ::SiteType"S=2", s::Index)
    sp = op("S+", s)
    sm = op("S-", s)
    Op .= -0.5im .* (sp .- sm)
end



# Sz² and S² (useful for single-ion aniostropy and total-spin checks)
function ITensors.op!(Op::ITensor, ::OpName"Sz2", ::SiteType"S=2", s::Index)
    Op[s'=>1, s=>1] = 4.0
    Op[s'=>2, s=>2] = 1.0
    Op[s'=>3, s=>3] = 0.0
    Op[s'=>4, s=>4] = 1.0
    Op[s'=>5, s=>5] = 4.0
end



# Identity operator
function ITensors.op!(Op::ITensor, ::OpName"Id", ::SiteType"S=2", s::Index)
    for idx in 1:5
        Op[s'=>idx, s=>idx] = 1.0   
    end
end



# Define states for product MPS construction
ITensors.state(::StateName"+2", ::SiteType"S=2") = [1, 0, 0, 0, 0]
ITensors.state(::StateName"+1", ::SiteType"S=2") = [0, 1, 0, 0, 0]
ITensors.state(::StateName"0",  ::SiteType"S=2") = [0, 0, 1, 0, 0]
ITensors.state(::StateName"-1", ::SiteType"S=2") = [0, 0, 0, 1, 0]
ITensors.state(::StateName"-2", ::SiteType"S=2") = [0, 0, 0, 0, 1]