abstract type Operator
end

struct OperatorList <: Operator
    ops::Vector{Operator}
end

function operate(ol::OperatorList, state::State)
    for op in ol.ops
        operate(op, state)
    end
end

"""
    ER()

Create an object that applies one iteration of Error Reduction (ER)
to the currently [`Mount`](@ref)ed peak. ER is an iterative projection 
algorithm that enforces two constraints, (1) the modulus constraint 
and (2) the support constraint:

1. When moved to reciprocal space, the reconstructed object must match the diffraction pattern.
2. The reconstructed object must fully lie within the support.

One iteration of ER first applies the modulus constraint, then the
support constraint to the object, then returnns.

Gradient descent is an alternate way to view the ER algorithm becausee
ER is equivalent to gradient descent with a step size of 0.5.

More information about the ER algorithm can be found in [Fienup1978,Marchesini2007](@cite).
"""
struct ER{T} <: Operator
    er::T
   
    function ER()
        er = BcdiTrad.ER()
        new{typeof(er)}(er)
    end
end

function operate(er::ER, state)
    BcdiTrad.operate(er.er, state.traditionals[state.currTrad[]])
end

"""
    HIO(beta)

Create an object that applies an iteration of hybrid input-output (HIO)
to the currently [`Mount`](@ref)ed peak. On the interior of the support, 
HIO is equivalent to applying the modulus constraint as described in the 
[`ER`](@ref) algorithm, and on the exterior of the support, HIO is equal 
to the current reconstruction minus a fraction of the output after applying 
the modulus constraint, that is,

```math
\\rho_{i+1} = \\begin{cases}
ER(\\rho_i) & \\rho \\in support \\\\
\\rho_i - \\beta * ER(\\rho_i) & \\rho \\notin support
\\end{cases}
```

Marchesini [Marchesini2007](@cite) has shown that the HIO algorithm is
equivalent to a mini-max problem.

More information about the HIO algorithm can be found in [Fienup1978,Marchesini2007](@cite).
"""
struct HIO{T} <: Operator
    hio::T

    function HIO(beta)
        hio = BcdiTrad.HIO(beta)
        new{typeof(hio)}(hio)
    end
end

function operate(hio::HIO, state)
    BcdiTrad.operate(hio.hio, state.traditionals[state.currTrad[]])
end

"""
    Shrink(threshold, sigma, state::State)

Create an object that applies one iteration of the shrinkwrap algorithm
to the current real space object. Shrinkwrap first applies a Gaussian 
blur to the current reconstruction using `sigma` as the width of the Gaussian. 
The support is then created from everything above the `threshold` times 
maximum value of the blurred object.

Further information about the shrinkwrap algorithm can be found in [Marchesini2003a](@cite).
"""
struct Shrink{T} <: Operator
    shrink::T

    function Shrink(threshold, sigma, state)
        shrink = BcdiTrad.Shrink(threshold, sigma, state.traditionals[state.currTrad[]])
        new{typeof(shrink)}(shrink)
    end
end

function operate(shrink::Shrink, state)
    BcdiTrad.operate(shrink.shrink, state.traditionals[state.currTrad[]])
end

"""
    Center(state)

Create an object that centers the current real space object.
The center of mass of the support is calculated and the object
is moved so the center of mass is centered in the Fourier transform
sense. In other words, the center of mass is moved to the zeroth
frequency, or the bottom left corner of the image.
"""
struct Center{T} <: Operator
    center::T

    function Center(state)
        center = BcdiTrad.Center(state.traditionals[state.currTrad[]])
        new{typeof(center)}(center)
    end
end

function operate(center::Center, state)
    BcdiTrad.operate(center.center, state.traditionals[state.currTrad[]])
end

function minDiffAngle(comp1::ComplexF64, angle2)
    if abs(comp1) < 1e-6
        return 0.0
    end
    return minDiffAngle(angle(comp1),angle2)
end

function minDiffAngle(angle1::Float64, angle2)
    diff = angle1-angle2
    if abs(diff) > abs(diff-2*pi)
        return diff-2*pi
    elseif abs(diff) > abs(diff+2*pi)
        return diff+2*pi
    else
        return diff
    end
end

"""
    Mount(beta, state, primitiveRecipLattice)

Create an object that mounts a new peak. The current real space
object is projected back to update the magnitude of the electron
density and the displacement field. A new peak is selected at
random and the current solution is projected out to this peak.

The paper that describes this algorithm is currently in submission.
"""
struct Mount <: Operator
    beta::Float64
    xArr::CuArray{Float64, 3, CUDA.Mem.DeviceBuffer}
    AInv::CuArray{Float64, 2, CUDA.Mem.DeviceBuffer}

    function Mount(beta, state, primitiveRecipLattice)
        xArr = CUDA.zeros(Float64, size(state.rho))
        new(beta, xArr, primitiveRecipLattice)
    end
end

function operate(mount::Mount, state::State)
    beta = mount.beta
    tct = state.traditionals[state.currTrad[]]
    currG = state.gVecs[state.currTrad[]]
    xArr = mount.xArr

    @views rspMul = 
        mapreduce(
            (re, rh, sup)-> sup ? abs(re) * rh : 0.0, +, 
            tct.realSpace, state.rho, 
            tct.support
        ) / 
        mapreduce(
            (re, sup)-> sup ? abs2(re) : 0.0, +, 
            tct.realSpace, 
            tct.support
        )
    if reduce(+, state.rho) < 1e-6
        beta = 1.0
        rspMul = 1.0
    end

    tct.realSpace .*= tct.support .* rspMul
    state.rho .*= (1.0 .- beta) .* tct.support
    state.rho .+= beta .* abs.(tct.realSpace)
    state.ux .*= tct.support
    state.uy .*= tct.support
    state.uz .*= tct.support

    
    xArr .= beta .* minDiffAngle.(
        tct.realSpace,
        currG[1] .* state.ux .+ currG[2] .* state.uy .+ currG[3] .* state.uz
    ) ./ dot(currG, currG)
    @views xArr .-= median(xArr[abs.(tct.realSpace) .> 1e-6])
    xArr .*= abs.(tct.realSpace) .> 1e-6

    state.ux .+= xArr .* currG[1]
    state.uy .+= xArr .* currG[2]
    state.uz .+= xArr .* currG[3]

    allU = CUDA.zeros(Float64, 3, reduce(+, tct.support))
    allU[1,:] .= state.ux[tct.support]
    allU[2,:] .= state.uy[tct.support]
    allU[3,:] .= state.uz[tct.support]

    B = floor.(Int64, (mount.AInv * allU) ./ (2 .* pi) .+ 0.5)
    allU .-= 2 .* pi .* (mount.AInv \ B)
    state.ux[tct.support] .= allU[1,:]
    state.uy[tct.support] .= allU[2,:]
    state.uz[tct.support] .= allU[3,:]
    
    state.currTrad[] = rand(1:length(state.traditionals))
    tct = state.traditionals[state.currTrad[]]
    currG = state.gVecs[state.currTrad[]]

    tct.realSpace .= state.rho .* exp.(1im .* (
         currG[1] .* state.ux .+ currG[2] .* state.uy .+ currG[3] .* state.uz
    ))
    tct.core.plan * tct.realSpace

    tct.realSpace .*= 
        mapreduce(
            (i,rsp,sup) -> sup ? sqrt(i) * abs(rsp) : 0.0, +, 
            tct.core.intens, tct.core.plan.recipSpace, tct.core.recSupport, init=0.0
        ) /
        mapreduce(
            (rsp,sup) -> sup ? abs2(rsp) : 0.0, +, 
            tct.core.plan.recipSpace, tct.core.recSupport, init=0.0
        )
end

function Base.:*(operator::Operator, state::State)
    operate(operator, state)
    return state
end

function Base.:*(operator1::Operator, operator2::Operator)
    return OperatorList([operator2, operator1])
end

function Base.:^(operator::Operator, pow::Int)
    return OperatorList([operator for i in 1:pow])
end
