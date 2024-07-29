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
