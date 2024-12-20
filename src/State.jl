"""
    State(intensities, gVecs, recSupport)
    State(intensities, gVecs, recSupport, support)

Create a reconstruction object. `intensities` is a vector of fully measured diffraction
peaks, `gVecs` is a vector of peak locations, and `recSupport` is a vector of masks over 
the intensities that removes those intenities from the reconstruction process.

The initialization process shifts each peak to be centered in the Fourier sense
(i.e. the center of mass of the peak is moved to the edge of the image, or the
zero frequency). If the support is not passed in, an initial guess of the support
is created by taking an IFFT of the intensities and including everything above
0.1 times the maximum value.
"""
struct State{T,I}
    rho::CuArray{Float64, 3, CUDA.Mem.DeviceBuffer}
    ux::CuArray{Float64, 3, CUDA.Mem.DeviceBuffer}
    uy::CuArray{Float64, 3, CUDA.Mem.DeviceBuffer}
    uz::CuArray{Float64, 3, CUDA.Mem.DeviceBuffer}
    xArr::CuArray{Float64, 3, CUDA.Mem.DeviceBuffer}
    yArr::CuArray{Float64, 3, CUDA.Mem.DeviceBuffer}
    zArr::CuArray{Float64, 3, CUDA.Mem.DeviceBuffer}
    traditionals::Vector{BcdiTrad.State}
    currTrad::Base.RefValue{Int64}
    gVecs::Vector{Vector{Float64}}
    rotations::T
    highStrain::Bool
    xPos::Vector{CuArray{Float64, 1, CUDA.Mem.DeviceBuffer}}
    yPos::Vector{CuArray{Float64, 1, CUDA.Mem.DeviceBuffer}}
    zPos::Vector{CuArray{Float64, 1, CUDA.Mem.DeviceBuffer}}
    keepInd::CuArray{CartesianIndex{3}, I, CUDA.Mem.DeviceBuffer}

    function State(intens, gVecs, recSupports; rotations=nothing, highStrain=false, truncRecSupport=true, loss="L2")
        s = size(intens[1])
        n = rotations == nothing && !highStrain ? size(intens[1]) : reduce(*, size(intens[1]))
        rho = CUDA.zeros(Float64, s)
        ux = CUDA.zeros(Float64, s)
        uy = CUDA.zeros(Float64, s)
        uz = CUDA.zeros(Float64, s)
        support = CUDA.zeros(Bool, n)

        emptyCore = BcdiCore.TradState(loss, false, CUDA.zeros(Float64, n), CUDA.zeros(Float64, s), CUDA.zeros(Float64, s))
        traditionals = [BcdiTrad.State(loss, intens[i], recSupports[i], support, emptyCore, truncRecSupport) for i in 1:length(intens)]
        currTrad = Ref(rand(1:length(intens)))

        xArr = CUDA.zeros(Float64, s)
        yArr = CUDA.zeros(Float64, s)
        zArr = CUDA.zeros(Float64, s)

        xPos = []
        yPos = []
        zPos = []
        if rotations != nothing || highStrain
            baseX = zeros(Float64, size(xArr))
            baseY = zeros(Float64, size(xArr))
            baseZ = zeros(Float64, size(xArr))
            for i in 1:size(xArr, 1)
                for j in 1:size(xArr, 2)
                    for k in 1:size(xArr, 3)
                        baseX[i,j,k] = 2*pi*(i-1)/size(xArr, 1)-pi
                        baseY[i,j,k] = 2*pi*(j-1)/size(xArr, 2)-pi
                        baseZ[i,j,k] = 2*pi*(k-1)/size(xArr, 3)-pi
                    end
                end
            end
            baseCuX = CuArray{Float64}(baseX)
            baseCuY = CuArray{Float64}(baseY)
            baseCuZ = CuArray{Float64}(baseZ)
            baseX = vec(baseX)
            baseY = vec(baseY)
            baseZ = vec(baseZ)

            cpuXPos = []
            cpuYPos = []
            cpuZPos = []
            if rotations != nothing
                deleteInd = []
                for i in 1:length(rotations)
                    rot = transpose(rotations[i])
                    x = rot[1,1] .* baseX .+ rot[1,2] .* baseY .+ rot[1,3] .* baseZ
                    y = rot[2,1] .* baseX .+ rot[2,2] .* baseY .+ rot[2,3] .* baseZ
                    z = rot[3,1] .* baseX .+ rot[3,2] .* baseY .+ rot[3,3] .* baseZ
                    
                    push!(cpuXPos, x)
                    push!(cpuYPos, y)
                    push!(cpuZPos, z)
                    append!(deleteInd, findall(x .> pi .|| x .< -pi .|| y .> pi .|| y .< -pi .|| z .> pi .|| z .< -pi))
                end
                deleteInd = sort(unique(deleteInd))
                keepInd = deleteat!(collect(vec(CartesianIndices(s))),deleteInd)
                for i in 1:length(rotations)
                    deleteat!(cpuXPos[i], deleteInd)
                    deleteat!(cpuYPos[i], deleteInd)
                    deleteat!(cpuZPos[i], deleteInd)
                    push!(xPos, CuArray{Float64}(cpuXPos[i]))
                    push!(yPos, CuArray{Float64}(cpuYPos[i]))
                    push!(zPos, CuArray{Float64}(cpuZPos[i]))
                end
                BcdiCore.setpts!(traditionals[currTrad[]].core, xPos[currTrad[]], yPos[currTrad[]], zPos[currTrad[]], true)
                resize!(traditionals[currTrad[]].support, length(xPos[1]))
                resize!(traditionals[currTrad[]].realSpace, length(xPos[1]))
            else
                baseX = CuArray{Float64}(baseX)
                baseY = CuArray{Float64}(baseY)
                baseZ = CuArray{Float64}(baseZ)
                BcdiCore.setpts!(traditionals[currTrad[]].core, baseX, baseY, baseZ, true)
                push!(xPos, baseX)
                push!(yPos, baseY)
                push!(zPos, baseZ)
                keepInd = collect(vec(CartesianIndices(s)))
            end
        else
            keepInd = collect(CartesianIndices(s))
        end
        BcdiTrad.initializeState(traditionals[currTrad[]].support, traditionals[currTrad[]].core)

        new{typeof(rotations),ndims(keepInd)}(
            rho,ux,uy,uz,xArr,yArr,zArr,
            traditionals,currTrad,gVecs,
            rotations, highStrain, xPos, yPos, zPos,
            keepInd
        )
    end
end
