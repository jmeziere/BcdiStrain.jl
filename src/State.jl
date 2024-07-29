struct State
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

    function State(intens, gVecs, recSupports)
#        support = CUDA.ones(Bool, size(intens[1]))
#        for i in 1:length(intens)
#            invInt = CUFFT.ifft(CuArray{Float64, 3, CUDA.Mem.DeviceBuffer}(intens[i]))
#            support .= support .&& (abs.(invInt) .> 0.1 * maximum(abs.(invInt)))
#        end
        invInt = CUFFT.ifft(CuArray{Float64, 3, CUDA.Mem.DeviceBuffer}(intens[1]))
        support = abs.(invInt) .> 0.1 * maximum(abs.(invInt))

        s = size(support)
        rho = CUDA.zeros(Float64, s)
        ux = CUDA.zeros(Float64, s)
        uy = CUDA.zeros(Float64, s)
        uz = CUDA.zeros(Float64, s)

        State(intens, gVecs, recSupports, rho, ux, uy, uz, support)
    end

    function State(intens, gVecs, recSupports, rho, ux, uy, uz, support)
        s = size(intens[1])

        emptyCore = BcdiCore.TradState("L2", false, CUDA.zeros(Float64, s), CUDA.zeros(Float64, s), CUDA.zeros(Float64, s))
        traditionals = [BcdiTrad.State(intens[i], recSupports[i], support, emptyCore) for i in 1:length(intens)]
        currTrad = Ref(rand(1:length(intens)))

        xArr = CUDA.zeros(Float64, s)
        yArr = CUDA.zeros(Float64, s)
        zArr = CUDA.zeros(Float64, s)
        new(
            rho,ux,uy,uz,xArr,yArr,zArr,
            traditionals,currTrad,gVecs
        )
    end
end
