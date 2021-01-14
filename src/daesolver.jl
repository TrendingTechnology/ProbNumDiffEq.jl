
abstract type GaussianDAEFilter <: OrdinaryDiffEq.DAEAlgorithm{0, true} end

Base.@kwdef struct DAE_EK1 <: GaussianDAEFilter
    prior::Symbol = :ibm
    order::Int = 1
    diffusionmodel::Symbol = :dynamic
    smooth::Bool = true
end


function OrdinaryDiffEq.alg_cache(
    alg::GaussianDAEFilter, du, u, res_prototype, rate_prototype, uEltypeNoUnits, uBottomEltypeNoUnits, tTypeNoUnits, uprev, uprev2, f, t, dt, reltol, p, calck, IIP)
    initialize_derivatives=true

    if !(u isa AbstractVector)
        error("Problems which are not scalar- or vector-valued (e.g. u0 is a scalar or a matrix) are currently not supported")
    end

    if (alg isa EK1 || alg isa IEKS) && isnothing(f.jac)
        error("""EK1 requires the Jacobian. To automatically generate it with ModelingToolkit.jl use ODEFilters.remake_prob_with_jac(prob).""")
    end

    q = alg.order
    u0 = u
    du0 = du
    t0 = t
    d = length(u)

    uType = typeof(u0)
    uElType = eltype(u0)
    matType = Matrix{uElType}

    # Projections
    Proj(deriv) = kron([i==(deriv+1) ? 1 : 0 for i in 1:q+1]', diagm(0 => ones(d)))
    SolProj = Proj(0)

    # Prior dynamics
    @assert alg.prior == :ibm "Only the ibm prior is implemented so far"
    Precond = preconditioner(d, q)
    A, Q = ibm(d, q, uElType)

    # Measurement model
    R = zeros(d, d)
    # Initial states
    m0, P0 = initialize_with_derivatives(u0, du0, f, p, t0, q)
    # @info "after init" m0 P0
    # @assert iszero(P0)
    @assert isdiag(P0)
    P0 = SRMatrix(sqrt.(P0))
    x0 = Gaussian(m0, P0)

    # Pre-allocate a bunch of matrices
    h = Proj(0) * x0.μ
    H = copy(Proj(0))
    du = copy(u0)
    ddu = zeros(uElType, d, d)
    v, S = copy(h), copy(ddu)
    measurement = Gaussian(v, S)
    K = copy(H')
    G = copy(Matrix(P0))
    covmatcache = copy(G)

    diffusion_models = Dict(
        :dynamic => DynamicDiffusion(),
        :dynamicMV => MVDynamicDiffusion(),
        :fixed => FixedDiffusion(),
        :fixedMV => MVFixedDiffusion(),
        :fixedMAP => MAPFixedDiffusion(),
    )
    diffmodel = diffusion_models[alg.diffusionmodel]
    initdiff = initial_diffusion(diffmodel, d, q, uEltypeNoUnits)

    return GaussianODEFilterCache{
        typeof(R), typeof(Proj), typeof(SolProj), typeof(Precond),
        uType, typeof(x0), typeof(A), typeof(Q), matType, typeof(initdiff),
        typeof(diffmodel), typeof(measurement), uEltypeNoUnits,
    }(
        # Constants
        d, q, A, Q, diffmodel, R, Proj, SolProj, Precond,
        # Mutable stuff
        copy(u0), copy(u0), copy(u0), copy(u0),
        copy(x0), copy(x0), copy(x0), copy(x0), copy(x0),
        measurement,
        H, du, ddu, K, G, covmatcache, initdiff,
        copy(u0),
        zero(uEltypeNoUnits),
    )
end



function DiffEqBase.initialize_dae!(
    integrator::OrdinaryDiffEq.ODEIntegrator{DAE_EK1})
end


function h!(integ::OrdinaryDiffEq.ODEIntegrator{DAE_EK1}, x_pred, t)
    @unpack f, p, dt = integ
    @unpack u_pred, du, Proj, Precond, measurement = integ.cache
    PI = inv(Precond(dt))
    z = measurement.μ
    E0, E1 = Proj(0), Proj(1)

    u_pred .= E0*PI*x_pred.μ
    du_pred = E1*PI*x_pred.μ

    @assert isinplace(integ.f)
    f(z, du_pred, u_pred, p, t)
    integ.destats.nf += 1

    return z
end

function H!(integ::OrdinaryDiffEq.ODEIntegrator{DAE_EK1}, x_pred, t)
    @unpack f, p, dt, alg = integ
    @unpack ddu, Proj, Precond, H, u_pred = integ.cache
    E0, E1 = Proj(0), Proj(1)
    PI = inv(Precond(dt))

    u_pred .= E0*PI*x_pred.μ
    du_pred = E1*PI*x_pred.μ

    @assert isinplace(integ.f)
    @assert !isnothing(integ.f.jac)

    Ju = copy(ddu)
    f.jac(Ju, du_pred, u_pred, p, 0.0, t)
    integ.destats.njacs += 1
    Jdu = copy(ddu)
    f.jac(Jdu, du_pred, u_pred, p, 1.0, t)
    integ.destats.njacs += 1
    Jdu -= Ju
    # @info "Jacobian of F wrt. u and du:" Ju Jdu
    H = (Jdu*E1 + Ju*E0) * PI

    return H
end
