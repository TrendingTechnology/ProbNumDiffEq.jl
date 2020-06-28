########################################################################################
# Structure following https://github.com/SciML/SimpleDiffEq.jl/tree/master/src/rk4
########################################################################################


########################################################################################
# Algorithm
########################################################################################
abstract type AbstractODEFilter <: DiffEqBase.AbstractODEAlgorithm end
mutable struct ODEFilter <: AbstractODEFilter
end
export ODEFilter


########################################################################################
# Integrator
########################################################################################
mutable struct ODEFilterIntegrator{IIP, S, X, T, P, F} <: DiffEqBase.AbstractODEIntegrator{ODEFilter, IIP, S, T}
    f::F                  # eom
    u::S                  # current functionvalue
    x::X                  # current state
    xprev::X              # previous state
    tmp::X                # dummy, same as state
    tprev::T              # previous time
    t::T                  # current time
    t0::T                 # initial time, only for reinit
    dt::T                 # step size
    tdir::T
    p::P                  # parameter container
    x_modified::Bool
    # ks::Vector{S}         # interpolants of the algorithm
    # cs::SVector{6, T}     # ci factors cache: time coefficients
    # as::SVector{21, T}    # aij factors cache: solution coefficients
    # rs::SVector{22, T}    # rij factors cache: interpolation coefficients

    # My additions
    d::Int
    q::Int
    dm
    mm
    sigma_estimator
    preconditioner
    steprule
end
DiffEqBase.isinplace(::ODEFilterIntegrator{IIP}) where {IIP} = IIP


########################################################################################
# Initialization
########################################################################################
function odefilter_init(f::F, IIP::Bool, u0::S, t0::T, dt::T, p::P, q::Int, sigmarule, steprule, abstol, reltol, ρ, prob_kwargs) where {F, P, T, S<:AbstractArray{T}}
    d = length(u0)
    dm = ibm(q, d)
    # if method == :ekf0
    #     mm = ekf0_measurement_model(d, q, ivp)
    # elseif method == :ekf1
    #     mm = ekf1_measurement_model(d, q, ivp)
    # else
    #     throw(Error("method argument not in [:ekf0, :ekf1]"))
    # end
    mm = ekf1_measurement_model(d, q, f, p, prob_kwargs)
    # mm = ekf0_measurement_model(d, q, f, p)

    initialize_derivatives = false
    initialize_derivatives = initialize_derivatives == :auto ? q <= 3 : false
    if initialize_derivatives
        derivatives = get_derivatives((x, t) -> f(x, p, t), d, q)
        m0 = vcat(u0, [_f(u0, t0) for _f in derivatives]...)
    else
        m0 = [u0; f(u0, p, t0); zeros(d*(q-1))]
    end
    P0 = diagm(0 => [zeros(d); ones(d*q)] .+ 1e-16)
    x0 = Gaussian(m0, P0)
    X = typeof(x0)

    precond = preconditioner(dt, d, q)
    apply_preconditioner!(precond, x0)

    steprules = Dict(
        :constant => constant_steprule(),
        :pvalue => pvalue_steprule(0.05),
        :baseline => classic_steprule(abstol, reltol; ρ=ρ),
        :measurement_error => measurement_error_steprule(;abstol=abstol, reltol=reltol, ρ=ρ),
        :measurement_scaling => measurement_scaling_steprule(),
        :schober16 => schober16_steprule(;ρ=ρ, abstol=abstol, reltol=reltol),
    )
    steprule = steprules[steprule]

    return ODEFilterIntegrator{IIP, S, X, T, P, F}(
        f, u0, _copy(x0), _copy(x0), _copy(x0), t0, t0, t0, dt, sign(dt), p, true,
        d, q, dm, mm, sigmarule, precond, steprule
    )
end


########################################################################################
# Solve
########################################################################################
function DiffEqBase.__solve(prob::DiffEqBase.AbstractODEProblem, alg::ODEFilter;
                            dt=0.1,
                            q=1,
                            saveat=nothing,
                            save_everystep=true,
                            abstol=1e-6, reltol=1e-3, ρ=0.95,
                            sigmarule=Schober16Sigma(),
                            steprule=:baseline,
                            progressbar=false,
                            maxiters=1e5,
                            smoothed=true,
                            kwargs...)
    # Init
    IIP = DiffEqBase.isinplace(prob)
    f = IIP ? IIP_to_OOP(prob.f) : prob.f
    integ = odefilter_init(f, false, prob.u0, prob.tspan[1], dt, prob.p, q, sigmarule, steprule, abstol, reltol, ρ, prob.kwargs)

    # More Initialization
    t_0, T = prob.tspan
    sol = StructArray([StateBelief(integ.t, integ.x)])
    proposals = []
    retcode = :Default

    # Solve
    if progressbar pbar_update, pbar_close = make_progressbar(0.1) end
    iter = 0
    while integ.t < T
        if progressbar pbar_update(fraction=(integ.t-t_0)/(T-t_0)) end

        # Here happens the main "work"
        proposal = predict_update(integ)

        accept, dt_proposal = integ.steprule(integ, proposal, proposals)
        push!(proposals, (proposal..., accept=accept, dt=integ.dt))
        integ.dt = min(dt_proposal, T-integ.t)

        if accept
            push!(sol, StateBelief(proposal.t, proposal.filter_estimate))
            integ.x = proposal.filter_estimate
            integ.t = proposal.t
        end

        iter += 1
        if iter >= maxiters
            break
            retcode = :MaxIters
        end
    end
    if progressbar pbar_close() end

    smoothed && smooth!(sol, proposals, integ)
    calibrate!(sol, proposals, integ)
    undo_preconditioner!(sol, proposals, integ)

    # Format Solution
    sol = DiffEqBase.build_solution(prob, alg, sol.t, StructArray(sol.x))

    return sol
end


########################################################################################
# Step
########################################################################################
function DiffEqBase.step!(integ::ODEFilterIntegrator{false, S, X, T}) where {S, X, T}

    # proposal = predict_update(solver, cache)
    # accept, dt_proposal = steprule(solver, cache, proposal, proposals)
    # push!(proposals, (proposal..., accept=accept, dt=cache.dt))
    # cache.dt = min(dt_proposal, T-cache.t)

    # if accept
    #     push!(sol, StateBelief(proposal.t, proposal.filter_estimate))
    #     cache.x = proposal.filter_estimate
    #     cache.t = proposal.t
    # end

    # iter += 1
    # if iter >= maxiters
    #     break
    #     retcode = :MaxIters
    # end
end


########################################################################################
# Solution handling
########################################################################################
abstract type AbstractProbODESolution{T,N,S} <: DiffEqBase.AbstractODESolution{T,N,S} end
struct ProbODESolution{T,N,uType,xType,tType,P,A,IType} <: AbstractProbODESolution{T,N,uType}
    u::uType
    x::xType
    t::tType
    prob::P
    alg::A
    dense::Bool
    interp::IType
    retcode::Symbol
end

function DiffEqBase.build_solution(
    prob::DiffEqBase.AbstractODEProblem,
    alg::ODEFilter,
    t,x;
    dense=false,
    retcode = :Default,
    kwargs...)

    d = length(prob.u0)
    function make_Measurement(state)
        # @assert isdiag(state.Σ[1:d,1:d]) state.Σ[1:d,1:d]
        return state.μ[1:d] .± sqrt.(diag(state.Σ)[1:d])
    end
    u = map(make_Measurement, x)
    # u = map(s -> s.μ[1:d], x)

    interp = DiffEqBase.LinearInterpolation(t,u)

    T = eltype(eltype(u))
    N = length((size(prob.u0)..., length(u)))

    return ProbODESolution{T,N,typeof(u),typeof(x),typeof(t),typeof(prob),typeof(alg),typeof(interp)}(
        u,x,t,prob,alg,dense,interp,retcode)
end


# Plot recipe for the solution: Plot with ribbon
@recipe function f(sol::AbstractProbODESolution; c=1.96)
    stack(x) = collect(reduce(hcat, x)')
    values = map(u -> Measurements.value.(u), sol.u)
    uncertainties = map(u -> Measurements.uncertainty.(u), sol.u)
    ribbon := stack(uncertainties) * c
    return sol.t, stack(values)
end
