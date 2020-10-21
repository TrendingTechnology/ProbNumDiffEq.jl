function DiffEqBase.__solve(prob::DiffEqBase.AbstractODEProblem,
                            alg::GaussianODEFilter, args...; kwargs...)
    @debug "Called solve with" args kwargs
    integrator = DiffEqBase.__init(prob, alg, args...; kwargs...)
    sol = DiffEqBase.solve!(integrator)
    return sol
end

function DiffEqBase.__init(prob::DiffEqBase.AbstractODEProblem,
                           alg::GaussianODEFilter;

                           steprule=:standard,
                           dt=eltype(prob.tspan)(0),
                           abstol=1e-6, reltol=1e-3,
                           gamma=9//10,
                           qmin=2//10, qmax=10,
                           dtmin=DiffEqBase.prob2dtmin(prob; use_end_time=true),
                           dtmax=eltype(prob.tspan)((prob.tspan[end]-prob.tspan[1])),
                           beta2 = beta2_default(alg),
                           beta1 = beta1_default(alg, beta2),
                           qoldinit = 1//10^4,

                           maxiters=1e5,
                           internalnorm = DiffEqBase.ODE_DEFAULT_NORM,
                           unstable_check = DiffEqBase.ODE_DEFAULT_UNSTABLE_CHECK,

                           dense=true,
                           callback=nothing,
                           calck = (callback !== nothing && callback != CallbackSet()) || (dense), # from OrdinaryDiffEq; not sure what it does

                           kwargs...)

    if alg isa EKF1 && isnothing(prob.f.jac)
        error("""EKF1 requires the Jacobian. To automatically generate it with ModelingToolkit.jl
               use ProbNumoDE.remake_prob_with_jac(prob).""")
    end

    if length(prob.u0) == 1 && size(prob.u0) == ()
        @warn "prob.u0 is a scalar; In order to run, we remake the problem with u0 = [u0]."
        prob = remake(prob, u0=[prob.u0])
    end

    f = prob.f
    u0 = copy(prob.u0)
    t0, tmax = prob.tspan
    p = prob.p
    d = length(u0)

    # Solver Options
    adaptive = steprule != :constant
    if !adaptive && iszero(dt)
        error("Fixed timestep methods require a choice of dt")
    end
    steprules = Dict(
        :constant => ConstantSteps(),
        :standard => StandardSteps(),
        :PI => PISteps(),
    )
    steprule = steprules[steprule]

    tType = eltype(prob.tspan)

    # Cache
    cache = OrdinaryDiffEq.alg_cache(
        alg, copy(u0), copy(u0), eltype(u0), eltype(u0), typeof(one(tType)), copy(u0),
        copy(u0), f, t0, dt, real.(reltol), p, calck, Val(isinplace(prob)))

    destats = DiffEqBase.DEStats(0)

    state_estimates = StructArray([copy(cache.x)])
    times = [t0]
    diffusions = []

    isnothing(dtmin) && (dtmin = DiffEqBase.prob2dtmin(prob; use_end_time=true))
    dt_init = dt != 0 ? dt : 1e-3
    QT = tType
    xType = typeof(cache.x)
    diffusionType = typeof(cache.diffmat)

    opts = DEOptions{
        typeof(maxiters), typeof(abstol), typeof(reltol), QT, typeof(internalnorm), tType,
        typeof(unstable_check)}(
            maxiters, adaptive, abstol, reltol, QT(gamma), QT(qmin), QT(qmax),
            QT(beta1), QT(beta2), QT(qoldinit),
            internalnorm, unstable_check, dtmin, dtmax, false, true)

    return ODEFilterIntegrator{
        DiffEqBase.isinplace(prob), typeof(u0), typeof(t0), typeof(p), typeof(f), QT,
        typeof(opts), typeof(cache), typeof(steprule),
        xType, diffusionType, typeof(prob), typeof(alg)
    }(
        nothing, f, u0, t0, t0, t0, tmax, dt_init, p, one(QT), QT(qoldinit), cache,
        opts, steprule, alg.smooth, state_estimates, times, diffusions,
        0, 0, false, :Default, prob, alg, destats,
    )
end


function DiffEqBase.solve!(integ::ODEFilterIntegrator)
    while integ.t < integ.tmax
        loopheader!(integ)
        if check_error!(integ) != :Success
            return integ.sol
        end
        perform_step!(integ, integ.cache)
        loopfooter!(integ)
    end
    postamble!(integ)
    if integ.sol.retcode == :Default
        integ.sol = solution_new_retcode(integ.sol, :Success)
    end
    return integ.sol
end
