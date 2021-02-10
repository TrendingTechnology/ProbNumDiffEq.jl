# Called in the OrdinaryDiffEQ.__init; All `OrdinaryDiffEqAlgorithm`s have one
function OrdinaryDiffEq.initialize!(integ, cache::GaussianODEFilterCache)
    @assert integ.opts.dense == integ.alg.smooth "`dense` and `smooth` should have the same value! "
    @assert integ.saveiter == 1
    OrdinaryDiffEq.copyat_or_push!(integ.sol.x, integ.saveiter, cache.x)
    OrdinaryDiffEq.copyat_or_push!(integ.sol.pu, integ.saveiter, cache.SolProj*cache.x)
end

"""Perform a step

Not necessarily successful! For that, see `step!(integ)`.

Basically consists of the following steps
- Coordinate change / Predonditioning
- Prediction step
- Measurement: Evaluate f and Jf; Build z, S, H
- Calibration; Adjust prediction / measurement covs if the diffusion model "dynamic"
- Update step
- Error estimation
- Undo the coordinate change / Predonditioning
"""
function OrdinaryDiffEq.perform_step!(integ, cache::GaussianODEFilterCache, repeat_step=false)
    @unpack t, dt = integ
    @unpack d, Proj, SolProj, Precond = integ.cache
    @unpack x, x_pred, u_pred, x_filt, u_filt, err_tmp = integ.cache
    @unpack A, Q = integ.cache

    tnew = t + dt
    @info "New perform_step!" t dt

    # Coordinate change / preconditioning
    P = Precond(dt)
    PI = inv(P)
    x = P * x

    if isdynamic(cache.diffusionmodel)  # Calibrate, then predict cov

        # Predict
        # predict_mean!(x_pred, x, A, Q)
        predict!(x_pred, x, A, Q)
        mul!(u_pred, SolProj, PI*x_pred.μ)

        # Proj
        if !isnothing(integ.alg.manifold) && integ.alg.mprojtime in (:before, :both)
            error("It's a bit unclear how to properly handle dynamic diffusion and manifold projections")
            manifold_update!(x_filt, (x) -> integ.alg.manifold(SolProj * PI * x), integ.alg.mprojmaxiters)
        end

        # Measure
        measure!(integ, x_pred, tnew)

        # Estimate diffusion
        integ.cache.diffusion = estimate_diffusion(cache.diffusionmodel, integ)
        # Adjust prediction and measurement
        predict!(x_pred, x, A, apply_diffusion(Q, integ.cache.diffusion))

        if !isnothing(integ.alg.manifold) && integ.alg.mprojtime in (:before, :both)
            error("It's a bit unclear how to properly handle dynamic diffusion and manifold projections")
            manifold_update!(x_pred, (x) -> integ.alg.manifold(SolProj * PI * x), integ.alg.mprojmaxiters)
        end

        copy!(integ.cache.measurement.Σ, Matrix(X_A_Xt(x_pred.Σ, integ.cache.H)))

    else  # Vanilla filtering order: Predict, measure, calibrate

        predict!(x_pred, x, A, Q)
        # @info "after predict!" integ.alg.manifold(SolProj * PI * x_pred.μ) |> norm
        if !isnothing(integ.alg.manifold) && integ.alg.mprojtime in (:before, :both)
            manifold_update!(x_pred, (x) -> integ.alg.manifold(SolProj * PI * x))
        end
        # @info "after manifold_update! 1" integ.alg.manifold(SolProj * PI * x_pred.μ) |> norm
        mul!(u_pred, SolProj, PI*x_pred.μ)
        measure!(integ, x_pred, tnew)
        integ.cache.diffusion = estimate_diffusion(cache.diffusionmodel, integ)

    end

    # Likelihood
    cache.log_likelihood = logpdf(cache.measurement, zeros(d))

    # Update
    x_filt = update!(integ, x_pred)

    # Project onto the manifold
    if !isnothing(integ.alg.manifold) && integ.alg.mprojtime in (:after, :both)
        @info "after update!" integ.alg.manifold(SolProj * PI * x_filt.μ) |> norm
        manifold_update!(x_filt, (x) -> integ.alg.manifold(SolProj * PI * x), integ.alg.mprojmaxiters, integ.alg.mprojtime==:both)
        @info "after manifold_update! 2" integ.alg.manifold(SolProj * PI * x_filt.μ) |> norm
    end

    # Save
    mul!(u_filt, SolProj, PI*x_filt.μ)
    integ.u .= u_filt

    # Undo the coordinate change / preconditioning
    copy!(integ.cache.x, PI * x)
    copy!(integ.cache.x_pred, PI * x_pred)
    copy!(integ.cache.x_filt, PI * x_filt)

    # Estimate error for adaptive steps
    if integ.opts.adaptive
        err_est_unscaled = estimate_errors(integ, integ.cache)
        DiffEqBase.calculate_residuals!(
            err_tmp, dt * err_est_unscaled, integ.u, u_filt,
            integ.opts.abstol, integ.opts.reltol, integ.opts.internalnorm, t)
        integ.EEst = integ.opts.internalnorm(err_tmp, t) # scalar

    end
    # stuff that would normally be in apply_step!
    if !integ.opts.adaptive || integ.EEst < one(integ.EEst)
        copy!(integ.cache.x, integ.cache.x_filt)
        integ.sol.log_likelihood += integ.cache.log_likelihood
    end
    # (t > 0) && error()
end


function manifold_update!(x, h, maxiters=1, check=false)
    z_before = h(x.μ)
    if iszero(z_before) || (check && z_before < eps(typeof(z_before)))
        return
    end

    for i in 1:maxiters
        if i > 1
            @warn "Second iteration of manifold projection!"
        end
        z = h(x.μ)
        H = ForwardDiff.gradient(h, x.μ)
        @assert H isa AbstractVector

        S = H' * x.Σ * H
        K = x.Σ * H * inv(S)

        SL = H'x.Σ.squareroot
        @info "manifold_update!" z S inv(S) SL SL*SL'
        K = x.Σ * H * inv(SL*SL')

        x.μ .= x.μ .+ K * (0 .- z)
        Pnew = X_A_Xt(x.Σ, (I-K*H'))
        copy!(x.Σ, Pnew)

        z_after = h(x.μ)
        # @info "Iteration" i z_before S z_after z_before ≈ z_after
        # @assert abs(z_after) <= abs(z_before)
        @assert abs(z_after) <= abs(z_before) || S < eps(typeof(S))
        if iszero(z_after) || S < eps(typeof(S)) break end
        z_before = z_after
    end
    # error()
end


function h!(integ, x_pred, t)
    @unpack f, p, dt = integ
    @unpack u_pred, du, Proj, Precond, measurement = integ.cache
    PI = inv(Precond(dt))
    z = measurement.μ
    E0, E1 = Proj(0), Proj(1)

    u_pred .= E0*PI*x_pred.μ
    IIP = isinplace(integ.f)
    if IIP
        f(du, u_pred, p, t)
    else
        du .= f(u_pred, p, t)
    end
    integ.destats.nf += 1

    z .= E1*PI*x_pred.μ .- du

    return z
end

function H!(integ, x_pred, t)
    @unpack f, p, dt, alg = integ
    @unpack ddu, Proj, Precond, H, u_pred = integ.cache
    E0, E1 = Proj(0), Proj(1)
    PI = inv(Precond(dt))

    if alg isa EK1 || alg isa IEKS
        if alg isa IEKS && !isnothing(alg.linearize_at)
            linearize_at = alg.linearize_at(t).μ
        else
            linearize_at = u_pred
        end

        if isinplace(integ.f)
            f.jac(ddu, linearize_at, p, t)
        else
            ddu .= f.jac(linearize_at, p, t)
            # WIP: Handle Jacobians as OrdinaryDiffEq.jl does
            # J = OrdinaryDiffEq.jacobian((u)-> f(u, p, t), u_pred, integ)
            # @assert J ≈ ddu
        end
        integ.destats.njacs += 1
        mul!(H, (E1 .- ddu * E0), PI)
    else
        mul!(H, E1, PI)
    end

    return H
end


function measure!(integ, x_pred, t)
    @unpack R = integ.cache
    @unpack u_pred, measurement, H = integ.cache

    z, S = measurement.μ, measurement.Σ
    z .= h!(integ, x_pred, t)
    H .= H!(integ, x_pred, t)
    # R .= Diagonal(eps.(z))
    @assert iszero(R)
    copy!(S, Matrix(X_A_Xt(x_pred.Σ, H)))

    return nothing
end


function update!(integ, prediction)
    @unpack measurement, H, R, x_filt = integ.cache
    update!(x_filt, prediction, measurement, H, R)
    # assert_nonnegative_diagonal(x_filt.Σ)
    return x_filt
end


function estimate_errors(integ, cache::GaussianODEFilterCache)
    @unpack diffusion, Q, H = integ.cache

    if diffusion isa Real && isinf(diffusion)
        return Inf
    end

    error_estimate = sqrt.(diag(Matrix(X_A_Xt(apply_diffusion(Q, diffusion), H))))

    return error_estimate
end
