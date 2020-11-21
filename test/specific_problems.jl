# Everytime I encounter something that raises some error and I fix it, I should add that
# specific problem to this list to make sure, that this specific run then works without
# bugs.
using ODEFilters
using Test
using LinearAlgebra
using UnPack


using DiffEqProblemLibrary.ODEProblemLibrary: importodeproblems; importodeproblems()
import DiffEqProblemLibrary.ODEProblemLibrary: prob_ode_fitzhughnagumo, prob_ode_vanstiff, prob_ode_mm_linear


@testset "Smoothing with small constant steps" begin
    prob = ODEFilters.remake_prob_with_jac(prob_ode_fitzhughnagumo)
    @test solve(prob, EKF0(order=4, diffusionmodel=:fixed, smooth=true),
                adaptive=false, dt=1e-3) isa ODEFilters.ProbODESolution
    @test solve(prob, EKF1(order=4, diffusionmodel=:fixed, smooth=true),
                adaptive=false, dt=1e-3) isa ODEFilters.ProbODESolution
end


@testset "Problem with analytic solution" begin
    prob = ODEFilters.remake_prob_with_jac(prob_ode_mm_linear)
    @test solve(prob, EKF0(order=4)) isa ODEFilters.ProbODESolution
    @test solve(prob, EKF1(order=4)) isa ODEFilters.ProbODESolution
end


@testset "Stiff Vanderpol" begin
    prob = ODEFilters.remake_prob_with_jac(prob_ode_vanstiff)
    @test solve(prob, EKF1(order=3)) isa ODEFilters.ProbODESolution
end


@testset "Big Float" begin
    prob = prob_ode_fitzhughnagumo
    prob = remake(prob, u0=big.(prob.u0))
    @test solve(prob, EKF0(order=3)) isa ODEFilters.ProbODESolution
end


@testset "OOP problem definition" begin
    prob = ODEProblem((u, p, t) -> ([p[1] * u[1] .* (1 .- u[1])]), [1e-1], (0.0, 5), [3.0])
    @test solve(prob, EKF0(order=4)) isa ODEFilters.ProbODESolution
    prob = ODEFilters.remake_prob_with_jac(prob)
    @test solve(prob, EKF1(order=4)) isa ODEFilters.ProbODESolution
end


@testset "Callback: Harmonic Oscillator with condition on E=2" begin
    u0 = ones(2)
    function harmonic_oscillator(du,u,p,t)
        du[1] = u[2]
        du[2] = -u[1]
    end
    prob = ODEProblem(harmonic_oscillator, u0, (0.0,100.0))

    function Callback()
        function affect!(integ)
            @unpack dt = integ
            @unpack x_filt, Proj, InvPrecond = integ.cache
            E0 = Proj(0)

            PI = InvPrecond(dt)
            x = x_filt

            m, P = x.μ, x.Σ

            m0, P0 = E0*m, ODEFilters.X_A_Xt(P, E0)

            e = m0'm0
            H = 2m0'E0
            S = H*P*H'

            S_inv = inv(S)
            K = P * H' * S_inv

            mnew = m + K * (2 .- e)
            Pnew = ODEFilters.X_A_Xt(P, (I-K*H)) # + X_A_Xt(R, K)

            # @info m P e S K mnew
            copy!(m, mnew)
            copy!(P, Pnew)
        end
        condtion = (t,u,integrator) -> true
        save_positions = (true, true)
        DiscreteCallback(condtion,affect!,save_positions=save_positions)
    end

    @test solve(prob, EKF0(order=3)) isa ODEFilters.ProbODESolution
    @test solve(prob, EKF0(order=3), callback=Callback()) isa ODEFilters.ProbODESolution
end
