var documenterSearchIndex = {"docs":
[{"location":"probints_comparison/#Comparison-to-ProbInts","page":"Comparison to ProbInts","title":"Comparison to ProbInts","text":"","category":"section"},{"location":"probints_comparison/","page":"Comparison to ProbInts","title":"Comparison to ProbInts","text":"The DifferentialEquations.jl documentation contains a section about Uncertainty Quantification. It describes the ProbInts method for quantification of numerical uncertainty, and provides an extension of ProbInts to adaptive step sizes.","category":"page"},{"location":"probints_comparison/","page":"Comparison to ProbInts","title":"Comparison to ProbInts","text":"In this example, we want to compare the uncertainty estimates of Tsit5+AdaptiveProbInts to the posterior computed with the EK1.","category":"page"},{"location":"probints_comparison/#.-Problem-definition:-FitzHugh-Nagumo","page":"Comparison to ProbInts","title":"1. Problem definition: FitzHugh-Nagumo","text":"","category":"section"},{"location":"probints_comparison/","page":"Comparison to ProbInts","title":"Comparison to ProbInts","text":"using ProbNumDiffEq\nusing ProbNumDiffEq: remake_prob_with_jac, stack\nusing DifferentialEquations\nusing DiffEqUncertainty\nusing Statistics\nusing Plots\n\n\nfunction fitz!(du,u,p,t)\n    V,R = u\n    a,b,c = p\n    du[1] = c*(V - V^3/3 + R)\n    du[2] = -(1/c)*(V -  a - b*R)\nend\nu0 = [-1.0;1.0]\ntspan = (0.0,20.0)\np = (0.2,0.2,3.0)\nprob = ODEProblem(fitz!,u0,tspan,p)\nprob = remake_prob_with_jac(prob)\nnothing # hide","category":"page"},{"location":"probints_comparison/#High-accuracy-reference-solution:","page":"Comparison to ProbInts","title":"High accuracy reference solution:","text":"","category":"section"},{"location":"probints_comparison/","page":"Comparison to ProbInts","title":"Comparison to ProbInts","text":"appxsol = solve(remake(prob, u0=big.(prob.u0)), abstol=1e-20, reltol=1e-20)\nplot(appxsol)\nsavefig(\"./figures/ex_pi_fitzhugh.svg\"); nothing # hide","category":"page"},{"location":"probints_comparison/","page":"Comparison to ProbInts","title":"Comparison to ProbInts","text":"(Image: Prob-Ints Errors)","category":"page"},{"location":"probints_comparison/#.-ProbInts","page":"Comparison to ProbInts","title":"2. ProbInts","text":"","category":"section"},{"location":"probints_comparison/","page":"Comparison to ProbInts","title":"Comparison to ProbInts","text":"Uncertainty quantification of Tsit5 with AdaptiveProbInts:","category":"page"},{"location":"probints_comparison/","page":"Comparison to ProbInts","title":"Comparison to ProbInts","text":"cb = AdaptiveProbIntsUncertainty(5)\nensemble_prob = EnsembleProblem(prob)\nsol = solve(prob, Tsit5())\nsim = solve(ensemble_prob, Tsit5(), trajectories=100, callback=cb)\n\np = plot(sol.t, stack(appxsol.(sol.t) - sol.u), color=[3 4], ylims=(-0.003, 0.003), ylabel=\"Error\")\nerrors = [(a.t, stack(appxsol.(a.t) .- a.u)) for a in sim.u]\nfor e in errors\n    plot!(p, e[1], e[2], color=[3 4], label=\"\", linewidth=0.2, linealpha=0.5)\nend\nsavefig(\"./figures/ex_pi_probints.svg\"); nothing # hide","category":"page"},{"location":"probints_comparison/","page":"Comparison to ProbInts","title":"Comparison to ProbInts","text":"(Image: Prob-Ints Errors)","category":"page"},{"location":"probints_comparison/#.-EK1","page":"Comparison to ProbInts","title":"3. EK1","text":"","category":"section"},{"location":"probints_comparison/","page":"Comparison to ProbInts","title":"Comparison to ProbInts","text":"Uncertainties provided by the EK1:","category":"page"},{"location":"probints_comparison/","page":"Comparison to ProbInts","title":"Comparison to ProbInts","text":"sol = solve(prob, EK1())\nplot(sol.t, stack(appxsol.(sol.t) - sol.u), ylabel=\"Error\")\nplot!(sol.t, zero(stack(sol.u)), ribbon=3stack(std(sol.pu)), color=[1 2], label=\"\")\nsavefig(\"./figures/ex_pi_ours.svg\"); nothing # hide","category":"page"},{"location":"probints_comparison/","page":"Comparison to ProbInts","title":"Comparison to ProbInts","text":"(Image: Our Errors)","category":"page"},{"location":"probints_comparison/","page":"Comparison to ProbInts","title":"Comparison to ProbInts","text":"Verdict: The provided credible bands are more calibrated!","category":"page"},{"location":"solvers/#Solvers-and-Options","page":"Solvers and Options","title":"Solvers and Options","text":"","category":"section"},{"location":"solvers/","page":"Solvers and Options","title":"Solvers and Options","text":"EK0: Zeroth order extended Kalman filter and smoother\nEK1: First order extended Kalman filter and smoother\nIEKS: Iterated extended Kalman smoother","category":"page"},{"location":"solvers/","page":"Solvers and Options","title":"Solvers and Options","text":"EK0\nEK1\nIEKS","category":"page"},{"location":"solvers/#ProbNumDiffEq.IEKS","page":"Solvers and Options","title":"ProbNumDiffEq.IEKS","text":"IEKS(; prior=:ibm, order=1, diffusionmodel=:dynamic, linearize_at=nothing)\n\nGaussian ODE filtering with iterated extended Kalman smoothing. To use it, use solve_ieks(prob, IEKS(), args...) instead of solve(prob, IEKS(), args...), since it is implemented as an outer loop around the solver.\n\nCurrently, only the integrated Brownian motion prior :ibm is supported. For the diffusionmodel, chose one of [:dynamic, :dynamicMV, :fixed, :fixedMV, :fixedMAP]. Just like the EK1 it requires that the Jacobian of the rhs function is available.\n\nSee also: EK0, EK1, solve_ieks\n\nReferences:\n\nF. Tronarp, S. Särkkä, and P. Hennig: Bayesian ODE Solvers: The Maximum A Posteriori Estimate\n\n\n\n\n\n","category":"type"},{"location":"#ProbNumDiffEq.jl","page":"Home","title":"ProbNumDiffEq.jl","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Probabilistic numerical methods for ordinary differential equations.","category":"page"},{"location":"","page":"Home","title":"Home","text":"ProbNumDiffEq.jl is a library for probabilistic numerical methods for solving differential equations. It provides drop-in replacements for classic ODE solvers from DifferentialEquations.jl by extending OrdinaryDiffEq.jl.","category":"page"},{"location":"#Installation","page":"Home","title":"Installation","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"The package can be installed directly from github:","category":"page"},{"location":"","page":"Home","title":"Home","text":"] add https://github.com/nathanaelbosch/ProbNumDiffEq.jl","category":"page"},{"location":"#Getting-Started","page":"Home","title":"Getting Started","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"If you are unfamiliar with DifferentialEquations.jl, check out the official tutorial on how to solve ordinary differential equations.","category":"page"},{"location":"#Step-1:-Defining-a-problem","page":"Home","title":"Step 1: Defining a problem","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"First, we set up an ODEProblem to solve the Fitzhugh-Nagumo model.","category":"page"},{"location":"","page":"Home","title":"Home","text":"using ProbNumDiffEq\n\nfunction fitz(u, p, t)\n    a, b, c = p\n    return [c*(u[1] - u[1]^3/3 + u[2])\n            -(1/c)*(u[1] -  a - b*u[2])]\nend\n\nu0 = [-1.0; 1.0]\ntspan = (0., 20.)\np = (0.2,0.2,3.0)\nprob = ODEProblem(fitz, u0, tspan, p)\nnothing # hide","category":"page"},{"location":"#Step-2:-Solving-a-problem","page":"Home","title":"Step 2: Solving a problem","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Currently, ProbNumDiffEq.jl implements two probabilistic numerical methods: EK0 and EK1. In this example we solve the ODE with the default EK0, for high tolerance levels.","category":"page"},{"location":"","page":"Home","title":"Home","text":"sol = solve(prob, EK0(), abstol=1e-1, reltol=1e-2)\nnothing # hide","category":"page"},{"location":"#Step-3:-Analyzing-the-solution","page":"Home","title":"Step 3: Analyzing the solution","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Just as in DifferentialEquations.jl, the result of solve is a solution object, and we can access the (mean) values and timesteps as usual","category":"page"},{"location":"","page":"Home","title":"Home","text":"sol[end]\nsol.u[5]\nsol.t[8]","category":"page"},{"location":"","page":"Home","title":"Home","text":"However, the solver returns a probabilistic solution, here a Gaussian distribution over solution values. These can be accessed similarly, with","category":"page"},{"location":"","page":"Home","title":"Home","text":"sol.pu[end]\nsol.pu[5]","category":"page"},{"location":"","page":"Home","title":"Home","text":"By default, the posterior distribution can be evaluated for arbitrary points in time t by treating sol as a function:","category":"page"},{"location":"","page":"Home","title":"Home","text":"sol(0.45)","category":"page"},{"location":"#Plotting-Solutions","page":"Home","title":"Plotting Solutions","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Finally, we can conveniently visualize the result through Plots.jl:","category":"page"},{"location":"","page":"Home","title":"Home","text":"using Plots\nplot(sol)\nsavefig(\"./figures/fitzhugh_nagumo.svg\"); nothing # hide","category":"page"},{"location":"","page":"Home","title":"Home","text":"(Image: Fitzhugh-Nagumo Solution)","category":"page"},{"location":"#References","page":"Home","title":"References","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Gaussian ODE Filters:","category":"page"},{"location":"","page":"Home","title":"Home","text":"M. Schober, S. Särkkä, and P. Hennig: A Probabilistic Model for the Numerical Solution of Initial Value Problems\nH. Kersting, T. J. Sullivan, and P. Hennig: Convergence Rates of Gaussian Ode Filters\nF. Tronarp, H. Kersting, S. Särkkä, and P. Hennig: Probabilistic Solutions To Ordinary Differential Equations As Non-Linear Bayesian Filtering: A New Perspective\nF. Tronarp, S. Särkkä, and P. Hennig: Bayesian ODE Solvers: The Maximum A Posteriori Estimate\nN. Bosch, P. Hennig, F. Tronarp: Calibrated Adaptive Probabilistic ODE Solvers\nN. Krämer, P. Hennig: Stable Implementation of Probabilistic ODE Solvers","category":"page"},{"location":"","page":"Home","title":"Home","text":"Probabilistic Numerics:","category":"page"},{"location":"","page":"Home","title":"Home","text":"http://probabilistic-numerics.org/\nP. Hennig, M. A. Osborne, and M. Girolami: Probabilistic numerics and uncertainty in computations\nC. J. Oates and T. J. Sullivan: A modern retrospective on probabilistic numerics","category":"page"},{"location":"internals/#Internals","page":"Internals","title":"Internals","text":"","category":"section"},{"location":"internals/","page":"Internals","title":"Internals","text":"Modules = [ProbNumDiffEq]","category":"page"},{"location":"internals/#ProbNumDiffEq.EK0","page":"Internals","title":"ProbNumDiffEq.EK0","text":"EK0(; prior=:ibm, order=1, diffusionmodel=:dynamic, smooth=true)\n\nGaussian ODE filtering with zeroth order extended Kalman filter.\n\nCurrently, only the integrated Brownian motion prior :ibm is supported. For the diffusionmodel, chose one of [:dynamic, :dynamicMV, :fixed, :fixedMV, :fixedMAP].\n\nSee also: EK1\n\nReferences:\n\nM. Schober, S. Särkkä, and P. Hennig: A Probabilistic Model for the Numerical Solution of Initial Value Problems\nF. Tronarp, H. Kersting, S. Särkkä, and P. Hennig: Probabilistic Solutions To Ordinary Differential Equations As Non-Linear Bayesian Filtering: A New Perspective\n\n\n\n\n\n","category":"type"},{"location":"internals/#ProbNumDiffEq.EK1","page":"Internals","title":"ProbNumDiffEq.EK1","text":"EK1(; prior=:ibm, order=1, diffusionmodel=:dynamic, smooth=true)\n\nGaussian ODE filtering with first order extended Kalman filter\n\nCurrently, only the integrated Brownian motion prior :ibm is supported. For the diffusionmodel, chose one of [:dynamic, :dynamicMV, :fixed, :fixedMV, :fixedMAP].\n\nSee also: EK0\n\nReferences:\n\nF. Tronarp, H. Kersting, S. Särkkä, and P. Hennig: Probabilistic Solutions To Ordinary Differential Equations As Non-Linear Bayesian Filtering: A New Perspective\n\n\n\n\n\n","category":"type"},{"location":"internals/#ProbNumDiffEq.IEKS-Tuple{}","page":"Internals","title":"ProbNumDiffEq.IEKS","text":"IEKS(; prior=:ibm, order=1, diffusionmodel=:dynamic, linearize_at=nothing)\n\nGaussian ODE filtering with iterated extended Kalman smoothing. To use it, use solve_ieks(prob, IEKS(), args...) instead of solve(prob, IEKS(), args...), since it is implemented as an outer loop around the solver.\n\nCurrently, only the integrated Brownian motion prior :ibm is supported. For the diffusionmodel, chose one of [:dynamic, :dynamicMV, :fixed, :fixedMV, :fixedMAP]. Just like the EK1 it requires that the Jacobian of the rhs function is available.\n\nSee also: EK0, EK1, solve_ieks\n\nReferences:\n\nF. Tronarp, S. Särkkä, and P. Hennig: Bayesian ODE Solvers: The Maximum A Posteriori Estimate\n\n\n\n\n\n","category":"method"},{"location":"internals/#ProbNumDiffEq.MAPFixedDiffusion","page":"Internals","title":"ProbNumDiffEq.MAPFixedDiffusion","text":"Maximum a-posteriori Diffusion estimate when using an InverseGamma(1/2,1/2) prior\n\nThe mode of an InverseGamma(α,β) distribution is given by β/(α+1) To compute this in an on-line basis from the previous Diffusion, we reverse the computation to get the previous sum of residuals from Diffusion, and then modify that sum and compute the new Diffusion.\n\n\n\n\n\n","category":"type"},{"location":"internals/#OrdinaryDiffEq.perform_step!","page":"Internals","title":"OrdinaryDiffEq.perform_step!","text":"Perform a step\n\nNot necessarily successful! For that, see step!(integ).\n\nBasically consists of the following steps\n\nCoordinate change / Predonditioning\nPrediction step\nMeasurement: Evaluate f and Jf; Build z, S, H\nCalibration; Adjust prediction / measurement covs if the diffusion model \"dynamic\"\nUpdate step\nError estimation\nUndo the coordinate change / Predonditioning\n\n\n\n\n\n","category":"function"},{"location":"internals/#ProbNumDiffEq._rand","page":"Internals","title":"ProbNumDiffEq._rand","text":"Helper function to sample from our covariances, which often have a \"cross\" of zeros For the 0-cov entries the outcome of the sampling is deterministic!\n\n\n\n\n\n","category":"function"},{"location":"internals/#ProbNumDiffEq.ibm","page":"Internals","title":"ProbNumDiffEq.ibm","text":"Generate the discrete dynamics for a q-IBM model. INCLUDES AUTOMATIC PRECONDITIONING!\n\nCareful: Dimensions are ordered differently than in probnum!\n\n\n\n\n\n","category":"function"},{"location":"internals/#ProbNumDiffEq.iip_to_oop-Tuple{Any}","page":"Internals","title":"ProbNumDiffEq.iip_to_oop","text":"Quick and dirty wrapper to make IIP functions OOP\n\n\n\n\n\n","category":"method"},{"location":"internals/#ProbNumDiffEq.initial_update!-Tuple{Any}","page":"Internals","title":"ProbNumDiffEq.initial_update!","text":"initialize x0 up to the provided order\n\n\n\n\n\n","category":"method"},{"location":"internals/#ProbNumDiffEq.predict!-Tuple{Gaussian,Gaussian,AbstractArray{T,2} where T,AbstractArray{T,2} where T}","page":"Internals","title":"ProbNumDiffEq.predict!","text":"predict!(x_out, x_curr, Ah, Qh)\n\nPREDICT step in Kalman filtering for linear dynamics models. In-place implementation of predict, saving the result in x_out.\n\nm_n+1^P = A(h)*m_n\nP_n+1^P = A(h)*P_n*A(h) + Q(h)\n\nSee also: predict\n\n\n\n\n\n","category":"method"},{"location":"internals/#ProbNumDiffEq.predict-Tuple{Gaussian,AbstractArray{T,2} where T,AbstractArray{T,2} where T}","page":"Internals","title":"ProbNumDiffEq.predict","text":"predict(x_curr::Gaussian, Ah::AbstractMatrix, Qh::AbstractMatrix)\n\nSee also: predict!\n\n\n\n\n\n","category":"method"},{"location":"internals/#ProbNumDiffEq.remake_prob_with_jac-Tuple{ODEProblem}","page":"Internals","title":"ProbNumDiffEq.remake_prob_with_jac","text":"remake_prob_with_jac(prob)\n\nAdd a jacobian function to the ODE function, using ModelingToolkit.jl.\n\n\n\n\n\n","category":"method"},{"location":"internals/#ProbNumDiffEq.smooth-Tuple{Gaussian,Gaussian,AbstractArray{T,2} where T,AbstractArray{T,2} where T}","page":"Internals","title":"ProbNumDiffEq.smooth","text":"smooth(x_curr::Gaussian, x_next_smoothed::Gaussian, Ah::AbstractMatrix, Qh::AbstractMatrix)\n\nSMOOTH step of the (extended) Kalman smoother, or (extended) Rauch-Tung-Striebel smoother. It is implemented in Joseph Form:\n\nm_n+1^P = A(h)*m_n\nP_n+1^P = A(h)*P_n*A(h) + Q(h)\n\nG = P_n * A(h)^T * (P_n+1^P)^-1\nm_n^S = m_n + G * (m_n+1^S - m_n+1^P)\nP_n^S = (I - G*A(h)) P_n (I - G*A(h))^T + G * Q(h) * G + G * P_n+1^S * G\n\n\n\n\n\n","category":"method"},{"location":"internals/#ProbNumDiffEq.solve_ieks-Tuple{SciMLBase.AbstractODEProblem,IEKS,Vararg{Any,N} where N}","page":"Internals","title":"ProbNumDiffEq.solve_ieks","text":"solve_ieks(prob::DiffEqBase.AbstractODEProblem, alg::IEKS, args...; iterations=10, kwargs...)\n\nSolve method to be used with the IEKS. The IEKS works essentially by solving the ODE multiple times. solve_ieks therefore wraps a call to the standard solve method, passing args... and kwargs....\n\n\n\n\n\n","category":"method"},{"location":"internals/#ProbNumDiffEq.update","page":"Internals","title":"ProbNumDiffEq.update","text":"update(x_pred, measurement, H, R=0)\n\nSee also: update!\n\n\n\n\n\n","category":"function"},{"location":"internals/#ProbNumDiffEq.update!","page":"Internals","title":"ProbNumDiffEq.update!","text":"update!(x_out, x_pred, measurement, H, R=0)\n\nUPDATE step in Kalman filtering for linear dynamics models, given a measurement Z=N(z, S). In-place implementation of update, saving the result in x_out.\n\nK = P_n+1^P * H^T * S^-1\nm_n+1 = m_n+1^P + K * (0 - z)\nP_n+1 = P_n+1^P - K*S*K^T\n\nImplemented in Joseph Form.\n\nSee also: predict\n\n\n\n\n\n","category":"function"},{"location":"internals/#ProbNumDiffEq.vanilla_ibm-Tuple{Integer,Integer}","page":"Internals","title":"ProbNumDiffEq.vanilla_ibm","text":"Same as above, but without the automatic preconditioning\n\n\n\n\n\n","category":"method"}]
}