# ProbNumODE.jl: Probabilistic Numerics for Ordinary Differential Equations

[![Build Status](https://travis-ci.org/nathanaelbosch/ProbNumODE.jl.svg?branch=master)](https://travis-ci.org/nathanaelbosch/ProbNumODE.jl)
[![Build Status](https://ci.appveyor.com/api/projects/status/github/nathanaelbosch/ProbNumODE.jl?svg=true)](https://ci.appveyor.com/project/nathanaelbosch/ProbNumODE-jl)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://nathanaelbosch.github.io/ProbNumODE.jl/dev)
[![Coverage](https://codecov.io/gh/nathanaelbosch/ProbNumODE.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/nathanaelbosch/ProbNumODE.jl)
<!-- [![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle) -->
<!-- [![](https://img.shields.io/badge/docs-stable-blue.svg)](https://nathanaelbosch.github.io/ProbNumODE.jl/stable) -->
<!-- [![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac) -->


ProbNumODE.jl is a library for probabilistic numerical methods for solving ordinary differential equations.
It provides drop-in replacements for classic ODE solvers from [DifferentialEquations.jl](https://docs.sciml.ai/stable/).


## Installation
The package can be installed directly from github:
```julia
] add https://github.com/nathanaelbosch/ProbNumODE.jl
```


## Example
Solving ODEs with probabilistic numerical methods is as simple as that!
```julia
using ProbNumODE

# Fitzhugh-Nagumo model
function f(u, p, t)
    V, R = u
    a, b, c = p
    return [
        c*(V - V^3/3 + R)
        -(1/c)*(V -  a - b*R)
    ]
end

u0 = [-1.0; 1.0]
tspan = (0., 20.)
p = (0.2,0.2,3.0)
prob = ODEProblem(f, u0, tspan, p)

sol = solve(prob, EKF0())

using Plots
plot(sol)
```
![Fitzhugh-Nagumo Solution](./docs/src/figures/fitzhugh_nagumo.svg?raw=true "Fitzhugh-Nagumo Solution")
