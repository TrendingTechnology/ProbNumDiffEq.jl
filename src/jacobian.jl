function remake_prob_with_jac(prob)
    try
        p = prob.p isa DiffEqBase.NullParameters ? [] : collect(prob.p)
        prob = remake(prob, p=p)
        sys = modelingtoolkitize(prob)
        jac = eval(ModelingToolkit.generate_jacobian(sys)[2])
        IIP = isinplace(prob.f)
        f = ODEFunction{IIP}(
            prob.f.f,
            jac=jac;
            analytic=prob.f.analytic,
        )
        return remake(prob, f=f)
    catch
        error("Could not generate a jacobian for the problem")
    end
end