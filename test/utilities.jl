"""
$(SIGNATURES)

Log jacobian abs determinant via automatic differentiation. For testing.
"""
AD_logjac(f, x) = log(abs(ForwardDiff.derivative(f, x)))

function AD_logjac(t::VectorTransform, x, vec_y)
    J = ForwardDiff.jacobian(x -> vec_y(transform(t, x)), x)
    n, n2 = size(J)
    if n == n2
        logabsdet(J)[1]
    else
        # for generalized Jacobian determinant, see
        # - https://encyclopediaofmath.org/wiki/Jacobian#Generalizations_of_the_Jacobian_determinant
        # - https://en.wikipedia.org/wiki/Area_formula_(geometric_measure_theory)
        # NOTE code below only works when the density is written wrt the Hausdorff measure.
        # see https://github.com/tpapp/TransformVariables.jl/pull/139#discussion_r2071715166
        logabsdet(J' * J)[1] / 2
    end
end


AD_logjac(t::ScalarTransform, x) = AD_logjac(x -> transform(t, x), x)

"""
$(SIGNATURES)

A random input, for testing.
"""
random_arg(t::ScalarTransform) = randn()
random_arg(t::VectorTransform) = randn(dimension(t))

"""
$(SIGNATURES)

Test transformation `t` with random values, `N` times.

`is_valid_y` checks the result of the transformation.

# Keyword arguments

`vec_y` converts the result to a vector, for checking the log Jacobian with automatic
differentiation.

`test_inverse` determines whether the inverse is tested.

`jac` determines whether `transform_and_logjac` is tested against the log Jacobian from
AD, true by default. The Jacobian is not defined for Unitful scaling.
"""
function test_transformation(t::AbstractTransform, is_valid_y;
                             vec_y = identity, N = 1000, test_inverse = true, jac=true)
    for _ in 1:N
        x = random_arg(t)
        x isa ScalarTransform && @test dimension(x) == 1
        y = @inferred transform(t, x)
        @test is_valid_y(y)
        @test transform(t, x) == y
        if jac
            y2, lj = @inferred transform_and_logjac(t, x)
            @test y2 == y
            jc = TransformVariables.logprior(t, y)
            if !iszero(jc)
                @test TransformVariables.nonzero_logprior(t) == true
            end
            if t isa ScalarTransform
                @test lj ≈ AD_logjac(t, x) + jc
            else
                @test lj ≈ AD_logjac(t, x, vec_y) + jc
            end
        end
        if test_inverse
            x2 = inverse(t, y)
            @test x ≈ x2 atol = 1e-6
            ι = inverse(t)
            @test x2 == ι(y)
        end
    end
end

"""
$(SIGNATURES)

Elements (strictly) above the diagonal as a vector.
"""
function vec_above_diagonal(U::UpperTriangular{T}) where T
    n = size(U, 1)
    index = 1
    x = Vector{T}(undef, unit_triangular_dimension(n))
    for col in 1:n
        for row in 1:(col-1)
            x[index] = U[row, col]
            index += 1
        end
    end
    x
end

"""
$(SIGNATURES)

Check if the argument `U` makes a valid correlation matrix `U'*U`.
"""
function is_valid_corr_cholesky(U::UpperTriangular)
    Ω = U'*U
    all(isapprox.(diag(Ω), 1)) && all(@. abs(Ω) ≤ (1+√eps()))
end
