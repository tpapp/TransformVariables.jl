AD_logjac(t::VectorTransform, x, vec_y) =
    logabsdet(ForwardDiff.jacobian(x -> vec_y(transform(t, x)), x))[1]

AD_logjac(t::ScalarTransform, x) =
    log(abs(ForwardDiff.derivative(x -> transform(t, x), x)))

"""
$(SIGNATURES)

Test transformation `t` with random values, `N` times.

`is_valid_y` checks the result of the transformation.

`vec_y` converts the result to a vector, for checking the log Jacobian with
automatic differentiation.

`test_inverse` determines whether the inverse is tested.
"""
function test_transformation(t::AbstractTransform, is_valid_y;
                             vec_y = identity, N = 1000, test_inverse = true)
    for _ in 1:N
        x = t isa ScalarTransform ? randn() : randn(dimension(t))
        x isa ScalarTransform && @test dimension(x) == 1
        y = transform(t, x)
        @test is_valid_y(y)
        y2, lj = transform_and_logjac(t, x)
        @test y2 == y
        if t isa ScalarTransform
            @test lj ≈ AD_logjac(t, x)
        else
            @test lj ≈ AD_logjac(t, x, vec_y)
        end
        if test_inverse
            x2 = inverse(t, y)
            @test x ≈ x2
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
