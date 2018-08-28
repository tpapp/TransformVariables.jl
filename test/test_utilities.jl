AD_logjac(t::TransformReals, x, vec_y) =
    logabsdet(jacobian(x -> vec_y(transform(t, x)), x))[1]

"""
    $SIGNATURES

Test transformation `t` with random values, `N` times.

`is_valid_y` checks the result of the transformation.

`vec_y` converts the result to a vector, for checking the log Jacobian with
automatic differentiation.

`test_inverse` determines whether the inverse is tested.
"""
function test_transformation(t::TransformReals, is_valid_y, vec_y;
                             N = 1000, test_inverse = true)
    for _ in 1:N
        x = randn(dimension(t))
        y = transform(t, x)
        @test is_valid_y(y)
        y2, lj = transform_and_logjac(t, x)
        @test y2 == y
        @test lj ≈ AD_logjac(t, x, vec_y)
        if test_inverse
            x2 = inverse(t, y)
            @test x ≈ x2
        end

    end
end

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
    $SIGNATURES

Check if the argument `U` makes a valid correlation matrix `U'*U`.
"""
function is_valid_corr_cholesky(U::UpperTriangular)
    Ω = U'*U
    all(isapprox.(diag(Ω), 1)) && all(@. abs(Ω) ≤ (1+√eps()))
end
