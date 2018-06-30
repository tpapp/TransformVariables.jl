@testset "to named tuple" begin
    t1 = to_ℝ
    t2 = to_𝕀
    t3 = to_corr_cholesky(7)
    tn = to_tuple((a = t1, b = t2, c = t3))
    @test length(tn) == length(t1) + length(t2) + length(t3)
    x = randn(length(tn))
    y = transform(tn, x)
    @test y isa NamedTuple{(:a,:b,:c)}
    @test inverse(tn, y) ≈ x
    index = 0
    ljacc = 0.0
    for (i, t) in enumerate((t1, t2, t3))
        d = length(t)
        xpart = x[index .+ (1:d)]
        @test y[i] == transform(t, xpart)
        ypart, ljpart = transform_and_logjac(t, xpart)
        @test ypart == y[i]
        ljacc += ljpart
        index += d
    end
    y2, lj2 = transform_and_logjac(tn, x)
    @test y == y2
    @test lj2 ≈ ljacc
end
