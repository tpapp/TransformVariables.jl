# Coverage submission script on travis-ci.com, delete or keep depending on whether you are using that service.
# Only pushes coverage from the given version and OS.
get(ENV, "TRAVIS_OS_NAME", nothing)       == "linux" || exit(0)
get(ENV, "TRAVIS_JULIA_VERSION", nothing) == "1.5"   || exit(0)

using Coverage

cd(joinpath(@__DIR__, "..", "..")) do
    Codecov.submit(Codecov.process_folder())
end
