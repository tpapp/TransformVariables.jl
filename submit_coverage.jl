using Pkg
pkg"add Coverage"
using Coverage
# push coverage results to Coveralls
Coveralls.submit(Coveralls.process_folder())
# push coverage results to Codecov
Codecov.submit(Codecov.process_folder())
