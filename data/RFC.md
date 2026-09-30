# Data management & quality control approach

The intent is to identify objectives of data management and control quality of
the data in a unified, reproducible way.

This document is called RFC not because it is a formal *request for comments*, 
but because it is a literal **request for comments**. A discussion is expected
to start here.

## Objectives

What data we should manage:
- Experiment data: matrices, right-hand side vectors, other useful data
- Experiment logs: configuration, parameters, basic run results
- GPU performance metrics (preferably on each CUDA kernel invoked)
- CPU performance metrics
- Code quality metrics (e.g., code coverage, static analysis, etc.)

What basic quality control requirements we may enforce:
- Experiment data:
        - Data should be taken from a trusted source or generated
        - If generated, the generation process should be reproducible
        - If taken from a trusted source, the source should be cited
        - Single source of truth for each experiment batch should be maintained
- Experiment logs:
        - Experiments for papers should have log files
        - Log files should be stored in a structured way
        - Every experiment should be perfectly reproducible from its log files
- Performance metrics quality:
		- List of perfrormance metrics to collect should be defined and 
enforced
        - Check for outliers or unexpected values in the metrics
        - Check for consistency across multiple runs of the same experiment
        - Check for platform-specific causes affecting performance metrics
(e.g., GPU warm-up, CPU throttling)
        - Keep in mind that profiling tools may not be perfectly accurate
        - Keep in mind that profiling tools may have their own overhead
- Code quality metrics quality (xd):
	- Metrics collection should be automated
	- Metrics collection should be reproducible
	- Metrics collection should be crossplatform
	- Metrics should be collected in a structured way
	- Standards should be defined and enforced
	- Standards should be based on the ones used in the industry (e.g., 
CUDA quality guidelines, etc.)
	- Standards should be reasonable for our use cases
---

## Architecture

Here is my proposal for the possible architecture patterns for achieving 
objectives above.
They are not full, not complete, and not final. They are just a starting point 
for discussion.

### Storing experiment data

How we may store experiment data:
- Single source of truth: an SFTP server on a DELL workstation
- Data format: text files, triplets, Matrix Market
- Metadata: each experiment data file should have a corresponding metadata
file. For example, matrix metadata file should store information about their
size, sparsity, source (for non-generated matrices), and generation parameters
(for generated matrices).

### Storing experiment logs
How we may store experiment logs:
- An experiment log is a JSON file containing all the parameters and
configuration of the experiment, as well as basic results (e.g., convergence,
time taken, number of iterations).
- An experiment log file is local to the experimental workstation and is
created in-place

### Collecting and storing performance metrics
- Performance metrics are collected using profiling tools (e.g., NVIDIA
Nsight Systems, NVIDIA Nsight Compute) and stored in a
structured (preferably text) format  for further analysis.

### Collecting and storing code quality metrics
- Static analysis tools are used (preferably clang-tidy)
- Sanitizer tools are used (both for host code and for device code)
- A test suite is created, code coverage is collected


P.S. `data` folder is now not in .gitignore - do not store data (e.g. matrices, 
vectors) here. Find yourself another folder, like `dataset`, `temp`, 
`experiments`, `sandbox`. We should probably define "playground" folder name 
later.
