
# Generate zero-mean, unit-variance spatially correlated random fields

rm(list = ls())

require(geoR, lib.loc="/pool0/data/yixinmao/Rpackages/")

args = commandArgs(trailingOnly=TRUE)
start_time_ind <- strtoi(args[1])
end_time_ind <- strtoi(args[2])

set.seed(234 + start_time_ind)  # ensure a different random number when starting from a different time

# --- Set parameters --- #
nx <- 125
ny <- 66
xlims <- c(1, 125) # the coords don't really matter,
                  # as long as phi matches them
ylims <- c(1, 66)
phi <- 12  # phi here needs to match coords

N <- 32 # ensemble member
ntile <- 12
nlayer <- 3
outdir <- "/pool0/data/yixinmao/data_assim/tools/prepare_perturbation/output/R_arrays/phi12.N32"


# --- 1) Generate an ensemble of 2D random fields --- #
# --- with unit variance and phi, for each time step --- #
# --- 2) Generate 2D fields for all layers/tiles that are spatially
# autocorrelated and also mutually correlated --- #
for (t in seq(start_time_ind, end_time_ind)) {
    # Generate ntile*nlayer of 2D fields
    sim <- grf(n=nx*ny, grid="reg", nx=nx, ny=ny,
                  xlims=xlims,
                  ylims=ylims,
                  cov.model="exponential",
                  cov.pars=c(1, phi),
                  mean=0,
                  nsim=ntile*nlayer*N)

#    image(sim.my, col  = gray((0:32)/32))
    # Save to file
    saveRDS(sim$data, file=paste(outdir, "/time", t, ".Rds", sep=""))
}

