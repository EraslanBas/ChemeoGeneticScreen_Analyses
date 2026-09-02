#!/usr/bin/env Rscript

suppressPackageStartupMessages(library(ashr))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) stop("Usage: run_ashr_on_chunk.R chunk_XXXXXX.csv")

infile <- args[1]
outfile <- sub("chunk_", "AshrResult_chunk_", infile)

a <- read.csv(infile)
myRes <- ash(a$mean_diff, a$se)
write.csv(myRes$result, outfile, row.names = FALSE)
