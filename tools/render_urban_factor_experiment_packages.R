#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(rmarkdown)
})

args_in <- commandArgs(trailingOnly = TRUE)
arg_value <- function(flag, default) {
  hit <- which(args_in == flag)
  if (length(hit) == 0 || hit[length(hit)] >= length(args_in)) default else args_in[hit[length(hit)] + 1]
}

results_root <- arg_value("--results_root", "results/urban_factor_experiments")
results_root_abs <- normalizePath(results_root, mustWork = FALSE)
Sys.setenv(URBAN_FACTOR_RESULTS_ROOT = results_root_abs)

input <- "reports/urban_factor_experiment_packages.Rmd"
output_dir <- "reports"
md_name <- "urban_factor_experiment_packages.md"
pdf_name <- "urban_factor_experiment_packages.pdf"
pdf_abs <- normalizePath(file.path(output_dir, pdf_name), mustWork = FALSE)

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(results_root_abs, "report"), recursive = TRUE, showWarnings = FALSE)

md_path <- rmarkdown::render(
  input = input,
  output_format = rmarkdown::md_document(variant = "markdown", preserve_yaml = FALSE),
  output_file = md_name,
  output_dir = output_dir,
  quiet = FALSE
)

old_wd <- getwd()
on.exit(setwd(old_wd), add = TRUE)
setwd(output_dir)

args <- c(
  md_name,
  "-o",
  pdf_name,
  "--pdf-engine=typst",
  "--metadata",
  "lang=zh-CN",
  "-V",
  "mainfont=Noto Sans CJK SC",
  "-V",
  "monofont=Noto Sans Mono CJK SC",
  "-V",
  "fontsize=10pt",
  "-V",
  "papersize=a4"
)

cmd <- paste("pandoc", paste(shQuote(args), collapse = " "))
status <- system(cmd)
if (!identical(status, 0L)) {
  stop("pandoc PDF export failed with status: ", status)
}

setwd(old_wd)
file.copy(file.path(output_dir, md_name), file.path(results_root_abs, "report", md_name), overwrite = TRUE)
file.copy(file.path(output_dir, pdf_name), file.path(results_root_abs, "report", pdf_name), overwrite = TRUE)

message("Wrote ", md_path)
message("Wrote ", normalizePath(pdf_abs, mustWork = TRUE))
message("Copied report outputs to ", file.path(results_root_abs, "report"))
