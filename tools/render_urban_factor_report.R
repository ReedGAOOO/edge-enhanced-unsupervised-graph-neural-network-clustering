#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(rmarkdown)
})

input <- "reports/urban_graph_factor_report.Rmd"
output_dir <- "reports"
md_name <- "urban_graph_factor_report.md"
pdf_name <- "urban_graph_factor_report.pdf"
pdf_abs <- normalizePath(file.path(output_dir, pdf_name), mustWork = FALSE)

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

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

message("Wrote ", md_path)
message("Wrote ", normalizePath(pdf_abs, mustWork = TRUE))
