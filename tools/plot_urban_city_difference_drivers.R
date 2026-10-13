#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(ggrepel)
  library(patchwork)
  library(readr)
  library(scales)
  library(tidyr)
})

base_dir <- "results/data_construction_series/urban_all_city_exploration"
out_dir <- file.path(base_dir, "figures")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

driver_frame <- read_csv(file.path(base_dir, "city_difference_driver_frame.csv"), show_col_types = FALSE)
correlations <- read_csv(file.path(base_dir, "city_difference_driver_correlations.csv"), show_col_types = FALSE)
city_types <- read_csv(file.path(base_dir, "city_difference_city_typology.csv"), show_col_types = FALSE)

group_palette <- c(
  "data_label" = "#4E79A7",
  "data_source" = "#9C755F",
  "graph_projection" = "#F28E2B",
  "urban_form_proxy" = "#59A14F"
)

type_palette <- c(
  "anchor candidate" = "#008837",
  "edge-rich but label-limited" = "#7B3294",
  "label-rich but edge-weak" = "#4393C3",
  "projection-risk diagnostic" = "#D6604D",
  "secondary diagnostic" = "#FDAE61",
  "low-priority or negative control" = "#999999"
)

city_df <- driver_frame |>
  left_join(city_types |> select(city, typology), by = "city")

top_drivers <- correlations |>
  filter(target %in% c("edge_probe_gain", "edge_probe_perm_drop", "overall_same_label_ratio")) |>
  group_by(driver, driver_label, driver_group) |>
  summarize(max_abs_rho = max(abs_spearman_rho, na.rm = TRUE), .groups = "drop") |>
  arrange(desc(max_abs_rho)) |>
  slice_head(n = 14)

heat_data <- correlations |>
  semi_join(top_drivers |> select(driver), by = "driver") |>
  filter(target %in% c("edge_probe_gain", "edge_probe_perm_drop", "overall_same_label_ratio")) |>
  mutate(
    target = recode(
      target,
      "edge_probe_gain" = "Edge gain",
      "edge_probe_perm_drop" = "Permutation drop",
      "overall_same_label_ratio" = "Same-label ratio"
    ),
    driver_label = factor(driver_label, levels = rev(top_drivers$driver_label))
  )

p_corr <- ggplot(heat_data, aes(target, driver_label, fill = spearman_rho)) +
  geom_tile(color = "white", linewidth = 0.35) +
  geom_text(aes(label = number(spearman_rho, accuracy = 0.01)), size = 2.5) +
  scale_fill_gradient2(
    low = "#2166AC",
    mid = "white",
    high = "#B2182B",
    midpoint = 0,
    limits = c(-1, 1),
    name = "Spearman"
  ) +
  labs(
    title = "A. Driver associations are mixed, not one-cause",
    subtitle = "n = 13 cities; correlations are diagnostic screens, not causal estimates.",
    x = NULL,
    y = NULL
  ) +
  theme_minimal(base_size = 9) +
  theme(panel.grid = element_blank())

p_building_tail <- ggplot(
  city_df,
  aes(building_area_tail, edge_probe_perm_drop, color = typology, size = known_label_ratio)
) +
  geom_hline(yintercept = 0, color = "grey70", linewidth = 0.25) +
  geom_point(alpha = 0.9) +
  geom_text_repel(aes(label = city_label), size = 2.6, max.overlaps = Inf) +
  scale_x_log10() +
  scale_y_continuous(labels = label_number(accuracy = 0.001)) +
  scale_size_area(max_size = 6, labels = percent_format(accuracy = 1)) +
  scale_color_manual(values = type_palette, drop = FALSE) +
  labs(
    title = "B. Urban-form proxy: building footprint heterogeneity",
    subtitle = "The strongest observed association is with edge_attr permutation sensitivity.",
    x = "Building area p99 / p50, log scale",
    y = "AUC drop after edge_attr permutation",
    color = "City type",
    size = "Known labels"
  ) +
  theme_minimal(base_size = 9) +
  theme(legend.position = "none")

p_projection <- ggplot(
  city_df,
  aes(projection_tail_score, edge_probe_gain, color = typology, size = known_label_ratio)
) +
  geom_hline(yintercept = 0, color = "grey70", linewidth = 0.25) +
  geom_point(alpha = 0.9) +
  geom_text_repel(aes(label = city_label), size = 2.6, max.overlaps = Inf) +
  scale_y_continuous(labels = label_number(accuracy = 0.001)) +
  scale_size_area(max_size = 6, labels = percent_format(accuracy = 1)) +
  scale_color_manual(values = type_palette, drop = FALSE) +
  labs(
    title = "C. Graph-construction proxy: projection-tail risk",
    subtitle = "High-tail cities can reflect projection artifacts as much as urban semantics.",
    x = "Projection-tail score",
    y = "AUC gain vs endpoint-only",
    color = "City type",
    size = "Known labels"
  ) +
  theme_minimal(base_size = 9) +
  theme(legend.position = "none")

type_counts <- city_types |>
  mutate(typology = factor(typology, levels = names(type_palette))) |>
  count(typology)

p_types <- ggplot(type_counts, aes(n, typology, fill = typology)) +
  geom_col(width = 0.7) +
  scale_fill_manual(values = type_palette, drop = FALSE) +
  scale_x_continuous(breaks = pretty_breaks()) +
  labs(
    title = "D. Practical city typology",
    subtitle = "Counts by diagnostic class.",
    x = "Number of cities",
    y = NULL,
    fill = "City type"
  ) +
  theme_minimal(base_size = 9) +
  theme(legend.position = "none")

fig_09 <- (p_corr | p_building_tail) / (p_projection | p_types) +
  plot_annotation(
    title = "Why do Urban graphs differ across cities?",
    subtitle = "The current evidence supports layered explanations: data/label quality, graph-projection design, and observable urban-form proxies.",
    theme = theme(plot.title = element_text(face = "bold", size = 14))
  )

png_path <- file.path(out_dir, "fig_09_city_difference_driver_audit.png")
pdf_path <- file.path(out_dir, "fig_09_city_difference_driver_audit.pdf")
ggsave(png_path, fig_09, width = 13, height = 10, dpi = 320, bg = "white")
ggsave(pdf_path, fig_09, width = 13, height = 10, bg = "white")
message("Wrote ", png_path)
