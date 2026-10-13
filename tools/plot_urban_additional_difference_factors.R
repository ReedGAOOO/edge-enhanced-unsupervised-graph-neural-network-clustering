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

factors <- read_csv(file.path(base_dir, "additional_city_difference_factors.csv"), show_col_types = FALSE)
priority <- read_csv(file.path(base_dir, "all_city_exploration_priority.csv"), show_col_types = FALSE) |>
  select(city, edge_probe_gain, edge_probe_perm_drop, exploration_priority, priority_group)
typology <- read_csv(file.path(base_dir, "city_difference_city_typology.csv"), show_col_types = FALSE) |>
  select(city, typology)

df <- factors |>
  left_join(priority, by = "city") |>
  left_join(typology, by = "city")

type_palette <- c(
  "anchor candidate" = "#008837",
  "edge-rich but label-limited" = "#7B3294",
  "label-rich but edge-weak" = "#4393C3",
  "projection-risk diagnostic" = "#D6604D",
  "secondary diagnostic" = "#FDAE61",
  "low-priority or negative control" = "#999999"
)

base_theme <- theme_minimal(base_size = 9) +
  theme(
    panel.grid.minor = element_blank(),
    plot.title = element_text(face = "bold", size = 11),
    plot.subtitle = element_text(size = 8.5, color = "grey30")
  )

p_lift <- ggplot(
  df,
  aes(same_label_lift_vs_node_baseline, edge_probe_gain, color = typology, size = known_label_ratio)
) +
  geom_hline(yintercept = 0, color = "grey70", linewidth = 0.25) +
  geom_point(alpha = 0.9) +
  geom_text_repel(aes(label = city_label), size = 2.6, max.overlaps = Inf) +
  scale_x_continuous(labels = label_number(accuracy = 0.01)) +
  scale_y_continuous(labels = label_number(accuracy = 0.001)) +
  scale_color_manual(values = type_palette, drop = FALSE) +
  scale_size_area(max_size = 6, labels = percent_format(accuracy = 1)) +
  labs(
    title = "A. Homophily lift is not the whole story",
    subtitle = "Same-label lift is positive in all cities, but it does not rank edge_attr usefulness.",
    x = "Observed same-label minus node-label baseline",
    y = "AUC gain vs endpoint-only",
    color = "City type",
    size = "Known labels"
  ) +
  base_theme +
  theme(legend.position = "none")

p_mi <- ggplot(
  df,
  aes(edge_label_mi_norm, heterophily_ratio, color = typology, size = top_cross_label_pair_share)
) +
  geom_point(alpha = 0.9) +
  geom_text_repel(aes(label = city_label), size = 2.6, max.overlaps = Inf) +
  scale_x_continuous(labels = label_number(accuracy = 0.01)) +
  scale_y_continuous(labels = percent_format(accuracy = 1)) +
  scale_color_manual(values = type_palette, drop = FALSE) +
  scale_size_area(max_size = 6, labels = percent_format(accuracy = 1)) +
  labs(
    title = "B. Heterophily can still be semantic",
    subtitle = "Commercial-Food and Food-Recreational pairs dominate cross-label edges in many cities.",
    x = "Normalized edge label-pair MI",
    y = "Heterophilous known-edge share",
    color = "City type",
    size = "Top cross-pair share"
  ) +
  base_theme +
  theme(legend.position = "none")

p_confound <- ggplot(
  df,
  aes(edge_attr_endpoint_r2_nonrelation, edge_probe_gain, color = typology, size = edge_probe_perm_drop)
) +
  geom_hline(yintercept = 0, color = "grey70", linewidth = 0.25) +
  geom_point(alpha = 0.9) +
  geom_text_repel(aes(label = city_label), size = 2.6, max.overlaps = Inf) +
  scale_x_continuous(labels = label_number(accuracy = 0.01)) +
  scale_y_continuous(labels = label_number(accuracy = 0.001)) +
  scale_color_manual(values = type_palette, drop = FALSE) +
  scale_size_area(max_size = 6, labels = label_number(accuracy = 0.001)) +
  labs(
    title = "C. Edge attributes can re-encode endpoint morphology",
    subtitle = "High R2 means stronger controls are needed before claiming independent edge semantics.",
    x = "R2: predict non-relation edge_attr from endpoint features",
    y = "AUC gain vs endpoint-only",
    color = "City type",
    size = "Permutation drop"
  ) +
  base_theme +
  theme(legend.position = "none")

p_spatial <- ggplot(
  df,
  aes(distance_only_same_label_auc, node_similarity_same_label_auc, color = typology, size = degree_label_eta2)
) +
  geom_hline(yintercept = 0.5, color = "grey70", linewidth = 0.25) +
  geom_vline(xintercept = 0.5, color = "grey70", linewidth = 0.25) +
  geom_point(alpha = 0.9) +
  geom_text_repel(aes(label = city_label), size = 2.6, max.overlaps = Inf) +
  scale_x_continuous(labels = label_number(accuracy = 0.01)) +
  scale_y_continuous(labels = label_number(accuracy = 0.01)) +
  scale_color_manual(values = type_palette, drop = FALSE) +
  scale_size_area(max_size = 6, labels = label_number(accuracy = 0.01)) +
  labs(
    title = "D. Spatial and topology controls",
    subtitle = "Distance and endpoint similarity are modest but non-negligible controls.",
    x = "Distance-only same-label AUC",
    y = "Node-similarity same-label AUC",
    color = "City type",
    size = "Degree-label eta2"
  ) +
  base_theme +
  theme(legend.position = "none")

fig_10 <- (p_lift | p_mi) / (p_confound | p_spatial) +
  plot_annotation(
    title = "Additional explanations beyond source, projection, and coarse urban form",
    subtitle = "Second-order diagnostics show task mismatch, baseline inflation, endpoint redundancy, spatial confounding, and degree effects.",
    theme = theme(plot.title = element_text(face = "bold", size = 14))
  )

png_path <- file.path(out_dir, "fig_10_additional_difference_factors.png")
pdf_path <- file.path(out_dir, "fig_10_additional_difference_factors.pdf")
ggsave(png_path, fig_10, width = 13, height = 10, dpi = 320, bg = "white")
ggsave(pdf_path, fig_10, width = 13, height = 10, bg = "white")
message("Wrote ", png_path)
