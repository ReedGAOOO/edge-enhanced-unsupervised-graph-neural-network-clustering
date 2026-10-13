#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(patchwork)
  library(scales)
})

args <- commandArgs(trailingOnly = TRUE)
arg_value <- function(flag, default) {
  hit <- which(args == flag)
  if (length(hit) == 0 || hit[length(hit)] >= length(args)) default else args[hit[length(hit)] + 1]
}

results_root <- arg_value("--results_root", "results/urban_data_paper")
fig_dir <- file.path(results_root, "figures")
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)

read_optional <- function(path) {
  if (!file.exists(path) || file.size(path) == 0) return(tibble())
  suppressWarnings(read_csv(path, show_col_types = FALSE))
}

save_both <- function(plot, name, width = 12, height = 8) {
  ggsave(file.path(fig_dir, paste0(name, ".png")), plot, width = width, height = height, dpi = 220, bg = "white")
  ggsave(file.path(fig_dir, paste0(name, ".pdf")), plot, width = width, height = height, bg = "white")
}

theme_data <- function() {
  theme_minimal(base_size = 10) +
    theme(
      panel.grid.minor = element_blank(),
      plot.title = element_text(face = "bold", size = 12),
      axis.text.x = element_text(angle = 35, hjust = 1),
      legend.position = "bottom"
    )
}

placeholder <- function(title) {
  ggplot(tibble(x = 0, y = 0), aes(x, y)) +
    geom_text(aes(label = "No data available"), size = 5) +
    labs(title = title, x = NULL, y = NULL) +
    theme_void()
}

tile <- read_optional(file.path(results_root, "tile_feature_table.csv"))
scores <- read_optional(file.path(results_root, "tile_anomaly_scores.csv"))
typology <- read_optional(file.path(results_root, "city_typology_summary.csv"))
weak <- read_optional(file.path(results_root, "weak_label_validation.csv"))
baseline <- read_optional(file.path(results_root, "city_baseline_profile.csv"))

detectors <- c(
  "relation_mix_outlier_score",
  "residual_edge_outlier_score",
  "geometry_confound_score",
  "graph_form_mismatch_score"
)

p21 <- if (nrow(scores) > 0) {
  scores |>
    select(city_label, all_of(detectors)) |>
    pivot_longer(all_of(detectors), names_to = "detector", values_to = "score") |>
    group_by(city_label, detector) |>
    summarise(score_p95 = quantile(score, 0.95, na.rm = TRUE), .groups = "drop") |>
    mutate(detector = recode(
      detector,
      relation_mix_outlier_score = "Relation mix",
      residual_edge_outlier_score = "Residual edge",
      geometry_confound_score = "Geometry confound",
      graph_form_mismatch_score = "Graph-form mismatch"
    )) |>
    ggplot(aes(detector, reorder(city_label, score_p95), fill = score_p95)) +
    geom_tile(color = "white", linewidth = 0.25) +
    geom_text(aes(label = number(score_p95, accuracy = 0.01)), size = 2.6) +
    scale_fill_gradient(low = "white", high = "#7b3294") +
    labs(title = "Figure 21. All-city tile anomaly heatmap", x = NULL, y = NULL, fill = "P95 score") +
    theme_data()
} else placeholder("Figure 21. All-city tile anomaly heatmap")
save_both(p21, "fig_21_all_city_tile_anomaly_heatmap", 12, 8)

p22 <- if (nrow(typology) > 0) {
  typology |>
    select(city_label, city_typology, relation_mix_outlier_p95, residual_edge_outlier_p95, geometry_confound_p95, graph_form_mismatch_p95) |>
    pivot_longer(ends_with("_p95"), names_to = "profile", values_to = "score") |>
    mutate(profile = recode(
      profile,
      relation_mix_outlier_p95 = "Relation mix",
      residual_edge_outlier_p95 = "Residual edge",
      geometry_confound_p95 = "Geometry",
      graph_form_mismatch_p95 = "Form mismatch"
    )) |>
    ggplot(aes(profile, score, fill = city_typology)) +
    geom_col(width = 0.72) +
    facet_wrap(~city_label, scales = "free_y") +
    labs(title = "Figure 22. City-level detector profile", x = NULL, y = "P95 detector score", fill = "Typology") +
    theme_data()
} else placeholder("Figure 22. City-level detector profile")
save_both(p22, "fig_22_city_detector_profile", 13, 9)

p23 <- if (nrow(scores) > 0) {
  top_rel <- scores |>
    group_by(city_label) |>
    slice_max(relation_mix_outlier_score, n = 40, with_ties = FALSE) |>
    ungroup()
  ggplot(top_rel, aes(lon_mean, lat_mean, color = relation_mix_outlier_score, size = n_nodes)) +
    geom_point(alpha = 0.82) +
    scale_color_viridis_c(option = "magma") +
    scale_size_area(max_size = 4) +
    facet_wrap(~city_label, scales = "free") +
    labs(title = "Figure 23. Relation-mix anomaly map", x = "Longitude", y = "Latitude", color = "Score", size = "Nodes") +
    theme_data()
} else placeholder("Figure 23. Relation-mix anomaly map")
save_both(p23, "fig_23_relation_mix_anomaly_map", 13, 9)

p24 <- if (nrow(scores) > 0) {
  top_res <- scores |>
    group_by(city_label) |>
    slice_max(residual_edge_outlier_score, n = 40, with_ties = FALSE) |>
    ungroup()
  ggplot(top_res, aes(lon_mean, lat_mean, color = edge_residual_z_mean, size = residual_edge_outlier_score)) +
    geom_point(alpha = 0.82) +
    scale_color_gradient2(low = "#b2182b", mid = "white", high = "#2166ac", midpoint = 0) +
    scale_size_area(max_size = 4) +
    facet_wrap(~city_label, scales = "free") +
    labs(title = "Figure 24. Residual graph anomaly map", x = "Longitude", y = "Latitude", color = "Residual z", size = "Score") +
    theme_data()
} else placeholder("Figure 24. Residual graph anomaly map")
save_both(p24, "fig_24_residual_graph_anomaly_map", 13, 9)

p25 <- if (nrow(scores) > 0) {
  scores |>
    group_by(city_label) |>
    summarise(
      geometry = mean(geom_edge_share, na.rm = TRUE),
      semantic = mean(semantic_edge_share, na.rm = TRUE),
      residual = quantile(residual_edge_outlier_score, 0.95, na.rm = TRUE),
      .groups = "drop"
    ) |>
    ggplot(aes(geometry, semantic, size = residual, label = city_label)) +
    geom_abline(slope = 1, intercept = 0, linetype = 2, color = "grey70") +
    geom_point(color = "#2166ac", alpha = 0.85) +
    geom_text(nudge_y = 0.015, size = 3, check_overlap = TRUE) +
    scale_size_area(max_size = 7) +
    labs(title = "Figure 25. Geometry-vs-semantic mechanism matrix", x = "Mean geometry edge share", y = "Mean semantic edge share", size = "Residual P95") +
    theme_data()
} else placeholder("Figure 25. Geometry-vs-semantic mechanism matrix")
save_both(p25, "fig_25_geometry_vs_semantic_matrix", 10, 7)

p26 <- if (nrow(weak) > 0) {
  weak |>
    mutate(
      detector = recode(
        detector,
        combined_anomaly_score = "Combined",
        relation_mix_outlier_score = "Relation mix",
        residual_edge_outlier_score = "Residual edge",
        geometry_confound_score = "Geometry",
        graph_form_mismatch_score = "Form mismatch"
      )
    ) |>
    ggplot(aes(weak_label_metric, detector, fill = spearman_rho)) +
    geom_tile(color = "white", linewidth = 0.25) +
    geom_text(aes(label = number(spearman_rho, accuracy = 0.01)), size = 2.4) +
    scale_fill_gradient2(low = "#2166ac", mid = "white", high = "#b2182b", midpoint = 0, limits = c(-1, 1), oob = squish) +
    facet_wrap(~city, ncol = 4) +
    labs(title = "Figure 26. Weak-label validation panel", x = NULL, y = NULL, fill = "Spearman") +
    theme_data()
} else placeholder("Figure 26. Weak-label validation panel")
save_both(p26, "fig_26_weak_label_validation_panel", 13, 9)

p27 <- if (nrow(typology) > 0) {
  typology |>
    select(city_label, city_typology, known_label_ratio_mean, semantic_edge_share_mean, geom_edge_share_mean, combined_anomaly_p95) |>
    pivot_longer(c(known_label_ratio_mean, semantic_edge_share_mean, geom_edge_share_mean, combined_anomaly_p95), names_to = "metric", values_to = "value") |>
    mutate(metric = recode(
      metric,
      known_label_ratio_mean = "Known label",
      semantic_edge_share_mean = "Semantic edge",
      geom_edge_share_mean = "Geometry edge",
      combined_anomaly_p95 = "Anomaly P95"
    )) |>
    ggplot(aes(metric, reorder(city_label, value), fill = value)) +
    geom_tile(color = "white", linewidth = 0.25) +
    geom_text(aes(label = number(value, accuracy = 0.01)), size = 2.4) +
    scale_fill_gradient(low = "white", high = "#008837") +
    facet_grid(. ~ city_typology, scales = "free_x", space = "free_x") +
    labs(title = "Figure 27. Cross-city typology matrix", x = NULL, y = NULL, fill = "Value") +
    theme_data()
} else placeholder("Figure 27. Cross-city typology matrix")
save_both(p27, "fig_27_cross_city_typology_matrix", 14, 8)

cat("Urban data-paper figures written to: ", fig_dir, "\n", sep = "")
