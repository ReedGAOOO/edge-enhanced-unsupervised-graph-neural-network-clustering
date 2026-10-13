#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(tidyr)
  library(ggplot2)
  library(scales)
})

args <- commandArgs(trailingOnly = TRUE)
arg_value <- function(flag, default) {
  hit <- which(args == flag)
  if (length(hit) == 0 || hit[length(hit)] >= length(args)) default else args[hit[length(hit)] + 1]
}

results_root <- arg_value("--results_root", "results/urban_data_paper")
top_n <- as.integer(arg_value("--top_n", "25"))
fig_dir <- file.path(results_root, "figures", "geom_fallback_spikes")
dir.create(fig_dir, recursive = TRUE, showWarnings = FALSE)

scores_path <- file.path(results_root, "tile_anomaly_scores.csv")
baseline_path <- file.path(results_root, "city_baseline_profile.csv")
if (!file.exists(scores_path) || !file.exists(baseline_path)) {
  stop("Required CSV files are missing under results_root.")
}

scores <- read_csv(scores_path, show_col_types = FALSE)
baseline <- read_csv(baseline_path, show_col_types = FALSE) |>
  select(city, city_label, global_geom_edge_share = geom_edge_share)

focus_cities <- c("bangkok", "beijing", "shanghai", "singapore", "sydney")

theme_spike <- function() {
  theme_minimal(base_size = 10) +
    theme(
      panel.grid.minor = element_blank(),
      plot.title = element_text(face = "bold", size = 12),
      plot.subtitle = element_text(size = 10),
      axis.text.x = element_text(angle = 35, hjust = 1),
      legend.position = "bottom"
    )
}

save_both <- function(plot, name, width = 12, height = 8) {
  ggsave(file.path(fig_dir, paste0(name, ".png")), plot, width = width, height = height, dpi = 220, bg = "white")
  ggsave(file.path(fig_dir, paste0(name, ".pdf")), plot, width = width, height = height, bg = "white")
}

top_tiles <- scores |>
  group_by(city) |>
  slice_min(city_anomaly_rank, n = top_n, with_ties = FALSE) |>
  ungroup()

summary <- scores |>
  group_by(city, city_label) |>
  summarise(
    n_tiles = n(),
    all_tile_geom_mean = mean(geom_edge_share, na.rm = TRUE),
    all_tile_geom_p95 = quantile(geom_edge_share, 0.95, na.rm = TRUE),
    all_tile_geom_max = max(geom_edge_share, na.rm = TRUE),
    .groups = "drop"
  ) |>
  left_join(
    top_tiles |>
      group_by(city) |>
      summarise(
        top_n = n(),
        top_anomaly_geom_mean = mean(geom_edge_share, na.rm = TRUE),
        top_anomaly_geom_p95 = quantile(geom_edge_share, 0.95, na.rm = TRUE),
        top_anomaly_geom_max = max(geom_edge_share, na.rm = TRUE),
        top_anomaly_semantic_mean = mean(semantic_edge_share, na.rm = TRUE),
        top_anomaly_nodes_median = median(n_nodes, na.rm = TRUE),
        top_anomaly_known_label_mean = mean(known_label_ratio, na.rm = TRUE),
        .groups = "drop"
      ),
    by = "city"
  ) |>
  left_join(baseline, by = c("city", "city_label")) |>
  mutate(
    spike_gap_vs_global = top_anomaly_geom_max - global_geom_edge_share,
    spike_ratio_vs_global = (top_anomaly_geom_max + 1e-6) / (global_geom_edge_share + 1e-6),
    city_label = factor(city_label, levels = city_label[order(top_anomaly_geom_max)])
  )

write_csv(summary, file.path(results_root, "geom_fallback_spike_summary.csv"))
write_csv(
  top_tiles |>
    arrange(city, city_anomaly_rank) |>
    select(
      city, city_label, tile_id, city_anomaly_rank, tile_typology,
      combined_anomaly_score, geom_edge_share, semantic_edge_share,
      relation_mix_outlier_score, residual_edge_outlier_score,
      geometry_confound_score, graph_form_mismatch_score,
      n_nodes, known_label_ratio, lon_mean, lat_mean
    ),
  file.path(results_root, "geom_fallback_top_anomaly_tiles.csv")
)

p28_data <- summary |>
  select(
    city, city_label, global_geom_edge_share,
    all_tile_geom_p95, top_anomaly_geom_mean, top_anomaly_geom_max
  ) |>
  pivot_longer(
    c(global_geom_edge_share, all_tile_geom_p95, top_anomaly_geom_mean, top_anomaly_geom_max),
    names_to = "metric",
    values_to = "geom_share"
  ) |>
  mutate(
    metric = recode(
      metric,
      global_geom_edge_share = "Global edge baseline",
      all_tile_geom_p95 = "All-tile P95",
      top_anomaly_geom_mean = paste0("Top-", top_n, " anomaly mean"),
      top_anomaly_geom_max = paste0("Top-", top_n, " anomaly max")
    ),
    metric = factor(
      metric,
      levels = c(
        "Global edge baseline",
        "All-tile P95",
        paste0("Top-", top_n, " anomaly mean"),
        paste0("Top-", top_n, " anomaly max")
      )
    )
  )

p28 <- ggplot(p28_data, aes(city_label, geom_share, color = metric, group = metric)) +
  geom_point(size = 2.1, alpha = 0.9) +
  geom_line(linewidth = 0.5, alpha = 0.6) +
  scale_y_continuous(labels = percent_format(accuracy = 1), limits = c(0, 1)) +
  scale_color_manual(values = c("#4d4d4d", "#2b8cbe", "#fdae61", "#d7191c")) +
  labs(
    title = "Figure 28. Global geometry baseline vs local anomaly-tile spikes",
    subtitle = "The global graph has low geometry fallback share, but top anomaly tiles can have much higher local geometry share.",
    x = NULL,
    y = "Geometry fallback edge share",
    color = NULL
  ) +
  theme_spike()
save_both(p28, "fig_28_global_vs_local_geom_fallback_spikes", 13, 7)

p29_data <- top_tiles |>
  filter(city %in% focus_cities) |>
  group_by(city) |>
  slice_min(city_anomaly_rank, n = 15, with_ties = FALSE) |>
  ungroup() |>
  left_join(baseline, by = c("city", "city_label")) |>
  mutate(
    rank_label = paste0("#", city_anomaly_rank),
    rank_label = factor(rank_label, levels = paste0("#", sort(unique(city_anomaly_rank)))),
    city_label = factor(city_label, levels = c("Bangkok", "Beijing", "Shanghai", "Singapore", "Sydney"))
  )

p29 <- ggplot(p29_data, aes(rank_label, geom_edge_share)) +
  geom_col(aes(fill = tile_typology), width = 0.7, alpha = 0.9) +
  geom_point(aes(y = semantic_edge_share), shape = 21, size = 1.9, fill = "white", color = "#2166ac", stroke = 0.6) +
  geom_hline(aes(yintercept = global_geom_edge_share), linetype = 2, linewidth = 0.35, color = "grey35") +
  facet_wrap(~city_label, ncol = 1) +
  scale_y_continuous(labels = percent_format(accuracy = 1), limits = c(0, 1)) +
  labs(
    title = "Figure 29. Top anomaly tiles where local geometry fallback spikes",
    subtitle = "Bars are local geometry share; blue dots are semantic edge share; dashed lines are city-level global geometry baseline.",
    x = "Anomaly rank within city",
    y = "Share",
    fill = "Tile typology"
  ) +
  theme_spike() +
  theme(axis.text.x = element_text(angle = 0, hjust = 0.5))
save_both(p29, "fig_29_focus_city_top_anomaly_geom_bars", 11, 12)

p30_data <- scores |>
  mutate(is_top_anomaly = city_anomaly_rank <= top_n) |>
  left_join(summary |> select(city, city_label_order = city_label), by = "city") |>
  mutate(city_label = city_label_order)

p30 <- ggplot(p30_data, aes(city_label, geom_edge_share)) +
  geom_boxplot(width = 0.55, outlier.shape = NA, fill = "grey90", color = "grey45", linewidth = 0.35) +
  geom_point(
    data = p30_data |> filter(is_top_anomaly),
    aes(color = combined_anomaly_score, size = n_nodes),
    alpha = 0.7,
    position = position_jitter(width = 0.16, height = 0, seed = 8)
  ) +
  scale_y_continuous(labels = percent_format(accuracy = 1), limits = c(0, 1)) +
  scale_color_gradient(low = "#fdae61", high = "#7f0000") +
  scale_size_area(max_size = 4, breaks = c(5, 20, 100, 300)) +
  labs(
    title = "Figure 30. Geometry fallback distribution with top anomaly tiles highlighted",
    subtitle = "Grey boxes show all tiles; colored points show top anomaly tiles. High points reveal local construction-rule spikes.",
    x = NULL,
    y = "Geometry fallback edge share",
    color = "Combined score",
    size = "Nodes"
  ) +
  theme_spike()
save_both(p30, "fig_30_geom_fallback_distribution_top_tiles", 13, 7)

p31_bg <- scores |>
  filter(city %in% focus_cities) |>
  mutate(city_label = factor(city_label, levels = c("Bangkok", "Beijing", "Shanghai", "Singapore", "Sydney")))

p31_top <- p31_bg |> filter(city_anomaly_rank <= top_n)

p31 <- ggplot() +
  geom_point(
    data = p31_bg,
    aes(lon_mean, lat_mean),
    color = "grey80",
    alpha = 0.25,
    size = 0.35
  ) +
  geom_point(
    data = p31_top,
    aes(lon_mean, lat_mean, color = geom_edge_share, size = combined_anomaly_score),
    alpha = 0.9
  ) +
  facet_wrap(~city_label, scales = "free", ncol = 2) +
  scale_color_gradient(low = "#fdae61", high = "#7f0000", labels = percent_format(accuracy = 1)) +
  scale_size_area(max_size = 4) +
  labs(
    title = "Figure 31. Spatial locations of high-geometry anomaly tiles",
    subtitle = "Grey points are all tiles; colored points are top anomaly tiles. Color encodes local geometry fallback share.",
    x = "Longitude",
    y = "Latitude",
    color = "Geom share",
    size = "Combined score"
  ) +
  theme_spike()
save_both(p31, "fig_31_spatial_geom_fallback_spikes", 12, 9)

cat("Geometry fallback spike figures written to: ", fig_dir, "\n", sep = "")
cat("Summary written to: ", file.path(results_root, "geom_fallback_spike_summary.csv"), "\n", sep = "")
