#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(arrow)
  library(dplyr)
  library(ggplot2)
  library(ggrepel)
  library(jsonlite)
  library(patchwork)
  library(readr)
  library(scales)
  library(sf)
  library(tidyr)
  library(viridis)
})

sf_use_s2(FALSE)

base_dir <- "results/data_construction_series/urban_all_city_exploration"
out_dir <- file.path(base_dir, "figures")
urban_root <- "/mnt/e/urban network datasets"
v3b_diag_path <- "results/data_diagnostics/urban_v3bsjg/urban_v3bsjg_diagnostics.csv"
probe_dir <- "results/edge_probe/urban_v3bsjg_same_label_allcity_cpu_v1"
probe_summary_path <- file.path(probe_dir, "edge_probe_summary.csv")

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
theme_set(theme_minimal(base_size = 10))

landuse_palette <- c(
  "Civic" = "#4E79A7",
  "Commercial" = "#F28E2B",
  "Entertainment" = "#E15759",
  "Food" = "#76B7B2",
  "Healthcare" = "#59A14F",
  "Institutional" = "#EDC948",
  "Recreational" = "#B07AA1",
  "Social" = "#FF9DA7",
  "Unknown" = "#D0D0D0"
)

relation_palette <- c(
  "Street" = "#386CB0",
  "Junction" = "#F0027F",
  "Building" = "#4DAF4A",
  "Geometry" = "#FF7F00"
)

city_display <- function(x) {
  recode(
    x,
    "washingtondc" = "WashingtonDC",
    "tokyo" = "Tokyo",
    "paris" = "Paris",
    "boston" = "Boston",
    "madrid" = "Madrid",
    "beijing" = "Beijing",
    "shanghai" = "Shanghai",
    "singapore" = "Singapore",
    "melbourne" = "Melbourne",
    "sydney" = "Sydney",
    "bangkok" = "Bangkok",
    "chicago" = "Chicago",
    "johannesburg" = "Johannesburg",
    .default = tools::toTitleCase(x)
  )
}

save_figure <- function(plot, stem, width, height) {
  png_path <- file.path(out_dir, paste0(stem, ".png"))
  pdf_path <- file.path(out_dir, paste0(stem, ".pdf"))
  ggsave(png_path, plot, width = width, height = height, dpi = 320, bg = "white")
  ggsave(pdf_path, plot, width = width, height = height, bg = "white")
  message("Wrote ", png_path)
}

clean_panel_theme <- function() {
  theme(
    panel.grid.minor = element_blank(),
    plot.title = element_text(face = "bold", size = 11),
    plot.subtitle = element_text(size = 9, color = "grey30"),
    legend.title = element_text(size = 9),
    legend.text = element_text(size = 8)
  )
}

read_csv_quiet <- function(path) {
  readr::read_csv(path, show_col_types = FALSE)
}

zscore <- function(x) {
  x <- as.numeric(x)
  s <- sd(x, na.rm = TRUE)
  if (!is.finite(s) || s == 0) {
    return(rep(0, length(x)))
  }
  (x - mean(x, na.rm = TRUE)) / s
}

raw <- read_csv_quiet(file.path(base_dir, "raw_city_diagnostics.csv")) |>
  mutate(city_label = city_display(city))

labels <- read_csv_quiet(file.path(base_dir, "raw_label_distribution.csv")) |>
  mutate(
    city_label = city_display(city),
    label = factor(label, levels = names(landuse_palette))
  )

tails <- read_csv_quiet(file.path(base_dir, "raw_membership_tail_summary.csv")) |>
  mutate(
    city_label = city_display(city),
    metric_label = recode(
      metric,
      "plot_street_membership" = "Plot-street",
      "plot_junction_membership" = "Plot-junction",
      "plot_building_membership" = "Plot-building",
      "street_entity_plot_load" = "Street load",
      "junction_entity_plot_load" = "Junction load",
      "building_entity_plot_load" = "Building load"
    )
  )

v3b <- read_csv_quiet(v3b_diag_path) |>
  mutate(city_label = city_display(city))

probe_summary <- if (file.exists(probe_summary_path)) {
  read_csv_quiet(probe_summary_path)
} else {
  tibble(
    dataset = character(),
    feature_set = character(),
    auc_mean = numeric(),
    auc_gain_vs_endpoint_mean = numeric(),
    auc_drop_vs_perm_mean = numeric()
  )
}

probe_edge <- probe_summary |>
  filter(feature_set == "endpoint_edge_attr") |>
  mutate(city = sub("^urban_([^_]+)_.*$", "\\1", dataset)) |>
  select(
    city,
    edge_probe_auc = auc_mean,
    edge_probe_gain = auc_gain_vs_endpoint_mean,
    edge_probe_perm_drop = auc_drop_vs_perm_mean
  )

city_join <- raw |>
  left_join(v3b |> select(city, overall_same_label_ratio, street_edge_frac, junction_edge_frac, geom_edge_frac, largest_component_ratio), by = "city") |>
  left_join(probe_edge, by = "city") |>
  mutate(
    raw_tail_penalty = zscore(log1p(street_entity_plot_load_p99)) + zscore(log1p(junction_entity_plot_load_p99)) +
      zscore(log1p(plot_building_membership_p99)),
    exploration_priority = zscore(known_label_ratio) +
      zscore(known_label_entropy_norm) +
      zscore(overall_same_label_ratio) +
      zscore(coalesce(edge_probe_gain, 0)) +
      zscore(coalesce(edge_probe_perm_drop, 0)) -
      0.5 * raw_tail_penalty,
    priority_group = case_when(
      known_label_ratio >= 0.20 & coalesce(edge_probe_gain, 0) > 0.003 & coalesce(edge_probe_perm_drop, 0) > 0.003 ~ "strong next",
      known_label_ratio >= 0.12 & coalesce(edge_probe_perm_drop, 0) > 0 ~ "diagnostic",
      TRUE ~ "low priority"
    )
  )

layer_counts <- raw |>
  select(city_label, n_plots, n_streets, n_buildings, n_intersections) |>
  pivot_longer(-city_label, names_to = "layer", values_to = "count") |>
  mutate(
    layer = recode(
      layer,
      "n_plots" = "Plots",
      "n_streets" = "Streets",
      "n_buildings" = "Buildings",
      "n_intersections" = "Intersections"
    ),
    layer = factor(layer, levels = c("Plots", "Streets", "Buildings", "Intersections"))
  )

p_layer_counts <- ggplot(layer_counts, aes(reorder(city_label, count), count, fill = layer)) +
  geom_col(position = position_dodge(width = 0.75), width = 0.68) +
  coord_flip() +
  scale_y_log10(labels = label_number(scale_cut = cut_short_scale())) +
  scale_fill_manual(values = c("#4E79A7", "#F28E2B", "#59A14F", "#B07AA1")) +
  labs(
    title = "A. Raw layer scale differs sharply",
    subtitle = "Graph construction starts from very different numbers of plots, streets, buildings, and junctions.",
    x = NULL,
    y = "Raw entities, log scale",
    fill = "Layer"
  ) +
  clean_panel_theme()

p_label_quality <- ggplot(
  city_join,
  aes(known_label_ratio, known_label_entropy_norm, size = n_plots, color = largest_known_class_ratio)
) +
  geom_point(alpha = 0.9) +
  geom_text_repel(aes(label = city_label), size = 2.7, max.overlaps = Inf) +
  scale_x_continuous(labels = percent_format(accuracy = 1)) +
  scale_y_continuous(labels = label_number(accuracy = 0.01), limits = c(0, 1)) +
  scale_size_area(max_size = 8, labels = label_number(scale_cut = cut_short_scale())) +
  scale_color_viridis_c(option = "D", labels = percent_format(accuracy = 1), direction = -1) +
  labs(
    title = "B. Label coverage and class balance",
    subtitle = "High coverage is useful only when labels are not dominated by one class.",
    x = "Known-label ratio",
    y = "Normalized label entropy",
    size = "Plots",
    color = "Largest class"
  ) +
  clean_panel_theme()

tail_focus <- tails |>
  filter(metric %in% c("street_entity_plot_load", "junction_entity_plot_load", "plot_building_membership")) |>
  mutate(metric_label = factor(metric_label, levels = c("Street load", "Junction load", "Plot-building")))

p_tail_heat <- ggplot(tail_focus, aes(city_label, metric_label, fill = log1p(p99))) +
  geom_tile(color = "white", linewidth = 0.35) +
  geom_text(aes(label = number(p99, accuracy = 1)), size = 2.4) +
  scale_fill_viridis_c(option = "C", name = "log1p(p99)") +
  labs(
    title = "C. Raw membership tails",
    subtitle = "Large p99 loads create projection risk: one raw entity can connect many plot nodes.",
    x = NULL,
    y = NULL
  ) +
  theme(axis.text.x = element_text(angle = 35, hjust = 1)) +
  clean_panel_theme()

label_comp <- labels |>
  group_by(city_label, label) |>
  summarize(fraction_all = sum(fraction_all), .groups = "drop")

p_label_stack <- ggplot(label_comp, aes(reorder(city_label, fraction_all), fraction_all, fill = label)) +
  geom_col(width = 0.72) +
  coord_flip() +
  scale_y_continuous(labels = percent_format(accuracy = 1)) +
  scale_fill_manual(values = landuse_palette, drop = FALSE) +
  labs(
    title = "D. Raw label composition",
    subtitle = "Unknown labels and dominant POI categories shape the diagnostic target.",
    x = NULL,
    y = "Fraction of plots",
    fill = "Label"
  ) +
  clean_panel_theme()

fig_06 <- (p_layer_counts | p_label_quality) / (p_tail_heat | p_label_stack) +
  plot_annotation(
    title = "All-city raw Urban data exploration",
    subtitle = "Before graph learning, cities differ in scale, label availability, class balance, and projection-tail risk.",
    theme = theme(plot.title = element_text(face = "bold", size = 14))
  )
save_figure(fig_06, "fig_06_all_city_raw_data_diagnostics", 13, 10)

relation_mix <- v3b |>
  select(city_label, street_edge_frac, junction_edge_frac, geom_edge_frac, shared_building_bit_frac) |>
  pivot_longer(-city_label, names_to = "relation", values_to = "fraction") |>
  mutate(
    relation = recode(
      relation,
      "street_edge_frac" = "Street",
      "junction_edge_frac" = "Junction",
      "geom_edge_frac" = "Geometry",
      "shared_building_bit_frac" = "Building"
    ),
    relation = factor(relation, levels = c("Street", "Junction", "Building", "Geometry"))
  )

p_edge_probe <- ggplot(
  city_join,
  aes(edge_probe_gain, edge_probe_perm_drop, color = priority_group, size = known_label_ratio)
) +
  geom_hline(yintercept = 0, linewidth = 0.25, color = "grey70") +
  geom_vline(xintercept = 0, linewidth = 0.25, color = "grey70") +
  geom_point(alpha = 0.9) +
  geom_text_repel(aes(label = city_label), size = 2.7, max.overlaps = Inf) +
  scale_x_continuous(labels = label_number(accuracy = 0.001)) +
  scale_y_continuous(labels = label_number(accuracy = 0.001)) +
  scale_size_area(max_size = 6, labels = percent_format(accuracy = 1)) +
  scale_color_manual(values = c("strong next" = "#008837", "diagnostic" = "#FDAE61", "low priority" = "#D7191C")) +
  labs(
    title = "A. All-city edge probe screen",
    subtitle = "Right/top cities have edge_attr evidence beyond endpoint features and permutation.",
    x = "AUC gain vs endpoint-only",
    y = "AUC drop after edge_attr permutation",
    color = "Screen",
    size = "Known labels"
  ) +
  clean_panel_theme()

p_relation_mix <- ggplot(relation_mix, aes(city_label, relation, fill = fraction)) +
  geom_tile(color = "white", linewidth = 0.35) +
  geom_text(aes(label = percent(fraction, accuracy = 1)), size = 2.35) +
  scale_fill_viridis_c(option = "E", labels = percent_format(accuracy = 1)) +
  labs(
    title = "B. Existing v3b relation mix",
    subtitle = "Cities differ in whether graph edges are mostly street, junction, building, or geometry driven.",
    x = NULL,
    y = NULL,
    fill = "Share"
  ) +
  theme(axis.text.x = element_text(angle = 35, hjust = 1)) +
  clean_panel_theme()

relation_same <- v3b |>
  select(city_label, street_same_label_ratio, junction_same_label_ratio, shared_building_same_label_ratio, geom_same_label_ratio) |>
  pivot_longer(-city_label, names_to = "relation", values_to = "same_label_ratio") |>
  mutate(
    relation = recode(
      relation,
      "street_same_label_ratio" = "Street",
      "junction_same_label_ratio" = "Junction",
      "shared_building_same_label_ratio" = "Building",
      "geom_same_label_ratio" = "Geometry"
    ),
    relation = factor(relation, levels = c("Street", "Junction", "Building", "Geometry"))
  ) |>
  filter(!is.na(same_label_ratio))

p_relation_same <- ggplot(relation_same, aes(reorder(city_label, same_label_ratio), same_label_ratio, color = relation)) +
  geom_point(position = position_dodge(width = 0.55), size = 2.4, alpha = 0.9) +
  coord_flip() +
  scale_y_continuous(labels = percent_format(accuracy = 1)) +
  scale_color_manual(values = relation_palette, drop = FALSE) +
  labs(
    title = "C. Relation-specific label alignment",
    subtitle = "Same-label ratios show which relation types are label-compatible in each city.",
    x = NULL,
    y = "Same-label ratio",
    color = "Relation"
  ) +
  clean_panel_theme()

p_priority <- ggplot(
  city_join,
  aes(reorder(city_label, exploration_priority), exploration_priority, fill = priority_group)
) +
  geom_hline(yintercept = 0, linewidth = 0.25, color = "grey70") +
  geom_col(width = 0.7) +
  coord_flip() +
  scale_fill_manual(values = c("strong next" = "#008837", "diagnostic" = "#FDAE61", "low priority" = "#D7191C")) +
  labs(
    title = "D. Exploration priority",
    subtitle = "Prototype ranking for deciding which cities deserve V4 graph construction first.",
    x = NULL,
    y = "Relative priority score",
    fill = "Screen"
  ) +
  clean_panel_theme()

fig_07 <- (p_edge_probe | p_relation_mix) / (p_relation_same | p_priority) +
  plot_annotation(
    title = "All-city edge-readiness screen",
    subtitle = "Combine raw diagnostics, existing graph diagnostics, and a CPU edge probe to choose next cities.",
    theme = theme(plot.title = element_text(face = "bold", size = 14))
  )
save_figure(fig_07, "fig_07_all_city_edge_readiness_screen", 13, 10)

derive_labels <- function(df) {
  label_cols <- intersect(names(landuse_palette)[names(landuse_palette) != "Unknown"], names(df))
  if (!length(label_cols)) {
    return(rep("Unknown", nrow(df)))
  }
  scores <- as.matrix(df[, label_cols, drop = FALSE])
  scores[is.na(scores)] <- 0
  max_id <- max.col(scores, ties.method = "first")
  max_val <- scores[cbind(seq_len(nrow(scores)), max_id)]
  out <- label_cols[max_id]
  out[max_val <= 0] <- "Unknown"
  out
}

read_city_map_points <- function(city, max_points = 12000) {
  plot_path <- file.path(urban_root, city, "plot.parquet")
  raw_plot <- arrow::read_parquet(plot_path)
  raw_plot$label <- derive_labels(raw_plot)
  geom <- st_as_sfc(raw_plot$geometry, EWKB = FALSE)
  plot_sf <- st_sf(raw_plot |> select(label), geometry = geom, crs = 4326)
  pts <- suppressWarnings(st_point_on_surface(plot_sf))
  xy <- st_coordinates(pts)
  df <- tibble(
    city = city,
    city_label = city_display(city),
    label = factor(pts$label, levels = names(landuse_palette)),
    known = pts$label != "Unknown",
    x = xy[, 1],
    y = xy[, 2]
  )
  if (nrow(df) > max_points) {
    set.seed(1000 + match(city, sort(unique(raw$city))))
    known_df <- filter(df, known)
    unknown_df <- filter(df, !known)
    n_known <- min(nrow(known_df), round(max_points * min(0.75, max(mean(df$known), 0.25))))
    n_unknown <- max_points - n_known
    df <- bind_rows(
      slice_sample(known_df, n = n_known),
      slice_sample(unknown_df, n = min(nrow(unknown_df), n_unknown))
    )
  }
  df
}

map_panel <- function(city) {
  pts <- read_city_map_points(city)
  known_ratio <- raw |> filter(city == !!city) |> pull(known_label_ratio)
  ggplot() +
    geom_point(data = filter(pts, !known), aes(x, y), color = landuse_palette[["Unknown"]], size = 0.08, alpha = 0.35) +
    geom_point(data = filter(pts, known), aes(x, y, color = label), size = 0.18, alpha = 0.85) +
    scale_color_manual(values = landuse_palette, drop = FALSE) +
    coord_equal(expand = FALSE) +
    labs(
      title = city_display(city),
      subtitle = paste0("Known: ", percent(known_ratio, accuracy = 1)),
      color = "Label"
    ) +
    theme_void(base_size = 8) +
    theme(
      plot.title = element_text(face = "bold", size = 9),
      plot.subtitle = element_text(size = 7, color = "grey35")
    )
}

ordered_cities <- city_join |> arrange(desc(exploration_priority)) |> pull(city)
map_plots <- lapply(ordered_cities, map_panel)
fig_08 <- wrap_plots(map_plots, ncol = 5, guides = "collect") +
  plot_annotation(
    title = "All-city sampled label maps",
    subtitle = "Maps are ordered by exploration priority; points are sampled for readability, preserving known labels where possible.",
    theme = theme(plot.title = element_text(face = "bold", size = 14))
  ) &
  theme(
    legend.position = "bottom",
    legend.title = element_text(size = 9),
    legend.text = element_text(size = 8)
  )
save_figure(fig_08, "fig_08_all_city_sampled_label_maps", 15, 8.8)

rank_table <- city_join |>
  arrange(desc(exploration_priority)) |>
  select(
    city,
    city_label,
    n_plots,
    known_label_ratio,
    known_label_entropy_norm,
    overall_same_label_ratio,
    edge_probe_gain,
    edge_probe_perm_drop,
    street_entity_plot_load_p99,
    junction_entity_plot_load_p99,
    plot_building_membership_p99,
    exploration_priority,
    priority_group
  )
write_csv(rank_table, file.path(base_dir, "all_city_exploration_priority.csv"))

index_lines <- c(
  "# Urban All-City Exploration Figures",
  "",
  "Generated by `tools/plot_urban_all_city_exploration.R`.",
  "",
  "## Figures",
  "",
  "- `fig_06_all_city_raw_data_diagnostics`: raw layer scale, label quality, membership-tail risk, and label composition.",
  "- `fig_07_all_city_edge_readiness_screen`: CPU edge-probe screen, v3b relation mix, relation-wise same-label ratios, and exploration priority.",
  "- `fig_08_all_city_sampled_label_maps`: sampled spatial label maps for all available cities.",
  "",
  "## Tables",
  "",
  "- `all_city_exploration_priority.csv`: ranked city table for deciding which cities should receive V4 construction next.",
  "",
  "Interpretation: use these figures to separate promising semantic-edge cities, diagnostic contrast cities, and cities where raw projection tails or label limitations make V4 construction lower priority."
)
writeLines(index_lines, file.path(out_dir, "README.md"))
message("Wrote ", file.path(out_dir, "README.md"))
