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

base_dir <- "results/data_construction_series/urban_v4_semantic_search"
out_dir <- file.path(base_dir, "figures_r_atlas")
map_layer_dir <- file.path(base_dir, "map_layers")
urban_root <- "/mnt/e/urban network datasets"
data_root <- "data"

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

theme_set(theme_minimal(base_size = 10))

city_palette <- c(
  "WashingtonDC" = "#2C7BB6",
  "Tokyo" = "#1A9850",
  "Paris" = "#D7191C",
  "Boston" = "#FDAE61",
  "Madrid" = "#7B3294"
)

relation_palette <- c(
  "Street" = "#386CB0",
  "Junction" = "#F0027F",
  "Building" = "#4DAF4A",
  "Geometry" = "#FF7F00"
)

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

city_display <- function(x) {
  recode(
    x,
    "washingtondc" = "WashingtonDC",
    "tokyo" = "Tokyo",
    "paris" = "Paris",
    "boston" = "Boston",
    "madrid" = "Madrid",
    .default = tools::toTitleCase(x)
  )
}

dataset_display <- function(x) {
  x |>
    gsub("^urban_", "", x = _) |>
    gsub("_plot_", " ", x = _) |>
    gsub("washingtondc", "WDC", x = _) |>
    gsub("tokyo", "Tokyo", x = _) |>
    gsub("paris", "Paris", x = _) |>
    gsub("boston", "Boston", x = _) |>
    gsub("madrid", "Madrid", x = _)
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

edge_level <- read_csv_quiet(file.path(base_dir, "plot_data_edge_level.csv")) |>
  mutate(
    city = factor(city, levels = c("WashingtonDC", "Tokyo", "Paris")),
    dataset_label = if_else(is.na(dataset_label), dataset_display(dataset), dataset_label),
    variant = factor(variant, levels = c("v4sl", "v4slj", "v4sljb", "v4sljbg", "v4g"))
  )

gnn_delta <- read_csv_quiet(file.path(base_dir, "plot_data_gnn_delta.csv")) |>
  mutate(
    dataset_label = if_else(is.na(dataset_label), dataset_display(dataset), dataset_label),
    comparison = factor(
      comparison,
      levels = c("EdgeSAGE - topology SAGE", "EdgeSAGE - permuted edge_attr")
    )
  )

label_limits <- read_csv_quiet(file.path(base_dir, "plot_data_label_limitations.csv")) |>
  mutate(dataset_label = if_else(is.na(dataset_label), dataset_display(dataset), dataset_label))

relation_prev <- read_csv_quiet(file.path(base_dir, "plot_data_relation_prevalence.csv")) |>
  mutate(
    dataset_label = if_else(is.na(dataset_label), dataset_display(dataset), dataset_label),
    relation = factor(relation, levels = c("Street", "Junction", "Building", "Geometry"))
  )

diag_files <- c(
  "results/data_diagnostics/urban_v4_semantic_washingtondc/urban_v3bsjg_diagnostics.csv",
  "results/data_diagnostics/urban_v4_semantic_tokyo/urban_v3bsjg_diagnostics.csv",
  "results/data_diagnostics/urban_v4_semantic_paris/urban_v3bsjg_diagnostics.csv"
)
diag_v4 <- bind_rows(lapply(diag_files, read_csv_quiet)) |>
  mutate(
    city_label = city_display(city),
    dataset_label = dataset_display(dataset),
    variant = factor(variant, levels = c("v4sl", "v4slj", "v4sljb", "v4sljbg", "v4g"))
  )

diag_v3b <- read_csv_quiet("results/data_diagnostics/urban_v3bsjg/urban_v3bsjg_diagnostics.csv") |>
  mutate(city_label = city_display(city))

edge_primary <- edge_level |>
  filter(feature_set == "endpoint_edge_attr") |>
  mutate(
    signal_class = case_when(
      auc_gain_vs_endpoint_mean > 0.005 & auc_drop_vs_perm_mean > 0.005 ~ "edge-useful",
      auc_gain_vs_endpoint_mean > 0 & auc_drop_vs_perm_mean > 0 ~ "weak-positive",
      TRUE ~ "limited"
    )
  )

label_wide <- label_limits |>
  pivot_wider(names_from = metric, values_from = value) |>
  left_join(
    edge_primary |>
      select(dataset, dataset_label, city, variant, auc_gain_vs_endpoint_mean, auc_drop_vs_perm_mean),
    by = c("dataset", "dataset_label")
  )

gnn_wide <- gnn_delta |>
  select(dataset, dataset_label, comparison, mean_delta) |>
  pivot_wider(names_from = comparison, values_from = mean_delta)

evi_data <- edge_primary |>
  left_join(gnn_wide, by = c("dataset", "dataset_label")) |>
  mutate(
    city_key = sub("^urban_([^_]+)_.*$", "\\1", dataset),
    geometry_penalty = if_else(variant == "v4g", auc_gain_vs_endpoint_mean, 0),
    edge_gain_z = as.numeric(scale(auc_gain_vs_endpoint_mean)),
    perm_drop_z = as.numeric(scale(auc_drop_vs_perm_mean)),
    sage_delta_z = as.numeric(scale(`EdgeSAGE - topology SAGE`)),
    perm_delta_z = as.numeric(scale(`EdgeSAGE - permuted edge_attr`)),
    evi = edge_gain_z + perm_drop_z + coalesce(sage_delta_z, 0) + coalesce(perm_delta_z, 0) -
      as.numeric(scale(geometry_penalty))
  )

p_edge_scatter <- ggplot(
  edge_primary,
  aes(auc_gain_vs_endpoint_mean, auc_drop_vs_perm_mean, color = city, shape = variant)
) +
  geom_hline(yintercept = 0, linewidth = 0.25, color = "grey70") +
  geom_vline(xintercept = 0, linewidth = 0.25, color = "grey70") +
  geom_point(size = 3, alpha = 0.9) +
  geom_text_repel(aes(label = dataset_label), size = 2.7, max.overlaps = Inf, show.legend = FALSE) +
  scale_color_manual(values = city_palette, drop = FALSE) +
  scale_x_continuous(labels = label_number(accuracy = 0.001)) +
  scale_y_continuous(labels = label_number(accuracy = 0.001)) +
  labs(
    title = "A. Edge-level validity",
    subtitle = "Useful edge features should improve over endpoint-only and degrade under permutation.",
    x = "AUC gain vs endpoint-only",
    y = "AUC drop after edge_attr permutation",
    color = "City",
    shape = "Graph"
  ) +
  clean_panel_theme()

p_gnn_delta <- ggplot(
  gnn_delta,
  aes(reorder(dataset_label, mean_delta), mean_delta, fill = comparison)
) +
  geom_hline(yintercept = 0, linewidth = 0.25, color = "grey70") +
  geom_col(position = position_dodge(width = 0.75), width = 0.68) +
  geom_errorbar(
    aes(ymin = mean_delta - sd_delta, ymax = mean_delta + sd_delta),
    position = position_dodge(width = 0.75),
    width = 0.18,
    linewidth = 0.25
  ) +
  coord_flip() +
  scale_fill_manual(values = c("#7B3294", "#008837")) +
  scale_y_continuous(labels = label_number(accuracy = 0.001)) +
  labs(
    title = "B. Independent GNN probe",
    subtitle = "EdgeSAGE is useful only when it beats topology-only and permuted-edge controls.",
    x = NULL,
    y = "Macro-F1 delta",
    fill = NULL
  ) +
  clean_panel_theme()

p_label_triangle <- ggplot(
  label_wide,
  aes(`Known-label ratio`, `Same-label edge ratio`, color = auc_gain_vs_endpoint_mean, shape = variant)
) +
  geom_point(size = 3.1, alpha = 0.95) +
  geom_text_repel(aes(label = dataset_label), size = 2.6, max.overlaps = Inf) +
  scale_color_viridis_c(option = "C", labels = label_number(accuracy = 0.001)) +
  scale_x_continuous(labels = percent_format(accuracy = 1)) +
  scale_y_continuous(labels = percent_format(accuracy = 1)) +
  labs(
    title = "C. Label availability is not edge validity",
    subtitle = "Paris is label-rich but edge-weak; WashingtonDC is sparse but more edge-aligned.",
    x = "Known-label ratio",
    y = "Same-label edge ratio",
    color = "AUC gain",
    shape = "Graph"
  ) +
  clean_panel_theme()

p_relation_heat <- ggplot(relation_prev, aes(dataset_label, relation, fill = prevalence)) +
  geom_tile(color = "white", linewidth = 0.4) +
  geom_text(aes(label = percent(prevalence, accuracy = 1)), size = 2.6, color = "grey10") +
  scale_fill_viridis_c(option = "E", labels = percent_format(accuracy = 1)) +
  labs(
    title = "D. Relation prevalence",
    subtitle = "Adding building or geometry changes what the homogeneous plot graph means.",
    x = NULL,
    y = NULL,
    fill = "Edge share"
  ) +
  theme(axis.text.x = element_text(angle = 35, hjust = 1)) +
  clean_panel_theme()

fig_01 <- (p_edge_scatter | p_gnn_delta) / (p_label_triangle | p_relation_heat) +
  plot_annotation(
    title = "Urban edge semantics: evidence and limitations",
    subtitle = "Prototype results show usable edge signal, city dependence, and a geometry confound.",
    theme = theme(plot.title = element_text(face = "bold", size = 14))
  )

save_figure(fig_01, "fig_01_edge_semantics_evidence_atlas", 12, 9)

relation_same <- diag_v4 |>
  select(
    dataset_label, city_label, variant,
    street_same_label_ratio, junction_same_label_ratio,
    shared_building_same_label_ratio, geom_same_label_ratio
  ) |>
  pivot_longer(
    cols = ends_with("_same_label_ratio"),
    names_to = "relation",
    values_to = "same_label_ratio"
  ) |>
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

wdc_projection <- edge_primary |>
  filter(city == "WashingtonDC") |>
  select(dataset_label, variant, auc_gain_vs_endpoint_mean, auc_drop_vs_perm_mean) |>
  pivot_longer(
    cols = c(auc_gain_vs_endpoint_mean, auc_drop_vs_perm_mean),
    names_to = "metric",
    values_to = "value"
  ) |>
  mutate(
    metric = recode(
      metric,
      "auc_gain_vs_endpoint_mean" = "AUC gain vs endpoint",
      "auc_drop_vs_perm_mean" = "Permutation drop"
    )
  )

p_all_city_context <- ggplot(
  diag_v3b,
  aes(known_label_ratio, overall_same_label_ratio, size = n_nodes, color = largest_known_class_ratio)
) +
  geom_point(alpha = 0.85) +
  geom_text_repel(aes(label = city_label), size = 2.6, max.overlaps = Inf) +
  scale_x_continuous(labels = percent_format(accuracy = 1)) +
  scale_y_continuous(labels = percent_format(accuracy = 1)) +
  scale_size_area(max_size = 8, labels = label_number(scale_cut = cut_short_scale())) +
  scale_color_viridis_c(option = "D", labels = percent_format(accuracy = 1)) +
  labs(
    title = "A. All-city diagnostic context",
    subtitle = "V3b diagnostics show that label coverage and edge homophily vary independently.",
    x = "Known-label ratio",
    y = "Overall same-label edge ratio",
    size = "Nodes",
    color = "Largest class"
  ) +
  clean_panel_theme()

p_relation_homophily <- ggplot(
  relation_same,
  aes(dataset_label, same_label_ratio, color = relation)
) +
  geom_point(size = 2.6, position = position_dodge(width = 0.55), alpha = 0.9) +
  scale_color_manual(values = relation_palette, drop = FALSE) +
  scale_y_continuous(labels = percent_format(accuracy = 1)) +
  labs(
    title = "B. Relation-specific label alignment",
    subtitle = "Street, junction, building, and geometry edges encode different label assumptions.",
    x = NULL,
    y = "Same-label ratio",
    color = "Relation"
  ) +
  theme(axis.text.x = element_text(angle = 35, hjust = 1)) +
  clean_panel_theme()

p_wdc_projection <- ggplot(
  wdc_projection,
  aes(variant, value, group = metric, color = metric)
) +
  geom_hline(yintercept = 0, linewidth = 0.25, color = "grey70") +
  geom_line(linewidth = 0.65) +
  geom_point(size = 2.4) +
  scale_color_manual(values = c("#386CB0", "#E66101")) +
  scale_y_continuous(labels = label_number(accuracy = 0.001)) +
  labs(
    title = "C. WashingtonDC projection path",
    subtitle = "Semantic variants and geometry-only control should not be interpreted the same way.",
    x = "Graph construction",
    y = "Edge-level value",
    color = NULL
  ) +
  clean_panel_theme()

p_evi <- ggplot(
  evi_data,
  aes(reorder(dataset_label, evi), evi, fill = signal_class)
) +
  geom_hline(yintercept = 0, linewidth = 0.25, color = "grey70") +
  geom_col(width = 0.68) +
  coord_flip() +
  scale_fill_manual(values = c("edge-useful" = "#008837", "weak-positive" = "#A6D96A", "limited" = "#D7191C")) +
  labs(
    title = "D. Prototype Edge Validity Index",
    subtitle = "A compact score for ranking graph-city pairs before main model ablations.",
    x = NULL,
    y = "Relative EVI",
    fill = "Class"
  ) +
  clean_panel_theme()

fig_02 <- (p_all_city_context | p_relation_homophily) / (p_wdc_projection | p_evi) +
  plot_annotation(
    title = "Why city graphs behave differently",
    subtitle = "The mechanism is label-edge alignment, relation projection, and spatial confounding.",
    theme = theme(plot.title = element_text(face = "bold", size = 14))
  )

save_figure(fig_02, "fig_02_city_graph_mechanism_dashboard", 12, 9)

label_lookup_from_meta <- function(dataset) {
  meta <- jsonlite::fromJSON(file.path(data_root, dataset, paste0(dataset, "_meta.json")))
  mapping <- unlist(meta$label_mapping)
  tibble(
    label_name = names(mapping),
    label = as.integer(mapping)
  ) |>
    mutate(label_name = if_else(label_name == "unknown", "Unknown", label_name))
}

read_plot_points <- function(city, dataset) {
  plot_path <- file.path(urban_root, city, "plot.parquet")
  node_path <- file.path(data_root, dataset, paste0(dataset, "_node_table.csv"))
  label_lookup <- label_lookup_from_meta(dataset)

  raw <- arrow::read_parquet(plot_path) |>
    select(plot_id, geometry)
  geom <- st_as_sfc(raw$geometry, EWKB = FALSE)
  plots <- st_sf(raw |> select(-geometry), geometry = geom, crs = 4326)

  node_table <- read_csv_quiet(node_path) |>
    left_join(label_lookup, by = "label") |>
    mutate(
      label_name = coalesce(label_name, "Unknown"),
      known = label_name != "Unknown"
    )

  plots <- plots |>
    left_join(node_table, by = "plot_id")

  points <- suppressWarnings(st_point_on_surface(plots)) |>
    mutate(
      label_name = factor(label_name, levels = names(landuse_palette)),
      city_label = city_display(city),
      dataset = dataset,
      dataset_label = dataset_display(dataset)
    )

  xy <- st_coordinates(points)
  node_xy <- points |>
    st_drop_geometry() |>
    transmute(
      node_idx,
      plot_id,
      label,
      label_name = as.character(label_name),
      known,
      x = xy[, 1],
      y = xy[, 2]
    )

  list(points = points, node_xy = node_xy)
}

city_map_specs <- tibble::tribble(
  ~city, ~dataset, ~title,
  "washingtondc", "urban_washingtondc_plot_v4sljb", "WashingtonDC",
  "tokyo", "urban_tokyo_plot_v4sljb", "Tokyo",
  "paris", "urban_paris_plot_v4slj", "Paris"
)

plot_cache <- list()
for (i in seq_len(nrow(city_map_specs))) {
  spec <- city_map_specs[i, ]
  plot_cache[[spec$dataset]] <- read_plot_points(spec$city, spec$dataset)
}

map_label_panel <- function(dataset, title) {
  pts <- plot_cache[[dataset]]$points
  known_ratio <- mean(pts$known, na.rm = TRUE)

  ggplot() +
    geom_sf(data = filter(pts, !known), color = landuse_palette[["Unknown"]], size = 0.08, alpha = 0.35) +
    geom_sf(data = filter(pts, known), aes(color = label_name), size = 0.18, alpha = 0.85) +
    scale_color_manual(values = landuse_palette, drop = FALSE) +
    guides(color = guide_legend(nrow = 2, byrow = TRUE, override.aes = list(size = 2.2, alpha = 1))) +
    coord_sf(datum = NA, expand = FALSE) +
    labs(
      title = title,
      subtitle = paste0("Known-label ratio: ", percent(known_ratio, accuracy = 0.1)),
      color = "Label"
    ) +
    theme_void(base_size = 9) +
    theme(
      plot.title = element_text(face = "bold", size = 10),
      plot.subtitle = element_text(size = 8, color = "grey35"),
      legend.position = "bottom",
      legend.key.width = unit(0.5, "cm")
    )
}

fig_03 <- wrap_plots(
  map_label_panel(city_map_specs$dataset[1], city_map_specs$title[1]),
  map_label_panel(city_map_specs$dataset[2], city_map_specs$title[2]),
  map_label_panel(city_map_specs$dataset[3], city_map_specs$title[3]),
  nrow = 1,
  guides = "collect"
) +
  plot_layout(guides = "collect") +
  plot_annotation(
    title = "Urban labels are spatially structured but unevenly observed",
    subtitle = "Maps show why label coverage alone cannot certify meaningful edge semantics.",
    theme = theme(plot.title = element_text(face = "bold", size = 14))
  ) &
  theme(legend.position = "bottom")

save_figure(fig_03, "fig_03_cross_city_label_coverage_maps", 14, 5)

edge_samples <- read_csv_quiet(file.path(map_layer_dir, "urban_v4_map_edge_samples.csv")) |>
  mutate(
    relation = factor(relation, levels = c("Street", "Junction", "Building", "Geometry")),
    dataset_label = dataset_display(dataset)
  )

build_edge_sf <- function(edge_df, node_xy) {
  e <- edge_df |>
    left_join(node_xy |> select(src = node_idx, x0 = x, y0 = y), by = "src") |>
    left_join(node_xy |> select(dst = node_idx, x1 = x, y1 = y), by = "dst") |>
    filter(!is.na(x0), !is.na(y0), !is.na(x1), !is.na(y1))

  lines <- Map(
    function(x0, y0, x1, y1) {
      st_linestring(matrix(c(x0, y0, x1, y1), ncol = 2, byrow = TRUE))
    },
    e$x0, e$y0, e$x1, e$y1
  )
  st_sf(e, geometry = st_sfc(lines, crs = 4326))
}

relation_key_plot <- function() {
  key <- tibble(
    relation = factor(names(relation_palette), levels = names(relation_palette)),
    x = seq_along(relation_palette) * 1.55
  )

  ggplot(key) +
    geom_segment(aes(x = x, xend = x + 0.42, y = 1, yend = 1, color = relation), linewidth = 1.1) +
    geom_text(aes(x = x + 0.52, y = 1, label = relation), hjust = 0, vjust = 0.5, size = 3.3) +
    scale_color_manual(values = relation_palette, limits = names(relation_palette), drop = FALSE) +
    coord_cartesian(xlim = c(1.2, 7.3), ylim = c(0.85, 1.15), clip = "off") +
    theme_void(base_size = 9) +
    theme(legend.position = "none", plot.margin = margin(0, 0, 0, 0))
}

map_edge_panel <- function(dataset, title, point_dataset = dataset, edge_alpha = 0.34, show_legend = FALSE) {
  pts <- plot_cache[[point_dataset]]$points
  node_xy <- plot_cache[[point_dataset]]$node_xy
  edge_sf <- build_edge_sf(filter(edge_samples, dataset == !!dataset), node_xy)

  ggplot() +
    geom_sf(data = pts, color = "grey82", size = 0.06, alpha = 0.34) +
    geom_sf(data = filter(pts, known), color = "grey40", size = 0.10, alpha = 0.35, show.legend = FALSE) +
    geom_sf(data = edge_sf, aes(color = relation), linewidth = 0.18, alpha = edge_alpha, show.legend = show_legend) +
    scale_color_manual(
      values = relation_palette,
      limits = names(relation_palette),
      breaks = names(relation_palette),
      drop = FALSE
    ) +
    guides(color = guide_legend(nrow = 1, override.aes = list(linewidth = 0.9, alpha = 1))) +
    coord_sf(datum = NA, expand = FALSE) +
    labs(title = title, color = "Relation") +
    theme_void(base_size = 9) +
    theme(
      plot.title = element_text(face = "bold", size = 10),
      legend.position = if (show_legend) "bottom" else "none",
      legend.key.width = unit(0.55, "cm")
    )
}

plot_cache[["urban_washingtondc_plot_v4slj"]] <- plot_cache[["urban_washingtondc_plot_v4sljb"]]
plot_cache[["urban_washingtondc_plot_v4g"]] <- plot_cache[["urban_washingtondc_plot_v4sljb"]]

fig_04_maps <- wrap_plots(
  map_edge_panel(
    "urban_washingtondc_plot_v4slj",
    "v4slj: street + junction",
    "urban_washingtondc_plot_v4sljb"
  ),
  map_edge_panel(
    "urban_washingtondc_plot_v4sljb",
    "v4sljb: plus building-neighbour",
    "urban_washingtondc_plot_v4sljb"
  ),
  map_edge_panel(
    "urban_washingtondc_plot_v4g",
    "v4g: geometry-only control",
    "urban_washingtondc_plot_v4sljb",
    edge_alpha = 0.42
  ),
  nrow = 1
)

fig_04 <- (fig_04_maps / relation_key_plot()) +
  plot_layout(heights = c(1, 0.08)) +
  plot_annotation(
    title = "Projection choices change the visible urban graph",
    subtitle = "Semantic relations and geometry-only edges create different spatial explanations for the same city.",
    theme = theme(plot.title = element_text(face = "bold", size = 14))
  )

save_figure(fig_04, "fig_04_washingtondc_projection_edge_maps", 14, 5)

fig_05_maps <- wrap_plots(
  map_edge_panel(
    "urban_washingtondc_plot_v4sljb",
    "WashingtonDC v4sljb",
    "urban_washingtondc_plot_v4sljb"
  ),
  map_edge_panel(
    "urban_tokyo_plot_v4sljb",
    "Tokyo v4sljb",
    "urban_tokyo_plot_v4sljb",
    edge_alpha = 0.26
  ),
  map_edge_panel(
    "urban_paris_plot_v4slj",
    "Paris v4slj",
    "urban_paris_plot_v4slj",
    edge_alpha = 0.36
  ),
  nrow = 1
)

fig_05 <- (fig_05_maps / relation_key_plot()) +
  plot_layout(heights = c(1, 0.08)) +
  plot_annotation(
    title = "The same relation vocabulary yields different city graphs",
    subtitle = "Edge maps expose why cross-city edge semantics should be studied as a phenomenon, not a preprocessing detail.",
    theme = theme(plot.title = element_text(face = "bold", size = 14))
  )

save_figure(fig_05, "fig_05_cross_city_semantic_edge_maps", 14, 5)

index_lines <- c(
  "# Urban V4 R Visualization Atlas",
  "",
  "Generated by `tools/plot_urban_v4_research_atlas.R`.",
  "",
  "## Figures",
  "",
  "- `fig_01_edge_semantics_evidence_atlas`: edge-level, GNN, label-limitation, and relation-prevalence evidence.",
  "- `fig_02_city_graph_mechanism_dashboard`: all-city diagnostic context, relation-specific homophily, WashingtonDC projection path, and prototype EVI.",
  "- `fig_03_cross_city_label_coverage_maps`: spatial label coverage for WashingtonDC, Tokyo, and Paris.",
  "- `fig_04_washingtondc_projection_edge_maps`: semantic vs building-neighbour vs geometry-only projection maps.",
  "- `fig_05_cross_city_semantic_edge_maps`: sampled relation-edge maps across WashingtonDC, Tokyo, and Paris.",
  "",
  "## Interpretation",
  "",
  "These figures support the current paper framing: Urban graph edges can be meaningful, but their validity is city- and projection-dependent, and geometry-only graphs are a necessary spatial-confounding control."
)
writeLines(index_lines, file.path(out_dir, "README.md"))
message("Wrote ", file.path(out_dir, "README.md"))
