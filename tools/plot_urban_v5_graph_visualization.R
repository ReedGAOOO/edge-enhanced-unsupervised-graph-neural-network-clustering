#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(ggplot2)
  library(sf)
  library(patchwork)
  library(scales)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
arg_value <- function(flag, default) {
  hit <- which(args == flag)
  if (length(hit) == 0 || hit[length(hit)] >= length(args)) default else args[hit[length(hit)] + 1]
}

vis_dir <- arg_value("--vis_dir", "results/urban_v5_graph_visualization")
out_dir <- file.path(vis_dir, "figures")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

save_both <- function(plot, name, width = 12, height = 8) {
  ggsave(file.path(out_dir, paste0(name, ".png")), plot, width = width, height = height, dpi = 240, bg = "white")
  ggsave(file.path(out_dir, paste0(name, ".pdf")), plot, width = width, height = height, bg = "white")
}

relation_cols <- c(
  "street-adjacency" = "#2b8cbe",
  "junction-path" = "#756bb1",
  "building-neighbor" = "#31a354",
  "mixed native" = "#111111",
  "geometry fallback" = "#d7301f",
  "other" = "#969696"
)

theme_work <- function(base_size = 10) {
  theme_minimal(base_size = base_size) +
    theme(
      panel.grid = element_line(color = "grey90", linewidth = 0.18),
      panel.grid.minor = element_blank(),
      plot.title = element_text(face = "bold", size = base_size + 3),
      plot.subtitle = element_text(size = base_size),
      axis.title = element_blank(),
      legend.position = "bottom",
      strip.text = element_text(face = "bold"),
      plot.margin = margin(8, 10, 8, 10)
    )
}

summary_path <- file.path(vis_dir, "v5_city_relation_summary.csv")
edges_path <- file.path(vis_dir, "v5_global_edge_samples.csv")
plots_path <- file.path(vis_dir, "v5_local_plots.geojson")
local_edges_path <- file.path(vis_dir, "v5_local_edges.geojson")
streets_path <- file.path(vis_dir, "v5_local_streets.geojson")
nodes_path <- file.path(vis_dir, "v5_local_nodes.csv")
manifest_path <- file.path(vis_dir, "v5_local_manifest.csv")

if (!file.exists(summary_path) || !file.exists(edges_path) || !file.exists(plots_path) || !file.exists(local_edges_path)) {
  stop("Missing V5 visualization export files. Run tools/export_urban_v5_graph_visualization.py first.")
}

rel_summary <- read_csv(summary_path, show_col_types = FALSE) |>
  mutate(
    relation = factor(relation, levels = names(relation_cols)),
    city_label = factor(city_label, levels = unique(city_label))
  )

p_rel <- rel_summary |>
  ggplot(aes(x = city_label, y = edge_share, fill = relation)) +
  geom_col(width = 0.68, color = "white", linewidth = 0.15) +
  geom_text(
    data = rel_summary |> filter(edge_share >= 0.06),
    aes(label = percent(edge_share, accuracy = 1)),
    position = position_stack(vjust = 0.5),
    color = "white",
    size = 3.2
  ) +
  scale_y_continuous(labels = percent_format(accuracy = 1), expand = expansion(mult = c(0, 0.03))) +
  scale_fill_manual(values = relation_cols, drop = FALSE) +
  labs(
    title = "Figure V5-1. Relation composition of the V5 native-object graph",
    subtitle = "V5 uses native Urbanity object paths. Geometry fallback is kept in the legend as a control and is zero.",
    y = "Share of undirected edges",
    fill = "V5 relation"
  ) +
  theme_work()
save_both(p_rel, "fig_v5_01_relation_composition", 10.5, 6)

global_edges <- read_csv(edges_path, show_col_types = FALSE) |>
  filter(
    is.finite(src_lon), is.finite(src_lat),
    is.finite(dst_lon), is.finite(dst_lat)
  ) |>
  mutate(
    relation = factor(relation, levels = names(relation_cols)),
    city_label = factor(city_label, levels = unique(city_label))
  )

p_global <- ggplot(global_edges) +
  geom_segment(
    aes(x = src_lon, y = src_lat, xend = dst_lon, yend = dst_lat, color = relation),
    linewidth = 0.17,
    alpha = 0.42
  ) +
  facet_wrap(~city_label, scales = "free", ncol = 1) +
  scale_color_manual(values = relation_cols, drop = FALSE) +
  labs(
    title = "Figure V5-2. Sampled V5 graph edges on city maps",
    subtitle = "Each panel shows sampled plot-to-plot edges induced by native object paths. Dense black/green/blue zones indicate mixed, building, or street mechanisms.",
    color = "V5 relation"
  ) +
  theme_work()
save_both(p_global, "fig_v5_02_global_sampled_edge_maps", 10.5, 13)

plots <- suppressWarnings(st_read(plots_path, quiet = TRUE)) |>
  mutate(
    city_label = as.character(city_label),
    is_target_tile = as.logical(is_target_tile)
  )
local_edges <- suppressWarnings(st_read(local_edges_path, quiet = TRUE)) |>
  mutate(
    city_label = as.character(city_label),
    relation = factor(relation, levels = names(relation_cols))
  )
streets <- if (file.exists(streets_path)) {
  suppressWarnings(st_read(streets_path, quiet = TRUE)) |> mutate(city_label = as.character(city_label))
} else {
  st_sf(city_label = character(), geometry = st_sfc(crs = 4326))
}
nodes <- read_csv(nodes_path, show_col_types = FALSE) |>
  mutate(city_label = as.character(city_label))
nodes_sf <- st_as_sf(nodes, coords = c("lon", "lat"), crs = 4326, remove = FALSE)
local_manifest <- read_csv(manifest_path, show_col_types = FALSE)

make_local_plot <- function(city_name) {
  p_plots <- plots |> filter(city == city_name)
  p_edges <- local_edges |> filter(city == city_name)
  p_streets <- streets |> filter(city == city_name)
  p_nodes <- nodes_sf |> filter(city == city_name)
  meta <- local_manifest |> filter(city == city_name) |> slice(1)
  if (nrow(p_plots) == 0) return(NULL)
  city_lab <- unique(p_plots$city_label)[1]
  subtitle <- paste0(
    "Target tile ", meta$target_tile,
    " | nodes ", meta$n_local_nodes,
    " | displayed edges ", meta$n_local_edges
  )
  ggplot() +
    {if (nrow(p_streets) > 0) geom_sf(data = p_streets, color = "#d0d0d0", linewidth = 0.16, alpha = 0.65)} +
    geom_sf(data = p_plots, fill = "grey94", color = "white", linewidth = 0.08) +
    geom_sf(data = p_nodes, color = "grey12", size = 0.14, alpha = 0.58) +
    geom_sf(data = p_edges, aes(color = relation), linewidth = 0.34, alpha = 0.78) +
    geom_sf(data = p_plots |> filter(is_target_tile), fill = NA, color = "#00bfc4", linewidth = 0.75) +
    scale_color_manual(values = relation_cols, drop = FALSE) +
    labs(title = city_lab, subtitle = subtitle, color = "V5 relation") +
    coord_sf(expand = FALSE) +
    theme_work(9) +
    theme(legend.position = "none")
}

local_plots <- lapply(unique(plots$city), make_local_plot)
local_plots <- local_plots[!vapply(local_plots, is.null, logical(1))]
p_local <- wrap_plots(local_plots, ncol = 1) +
  plot_annotation(
    title = "Figure V5-3. Local V5 graph construction examples",
    subtitle = "Grey polygons are urban plots; black points are plot nodes; colored lines are V5 edges. Cyan outline marks the selected tile window center."
  )
save_both(p_local, "fig_v5_03_local_graph_examples", 11.5, 15)

make_relation_panel <- function(city_name, rel_name) {
  p_plots <- plots |> filter(city == city_name)
  p_edges <- local_edges |> filter(city == city_name, relation == rel_name)
  p_streets <- streets |> filter(city == city_name)
  p_nodes <- nodes_sf |> filter(city == city_name)
  if (nrow(p_plots) == 0) return(NULL)
  city_lab <- unique(p_plots$city_label)[1]
  rel_lab <- gsub("-", " ", rel_name)
  ggplot() +
    {if (nrow(p_streets) > 0) geom_sf(data = p_streets, color = "#e0e0e0", linewidth = 0.12, alpha = 0.52)} +
    geom_sf(data = p_plots, fill = "grey96", color = "white", linewidth = 0.06) +
    geom_sf(data = p_nodes, color = "grey15", size = 0.11, alpha = 0.42) +
    geom_sf(data = p_edges, color = relation_cols[[rel_name]], linewidth = 0.56, alpha = 0.88) +
    geom_sf(data = p_plots |> filter(is_target_tile), fill = NA, color = "#00bfc4", linewidth = 0.65) +
    labs(title = paste(city_lab, rel_lab, sep = " | ")) +
    coord_sf(expand = FALSE) +
    theme_work(8) +
    theme(
      legend.position = "none",
      plot.title = element_text(face = "bold", size = 9),
      axis.text = element_blank(),
      panel.grid = element_blank()
    )
}

relation_panels <- list()
for (city_name in unique(plots$city)) {
  for (rel_name in c("street-adjacency", "junction-path", "building-neighbor")) {
    relation_panels[[length(relation_panels) + 1]] <- make_relation_panel(city_name, rel_name)
  }
}
relation_panels <- relation_panels[!vapply(relation_panels, is.null, logical(1))]
p_local_rel <- wrap_plots(relation_panels, ncol = 3) +
  plot_annotation(
    title = "Figure V5-4. Local V5 graph separated by native relation family",
    subtitle = "Columns separate plot-street-plot, plot-street-intersection-street-plot, and plot-building-building-plot induced edges."
  )
save_both(p_local_rel, "fig_v5_04_local_relation_family_examples", 13, 13)

box_df <- tibble::tribble(
  ~id, ~x, ~y, ~w, ~h, ~label, ~group,
  "raw", 0.10, 0.78, 0.25, 0.17, "Raw Urbanity objects\nplots / streets / intersections / buildings", "input",
  "street", 0.43, 0.80, 0.28, 0.13, "plot-street-plot\nshared street adjacency", "relation",
  "junction", 0.43, 0.58, 0.28, 0.13, "plot-street-intersection-street-plot\njunction path", "relation",
  "building", 0.43, 0.36, 0.28, 0.13, "plot-building-building-plot\nbuilding-neighbor path", "relation",
  "nogeom", 0.43, 0.16, 0.28, 0.12, "no geometry fallback\nis_geom_fallback = 0", "control",
  "graph", 0.81, 0.52, 0.30, 0.22, "V5 plot graph\ncompatible with tile detectors\nbut grounded in native object paths", "output"
)
arrow_df <- tibble::tribble(
  ~x, ~y, ~xend, ~yend,
  0.225, 0.78, 0.29, 0.80,
  0.225, 0.78, 0.29, 0.58,
  0.225, 0.78, 0.29, 0.36,
  0.225, 0.78, 0.29, 0.16,
  0.57, 0.80, 0.66, 0.58,
  0.57, 0.58, 0.66, 0.54,
  0.57, 0.36, 0.66, 0.49,
  0.57, 0.16, 0.66, 0.45
)
p_flow <- ggplot() +
  geom_segment(
    data = arrow_df,
    aes(x = x, y = y, xend = xend, yend = yend),
    arrow = arrow(length = grid::unit(0.017, "npc"), type = "closed"),
    linewidth = 0.45,
    color = "grey35"
  ) +
  geom_rect(
    data = box_df,
    aes(xmin = x - w / 2, xmax = x + w / 2, ymin = y - h / 2, ymax = y + h / 2, fill = group),
    color = "grey25",
    linewidth = 0.45
  ) +
  geom_text(data = box_df, aes(x = x, y = y, label = label), size = 3.2, lineheight = 0.95) +
  scale_fill_manual(values = c(input = "#f0f0f0", relation = "#b3cde3", control = "#fbb4ae", output = "#ccebc5")) +
  coord_cartesian(xlim = c(0, 1), ylim = c(0, 1), expand = FALSE) +
  labs(
    title = "Figure V5-4. V5 construction mechanism",
    subtitle = "V5 is not a new geometric kNN graph. It is a plot-level view induced by native Urbanity object relationships."
  ) +
  theme_void(base_size = 10) +
  theme(
    plot.title = element_text(face = "bold", size = 13),
    plot.subtitle = element_text(size = 10),
    legend.position = "none",
    plot.margin = margin(10, 12, 10, 12)
  )
save_both(p_flow, "fig_v5_05_native_construction_flow", 12, 7)

cat("V5 graph visualization figures written to: ", out_dir, "\n", sep = "")
