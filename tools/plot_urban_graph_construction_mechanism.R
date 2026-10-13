#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(ggplot2)
  library(sf)
  library(patchwork)
  library(jsonlite)
  library(scales)
})

args <- commandArgs(trailingOnly = TRUE)
arg_value <- function(flag, default) {
  hit <- which(args == flag)
  if (length(hit) == 0 || hit[length(hit)] >= length(args)) default else args[hit[length(hit)] + 1]
}

example_dir <- arg_value("--example_dir", "results/urban_data_paper/graph_construction_mechanism")
out_dir <- file.path(example_dir, "figures")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

save_both <- function(plot, name, width = 12, height = 8) {
  ggsave(file.path(out_dir, paste0(name, ".png")), plot, width = width, height = height, dpi = 220, bg = "white")
  ggsave(file.path(out_dir, paste0(name, ".pdf")), plot, width = width, height = height, bg = "white")
}

theme_clean <- function() {
  theme_void(base_size = 10) +
    theme(
      plot.title = element_text(face = "bold", size = 13),
      plot.subtitle = element_text(size = 10),
      legend.position = "bottom",
      plot.margin = margin(10, 12, 10, 12)
    )
}

theme_map <- function() {
  theme_minimal(base_size = 10) +
    theme(
      panel.grid = element_line(color = "grey90", linewidth = 0.2),
      panel.grid.minor = element_blank(),
      axis.title = element_blank(),
      plot.title = element_text(face = "bold", size = 11),
      plot.subtitle = element_text(size = 9),
      legend.position = "bottom",
      strip.text = element_text(face = "bold")
    )
}

box_df <- tibble::tribble(
  ~id, ~x, ~y, ~w, ~h, ~label, ~group,
  "raw", 0.08, 0.78, 0.23, 0.16, "Raw Urban layers\nplot.parquet\nstreet.parquet\nplot_street_id.npy\nplot_bid.npy", "input",
  "nodes", 0.43, 0.88, 0.25, 0.12, "Nodes\none plot = one graph node\nnode features from plot form / density", "node",
  "street", 0.43, 0.66, 0.25, 0.12, "Shared-street rule\nplots attached to same street\nbecome candidate edges", "edge",
  "junction", 0.43, 0.48, 0.25, 0.12, "Same-junction rule\nstreet endpoints are projected\ninto junction context edges", "edge",
  "geom", 0.43, 0.30, 0.25, 0.12, "Geometry fallback\nonly for under-connected plots\nnearby centroids add weak edges", "fallback",
  "building", 0.43, 0.12, 0.25, 0.12, "Shared-building signal\nstored as edge attribute\nnot the main v3 edge source", "attr",
  "graph", 0.80, 0.50, 0.27, 0.20, "Final plot graph\nedge_index + edge_attr + edge_weight\nrelation bits: street / junction / geom / building", "output"
)

arrow_df <- tibble::tribble(
  ~x, ~y, ~xend, ~yend,
  0.195, 0.78, 0.305, 0.88,
  0.195, 0.78, 0.305, 0.66,
  0.195, 0.78, 0.305, 0.48,
  0.195, 0.78, 0.305, 0.30,
  0.195, 0.78, 0.305, 0.12,
  0.555, 0.88, 0.665, 0.55,
  0.555, 0.66, 0.665, 0.55,
  0.555, 0.48, 0.665, 0.52,
  0.555, 0.30, 0.665, 0.47,
  0.555, 0.12, 0.665, 0.45
)

p34 <- ggplot() +
  geom_segment(
    data = arrow_df,
    aes(x = x, y = y, xend = xend, yend = yend),
    arrow = arrow(length = grid::unit(0.018, "npc"), type = "closed"),
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
  scale_fill_manual(
    values = c(
      input = "#f0f0f0",
      node = "#d9f0a3",
      edge = "#a6bddb",
      fallback = "#fdbb84",
      attr = "#d8b365",
      output = "#c7eae5"
    )
  ) +
  coord_cartesian(xlim = c(0, 1), ylim = c(0, 1), expand = FALSE) +
  labs(
    title = "Figure 34. How the Urban plot graph is constructed",
    subtitle = "The graph is semantics-first: street and junction rules create most edges; geometry is a fallback for under-connected plots."
  ) +
  theme_clean() +
  theme(legend.position = "none")
save_both(p34, "fig_34_graph_construction_mechanism_flow", 13, 7)

plots_path <- file.path(example_dir, "construction_plots.geojson")
streets_path <- file.path(example_dir, "construction_streets.geojson")
edges_path <- file.path(example_dir, "construction_edges.geojson")
nodes_path <- file.path(example_dir, "construction_nodes.csv")
meta_path <- file.path(example_dir, "construction_example_meta.json")

if (!file.exists(plots_path) || !file.exists(edges_path) || !file.exists(nodes_path)) {
  stop("Example files are missing. Run tools/export_urban_graph_construction_example.py first.")
}

plots <- suppressWarnings(st_read(plots_path, quiet = TRUE))
edges <- suppressWarnings(st_read(edges_path, quiet = TRUE))
nodes <- read_csv(nodes_path, show_col_types = FALSE)
meta <- jsonlite::fromJSON(meta_path)
nodes_sf <- st_as_sf(nodes, coords = c("lon", "lat"), crs = 4326, remove = FALSE)
streets <- if (file.exists(streets_path)) suppressWarnings(st_read(streets_path, quiet = TRUE)) else st_sf()

plots$is_target_tile <- as.logical(plots$is_target_tile)
target_plots <- plots |> filter(is_target_tile)
edge_cols <- c("shared street" = "#2b8cbe", "same junction" = "#756bb1", "geometry fallback" = "#d7301f", "shared building attribute" = "#31a354", "other" = "grey50")
present_rels <- intersect(names(edge_cols), sort(unique(as.character(edges$relation_type))))

base_layers <- list(
  geom_sf(data = plots, fill = "grey93", color = "white", linewidth = 0.08),
  geom_sf(data = target_plots, fill = NA, color = "#00bfc4", linewidth = 0.65)
)

p35a <- ggplot() +
  base_layers +
  {if (nrow(streets) > 0) geom_sf(data = streets, color = "#74a9cf", linewidth = 0.25, alpha = 0.75)} +
  geom_sf(data = nodes_sf, color = "grey15", size = 0.25, alpha = 0.75) +
  coord_sf(expand = FALSE) +
  labs(
    title = "A. Raw map layers",
    subtitle = "Plots become nodes; street geometries provide urban context."
  ) +
  theme_map()

p35b <- ggplot() +
  base_layers +
  geom_sf(
    data = edges |> filter(relation_type %in% c("shared street", "same junction")),
    aes(color = relation_type),
    linewidth = 0.22,
    alpha = 0.62
  ) +
  geom_sf(data = nodes_sf, color = "grey10", size = 0.22, alpha = 0.72) +
  scale_color_manual(values = edge_cols, breaks = present_rels) +
  coord_sf(expand = FALSE) +
  labs(
    title = "B. Semantic relation edges",
    subtitle = "Plots sharing a street or projected junction are connected.",
    color = "Relation"
  ) +
  theme_map()

p35c <- ggplot() +
  base_layers +
  geom_sf(
    data = edges |> filter(relation_type == "geometry fallback"),
    aes(color = relation_type),
    linewidth = 0.28,
    alpha = 0.72
  ) +
  geom_sf(data = nodes_sf, color = "grey10", size = 0.22, alpha = 0.72) +
  scale_color_manual(values = edge_cols, breaks = present_rels) +
  coord_sf(expand = FALSE) +
  labs(
    title = "C. Geometry fallback edges",
    subtitle = "Only under-connected plots receive nearby-centroid fallback edges.",
    color = "Relation"
  ) +
  theme_map()

p35d <- ggplot() +
  base_layers +
  geom_sf(
    data = edges,
    aes(color = relation_type),
    linewidth = 0.24,
    alpha = 0.66
  ) +
  geom_sf(data = nodes_sf, color = "grey10", size = 0.18, alpha = 0.65) +
  scale_color_manual(values = edge_cols, breaks = present_rels) +
  coord_sf(expand = FALSE) +
  labs(
    title = "D. Final local graph",
    subtitle = "Diagnostics measure how much each tile relies on each edge mechanism.",
    color = "Relation"
  ) +
  theme_map()

p35 <- ((p35a | (p35b + theme(legend.position = "none"))) / ((p35c + theme(legend.position = "none")) | p35d)) +
  plot_annotation(
    title = paste0("Figure 35. Local construction example: ", meta$city_label, " target tile ", meta$target_tile),
    subtitle = paste0("Cyan outline marks the target tile. Example contains ", meta$n_nodes, " plot nodes and ", meta$n_edges, " displayed graph edges.")
  )
save_both(p35, "fig_35_local_graph_construction_example", 14, 11)

cat("Graph construction mechanism figures written to: ", out_dir, "\n", sep = "")
