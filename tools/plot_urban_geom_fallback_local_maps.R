#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(ggplot2)
  library(sf)
  library(scales)
  library(leaflet)
  library(htmlwidgets)
})

args <- commandArgs(trailingOnly = TRUE)
arg_value <- function(flag, default) {
  hit <- which(args == flag)
  if (length(hit) == 0 || hit[length(hit)] >= length(args)) default else args[hit[length(hit)] + 1]
}

results_root <- arg_value("--results_root", "results/urban_data_paper")
top_n <- as.integer(arg_value("--top_n", "25"))
window_tiles <- as.integer(arg_value("--window_tiles", "4"))
focus_cities <- strsplit(arg_value("--focus_cities", "bangkok,beijing,shanghai,singapore,sydney"), ",")[[1]]
focus_cities <- trimws(focus_cities)

out_dir <- file.path(results_root, "figures", "geom_fallback_local_maps")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

scores_path <- file.path(results_root, "tile_anomaly_scores.csv")
if (!file.exists(scores_path)) {
  stop("Missing tile_anomaly_scores.csv under results_root.")
}

scores <- read_csv(scores_path, show_col_types = FALSE) |>
  mutate(
    geom_edge_share = pmin(pmax(geom_edge_share, 0), 1),
    semantic_edge_share = pmin(pmax(semantic_edge_share, 0), 1),
    is_top_anomaly = city_anomaly_rank <= top_n,
    local_tile_label = if_else(is_top_anomaly, paste0("#", city_anomaly_rank), "")
  ) |>
  filter(
    is.finite(lon_min), is.finite(lon_max),
    is.finite(lat_min), is.finite(lat_max),
    !(abs(lon_mean) < 1e-8 & abs(lat_mean) < 1e-8)
  )

make_tile_sf <- function(df) {
  eps <- 1e-5
  polys <- Map(
    function(xmin, xmax, ymin, ymax) {
      if (!is.finite(xmin) || !is.finite(xmax) || !is.finite(ymin) || !is.finite(ymax)) {
        return(st_polygon())
      }
      if (abs(xmax - xmin) < eps) {
        xmin <- xmin - eps
        xmax <- xmax + eps
      }
      if (abs(ymax - ymin) < eps) {
        ymin <- ymin - eps
        ymax <- ymax + eps
      }
      st_polygon(list(matrix(
        c(
          xmin, ymin,
          xmax, ymin,
          xmax, ymax,
          xmin, ymax,
          xmin, ymin
        ),
        ncol = 2,
        byrow = TRUE
      )))
    },
    df$lon_min, df$lon_max, df$lat_min, df$lat_max
  )
  st_as_sf(df, geometry = st_sfc(polys, crs = 4326))
}

tile_sf <- make_tile_sf(scores)
focus_sf <- tile_sf |> filter(city %in% focus_cities)
top_focus_sf <- focus_sf |> filter(is_top_anomaly)

theme_map <- function() {
  theme_minimal(base_size = 10) +
    theme(
      panel.grid = element_line(color = "grey88", linewidth = 0.2),
      panel.grid.minor = element_blank(),
      plot.title = element_text(face = "bold", size = 12),
      plot.subtitle = element_text(size = 10),
      axis.title = element_blank(),
      legend.position = "bottom",
      strip.text = element_text(face = "bold")
    )
}

save_both <- function(plot, name, width = 12, height = 8) {
  ggsave(file.path(out_dir, paste0(name, ".png")), plot, width = width, height = height, dpi = 220, bg = "white")
  ggsave(file.path(out_dir, paste0(name, ".pdf")), plot, width = width, height = height, bg = "white")
}

city_order <- focus_sf |>
  st_drop_geometry() |>
  distinct(city, city_label) |>
  arrange(match(city, focus_cities)) |>
  pull(city_label)

focus_sf <- focus_sf |>
  mutate(city_label = factor(city_label, levels = city_order))
top_focus_sf <- top_focus_sf |>
  mutate(city_label = factor(city_label, levels = city_order))

p32 <- ggplot() +
  geom_rect(
    data = focus_sf |> st_drop_geometry(),
    aes(xmin = lon_min, xmax = lon_max, ymin = lat_min, ymax = lat_max),
    fill = "grey93",
    color = "white",
    linewidth = 0.06
  ) +
  geom_rect(
    data = top_focus_sf |> st_drop_geometry(),
    aes(xmin = lon_min, xmax = lon_max, ymin = lat_min, ymax = lat_max, fill = geom_edge_share),
    color = "grey20",
    linewidth = 0.16,
    alpha = 0.92
  ) +
  facet_wrap(~city_label, scales = "free", ncol = 2) +
  scale_fill_gradient(low = "#fee8c8", high = "#b30000", labels = percent_format(accuracy = 1), limits = c(0, 1)) +
  labs(
    title = "Figure 32. Local geometry fallback anomaly tiles on city maps",
    subtitle = "Grey polygons are all valid tiles; colored polygons are top anomaly tiles. Color encodes local geometry fallback share.",
    fill = "Geom share"
  ) +
  theme_map()
save_both(p32, "fig_32_focus_city_geom_fallback_tile_maps", 12, 10)

zoom_rows <- scores |>
  filter(city %in% focus_cities, is_top_anomaly) |>
  group_by(city) |>
  slice_max(geom_edge_share, n = 1, with_ties = FALSE) |>
  ungroup() |>
  select(city, target_tile_id = tile_id, target_tile_x = tile_x, target_tile_y = tile_y)

zoom_df <- scores |>
  inner_join(zoom_rows, by = "city") |>
  filter(
    abs(tile_x - target_tile_x) <= window_tiles,
    abs(tile_y - target_tile_y) <= window_tiles
  ) |>
  mutate(
    is_target_tile = tile_id == target_tile_id,
    in_top_anomaly_window = city_anomaly_rank <= top_n
  )

zoom_sf <- make_tile_sf(zoom_df) |>
  mutate(city_label = factor(city_label, levels = city_order))

zoom_labels <- zoom_sf |>
  st_drop_geometry() |>
  filter(in_top_anomaly_window) |>
  mutate(label = paste0("#", city_anomaly_rank))

p33 <- ggplot() +
  geom_rect(
    data = zoom_sf |> st_drop_geometry(),
    aes(xmin = lon_min, xmax = lon_max, ymin = lat_min, ymax = lat_max, fill = geom_edge_share),
    color = "white",
    linewidth = 0.12
  ) +
  geom_rect(
    data = zoom_sf |> filter(in_top_anomaly_window) |> st_drop_geometry(),
    aes(xmin = lon_min, xmax = lon_max, ymin = lat_min, ymax = lat_max),
    fill = NA,
    color = "#252525",
    linewidth = 0.35
  ) +
  geom_rect(
    data = zoom_sf |> filter(is_target_tile) |> st_drop_geometry(),
    aes(xmin = lon_min, xmax = lon_max, ymin = lat_min, ymax = lat_max),
    fill = NA,
    color = "#00bfc4",
    linewidth = 1.1
  ) +
  geom_text(
    data = zoom_labels,
    aes(x = lon_mean, y = lat_mean, label = label),
    size = 2.6,
    color = "black",
    check_overlap = TRUE
  ) +
  facet_wrap(~city_label, scales = "free", ncol = 2) +
  scale_fill_gradient(low = "#fff7ec", high = "#b30000", labels = percent_format(accuracy = 1), limits = c(0, 1)) +
  labs(
    title = "Figure 33. Zoom-in maps around each city's strongest geometry fallback tile",
    subtitle = paste0("Window uses +/-", window_tiles, " tiles around the strongest high-geometry anomaly tile. Cyan border marks the target tile."),
    fill = "Geom share"
  ) +
  theme_map()
save_both(p33, "fig_33_zoom_in_strongest_geom_fallback_tiles", 12, 10)

write_csv(
  zoom_df |>
    arrange(city, desc(is_target_tile), city_anomaly_rank) |>
    select(
      city, city_label, tile_id, target_tile_id,
      city_anomaly_rank, tile_typology,
      geom_edge_share, semantic_edge_share,
      combined_anomaly_score, n_nodes, known_label_ratio,
      lon_min, lon_max, lat_min, lat_max
    ),
  file.path(results_root, "geom_fallback_local_map_tiles.csv")
)

make_leaflet_city <- function(city_name) {
  city_sf <- zoom_sf |> filter(city == city_name)
  if (nrow(city_sf) == 0) return(invisible(NULL))
  city_label <- as.character(city_sf$city_label[1])
  pal <- colorNumeric("YlOrRd", domain = c(0, 1), na.color = "#cccccc")
  popup <- paste0(
    "<b>", city_label, "</b><br/>",
    "tile: ", city_sf$tile_id, "<br/>",
    "rank: #", city_sf$city_anomaly_rank, "<br/>",
    "typology: ", city_sf$tile_typology, "<br/>",
    "geom share: ", percent(city_sf$geom_edge_share, accuracy = 0.1), "<br/>",
    "semantic share: ", percent(city_sf$semantic_edge_share, accuracy = 0.1), "<br/>",
    "nodes: ", city_sf$n_nodes, "<br/>",
    "known label: ", percent(city_sf$known_label_ratio, accuracy = 0.1)
  )
  m <- leaflet(city_sf, options = leafletOptions(preferCanvas = TRUE)) |>
    addProviderTiles(providers$CartoDB.Positron) |>
    addPolygons(
      fillColor = ~pal(geom_edge_share),
      fillOpacity = 0.72,
      color = ~ifelse(is_target_tile, "#00bfc4", ifelse(in_top_anomaly_window, "#252525", "#ffffff")),
      weight = ~ifelse(is_target_tile, 4, ifelse(in_top_anomaly_window, 2, 0.6)),
      popup = popup,
      label = ~paste0("tile ", tile_id, " | geom ", percent(geom_edge_share, accuracy = 1))
    ) |>
    addLegend(
      "bottomright",
      pal = pal,
      values = c(0, 1),
      title = "Geom fallback share",
      opacity = 0.8
    )
  htmlwidgets::saveWidget(
    m,
    file.path(out_dir, paste0("local_geom_fallback_leaflet_", city_name, ".html")),
    selfcontained = TRUE
  )
}

invisible(lapply(focus_cities, make_leaflet_city))

cat("Local geometry fallback maps written to: ", out_dir, "\n", sep = "")
cat("Zoom tile table written to: ", file.path(results_root, "geom_fallback_local_map_tiles.csv"), "\n", sep = "")
