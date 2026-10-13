#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(ggrepel)
  library(patchwork)
  library(readr)
  library(tidyr)
})

out_dir <- "results/data_construction_series/urban_v4_semantic_search"
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

read_if_exists <- function(path) {
  if (!file.exists(path)) {
    warning(sprintf("Missing file: %s", path))
    return(tibble())
  }
  read_csv(path, show_col_types = FALSE)
}

city_from_dataset <- function(x) {
  city <- sub("^urban_", "", x)
  city <- sub("_plot_.*$", "", city)
  ifelse(city == "washingtondc", "WashingtonDC", tools::toTitleCase(city))
}

city_short_from_dataset <- function(x) {
  city <- city_from_dataset(x)
  ifelse(city == "WashingtonDC", "WDC", city)
}

variant_from_dataset <- function(x) {
  sub("^urban_.*_plot_", "", x)
}

label_from_dataset <- function(x) {
  paste(city_short_from_dataset(x), variant_from_dataset(x))
}

plot_order <- c(
  "WDC v4sl",
  "WDC v4slj",
  "WDC v4sljb",
  "WDC v4g",
  "Tokyo v4slj",
  "Tokyo v4sljb",
  "Paris v4slj",
  "Paris v4sljb"
)

selected_edge <- c(
  "urban_washingtondc_plot_v4sl",
  "urban_washingtondc_plot_v4slj",
  "urban_washingtondc_plot_v4sljb",
  "urban_washingtondc_plot_v4g",
  "urban_tokyo_plot_v4slj",
  "urban_tokyo_plot_v4sljb",
  "urban_paris_plot_v4slj",
  "urban_paris_plot_v4sljb"
)

edge_paths <- c(
  "results/edge_probe/urban_v4_semantic_washingtondc_edgelevel/edge_probe_summary.csv",
  "results/edge_probe/urban_v4_semantic_tokyo_edgelevel/edge_probe_summary.csv",
  "results/edge_probe/urban_v4_semantic_paris_edgelevel/edge_probe_summary.csv",
  "results/edge_probe/urban_v4_semantic_smoke_edgelevel/edge_probe_summary.csv"
)

edge_summary <- bind_rows(lapply(edge_paths, read_if_exists)) %>%
  filter(dataset %in% selected_edge, feature_set == "endpoint_edge_attr") %>%
  mutate(
    city = city_from_dataset(dataset),
    variant = variant_from_dataset(dataset),
    dataset_label = factor(label_from_dataset(dataset), levels = plot_order),
    status = case_when(
      auc_gain_vs_endpoint_mean > 0 & auc_drop_vs_perm_mean > 0 ~ "usable edge signal",
      auc_drop_vs_perm_mean > 0 ~ "permutation-sensitive only",
      TRUE ~ "weak / negative"
    )
  ) %>%
  distinct(dataset, .keep_all = TRUE)

gnn_paths <- c(
  "results/edge_probe/urban_v4_semantic_washingtondc_gnn/edge_gnn_probe_runs.csv",
  "results/edge_probe/urban_v4_semantic_tokyo_gnn_light/edge_gnn_probe_runs.csv"
)

gnn_runs <- bind_rows(lapply(gnn_paths, read_if_exists)) %>%
  filter(condition %in% c("sage", "edge_sage", "edge_sage_perm")) %>%
  mutate(
    city = city_from_dataset(dataset),
    variant = variant_from_dataset(dataset),
    dataset_label = factor(label_from_dataset(dataset), levels = plot_order)
  )

gnn_delta <- gnn_runs %>%
  select(dataset, dataset_label, seed, condition, test_macro_f1, test_acc) %>%
  pivot_wider(names_from = condition, values_from = c(test_macro_f1, test_acc)) %>%
  filter(!is.na(test_macro_f1_sage), !is.na(test_macro_f1_edge_sage), !is.na(test_macro_f1_edge_sage_perm)) %>%
  transmute(
    dataset,
    dataset_label,
    seed,
    `EdgeSAGE - topology SAGE` = test_macro_f1_edge_sage - test_macro_f1_sage,
    `EdgeSAGE - permuted edge_attr` = test_macro_f1_edge_sage - test_macro_f1_edge_sage_perm
  ) %>%
  pivot_longer(
    cols = c(`EdgeSAGE - topology SAGE`, `EdgeSAGE - permuted edge_attr`),
    names_to = "comparison",
    values_to = "delta_macro_f1"
  )

gnn_delta_summary <- gnn_delta %>%
  group_by(dataset, dataset_label, comparison) %>%
  summarise(
    mean_delta = mean(delta_macro_f1, na.rm = TRUE),
    sd_delta = sd(delta_macro_f1, na.rm = TRUE),
    .groups = "drop"
  )

diag_paths <- c(
  "results/data_diagnostics/urban_v4_semantic_washingtondc/urban_v3bsjg_diagnostics.csv",
  "results/data_diagnostics/urban_v4_semantic_tokyo/urban_v3bsjg_diagnostics.csv",
  "results/data_diagnostics/urban_v4_semantic_paris/urban_v3bsjg_diagnostics.csv",
  "results/data_diagnostics/urban_v4_semantic_smoke/urban_v3bsjg_diagnostics.csv"
)

diagnostics <- bind_rows(lapply(diag_paths, read_if_exists)) %>%
  filter(dataset %in% selected_edge) %>%
  mutate(
    city = city_from_dataset(dataset),
    variant = variant_from_dataset(dataset),
    dataset_label = factor(label_from_dataset(dataset), levels = plot_order)
  ) %>%
  arrange(dataset, desc(!is.na(shared_building_bit_frac))) %>%
  distinct(dataset, .keep_all = TRUE)

limitation_data <- diagnostics %>%
  select(dataset, dataset_label, known_label_ratio, overall_same_label_ratio) %>%
  pivot_longer(
    cols = c(known_label_ratio, overall_same_label_ratio),
    names_to = "metric",
    values_to = "value"
  ) %>%
  mutate(
    metric = recode(
      metric,
      known_label_ratio = "Known-label ratio",
      overall_same_label_ratio = "Same-label edge ratio"
    )
  )

relation_data <- diagnostics %>%
  transmute(
    dataset,
    dataset_label,
    Street = street_edge_frac,
    Junction = junction_edge_frac,
    Building = ifelse(is.na(shared_building_bit_frac), 0, shared_building_bit_frac),
    Geometry = geom_edge_frac
  ) %>%
  pivot_longer(
    cols = c(Street, Junction, Building, Geometry),
    names_to = "relation",
    values_to = "prevalence"
  ) %>%
  mutate(relation = factor(relation, levels = c("Street", "Junction", "Building", "Geometry")))

write_csv(edge_summary, file.path(out_dir, "plot_data_edge_level.csv"))
write_csv(gnn_delta_summary, file.path(out_dir, "plot_data_gnn_delta.csv"))
write_csv(limitation_data, file.path(out_dir, "plot_data_label_limitations.csv"))
write_csv(relation_data, file.path(out_dir, "plot_data_relation_prevalence.csv"))

fmt_signed <- function(x) sprintf("%+.3f", x)
fmt_frac <- function(x) sprintf("%.2f", x)
break_dataset_label <- function(x) sub(" ", "\n", x)

city_cols <- c(
  WashingtonDC = "#2C7FB8",
  Tokyo = "#41AB5D",
  Paris = "#D95F0E"
)

p_edge <- ggplot(edge_summary, aes(
  x = auc_gain_vs_endpoint_mean,
  y = auc_drop_vs_perm_mean,
  color = city,
  shape = variant,
  label = as.character(dataset_label)
)) +
  annotate("rect", xmin = 0, xmax = Inf, ymin = 0, ymax = Inf, fill = "#74C476", alpha = 0.10) +
  geom_hline(yintercept = 0, linewidth = 0.35, linetype = "dashed", color = "grey40") +
  geom_vline(xintercept = 0, linewidth = 0.35, linetype = "dashed", color = "grey40") +
  geom_point(size = 3.2, alpha = 0.95) +
  geom_text_repel(size = 3.0, max.overlaps = 30, show.legend = FALSE, box.padding = 0.35) +
  scale_color_manual(values = city_cols) +
  scale_x_continuous(labels = fmt_signed) +
  scale_y_continuous(labels = fmt_signed) +
  labs(
    title = "A. Edge-level evidence",
    subtitle = "Positive x: edge_attr helps beyond endpoints.\nPositive y: the gain drops after permutation.",
    x = "AUC gain vs endpoint-only",
    y = "AUC drop vs permuted edge_attr",
    color = "City",
    shape = "Graph"
  ) +
  theme_minimal(base_size = 10) +
  theme(
    legend.position = "none",
    panel.grid.minor = element_blank(),
    plot.subtitle = element_text(size = 9.2),
    plot.title = element_text(size = 13)
  )

p_gnn <- ggplot(gnn_delta_summary, aes(
  x = dataset_label,
  y = mean_delta,
  fill = comparison
)) +
  geom_hline(yintercept = 0, linewidth = 0.35, linetype = "dashed", color = "grey35") +
  geom_col(position = position_dodge(width = 0.72), width = 0.64) +
  geom_errorbar(
    aes(ymin = mean_delta - sd_delta, ymax = mean_delta + sd_delta),
    position = position_dodge(width = 0.72),
    width = 0.18,
    linewidth = 0.35
  ) +
  scale_y_continuous(labels = fmt_signed) +
  scale_x_discrete(labels = break_dataset_label) +
  scale_fill_manual(values = c(
    "EdgeSAGE - topology SAGE" = "#756BB1",
    "EdgeSAGE - permuted edge_attr" = "#31A354"
  )) +
  labs(
    title = "B. Independent GNN probe",
    subtitle = "WashingtonDC is clearly positive.\nTokyo is positive but small.",
    x = NULL,
    y = "Test macro-F1 delta",
    fill = NULL
  ) +
  theme_minimal(base_size = 10) +
  theme(
    legend.position = "bottom",
    panel.grid.minor = element_blank(),
    axis.text.x = element_text(size = 8.5),
    plot.subtitle = element_text(size = 9.2),
    plot.title = element_text(size = 13)
  )

p_limits <- ggplot(limitation_data, aes(
  x = dataset_label,
  y = value,
  fill = metric
)) +
  geom_col(position = position_dodge(width = 0.72), width = 0.64) +
  scale_y_continuous(labels = fmt_frac, limits = c(0, 0.65)) +
  scale_x_discrete(labels = break_dataset_label) +
  scale_fill_manual(values = c(
    "Known-label ratio" = "#9ECAE1",
    "Same-label edge ratio" = "#FDD0A2"
  )) +
  labs(
    title = "C. Dataset limitations",
    subtitle = "Labels are sparse for the strongest city, and edge homophily is only moderate.",
    x = NULL,
    y = "Fraction",
    fill = NULL
  ) +
  theme_minimal(base_size = 10) +
  theme(
    legend.position = "bottom",
    panel.grid.minor = element_blank(),
    axis.text.x = element_text(size = 8.5),
    plot.subtitle = element_text(size = 9.2),
    plot.title = element_text(size = 13)
  )

p_rel <- ggplot(relation_data, aes(
  x = relation,
  y = dataset_label,
  fill = prevalence
)) +
  geom_tile(color = "white", linewidth = 0.7) +
  geom_text(aes(label = fmt_frac(prevalence)), size = 3.0) +
  scale_fill_gradient(low = "#F7FBFF", high = "#08519C", limits = c(0, 1)) +
  scale_y_discrete(labels = break_dataset_label) +
  labs(
    title = "D. Construction dependence",
    subtitle = "Signal depends on how raw street, junction, building, and geometry layers are projected.",
    x = NULL,
    y = NULL,
    fill = "Prevalence"
  ) +
  theme_minimal(base_size = 10) +
  theme(
    panel.grid = element_blank(),
    plot.subtitle = element_text(size = 9.2),
    plot.title = element_text(size = 13)
  )

combined <- (p_edge | p_gnn) / (p_limits | p_rel) +
  plot_annotation(
    title = "Urban V4 edge data: valid signal, but not a universal benchmark",
    subtitle = "Validity: edge_attr beats endpoint-only or permutation in selected graphs. Limitation: gains are city- and construction-dependent, and geometry can be a confound.",
    caption = "Rule of thumb: use WashingtonDC v4slj/v4sljb as Urban case studies; use Tokyo v4sljb as weak cross-city sanity; keep v4g as spatial-control, not semantic evidence."
  ) &
  theme(
    plot.title = element_text(size = 17),
    plot.subtitle = element_text(size = 10.5),
    plot.caption = element_text(size = 8.5)
  )

ggsave(
  filename = file.path(out_dir, "fig_urban_v4_validity_limitations.png"),
  plot = combined,
  width = 16,
  height = 11,
  dpi = 300,
  bg = "white"
)

ggsave(
  filename = file.path(out_dir, "fig_urban_v4_validity_limitations.pdf"),
  plot = combined,
  width = 16,
  height = 11,
  bg = "white"
)

message("Wrote: ", file.path(out_dir, "fig_urban_v4_validity_limitations.png"))
message("Wrote: ", file.path(out_dir, "fig_urban_v4_validity_limitations.pdf"))
